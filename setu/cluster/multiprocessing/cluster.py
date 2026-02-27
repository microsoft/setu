"""Multiprocessing-based Setu cluster.

Spawns coordinator and node agents as child processes using
``torch.multiprocessing``.  All processes run on the same physical
machine, making this suitable for testing and single-node benchmarks.
"""

import logging
import socket
import time
import uuid
from contextlib import closing
from logging.handlers import QueueHandler, QueueListener
from typing import Dict, Optional, TypeVar

import torch.multiprocessing as mp

from setu._commons.datatypes import Device
from setu._coordinator import Participant
from setu.cluster.handle import ClientHandle
from setu.cluster.info import ClusterInfo, NodeInfo
from setu.cluster.multiprocessing.info import MultiprocessingClusterInfo
from setu.cluster.protocol import Cluster as ClusterProto
from setu.cluster.spec import ClusterSpec
from setu.telemetry.server import MetricsServer

T = TypeVar("T")


def _find_free_port() -> int:
    """Find a free port on the current machine using OS assignment."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def _redirect_native_stderr(log_dir: str, label: str) -> None:
    """Redirect fd 2 (stderr) to a file so C++ LOG_* output is captured.

    Must be called early in a child process, before any C++ code runs.
    """
    import os
    import sys

    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{label}_pid{os.getpid()}.log")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, 2)  # replace stderr fd
    os.close(fd)
    sys.stderr = os.fdopen(2, "w", buffering=1)


def _setup_child_logging(log_queue, log_dir: str = "", label: str = "") -> None:
    """Replace handlers on the 'setu' logger with a QueueHandler.

    Called at the start of every child process so that log records are
    forwarded to the parent through *log_queue* instead of being written
    to the child's (invisible) stderr.

    If *log_dir* is set, also redirects native (C++) stderr to a file.
    """
    if log_dir:
        _redirect_native_stderr(log_dir, label)

    root = logging.getLogger("setu")
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(QueueHandler(log_queue))
    root.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Process targets for coordinator and node agents
# ---------------------------------------------------------------------------


def _run_coordinator_process(
    spec: ClusterSpec,
    ready_event,
    stop_event,
    log_queue,
    log_dir="",
    metrics_endpoint="",
):
    """Coordinator process target. Builds Planner from spec."""
    _setup_child_logging(log_queue, log_dir=log_dir, label="coordinator")
    from setu._coordinator import (
        Coordinator,
        NCCLBackend,
        PassManager,
        Planner,
    )
    from setu.cluster.passes import resolve_passes

    pass_manager = PassManager()
    for p in resolve_passes(spec.passes, topology=spec.topology):
        pass_manager.add_pass(p)

    register_sets = {}
    for node_id, (_, device_specs) in spec.nodes.items():
        for ds in device_specs:
            if ds.register_set is not None:
                p = Participant(node_id, ds.device)
                register_sets[p] = ds.register_set

    backend = NCCLBackend(register_sets)

    planner = Planner(backend, pass_manager)
    coordinator = Coordinator(spec.coordinator_port, planner, metrics_endpoint)
    coordinator.start()
    ready_event.set()

    while not stop_event.is_set():
        time.sleep(0.05)

    coordinator.stop()


def _run_node_agent_process(
    node_id,
    port,
    coordinator_endpoint,
    devices,
    ready_event,
    stop_event,
    log_queue,
    log_dir="",
    metrics_endpoint="",
):
    """NodeAgent process target. Receives picklable Device objects directly."""
    _setup_child_logging(log_queue, log_dir=log_dir, label=f"node_agent_{port}")
    from setu._node_manager import NodeAgent

    agent = NodeAgent(
        node_id=node_id,
        port=port,
        coordinator_endpoint=coordinator_endpoint,
        devices=devices,
        metrics_endpoint=metrics_endpoint,
    )
    agent.start()
    ready_event.set()

    while not stop_event.is_set():
        time.sleep(0.05)

    agent.stop()


# ---------------------------------------------------------------------------
# Process-based ClientHandle
# ---------------------------------------------------------------------------


_ERROR_TAG = "__ERROR__"


def _warmup_cuda(device) -> None:
    """Force CUDA runtime init on *device* so later ops aren't penalised."""
    import torch

    if device.torch_device.type == "cuda":
        with torch.cuda.device(device.torch_device):
            torch.cuda.init()
            # Small alloc+free to fully warm the allocator.
            torch.empty(1, device=device.torch_device)


def _client_process_target(
    endpoint,
    participant,
    body,
    result_queue,
    stop_event,
):
    """Process target: create Client, run body, put result, wait for stop.

    The body is a plain function that runs to completion and returns a value.
    The return value (or an error tuple) is placed on *result_queue*.
    """
    import time
    import traceback

    from setu.logger import init_logger

    logger = init_logger(__name__)
    t_proc_start = time.monotonic()

    t0 = time.monotonic()
    _warmup_cuda(participant.device)
    logger.debug(
        "client_process_target: CUDA warmup took %.3fs (pid=%d, device=%s)",
        time.monotonic() - t0,
        __import__("os").getpid(),
        participant.device,
    )

    client = None
    try:
        from setu.client import Client

        t0 = time.monotonic()
        logger.debug("client_process_target: creating Client(%s)", endpoint)
        client = Client(endpoint)
        logger.debug(
            "client_process_target: client created in %.3fs, running body",
            time.monotonic() - t0,
        )
        result = body(client, participant)
        result_queue.put(result)
    except Exception:
        tb = traceback.format_exc()
        logger.error("client_process_target: failed:\n%s", tb)
        result_queue.put((_ERROR_TAG, tb))
    finally:
        if client is not None:
            t0 = time.monotonic()
            client.disconnect()
            logger.debug(
                "client_process_target: disconnect took %.3fs", time.monotonic() - t0
            )

    logger.debug(
        "client_process_target: body finished, total=%.3fs, waiting for stop_event",
        time.monotonic() - t_proc_start,
    )
    stop_event.wait()


class _ProcessClientHandle(ClientHandle[T]):
    """Handle wrapping a multiprocessing.Process."""

    def __init__(self, process, result_queue, stop_event) -> None:
        self._process = process
        self._result_queue = result_queue
        self._stop_event = stop_event

    def result(self, timeout: Optional[float] = None) -> T:
        if not self._process.is_alive() and self._result_queue.empty():
            raise RuntimeError(
                f"Client process died (exit code {self._process.exitcode}) "
                "with no result on the queue"
            )
        value = self._result_queue.get(timeout=timeout)
        if isinstance(value, tuple) and len(value) == 2 and value[0] == _ERROR_TAG:
            raise RuntimeError(f"Remote process error:\n{value[1]}")
        return value

    def stop(self) -> None:
        self._stop_event.set()
        self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.kill()


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------


class Cluster(ClusterProto):
    """Multiprocessing-based Setu cluster.

    Spawns a coordinator process and one node-agent process per entry in the
    ClusterSpec using ``torch.multiprocessing``.  All processes run on the
    same physical machine.  Use as a context manager for automatic cleanup.

    Validates that no two node agents own the same device.

    Example::

        from setu.cluster.multiprocessing import Cluster
        from setu.bench.runner import run_experiment

        with Cluster(spec) as cluster:
            result = run_experiment(
                cluster.cluster_info, src=src, dst=dst,
            )
    """

    def __init__(
        self,
        spec: ClusterSpec,
        startup_timeout: float = 10.0,
        settle_time: float = 0.5,
        log_dir: str = "",
        metrics_endpoint: str = "",
    ):
        self._validate_unique_devices(spec)
        self._spec = spec
        self._startup_timeout = startup_timeout
        self._settle_time = settle_time
        self._log_dir = log_dir
        self._metrics_endpoint = metrics_endpoint
        self._ctx = mp.get_context("spawn")
        self._stop_event = self._ctx.Event()
        self._processes: list = []
        self._cluster_info: Optional[ClusterInfo] = None
        self._metrics_server: Optional[MetricsServer] = None

        # Forward child-process log records to the parent's handlers.
        self._log_queue = self._ctx.Queue()
        parent_handlers = logging.getLogger("setu").handlers
        self._log_listener = QueueListener(
            self._log_queue, *parent_handlers, respect_handler_level=True
        )
        self._log_listener.start()

    @staticmethod
    def _validate_unique_devices(spec: ClusterSpec) -> None:
        """Ensure no device is claimed by more than one node agent."""
        seen: Dict[Device, uuid.UUID] = {}
        for node_id, (_, device_specs) in spec.nodes.items():
            for ds in device_specs:
                if ds.device in seen:
                    raise ValueError(
                        f"Device {ds.device} is owned by both node "
                        f"{seen[ds.device]} and node {node_id}"
                    )
                seen[ds.device] = node_id

    @property
    def spec(self) -> ClusterSpec:
        return self._spec

    @property
    def coordinator_endpoint(self) -> str:
        return self._spec.coordinator_endpoint

    def client_endpoint(self, node_id: uuid.UUID) -> str:
        return self._spec.client_endpoint(node_id)

    @property
    def cluster_info(self) -> Optional[ClusterInfo]:
        return self._cluster_info

    @property
    def metrics_server(self) -> Optional[MetricsServer]:
        return self._metrics_server

    def start(self) -> ClusterInfo:
        """Start coordinator and all node agents, build ClusterInfo."""
        # Start metrics server before child processes so the ZMQ endpoint is ready.
        # The server binds to the endpoint (may use "*"); child processes need a
        # connect-friendly endpoint with a real address.
        metrics_http_url = ""
        metrics_connect_endpoint = ""
        if self._metrics_endpoint:
            http_port = _find_free_port()
            self._metrics_server = MetricsServer(
                endpoint=self._metrics_endpoint,
                http_port=http_port,
            )
            self._metrics_server.start()
            metrics_http_url = f"http://localhost:{self._metrics_server.http_port}"
            metrics_connect_endpoint = self._metrics_endpoint.replace("*", "localhost")

        coordinator_ready = self._ctx.Event()
        coordinator_proc = self._ctx.Process(
            target=_run_coordinator_process,
            args=(
                self._spec,
                coordinator_ready,
                self._stop_event,
                self._log_queue,
                self._log_dir,
                metrics_connect_endpoint,
            ),
        )
        coordinator_proc.start()
        self._processes.append(coordinator_proc)
        assert coordinator_ready.wait(
            timeout=self._startup_timeout
        ), "Coordinator failed to start"

        nodes = []
        for node_id, (port, device_specs) in self._spec.nodes.items():
            devices = [ds.device for ds in device_specs]
            node_ready = self._ctx.Event()
            node_proc = self._ctx.Process(
                target=_run_node_agent_process,
                args=(
                    node_id,
                    port,
                    self._spec.coordinator_endpoint,
                    devices,
                    node_ready,
                    self._stop_event,
                    self._log_queue,
                    self._log_dir,
                    metrics_connect_endpoint,
                ),
            )
            node_proc.start()
            self._processes.append(node_proc)
            assert node_ready.wait(
                timeout=self._startup_timeout
            ), f"NodeAgent for {node_id} failed to start"

            nodes.append(
                NodeInfo(
                    node_id=str(node_id),
                    node_agent_endpoint=f"tcp://localhost:{port}",
                    devices=devices,
                )
            )

        time.sleep(self._settle_time)

        self._cluster_info = MultiprocessingClusterInfo(
            coordinator_endpoint=self._spec.coordinator_endpoint,
            nodes=nodes,
            metrics_endpoint=metrics_connect_endpoint,
            metrics_http_url=metrics_http_url,
        )
        return self._cluster_info

    def stop(self) -> None:
        """Signal stop, terminate, and join all processes."""
        self._stop_event.set()
        time.sleep(0.2)

        for proc in self._processes:
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()

        self._processes.clear()
        self._cluster_info = None
        self._log_listener.stop()

        if self._metrics_server is not None:
            self._metrics_server.stop()
            self._metrics_server = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
