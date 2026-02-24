"""Single-node Setu cluster with spawn_client support."""

import logging
import time
import uuid
from logging.handlers import QueueHandler, QueueListener
from typing import Callable, Dict, List, Optional, TypeVar

import torch.multiprocessing as mp

from setu._commons.datatypes import Device
from setu._coordinator import Participant
from setu.cluster.barrier import Barrier, MultiprocessingBarrier
from setu.cluster.handle import ClientHandle
from setu.cluster.info import ClusterInfo, NodeInfo
from setu.cluster.protocol import Cluster as ClusterProto
from setu.cluster.spec import ClusterSpec

T = TypeVar("T")


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


def _run_coordinator_process(spec: ClusterSpec, ready_event, stop_event, log_queue,
                             log_dir=""):
    """Coordinator process target. Builds Planner from spec."""
    _setup_child_logging(log_queue, log_dir=log_dir, label="coordinator")
    from setu._coordinator import (
        Coordinator,
        NCCLBackend,
        PassManager,
        Planner,
        ShortestPathRouting,
    )

    pass_manager = PassManager()
    if spec.topology is not None:
        pass_manager.add_pass(ShortestPathRouting(spec.topology))

    register_sets = {}
    for node_id, (_, device_specs) in spec.nodes.items():
        for ds in device_specs:
            if ds.register_set is not None:
                p = Participant(node_id, ds.device)
                register_sets[p] = ds.register_set

    backend = NCCLBackend(register_sets)

    planner = Planner(backend, pass_manager)
    coordinator = Coordinator(spec.coordinator_port, planner)
    coordinator.start()
    ready_event.set()

    while not stop_event.is_set():
        time.sleep(0.05)

    coordinator.stop()


def _run_node_agent_process(
    node_id, port, coordinator_endpoint, devices, ready_event, stop_event, log_queue,
    log_dir=""
):
    """NodeAgent process target. Receives picklable Device objects directly."""
    _setup_child_logging(log_queue, log_dir=log_dir, label=f"node_agent_{port}")
    from setu._node_manager import NodeAgent

    agent = NodeAgent(
        node_id=node_id,
        port=port,
        coordinator_endpoint=coordinator_endpoint,
        devices=devices,
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
    endpoint, participant, body, result_queue, stop_event, log_queue,
    log_dir=""
):
    """Process target: create Client, run body, put result, wait for stop.

    The body is a plain function that runs to completion and returns a value.
    The return value (or an error tuple) is placed on *result_queue*.
    """
    _setup_child_logging(
        log_queue, log_dir=log_dir,
        label=f"client_{participant.device}".replace(":", "_"),
    )
    import time
    import traceback

    from setu.logger import init_logger

    logger = init_logger(__name__)
    t_proc_start = time.monotonic()

    t0 = time.monotonic()
    _warmup_cuda(participant.device)
    logger.debug(
        "client_process_target: CUDA warmup took %.3fs (pid=%d, device=%s)",
        time.monotonic() - t0, __import__("os").getpid(), participant.device,
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
# SingleNodeCluster
# ---------------------------------------------------------------------------


class SingleNodeCluster(ClusterProto):
    """Manages a single-node Setu cluster for testing.

    All node agents run on the same physical machine. Spawns a coordinator
    process and one node-agent process per entry in the ClusterSpec.
    Use as a context manager for automatic cleanup.

    Validates that no two node agents own the same device.

    Example::

        from functools import partial

        with SingleNodeCluster(spec) as cluster:
            body = partial(my_fn, extra_arg=value)
            handle = cluster.spawn_client(participant, body)
            result = handle.result()
            handle.stop()
    """

    def __init__(
        self,
        spec: ClusterSpec,
        startup_timeout: float = 10.0,
        settle_time: float = 0.5,
        log_dir: str = "",
    ):
        self._validate_unique_devices(spec)
        self._spec = spec
        self._startup_timeout = startup_timeout
        self._settle_time = settle_time
        self._log_dir = log_dir
        self._ctx = mp.get_context("spawn")
        self._stop_event = self._ctx.Event()
        self._processes: list = []
        self._cluster_info: Optional[ClusterInfo] = None

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
    def mp_context(self):
        return self._ctx

    @property
    def cluster_info(self) -> Optional[ClusterInfo]:
        return self._cluster_info

    def start(self) -> ClusterInfo:
        """Start coordinator and all node agents, build ClusterInfo."""
        coordinator_ready = self._ctx.Event()
        coordinator_proc = self._ctx.Process(
            target=_run_coordinator_process,
            args=(self._spec, coordinator_ready, self._stop_event, self._log_queue,
                  self._log_dir),
        )
        coordinator_proc.start()
        self._processes.append(coordinator_proc)
        assert coordinator_ready.wait(timeout=self._startup_timeout), (
            "Coordinator failed to start"
        )

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
                ),
            )
            node_proc.start()
            self._processes.append(node_proc)
            assert node_ready.wait(timeout=self._startup_timeout), (
                f"NodeAgent for {node_id} failed to start"
            )

            nodes.append(
                NodeInfo(
                    node_id=str(node_id),
                    node_agent_endpoint=f"tcp://localhost:{port}",
                    devices=devices,
                )
            )

        time.sleep(self._settle_time)

        self._cluster_info = ClusterInfo(
            coordinator_endpoint=self._spec.coordinator_endpoint,
            nodes=nodes,
        )
        return self._cluster_info

    def spawn_client(
        self,
        participant: Participant,
        body: Callable[..., T],
    ) -> ClientHandle[T]:
        """Spawn a Client in a subprocess connected to the correct node.

        The subprocess creates a ``Client``, runs
        ``body(client, participant)``, puts the result in a queue, then
        blocks until ``handle.stop()`` is called.

        Use ``functools.partial`` to bind extra arguments into *body*.
        """
        assert self._cluster_info is not None, "Cluster has not been started"

        node_info = self._cluster_info.node_info_for_participant(participant)
        result_queue = self._ctx.Queue()
        stop_event = self._ctx.Event()

        proc = self._ctx.Process(
            target=_client_process_target,
            args=(
                node_info.node_agent_endpoint,
                participant,
                body,
                result_queue,
                stop_event,
                self._log_queue,
                self._log_dir,
            ),
        )
        proc.start()
        return _ProcessClientHandle(proc, result_queue, stop_event)

    def create_barrier(self, num_clients: int) -> List[Barrier]:
        """Create a shared-memory barrier for *num_clients* SPMD participants.

        All returned handles wrap the same ``mp.Barrier`` object.
        """
        mp_barrier = self._ctx.Barrier(num_clients)
        shared = MultiprocessingBarrier(mp_barrier)
        return [shared] * num_clients

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

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
