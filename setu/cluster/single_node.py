"""Single-node Setu cluster with spawn_client support."""

import time
import uuid
from typing import Any, Callable, Dict, Optional, TypeVar

import torch.multiprocessing as mp

from setu._commons.datatypes import Device
from setu._coordinator import Participant
from setu.cluster.handle import ClientHandle
from setu.cluster.info import ClusterInfo, NodeInfo
from setu.cluster.protocol import Cluster
from setu.cluster.spec import ClusterSpec

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Process targets for coordinator and node agents
# ---------------------------------------------------------------------------


def _run_coordinator_process(spec: ClusterSpec, ready_event, stop_event):
    """Coordinator process target. Builds Planner from spec."""
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

    if register_sets:
        backend = NCCLBackend(register_sets)
    else:
        backend = NCCLBackend()

    planner = Planner(backend, pass_manager)
    coordinator = Coordinator(spec.coordinator_port, planner)
    coordinator.start()
    ready_event.set()

    while not stop_event.is_set():
        time.sleep(0.05)

    coordinator.stop()


def _run_node_agent_process(
    node_id, port, coordinator_endpoint, devices, ready_event, stop_event
):
    """NodeAgent process target. Receives picklable Device objects directly."""
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


def _client_process_target(
    endpoint, participant, body, args, kwargs, result_queue, stop_event
):
    """Process target: create Client, run body, put result, wait for stop."""
    from setu.client import Client

    client = Client(endpoint)
    try:
        result = body(client, participant, *args, **kwargs)
        result_queue.put(result)
        stop_event.wait()
    finally:
        client.disconnect()


class _ProcessClientHandle(ClientHandle[T]):
    """Handle wrapping a multiprocessing.Process."""

    def __init__(self, process, result_queue, stop_event) -> None:
        self._process = process
        self._result_queue = result_queue
        self._stop_event = stop_event

    def result(self, timeout: Optional[float] = None) -> T:
        return self._result_queue.get(timeout=timeout)

    def stop(self) -> None:
        self._stop_event.set()
        self._process.join(timeout=2)
        if self._process.is_alive():
            self._process.kill()


# ---------------------------------------------------------------------------
# SingleNodeCluster
# ---------------------------------------------------------------------------


class SingleNodeCluster(Cluster):
    """Manages a single-node Setu cluster for testing.

    All node agents run on the same physical machine. Spawns a coordinator
    process and one node-agent process per entry in the ClusterSpec.
    Use as a context manager for automatic cleanup.

    Validates that no two node agents own the same device.

    Example::

        with SingleNodeCluster(spec) as cluster:
            handle = cluster.spawn_client(participant, my_fn)
            result = handle.result()
            handle.stop()
    """

    def __init__(
        self,
        spec: ClusterSpec,
        startup_timeout: float = 10.0,
        settle_time: float = 0.5,
    ):
        self._validate_unique_devices(spec)
        self._spec = spec
        self._startup_timeout = startup_timeout
        self._settle_time = settle_time
        self._ctx = mp.get_context("spawn")
        self._stop_event = self._ctx.Event()
        self._processes: list = []
        self._cluster_info: Optional[ClusterInfo] = None

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
            args=(self._spec, coordinator_ready, self._stop_event),
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

        self._cluster_info = ClusterInfo(
            coordinator_endpoint=self._spec.coordinator_endpoint,
            nodes=nodes,
        )
        return self._cluster_info

    def spawn_client(
        self,
        participant: Participant,
        body: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> ClientHandle[T]:
        """Spawn a Client in a subprocess connected to the correct node.

        The subprocess creates a ``Client``, runs
        ``body(client, participant, *args, **kwargs)``, puts the result
        in a queue, then blocks until ``handle.stop()`` is called.
        """
        assert self._cluster_info is not None, "Cluster has not been started"

        node = self._cluster_info.node_for_device(participant.device)
        result_queue = self._ctx.Queue()
        stop_event = self._ctx.Event()

        proc = self._ctx.Process(
            target=_client_process_target,
            args=(
                node.node_agent_endpoint,
                participant,
                body,
                args,
                kwargs,
                result_queue,
                stop_event,
            ),
        )
        proc.start()
        return _ProcessClientHandle(proc, result_queue, stop_event)

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

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
