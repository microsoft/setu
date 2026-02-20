"""
Cluster abstraction for E2E tests.

Provides SetuTestCluster which spins up coordinator + node agents from a
ClusterSpec for integration testing.
"""

import time
import uuid

import torch.multiprocessing as mp

from setu.cluster import ClusterSpec


def _run_coordinator_process(spec, ready_event, stop_event):
    """Coordinator process target. Builds Planner from spec."""
    from setu._coordinator import (
        Coordinator,
        NCCLBackend,
        Participant,
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


class SetuTestCluster:
    """Manages a Setu cluster for testing.

    Spawns coordinator and node agent processes from a ClusterSpec.
    Use as a context manager for automatic cleanup.

    Example::

        with SetuTestCluster(spec) as cluster:
            ctx = cluster.mp_context
            # spawn client processes...
    """

    def __init__(
        self,
        spec: ClusterSpec,
        startup_timeout: float = 10.0,
        settle_time: float = 0.5,
    ):
        self._spec = spec
        self._startup_timeout = startup_timeout
        self._settle_time = settle_time
        self._ctx = mp.get_context("spawn")
        self._stop_event = self._ctx.Event()
        self._processes: list = []

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

    def start(self):
        """Start coordinator and all node agents."""
        # Start coordinator
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

        # Start node agents
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

        time.sleep(self._settle_time)

    def stop(self):
        """Signal stop, terminate, and join all processes."""
        self._stop_event.set()
        time.sleep(0.2)

        for proc in self._processes:
            proc.terminate()
            proc.join(timeout=2)
            if proc.is_alive():
                proc.kill()

        self._processes.clear()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
