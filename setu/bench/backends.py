"""Pluggable backends for experiment client spawning and synchronization.

An ``ExperimentBackend`` encapsulates *how* client processes are created
and how barriers are set up — so that
:func:`setu.bench.runner.run_experiment` stays backend-agnostic.

Two implementations are provided:

* **RayBackend** — spawns Ray actors (the default for Ray clusters).
* **MultiprocessingBackend** — spawns ``mp.Process`` children (used by
  :class:`~setu.cluster.multiprocessing.Cluster`).
"""

from typing import Callable, List, Protocol, Tuple, TypeVar

from setu._coordinator import Participant
from setu.cluster.barrier import Barrier
from setu.cluster.handle import ClientHandle
from setu.cluster.info import ClusterInfo

T = TypeVar("T")


def backend_for(cluster_info: ClusterInfo) -> "ExperimentBackend":
    """Return the appropriate ExperimentBackend for a ClusterInfo subclass."""
    from setu.cluster.multiprocessing.info import MultiprocessingClusterInfo
    from setu.cluster.ray.info import RayClusterInfo

    if isinstance(cluster_info, RayClusterInfo):
        return RayBackend()
    if isinstance(cluster_info, MultiprocessingClusterInfo):
        return MultiprocessingBackend()
    raise TypeError(f"No experiment backend for {type(cluster_info).__name__}")


class ExperimentBackend(Protocol):
    """Protocol for experiment spawning backends."""

    def spawn_client(
        self,
        cluster_info: ClusterInfo,
        participant: Participant,
        body: Callable[..., T],
    ) -> ClientHandle[T]:
        """Spawn a Client connected to the node owning *participant*."""
        ...

    def create_barrier(self, num_clients: int) -> List[Barrier]:
        """Create barriers for *num_clients* SPMD participants."""
        ...


# ---------------------------------------------------------------------------
# Ray backend
# ---------------------------------------------------------------------------


class RayBackend:
    """Spawns clients as Ray actors with Ray actor-backed barriers."""

    def spawn_client(
        self,
        cluster_info: ClusterInfo,
        participant: Participant,
        body: Callable,
    ) -> ClientHandle:
        import ray
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        from setu.cluster.ray.cluster import _ClientActor, _RayClientHandle

        from setu.cluster.ray.info import RayNodeInfo

        node_info = cluster_info.node_info_for_participant(participant)
        assert isinstance(node_info, RayNodeInfo), (
            f"RayBackend requires RayNodeInfo, got {type(node_info).__name__}"
        )
        assert node_info.ray_node_id is not None, (
            f"RayNodeInfo for device {participant.device} has no ray_node_id"
        )

        scheduling = NodeAffinitySchedulingStrategy(
            node_id=node_info.ray_node_id,
            soft=False,
        )
        cuda_devices = ",".join(
            str(idx)
            for idx in sorted(d.torch_device.index for d in node_info.devices)
        )
        actor = _ClientActor.options(
            scheduling_strategy=scheduling,
        ).remote(node_info.node_agent_endpoint, cuda_devices)

        result_ref = actor.run.remote(body, participant)
        return _RayClientHandle(actor, result_ref)

    def create_barrier(self, num_clients: int) -> List[Barrier]:
        from setu.cluster.barrier import RayActorBarrier, create_ray_barrier_actor

        actor = create_ray_barrier_actor(num_clients)
        return [RayActorBarrier(actor, rank) for rank in range(num_clients)]


# ---------------------------------------------------------------------------
# Multiprocessing backend
# ---------------------------------------------------------------------------


class MultiprocessingBackend:
    """Spawns clients as child processes with ``mp.Barrier``-based sync.

    Used by :class:`~setu.cluster.multiprocessing.Cluster`.
    """

    def __init__(self) -> None:
        import torch.multiprocessing as mp

        self._ctx = mp.get_context("spawn")

    def spawn_client(
        self,
        cluster_info: ClusterInfo,
        participant: Participant,
        body: Callable,
    ) -> ClientHandle:
        from setu.cluster.multiprocessing.cluster import (
            _ProcessClientHandle,
            _client_process_target,
        )

        node_info = cluster_info.node_info_for_participant(participant)
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
            ),
        )
        proc.start()
        return _ProcessClientHandle(proc, result_queue, stop_event)

    def create_barrier(self, num_clients: int) -> List[Barrier]:
        from setu.cluster.barrier import MultiprocessingBarrier

        mp_barrier = self._ctx.Barrier(num_clients)
        shared = MultiprocessingBarrier(mp_barrier)
        return [shared] * num_clients
