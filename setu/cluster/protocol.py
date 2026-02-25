"""Cluster protocol that all backends implement."""

from typing import Callable, List, Optional, Protocol, TypeVar, runtime_checkable

from setu._coordinator import Participant
from setu.cluster.barrier import Barrier
from setu.cluster.handle import ClientHandle
from setu.cluster.info import ClusterInfo

T = TypeVar("T")


@runtime_checkable
class Cluster(Protocol):
    """Protocol that all cluster backends implement.

    A cluster owns placement and client lifecycle.  Call ``spawn_client``
    to create a ``Client`` connected to the correct node, run an
    arbitrary body function, and get a handle back.
    """

    def start(self) -> ClusterInfo: ...

    def stop(self) -> None: ...

    @property
    def cluster_info(self) -> Optional[ClusterInfo]: ...

    def spawn_client(
        self,
        participant: Participant,
        body: Callable[..., T],
    ) -> ClientHandle[T]:
        """Spawn a Client connected to the node owning *participant*.

        The cluster:
        1. Creates a Client and connects it to the correct endpoint.
        2. Calls ``body(client, participant)``.
        3. Returns a handle -- the client stays alive until ``handle.stop()``.

        *body* may be a plain function or a generator.  If it is a
        generator, each yielded value and the final return value are
        available via ``handle.next_result()``.

        Use ``functools.partial`` to bind extra arguments into *body*.
        """
        ...

    def create_barrier(self, num_clients: int) -> List[Barrier]:
        """Create a synchronization barrier for SPMD client bodies.

        Returns *num_clients* :class:`Barrier` handles -- pass one to each
        client body via ``functools.partial`` so they can coordinate phases.
        """
        ...

    def add_hint(self, hint) -> None:
        """Send a routing hint to the coordinator."""
        ...

    def clear_hints(self) -> None:
        """Clear all routing hints from the coordinator."""
        ...
