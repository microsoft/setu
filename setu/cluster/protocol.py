"""Cluster protocol that all backends implement."""

from typing import Callable, Optional, Protocol, TypeVar, runtime_checkable

from setu._coordinator import Participant
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
