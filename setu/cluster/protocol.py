"""Abstract Cluster interface that all backends implement."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, TypeVar

from setu._coordinator import Participant
from setu.cluster.handle import ClientHandle
from setu.cluster.info import ClusterInfo

T = TypeVar("T")


class Cluster(ABC):
    """Abstract base that all cluster backends implement.

    A cluster owns placement and client lifecycle.  Call ``spawn_client``
    to create a ``Client`` connected to the correct node, run an
    arbitrary body function, and get a handle back.
    """

    @abstractmethod
    def start(self) -> ClusterInfo: ...

    @abstractmethod
    def stop(self) -> None: ...

    @property
    @abstractmethod
    def cluster_info(self) -> Optional[ClusterInfo]: ...

    @abstractmethod
    def spawn_client(
        self,
        participant: Participant,
        body: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> ClientHandle[T]:
        """Spawn a Client connected to the node owning *participant*.

        The cluster:
        1. Creates a Client and connects it to the correct endpoint.
        2. Calls ``body(client, participant, *args, **kwargs)``.
        3. Returns a handle -- the client stays alive until ``handle.stop()``.
        """
        ...
