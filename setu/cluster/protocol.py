"""Cluster protocol that all backends implement."""

from typing import Optional, Protocol, runtime_checkable

from setu.cluster.info import ClusterInfo


@runtime_checkable
class Cluster(Protocol):
    """Protocol that all cluster backends implement.

    A cluster owns the lifecycle of a coordinator and node agents.
    Call ``start()`` to bring up infrastructure and ``stop()`` to tear
    it down.  Use ``cluster_info`` to get connection details for the
    running cluster.
    """

    def start(self) -> ClusterInfo: ...

    def stop(self) -> None: ...

    @property
    def cluster_info(self) -> Optional[ClusterInfo]: ...
