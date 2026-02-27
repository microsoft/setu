"""Multiprocessing-specific cluster info types."""

from dataclasses import dataclass

from setu.cluster.info import ClusterInfo


@dataclass(frozen=True)
class MultiprocessingClusterInfo(ClusterInfo):
    """ClusterInfo for multiprocessing-based clusters."""
