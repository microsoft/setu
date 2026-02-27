"""Ray-specific cluster info types."""

from dataclasses import dataclass
from typing import Optional

from setu.cluster.info import ClusterInfo, NodeInfo


@dataclass(frozen=True)
class RayNodeInfo(NodeInfo):
    """NodeInfo with Ray scheduling metadata."""

    ray_node_id: Optional[str] = None


@dataclass(frozen=True)
class RayClusterInfo(ClusterInfo):
    """ClusterInfo with Ray-specific fields."""

    _node_info_cls = RayNodeInfo

    ray_address: Optional[str] = None

    def connect(self) -> None:
        """Connect to a pre-spawned Ray cluster."""
        import ray

        if not ray.is_initialized():
            assert self.ray_address is not None, (
                "RayClusterInfo has no ray_address — "
                "was it produced by an older version of setu-cluster?"
            )
            ray.init(address=self.ray_address)
