"""
Setu Ray integration for distributed cluster management.

Provides Ray-based orchestration for Coordinator and NodeAgent processes
across a pre-existing Ray cluster.
"""

from setu.cluster.info import ClusterInfo, NodeInfo
from setu.cluster.ray.cluster import Cluster

__all__ = [
    "Cluster",
    "ClusterInfo",
    "NodeInfo",
]
