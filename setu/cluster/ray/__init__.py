"""
Setu Ray integration for distributed cluster management.

Provides Ray-based orchestration for Coordinator and NodeAgent processes
across a pre-existing Ray cluster.
"""

from setu.cluster.ray.cluster import Cluster, ClusterInfo, NodeAgentInfo

__all__ = [
    "Cluster",
    "ClusterInfo",
    "NodeAgentInfo",
]
