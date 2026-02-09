"""
Setu Ray integration for distributed cluster management.

Provides Ray-based orchestration for Coordinator and NodeAgent processes
across a pre-existing Ray cluster.
"""

from setu.ray.cluster import ClusterInfo, NodeAgentInfo, SetuCluster

__all__ = [
    "SetuCluster",
    "ClusterInfo",
    "NodeAgentInfo",
]
