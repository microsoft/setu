"""
Setu Ray integration for distributed cluster management.

Provides Ray-based orchestration for Coordinator and NodeAgent processes
across a pre-existing Ray cluster.
"""

from setu.cluster import ClusterSpec, DeviceSpec
from setu.ray.__main__ import main as start_cluster_cli
from setu.ray.cluster import ClusterInfo, NodeAgentInfo, SetuCluster

__all__ = [
    "ClusterSpec",
    "DeviceSpec",
    "SetuCluster",
    "ClusterInfo",
    "NodeAgentInfo",
    "start_cluster_cli",
]
