"""
Setu Ray integration for distributed cluster management.

Provides Ray-based orchestration for Coordinator and NodeAgent processes
across a pre-existing Ray cluster.
"""

from setu.cluster.ray.actors import (
    COORDINATOR_ACTOR_NAME,
    COORDINATOR_ACTOR_NAMESPACE,
    get_coordinator_actor,
)
from setu.cluster.ray.cluster import Cluster
from setu.cluster.ray.info import RayClusterInfo, RayNodeInfo

__all__ = [
    "COORDINATOR_ACTOR_NAME",
    "COORDINATOR_ACTOR_NAMESPACE",
    "Cluster",
    "RayClusterInfo",
    "RayNodeInfo",
    "get_coordinator_actor",
]
