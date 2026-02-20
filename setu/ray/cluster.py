"""
Main orchestration for Setu on Ray.

Provides SetuCluster which manages the lifecycle of Coordinator and
NodeAgent actors across a Ray cluster, along with ClusterInfo and
NodeAgentInfo data classes for describing cluster topology.
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from setu.logger import init_logger
from setu.ray.actors import CoordinatorActor, NodeAgentActor

logger = init_logger(__name__)

# Timeout in seconds for actor creation and start calls.
_ACTOR_TIMEOUT_S = 60


@dataclass(frozen=True)
class NodeAgentInfo:
    """Information about a running NodeAgent in the cluster."""

    node_id: str
    ip_address: str
    node_agent_endpoint: str
    num_gpus: int


@dataclass(frozen=True)
class ClusterInfo:
    """Information about the running Setu cluster."""

    coordinator_endpoint: str
    node_agents: List[NodeAgentInfo]

    @property
    def node_agent_endpoints(self) -> List[str]:
        """All NodeAgent endpoints in the cluster."""
        return [na.node_agent_endpoint for na in self.node_agents]

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the cluster."""
        return len(self.node_agents)

    @property
    def total_gpus(self) -> int:
        """Total number of GPUs across all nodes."""
        return sum(na.num_gpus for na in self.node_agents)


def _discover_ray_nodes() -> List[Dict]:
    """Discover alive nodes in the Ray cluster.

    Returns:
        List of dicts with node_id, ip, num_gpus for each unique node.
    """
    nodes = ray.nodes()
    seen_ips = set()
    result = []

    for node in nodes:
        if not node.get("Alive", False):
            continue

        ip = node.get("NodeManagerAddress", "")
        if not ip or ip in seen_ips:
            continue
        seen_ips.add(ip)

        resources = node.get("Resources", {})
        num_gpus = int(resources.get("GPU", 0))

        result.append(
            {
                "ray_node_id": node["NodeID"],
                "ip": ip,
                "num_gpus": num_gpus,
            }
        )

    logger.info(
        "Discovered %d Ray node(s): %s",
        len(result),
        ", ".join(f"{n['ip']} ({n['num_gpus']} GPUs)" for n in result),
    )
    return result


class SetuCluster:
    """Manages the lifecycle of Setu components on a Ray cluster.

    Creates one CoordinatorActor (cluster-wide) and one NodeAgentActor
    per physical node. The coordinator node also runs a NodeAgentActor.

    Usage::

        cluster = SetuCluster()
        info = cluster.start()
        # info.coordinator_endpoint, info.node_agent_endpoints, etc.
        cluster.stop()
    """

    def __init__(self, env_vars: Optional[Dict[str, str]] = None) -> None:
        self._coordinator_actor: Optional[ray.actor.ActorHandle] = None
        self._node_agent_actors: List[ray.actor.ActorHandle] = []
        self._cluster_info: Optional[ClusterInfo] = None
        self._started: bool = False
        self._env_vars = env_vars

    @property
    def cluster_info(self) -> Optional[ClusterInfo]:
        """Returns the ClusterInfo if the cluster has been started."""
        return self._cluster_info

    def start(self) -> ClusterInfo:
        """Start the Setu cluster on Ray.

        Discovers all Ray nodes, places a CoordinatorActor on the node
        with the most GPUs, then places a NodeAgentActor on every node
        (including the coordinator node).

        Returns:
            ClusterInfo describing the running cluster.
        """
        if self._started:
            raise RuntimeError("SetuCluster is already started")

        ray_nodes = _discover_ray_nodes()
        if not ray_nodes:
            raise RuntimeError("No alive Ray nodes found")

        coordinator_node = random.choice(ray_nodes)
        logger.info(
            "Selected coordinator node: %s (%d GPUs)",
            coordinator_node["ip"],
            coordinator_node["num_gpus"],
        )

        # Start CoordinatorActor on chosen node (0 GPUs)
        coordinator_scheduling = NodeAffinitySchedulingStrategy(
            node_id=coordinator_node["ray_node_id"],
            soft=False,
        )
        coordinator_options: Dict = {
            "num_gpus": 0,
            "scheduling_strategy": coordinator_scheduling,
        }
        if self._env_vars:
            coordinator_options["runtime_env"] = {"env_vars": self._env_vars}
        self._coordinator_actor = CoordinatorActor.options(
            **coordinator_options,
        ).remote()

        coordinator_result = ray.get(self._coordinator_actor.start.remote())
        coordinator_endpoint = coordinator_result["coordinator_endpoint"]
        logger.info("Coordinator started at %s", coordinator_endpoint)

        # Create NodeAgentActors on every node (including coordinator node)
        for node in ray_nodes:
            node_scheduling = NodeAffinitySchedulingStrategy(
                node_id=node["ray_node_id"],
                soft=False,
            )
            node_options: Dict = {
                "num_gpus": node["num_gpus"],
                "scheduling_strategy": node_scheduling,
            }
            if self._env_vars:
                node_options["runtime_env"] = {"env_vars": self._env_vars}
            actor = NodeAgentActor.options(
                **node_options,
            ).remote(coordinator_endpoint)
            self._node_agent_actors.append(actor)

        # Start all NodeAgentActors in parallel
        start_futures = [actor.start.remote() for actor in self._node_agent_actors]
        try:
            node_agent_results = ray.get(
                start_futures,
                timeout=_ACTOR_TIMEOUT_S,
            )
        except ray.exceptions.GetTimeoutError:
            self._kill_all_actors()
            raise RuntimeError(
                "Timed out starting NodeAgentActors — GPUs may be held by "
                "another cluster. Stop existing clusters before starting a "
                "new one."
            )

        # Build ClusterInfo
        node_agents = [
            NodeAgentInfo(
                node_id=result["node_id"],
                ip_address=result["ip_address"],
                node_agent_endpoint=result["node_agent_endpoint"],
                num_gpus=result["num_gpus"],
            )
            for result in node_agent_results
        ]

        self._cluster_info = ClusterInfo(
            coordinator_endpoint=coordinator_endpoint,
            node_agents=node_agents,
        )
        self._started = True

        logger.info(
            "Setu cluster started: %d node(s), %d total GPU(s), " "coordinator at %s",
            self._cluster_info.num_nodes,
            self._cluster_info.total_gpus,
            coordinator_endpoint,
        )
        return self._cluster_info

    def _kill_all_actors(self) -> None:
        """Force-kill all actors and reset state. Used for cleanup on failure."""
        for actor in self._node_agent_actors:
            ray.kill(actor)
        if self._coordinator_actor is not None:
            ray.kill(self._coordinator_actor)

        self._node_agent_actors = []
        self._coordinator_actor = None
        self._cluster_info = None
        self._started = False

        logger.info("Killed all actors during cleanup")

    def stop(self) -> None:
        """Stop the Setu cluster.

        Stops all NodeAgentActors first (closing ZMQ connections to the
        Coordinator), then stops the CoordinatorActor. Actor handles are
        dropped so Ray can garbage-collect the processes and run C++
        destructors, which close ZMQ sockets and contexts.
        """
        if not self._started:
            return

        # Stop NodeAgentActors in parallel
        if self._node_agent_actors:
            stop_futures = [actor.stop.remote() for actor in self._node_agent_actors]
            ray.get(stop_futures)
            logger.info("All NodeAgentActors stopped")

        # Stop CoordinatorActor
        if self._coordinator_actor is not None:
            ray.get(self._coordinator_actor.stop.remote())
            logger.info("CoordinatorActor stopped")

        # Drop actor handles — Ray will GC the processes and run destructors,
        # which close ZMQ sockets and contexts cleanly.
        self._node_agent_actors = []
        self._coordinator_actor = None
        self._cluster_info = None
        self._started = False

        logger.info("Setu cluster fully shut down")

    def add_hint(self, hint) -> None:
        """Add a compiler hint to the Coordinator.

        Args:
            hint: A compiler hint (e.g. RoutingHint).
        """
        if self._coordinator_actor is None:
            raise RuntimeError("SetuCluster is not started")
        ray.get(self._coordinator_actor.add_hint.remote(hint))

    def clear_hints(self) -> None:
        """Clear all compiler hints from the Coordinator."""
        if self._coordinator_actor is None:
            raise RuntimeError("SetuCluster is not started")
        ray.get(self._coordinator_actor.clear_hints.remote())
