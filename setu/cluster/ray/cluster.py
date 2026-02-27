"""
Main orchestration for Setu on Ray.

Provides Cluster which manages the lifecycle of Coordinator and
NodeAgent actors across a Ray cluster.
"""

import random
import socket
from contextlib import closing
from typing import Callable, Dict, List, Optional, TypeVar

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from setu._coordinator import Participant
from setu.cluster.handle import ClientHandle
from setu.cluster.info import ClusterInfo
from setu.cluster.protocol import Cluster as ClusterProto
from setu.cluster.ray.actors import (
    COORDINATOR_ACTOR_NAME,
    COORDINATOR_ACTOR_NAMESPACE,
    CoordinatorActor,
    NodeAgentActor,
)
from setu.cluster.ray.info import RayClusterInfo, RayNodeInfo
from setu.logger import init_logger
from setu.telemetry.server import MetricsServer

logger = init_logger(__name__)

T = TypeVar("T")

# Timeout in seconds for actor creation and start calls.
_ACTOR_TIMEOUT_S = 60


def _find_free_port() -> int:
    """Find a free port on the current machine using OS assignment."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


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


# ---------------------------------------------------------------------------
# Ray client actor (used by spawn_client)
# ---------------------------------------------------------------------------


@ray.remote(num_gpus=0)
class _ClientActor:
    """Ray actor that creates a Client, runs a body function, and returns the result.

    The body is a plain function: ``body(client, participant) -> T``.
    Uses num_gpus=0 to bypass Ray GPU scheduling.  The caller passes
    *cuda_visible_devices* which is set in ``os.environ`` before any
    CUDA code runs, making all node GPUs visible to this process.
    """

    def __init__(self, endpoint: str, cuda_visible_devices: str) -> None:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        self._endpoint = endpoint
        self._client = None

    def run(self, body: Callable, participant: Participant):
        """Connect, run body(client, participant), return the result."""
        from setu.client import Client

        self._client = Client(self._endpoint)
        return body(self._client, participant)

    def stop(self) -> None:
        if self._client is not None:
            self._client.disconnect()
            self._client = None


class _RayClientHandle(ClientHandle[T]):
    """Handle wrapping a Ray actor."""

    def __init__(self, actor, result_ref) -> None:
        self._actor = actor
        self._result_ref = result_ref

    def result(self, timeout: Optional[float] = None) -> T:
        import queue as _queue

        try:
            return ray.get(self._result_ref, timeout=timeout)
        except ray.exceptions.GetTimeoutError:
            raise _queue.Empty("ray.get timed out") from None

    def stop(self) -> None:
        try:
            ray.get(self._actor.stop.remote(), timeout=10)
        except Exception:
            pass
        ray.kill(self._actor, no_restart=True)


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------


class Cluster(ClusterProto):
    """Manages the lifecycle of Setu components on a Ray cluster.

    Creates one CoordinatorActor (cluster-wide) and one NodeAgentActor
    per physical node. The coordinator node also runs a NodeAgentActor.

    Usage::

        cluster = Cluster()
        info = cluster.start()
        # info.coordinator_endpoint, info.node_agent_endpoints, etc.
        cluster.stop()
    """

    def __init__(
        self,
        env_vars: Optional[Dict[str, str]] = None,
        passes: Optional[List[str]] = None,
        metrics_endpoint: str = "",
        register_size: int = 0,
    ) -> None:
        self._coordinator_actor: Optional[ray.actor.ActorHandle] = None
        self._node_agent_actors: List[ray.actor.ActorHandle] = []
        self._cluster_info: Optional[ClusterInfo] = None
        self._started: bool = False
        self._env_vars = env_vars
        self._passes = passes
        self._metrics_endpoint = metrics_endpoint
        self._metrics_server: Optional[MetricsServer] = None
        self._register_size = register_size

    @property
    def cluster_info(self) -> Optional[ClusterInfo]:
        """Returns the ClusterInfo if the cluster has been started."""
        return self._cluster_info

    @property
    def metrics_server(self) -> Optional[MetricsServer]:
        return self._metrics_server

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

        # Start metrics server before actors so the ZMQ endpoint is ready.
        # The server binds to the endpoint (may use "*"); actors need a
        # connect-friendly endpoint with a real IP.
        metrics_http_url = ""
        metrics_connect_endpoint = ""
        if self._metrics_endpoint:
            http_port = _find_free_port()
            self._metrics_server = MetricsServer(
                endpoint=self._metrics_endpoint,
                http_port=http_port,
            )
            self._metrics_server.start()
            metrics_http_url = f"http://localhost:{self._metrics_server.http_port}"

            # Replace "*" with the node's actual IP so remote actors can connect.
            head_ip = ray.util.get_node_ip_address()
            metrics_connect_endpoint = self._metrics_endpoint.replace("*", head_ip)

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
            name=COORDINATOR_ACTOR_NAME,
            namespace=COORDINATOR_ACTOR_NAMESPACE,
            **coordinator_options,
        ).remote(
            metrics_endpoint=metrics_connect_endpoint,
        )

        coordinator_result = ray.get(
            self._coordinator_actor.start.remote(passes=self._passes)
        )
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
            ).remote(
                coordinator_endpoint,
                metrics_endpoint=metrics_connect_endpoint,
                register_size=self._register_size,
            )
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

        # Build RayClusterInfo with Ray-specific scheduling metadata.
        nodes = [
            RayNodeInfo(
                node_id=result["node_id"],
                node_agent_endpoint=result["node_agent_endpoint"],
                devices=result["devices"],
                ray_node_id=result["ray_node_id"],
            )
            for result in node_agent_results
        ]

        # Record the Ray address so external processes (e.g. bench_setu)
        # can connect to the same Ray cluster.
        ray_ctx = ray.get_runtime_context()
        ray_address = ray_ctx.gcs_address if hasattr(ray_ctx, "gcs_address") else None

        self._cluster_info = RayClusterInfo(
            coordinator_endpoint=coordinator_endpoint,
            nodes=nodes,
            metrics_endpoint=metrics_connect_endpoint,
            metrics_http_url=metrics_http_url,
            ray_address=ray_address,
        )
        self._started = True

        logger.info(
            "Setu cluster started: %d node(s), %d total GPU(s), coordinator at %s",
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

        Attempts graceful stop with a timeout, then force-kills all actors.
        """
        if not self._started:
            return

        _STOP_TIMEOUT_S = 5

        # Try graceful stop of NodeAgentActors
        if self._node_agent_actors:
            stop_futures = [actor.stop.remote() for actor in self._node_agent_actors]
            try:
                ray.get(stop_futures, timeout=_STOP_TIMEOUT_S)
                logger.info("All NodeAgentActors stopped gracefully")
            except Exception:
                logger.warning(
                    "NodeAgentActors did not stop within %ds, force-killing",
                    _STOP_TIMEOUT_S,
                )

        # Try graceful stop of CoordinatorActor
        if self._coordinator_actor is not None:
            try:
                ray.get(
                    self._coordinator_actor.stop.remote(),
                    timeout=_STOP_TIMEOUT_S,
                )
                logger.info("CoordinatorActor stopped gracefully")
            except Exception:
                logger.warning(
                    "CoordinatorActor did not stop within %ds, force-killing",
                    _STOP_TIMEOUT_S,
                )

        # Force-kill everything to ensure cleanup
        self._kill_all_actors()
        self._started = False

        if self._metrics_server is not None:
            self._metrics_server.stop()
            self._metrics_server = None

        logger.info("Setu cluster fully shut down")
