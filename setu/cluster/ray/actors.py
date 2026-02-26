"""
Ray actor definitions for Setu cluster components.

Defines CoordinatorActor and NodeAgentActor, which wrap the native
Coordinator and NodeAgent C++ classes as Ray actors for distributed
process management.
"""

import socket
import uuid
from contextlib import closing

import ray
import torch

from setu.logger import init_logger

logger = init_logger(__name__)

COORDINATOR_ACTOR_NAME = "setu_coordinator"
COORDINATOR_ACTOR_NAMESPACE = "setu"


def get_coordinator_actor():
    """Get the named CoordinatorActor handle. Returns None if not found."""
    try:
        return ray.get_actor(COORDINATOR_ACTOR_NAME, namespace=COORDINATOR_ACTOR_NAMESPACE)
    except ValueError:
        return None


def _find_free_port() -> int:
    """Find a free port on the current node using OS assignment."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("", 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


@ray.remote
class CoordinatorActor:
    """Ray actor wrapping the native Coordinator.

    Manages a single Coordinator instance per cluster. Binds to an
    OS-assigned port and exposes its endpoint for NodeAgentActors
    to connect to.
    """

    def __init__(self, metrics_endpoint: str = "", register_size: int = 0) -> None:
        self._coordinator = None
        self._port: int = 0
        self._ip_address: str = ""
        self._metrics_endpoint = metrics_endpoint
        self._register_size = register_size

    def start(self, passes=None) -> dict:
        """Start the Coordinator on an OS-assigned port.

        Args:
            passes: Optional list of pass name strings. ``None`` means
                default, ``[]`` means no passes.

        Returns:
            Dict with coordinator_endpoint and ip_address.
        """
        from setu._coordinator import Coordinator, NCCLBackend, PassManager, Planner
        from setu.cluster.passes import resolve_passes

        self._ip_address = ray.util.get_node_ip_address()

        self._port = _find_free_port()

        pass_manager = PassManager()
        for p in resolve_passes(passes):
            pass_manager.add_pass(p)

        if self._register_size > 0:
            from setu._coordinator import RegisterSet
            from setu._commons.datatypes import Device

            # Discover local devices and build per-device register sets
            register_sets = {}
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                dev = Device(torch_device=torch.device(f"cuda:{i}"))
                register_sets[dev] = RegisterSet.uniform(1, self._register_size)
            backend = NCCLBackend(register_sets)
        else:
            backend = NCCLBackend()

        planner = Planner(backend, pass_manager)
        self._coordinator = Coordinator(
            self._port, planner, self._metrics_endpoint
        )
        self._coordinator.start()

        endpoint = f"tcp://{self._ip_address}:{self._port}"
        logger.info(
            "CoordinatorActor started on %s",
            endpoint,
        )

        return {
            "coordinator_endpoint": endpoint,
            "ip_address": self._ip_address,
        }

    def stop(self) -> None:
        """Stop the Coordinator."""
        if self._coordinator is not None:
            self._coordinator.stop()
            self._coordinator = None
            logger.info("CoordinatorActor stopped")

    def is_alive(self) -> bool:
        """Check if the Coordinator is running."""
        return self._coordinator is not None


@ray.remote
class NodeAgentActor:
    """Ray actor wrapping the native NodeAgent.

    Manages a single NodeAgent instance per physical node. Auto-detects
    all CUDA GPUs on the node and creates Device objects for each.
    """

    def __init__(
        self,
        coordinator_endpoint: str,
        metrics_endpoint: str = "",
        register_size: int = 0,
    ) -> None:
        self._coordinator_endpoint = coordinator_endpoint
        self._metrics_endpoint = metrics_endpoint
        self._register_size = register_size
        self._node_agent = None
        self._port: int = 0
        self._ip_address: str = ""
        self._node_id = None
        self._num_gpus: int = 0

    def start(self) -> dict:
        """Start the NodeAgent with auto-detected GPUs.

        Returns:
            Dict with node_agent_endpoint, node_id, ip_address, num_gpus.
        """
        from setu._commons.datatypes import Device
        from setu._node_manager import NodeAgent

        self._ip_address = ray.util.get_node_ip_address()
        self._node_id = uuid.uuid4()

        self._port = _find_free_port()

        self._num_gpus = torch.cuda.device_count()
        devices = [
            Device(torch_device=torch.device(f"cuda:{i}"))
            for i in range(self._num_gpus)
        ]

        kwargs = dict(
            node_id=self._node_id,
            port=self._port,
            coordinator_endpoint=self._coordinator_endpoint,
            devices=devices,
            metrics_endpoint=self._metrics_endpoint,
        )
        if self._register_size > 0:
            kwargs["register_size"] = self._register_size
        self._node_agent = NodeAgent(**kwargs)
        self._node_agent.start()

        endpoint = f"tcp://{self._ip_address}:{self._port}"
        logger.info(
            "NodeAgentActor started on %s with %d GPUs (node_id=%s)",
            endpoint,
            self._num_gpus,
            self._node_id,
        )

        return {
            "node_agent_endpoint": endpoint,
            "node_id": str(self._node_id),
            "ip_address": self._ip_address,
            "num_gpus": self._num_gpus,
            "devices": devices,
            "ray_node_id": ray.get_runtime_context().get_node_id(),
        }

    def stop(self) -> None:
        """Stop the NodeAgent."""
        if self._node_agent is not None:
            self._node_agent.stop()
            self._node_agent = None
            logger.info("NodeAgentActor stopped (node_id=%s)", self._node_id)

    def is_alive(self) -> bool:
        """Check if the NodeAgent is running."""
        return self._node_agent is not None
