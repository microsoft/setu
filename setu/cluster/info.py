"""Shared cluster topology types used by all backends."""

from dataclasses import dataclass
from typing import List, Optional

from setu._commons.datatypes import Device
from setu._coordinator import Participant


@dataclass(frozen=True)
class NodeInfo:
    """Information about a single node in the cluster."""

    node_id: str
    node_agent_endpoint: str
    devices: List[Device]
    ray_node_id: Optional[str] = None


@dataclass(frozen=True)
class ClusterInfo:
    """Describes a running Setu cluster regardless of backend."""

    coordinator_endpoint: str
    nodes: List[NodeInfo]

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def total_gpus(self) -> int:
        return sum(len(n.devices) for n in self.nodes)

    @property
    def node_agent_endpoints(self) -> List[str]:
        return [n.node_agent_endpoint for n in self.nodes]

    def node_for_device(self, device: Device) -> NodeInfo:
        """Find the NodeInfo that owns the given Device."""
        for node in self.nodes:
            if device in node.devices:
                return node
        raise ValueError(f"Device {device} not found in any node of the cluster")

    def endpoint_for_participant(self, participant: Participant) -> str:
        """Return the node-agent endpoint for a participant's node."""
        node_id_str = str(participant.node_id)
        for node in self.nodes:
            if node.node_id == node_id_str:
                return node.node_agent_endpoint
        raise ValueError(f"Participant node {participant.node_id} not found in cluster")
