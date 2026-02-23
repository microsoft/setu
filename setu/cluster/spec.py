"""
Cluster specification types for Setu.

Provides DeviceSpec and ClusterSpec — picklable dataclasses that
describe a Setu cluster topology.
"""

import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from setu._commons.datatypes import Device
from setu._coordinator import Participant, RegisterSet, Topology


@dataclass
class DeviceSpec:
    """A device with optional register set configuration."""

    device: Device
    register_set: Optional[RegisterSet] = None


@dataclass
class ClusterSpec:
    """Setu cluster specification

    Attributes:
        coordinator_port: Port for the coordinator to bind on.
        nodes: Mapping from node_id to (port, device_specs).
        topology: Optional topology.
    """

    coordinator_port: int
    nodes: Dict[uuid.UUID, Tuple[int, List[DeviceSpec]]]
    topology: Optional[Topology] = None

    @property
    def coordinator_endpoint(self) -> str:
        return f"tcp://localhost:{self.coordinator_port}"

    def client_endpoint(self, node_id: uuid.UUID) -> str:
        port, _ = self.nodes[node_id]
        return f"tcp://localhost:{port}"

    def __post_init__(self) -> None:
        if self.topology is None:
            return

        topo_participants = set()
        for src, dst, _ in self.topology.get_edges():
            topo_participants.add(src)
            topo_participants.add(dst)

        for node_id, (_, device_specs) in self.nodes.items():
            for ds in device_specs:
                p = Participant(node_id, ds.device)
                if p not in topo_participants:
                    raise ValueError(
                        f"Participant {p} from cluster spec is missing "
                        f"from the topology"
                    )

    def with_topology(self, topology: Topology) -> "ClusterSpec":
        """Return a new ClusterSpec with the given topology attached."""
        return ClusterSpec(
            coordinator_port=self.coordinator_port,
            nodes=self.nodes,
            topology=topology,
        )
