"""
Cluster specification types for Setu.

Provides DeviceSpec and ClusterSpec, picklable dataclasses that describe
a Setu cluster topology using real C++ types.
"""

import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from setu._commons.datatypes import Device
from setu._coordinator import RegisterSet, Topology


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
