"""Shared cluster topology types used by all backends."""

import dataclasses
from dataclasses import dataclass
from typing import List

from setu._commons.datatypes import Device
from setu._coordinator import Participant

_BASE_NODE_FIELDS = ("node_id", "node_agent_endpoint", "devices")
_BASE_CLUSTER_FIELDS = ("coordinator_endpoint", "nodes", "metrics_endpoint", "metrics_http_url")


@dataclass(frozen=True)
class NodeInfo:
    """Information about a single node in the cluster."""

    node_id: str
    node_agent_endpoint: str
    devices: List[Device]


@dataclass(frozen=True)
class ClusterInfo:
    """Describes a running Setu cluster."""

    coordinator_endpoint: str
    nodes: List[NodeInfo]
    metrics_endpoint: str = ""
    metrics_http_url: str = ""

    # Subclasses should set this to their NodeInfo subclass.
    _node_info_cls = NodeInfo

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def total_gpus(self) -> int:
        return sum(len(n.devices) for n in self.nodes)

    @property
    def node_agent_endpoints(self) -> List[str]:
        return [n.node_agent_endpoint for n in self.nodes]

    def node_info_for_participant(self, participant: Participant) -> NodeInfo:
        """Find the NodeInfo for *participant*'s node."""
        node_id_str = str(participant.node_id)
        for node in self.nodes:
            if node.node_id == node_id_str:
                return node
        raise ValueError(f"Participant node {participant.node_id} not found in cluster")

    def to_yaml(self) -> str:
        """Serialize to YAML. Handles subclass fields automatically."""
        import yaml

        def _sort_key(n: NodeInfo):
            parts = n.node_agent_endpoint.split("//")
            host_port = parts[1] if len(parts) > 1 else parts[0]
            host, _, port_str = host_port.rpartition(":")
            try:
                port = int(port_str)
            except ValueError:
                port = 0
            return (host, port)

        def _node_to_dict(node: NodeInfo) -> dict:
            d = {
                "node_id": node.node_id,
                "node_agent_endpoint": node.node_agent_endpoint,
                "devices": [
                    {"type": dev.torch_device.type, "index": dev.torch_device.index}
                    for dev in node.devices
                ],
            }
            for f in dataclasses.fields(node):
                if f.name not in _BASE_NODE_FIELDS:
                    d[f.name] = getattr(node, f.name)
            return d

        data = {
            "_type": f"{type(self).__module__}.{type(self).__qualname__}",
            "coordinator_endpoint": self.coordinator_endpoint,
            "metrics_endpoint": self.metrics_endpoint,
            "metrics_http_url": self.metrics_http_url,
            "nodes": [_node_to_dict(n) for n in sorted(self.nodes, key=_sort_key)],
        }
        for f in dataclasses.fields(self):
            if f.name not in _BASE_CLUSTER_FIELDS:
                data[f.name] = getattr(self, f.name)

        return yaml.dump(data, default_flow_style=False)

    def connect(self) -> None:
        """Connect to a pre-spawned cluster. No-op for local clusters."""
        pass

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ClusterInfo":
        """Deserialize from YAML.

        When called on the base class (``ClusterInfo.from_yaml(...)``), the
        ``_type`` tag in the YAML is used to find the correct subclass.
        When called on a subclass directly, uses that subclass.
        """
        import importlib
        import torch
        import yaml

        data = yaml.safe_load(yaml_str)

        # Dispatch to the right subclass when called on the base class.
        if cls is ClusterInfo and "_type" in data:
            type_path = data["_type"]
            module_path, _, class_name = type_path.rpartition(".")
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)

        node_cls = cls._node_info_cls

        def _node_from_dict(d: dict) -> NodeInfo:
            base = {
                "node_id": d["node_id"],
                "node_agent_endpoint": d["node_agent_endpoint"],
                "devices": [
                    Device(torch_device=torch.device(dev["type"], dev.get("index")))
                    for dev in d["devices"]
                ],
            }
            extras = {
                f.name: d[f.name]
                for f in dataclasses.fields(node_cls)
                if f.name not in _BASE_NODE_FIELDS and f.name in d
            }
            return node_cls(**base, **extras)

        nodes = [_node_from_dict(n) for n in data["nodes"]]
        extras = {
            f.name: data[f.name]
            for f in dataclasses.fields(cls)
            if f.name not in _BASE_CLUSTER_FIELDS and f.name in data
        }
        return cls(
            coordinator_endpoint=data["coordinator_endpoint"],
            nodes=nodes,
            metrics_endpoint=data.get("metrics_endpoint", ""),
            metrics_http_url=data.get("metrics_http_url", ""),
            **extras,
        )
