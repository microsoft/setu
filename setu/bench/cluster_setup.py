"""Shared cluster setup helpers for Setu benchmarking.

Extracted from ``setu.bench.__main__`` so that experiment scripts can
import them without pulling in the CLI argument parser.
"""

import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from setu._commons.datatypes import TensorDim
from setu._coordinator import Participant
from setu.bench.helpers import ShardedTensor
from setu.cluster.info import ClusterInfo
from setu.cluster.mesh import Mesh, P


def parse_device_spec(spec: str) -> List[Tuple[int, Optional[int]]]:
    """Parse a single device spec string into (node_idx, device_idx) pairs.

    Supported forms:
        "0"       → all devices on node 0 (device_idx=None)
        "0:2"     → node 0, device 2
        "0:1-3"   → node 0, devices 1, 2, 3
        "0:1,4,5" → node 0, devices 1, 4, 5
    """
    if ":" not in spec:
        return [(int(spec), None)]

    node_str, dev_str = spec.split(":", 1)
    node_idx = int(node_str)

    if "-" in dev_str:
        start, end = dev_str.split("-", 1)
        dev_indices = list(range(int(start), int(end) + 1))
    elif "," in dev_str:
        dev_indices = [int(d) for d in dev_str.split(",")]
    else:
        dev_indices = [int(dev_str)]

    return [(node_idx, d) for d in dev_indices]


def resolve_device_specs(
    specs: List[str],
    cluster_info: ClusterInfo,
) -> List[Participant]:
    """Resolve device spec strings into Participant objects."""
    participants = []
    for spec in specs:
        for node_idx, dev_idx in parse_device_spec(spec):
            assert (
                0 <= node_idx < len(cluster_info.nodes)
            ), f"Node index {node_idx} out of range [0, {len(cluster_info.nodes)})"
            node = cluster_info.nodes[node_idx]
            node_id = uuid.UUID(node.node_id)

            if dev_idx is None:
                for dev in node.devices:
                    participants.append(Participant(node_id, dev))
            else:
                assert 0 <= dev_idx < len(node.devices), (
                    f"Device index {dev_idx} out of range "
                    f"[0, {len(node.devices)}) on node {node_idx}"
                )
                participants.append(Participant(node_id, node.devices[dev_idx]))
    return participants


def build_sharded_tensor(
    name: str,
    cluster_info: ClusterInfo,
    nbytes: int,
    specs: Optional[List[str]] = None,
    dtype: torch.dtype = torch.float32,
) -> ShardedTensor:
    """Build a 1-D sharded tensor.

    Args:
        name: Tensor name.
        cluster_info: Running cluster description.
        nbytes: Total tensor size in bytes.
        specs: Device spec strings.  Defaults to all devices on node 0.
        dtype: Element type (default: float32).
    """
    if specs is not None:
        participants = resolve_device_specs(specs, cluster_info)
    else:
        node = cluster_info.nodes[0]
        node_id = uuid.UUID(node.node_id)
        participants = [Participant(node_id, dev) for dev in node.devices]

    element_size = dtype.itemsize
    assert (
        nbytes % element_size == 0
    ), f"Size {nbytes} bytes not divisible by element size {element_size}"
    n_elements = nbytes // element_size

    mesh = Mesh(participants, axis_names=("devices",))
    dims = [TensorDim("dim0", n_elements)]
    return ShardedTensor(
        name=name,
        dims=dims,
        mesh=mesh,
        partition=P("devices"),
        dtype=dtype,
    )


def connect_prespawned(yaml_path: str) -> ClusterInfo:
    """Connect to a pre-spawned Setu cluster described by a YAML file."""
    yaml_text = Path(yaml_path).read_text()
    cluster_info = ClusterInfo.from_yaml(yaml_text)
    cluster_info.connect()
    return cluster_info


def spawn_local(
    gpus: Optional[int],
    passes: Optional[List[str]],
    metrics_endpoint: str = "",
):
    """Spawn a local Ray + Setu cluster.

    Returns ``(cluster_info, cluster)`` where *cluster* must be stopped
    by the caller.
    """
    import ray

    from setu.cluster.ray.cluster import Cluster

    assert torch.cuda.is_available(), "CUDA not available"
    n_available = torch.cuda.device_count()
    n_gpus = gpus if gpus is not None else n_available
    assert (
        n_gpus <= n_available
    ), f"Requested {n_gpus} GPUs but only {n_available} available"
    assert n_gpus >= 2, f"Need >= 2 GPUs, got {n_gpus}"

    if not ray.is_initialized():
        ray.init()

    cluster = Cluster(passes=passes, metrics_endpoint=metrics_endpoint)
    cluster_info = cluster.start()
    return cluster_info, cluster
