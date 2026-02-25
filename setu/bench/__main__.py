"""Setu Benchmarking Tool

Supports two flows:
  1. Pre-spawned cluster: Load a ClusterInfo YAML dumped by
     ``python -m setu.cluster.ray --dump-info cluster.yaml``.
  2. Auto-spawn local (default): Starts a single-node Setu
     cluster automatically.

Usage::

    # auto-spawn local, default:
    python -m setu.bench
    python -m setu.bench --topo nccl_topo.xml
    python -m setu.bench --src 0:0-1 --dst 0:0-3

    # pre-spawned cluster:
    # Terminal 1:
    python -m setu.cluster.ray --dump-info cluster.yaml
    # Terminal 2:
    python -m setu.bench --cluster-info cluster.yaml

Device spec format for --src / --dst:
    "0"       — all devices on node 0
    "0:2"     — node 0, device 2
    "0:1-3"   — node 0, devices 1, 2, 3 (range)
    "0:1,4,5" — node 0, devices 1, 4, 5 (comma-separated)
"""

import argparse
import logging
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from setu._commons.datatypes import TensorDim
from setu._coordinator import Participant

from setu.cluster.info import ClusterInfo
from setu.cluster.mesh import Mesh, P
from setu.experiment.helpers import ShardedTensor
from setu.experiment.result import CopyMode
from setu.experiment.runner import run_experiment

# ---------------------------------------------------------------------------
# Setup logging so Python logs are visible on stdout
# ---------------------------------------------------------------------------


def setup_logging():
    import os

    level = getattr(logging, os.getenv("SETU_LOG_LEVEL", "INFO").upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(fmt)
    root = logging.getLogger("setu")
    root.addHandler(handler)
    root.setLevel(level)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Setu copy benchmark")

    # Flow selection
    parser.add_argument(
        "--cluster-info",
        type=str,
        default=None,
        help="Path to ClusterInfo YAML (pre-spawned cluster). "
        "If not given, auto-spawns a local Ray cluster.",
    )

    # Auto-spawn options (Flow 2)
    parser.add_argument(
        "--topo",
        type=str,
        default=None,
        help="Path to NCCL topology XML dump (from NCCL_TOPO_DUMP_FILE). "
        "Enables ShortestPathRouting.",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available).",
    )
    from setu.cluster.passes import AVAILABLE_PASSES

    parser.add_argument(
        "--passes",
        type=str,
        nargs="*",
        default=None,
        choices=AVAILABLE_PASSES,
        help="Planner passes to enable. "
        "Omit for default, pass none with '--passes' for ablation. "
        f"Available: {', '.join(AVAILABLE_PASSES)}.",
    )

    # Tensor / mesh
    parser.add_argument(
        "--size",
        type=str,
        default="256M",
        help="Total tensor size in bytes, e.g. '256M', '1G', '512K'. "
        "Suffix: K=KiB, M=MiB, G=GiB (default: 256 MiB).",
    )
    parser.add_argument(
        "--src",
        type=str,
        nargs="+",
        default=None,
        help="Source device specs, e.g. '0:0-1' or '0:0 0:1'. "
        "Default: all devices on node 0.",
    )
    parser.add_argument(
        "--dst",
        type=str,
        nargs="+",
        default=None,
        help="Dest device specs, e.g. '0:0-3' or '0'. Default: all devices on node 0.",
    )

    # Experiment parameters
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of measured copy rounds (default: 10).",
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=1,
        help="Number of warmup rounds before measured rounds (default: 1).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pull", "copy"],
        default="pull",
        help="Copy mode: 'pull' (one-sided) or 'copy' (two-sided) (default: pull).",
    )
    parser.add_argument(
        "--init-value",
        type=float,
        default=7.0,
        help="Value to fill source tensors with (default: 7.0).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout in seconds for the experiment (default: 600).",
    )
    return parser.parse_args()


def _parse_size(s: str) -> int:
    """Parse a human-readable size string into bytes.

    Examples: '256M' → 268435456, '1G' → 1073741824, '512K' → 524288.
    Plain integers are treated as bytes.
    """
    s = s.strip().upper()
    multipliers = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
    if s[-1] in multipliers:
        return int(float(s[:-1]) * multipliers[s[-1]])
    return int(s)


# ---------------------------------------------------------------------------
# Device spec parsing
# ---------------------------------------------------------------------------


def _parse_device_spec(spec: str) -> List[Tuple[int, Optional[int]]]:
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


def _resolve_device_specs(
    specs: List[str],
    cluster_info: ClusterInfo,
) -> List[Participant]:
    """Resolve device spec strings into Participant objects."""
    participants = []
    for spec in specs:
        for node_idx, dev_idx in _parse_device_spec(spec):
            assert 0 <= node_idx < len(cluster_info.nodes), (
                f"Node index {node_idx} out of range [0, {len(cluster_info.nodes)})"
            )
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


def _build_sharded_tensor(
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
        participants = _resolve_device_specs(specs, cluster_info)
    else:
        node = cluster_info.nodes[0]
        node_id = uuid.UUID(node.node_id)
        participants = [Participant(node_id, dev) for dev in node.devices]

    element_size = dtype.itemsize
    assert nbytes % element_size == 0, (
        f"Size {nbytes} bytes not divisible by element size {element_size}"
    )
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


# ---------------------------------------------------------------------------
# Cluster setup
# ---------------------------------------------------------------------------


def _connect_prespawned(yaml_path: str) -> ClusterInfo:
    """Connect to a pre-spawned Setu cluster described by a YAML file."""
    yaml_text = Path(yaml_path).read_text()
    cluster_info = ClusterInfo.from_yaml(yaml_text)
    cluster_info.connect()
    return cluster_info


def _spawn_local(gpus: Optional[int], passes: Optional[List[str]]):
    """Spawn a local Ray + Setu cluster.

    Returns ``(cluster_info, cluster)`` where *cluster* must be stopped
    by the caller.
    """
    import ray

    from setu.cluster.ray.cluster import Cluster

    assert torch.cuda.is_available(), "CUDA not available"
    n_available = torch.cuda.device_count()
    n_gpus = gpus if gpus is not None else n_available
    assert n_gpus <= n_available, (
        f"Requested {n_gpus} GPUs but only {n_available} available"
    )
    assert n_gpus >= 2, f"Need >= 2 GPUs, got {n_gpus}"

    if not ray.is_initialized():
        ray.init()

    cluster = Cluster(passes=passes)
    cluster_info = cluster.start()
    return cluster_info, cluster


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    setup_logging()

    tensor_bytes = _parse_size(args.size)
    copy_mode = CopyMode(args.mode)

    cluster = None
    if args.cluster_info:
        cluster_info = _connect_prespawned(args.cluster_info)
    else:
        cluster_info, cluster = _spawn_local(args.gpus, args.passes)

    print("=== Setu copy benchmark ===")
    print(
        f"Nodes: {cluster_info.num_nodes}, "
        f"GPUs: {cluster_info.total_gpus}, "
        f"tensor: {tensor_bytes / (1 << 20):.0f} MiB"
    )
    print(f"Src: {args.src or 'all of node 0'}")
    print(f"Dst: {args.dst or 'all of node 0'}")
    print(
        f"Mode: {copy_mode.value}, rounds: {args.rounds} + {args.warmup_rounds} warmup"
    )
    print()

    try:
        src = _build_sharded_tensor("src_t", cluster_info, tensor_bytes, args.src)
        dst = _build_sharded_tensor("dst_t", cluster_info, tensor_bytes, args.dst)

        result = run_experiment(
            cluster_info=cluster_info,
            src=src,
            dst=dst,
            copy_mode=copy_mode,
            init_value=args.init_value,
            timeout=args.timeout,
            n_copy_rounds=args.rounds,
            n_warmup_rounds=args.warmup_rounds,
        )
        print(result.pretty_print())

        assert result.success, f"Experiment failed: {result.errors}"
        for dr in result.dest_results:
            assert dr["values_match"], (
                f"Shard {dr['shard_name']}: expected={dr['expected_value']} "
                f"actual={dr['actual_value']}"
            )
    finally:
        if cluster is not None:
            cluster.stop()

    print("PASS")


if __name__ == "__main__":
    main()
