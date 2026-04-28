import argparse
import os
import uuid
from dataclasses import dataclass
from typing import List

import torch

from setu._commons.datatypes import TensorDim, TensorDimSpec, TensorShardSpec
from setu._coordinator import Participant, ReplicationHint, ReplicationStrategy
from setu.bench.cluster_setup import connect_prespawned
from setu.bench.result import CopyMode
from setu.bench.runner import run_experiment
from setu.cluster.info import ClusterInfo
from setu.cluster.mesh import Mesh
from setu.schedule import Schedule
from setu.utils.parsing import parse_num_bytes as _parse_size


# ---------------------------------------------------------------------------
# ReplicatedTensor: matches the interface run_experiment expects from
# ShardedTensor (.name, .shards, .mesh, .partition, .shard_bytes)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplicatedTensor:
    """A tensor replicated across N GPUs (one full copy per GPU).

    Provides the same interface as ShardedTensor so it can be passed
    directly to ``run_experiment``.
    """

    name: str
    mesh: Mesh
    partition: None  # Not used, but run_experiment logs it
    dtype: torch.dtype
    _shards: List[TensorShardSpec]
    _shard_bytes: List[int]

    @property
    def shards(self) -> List[TensorShardSpec]:
        return self._shards

    @property
    def shard_bytes(self) -> List[int]:
        return self._shard_bytes


def _build_replicated_tensor(
    name: str,
    cluster_info: ClusterInfo,
    num_pieces: int,
    piece_elements: int,
    gpu_indices: List[int],
    dtype: torch.dtype = torch.float32,
) -> ReplicatedTensor:
    """Build a replicated tensor: full copy on each of the specified GPUs.

    The tensor is shaped ``(2 * num_pieces, piece_elements)``. Selecting
    every other index on the outer dim (see CLI) gives ``num_pieces``
    non-contiguous pieces per replica. The trick mirrors the one in
    ``experiments/pack/client.py``: same layout, but force the planner to
    treat the per-replica copy as several disjoint runs in the shard
    buffer so we can exercise multi-piece AllGather / BatchedCopy paths.

    Args:
        name: Tensor name.
        cluster_info: Running cluster description.
        num_pieces: Number of non-contiguous pieces per replica (>= 1).
        piece_elements: Elements per piece.
        gpu_indices: GPU device indices (all on node 0).
        dtype: Element type.
    """
    node = cluster_info.nodes[0]
    node_id = uuid.UUID(node.node_id)

    assert num_pieces >= 1, f"num_pieces must be >= 1, got {num_pieces}"
    outer = 2 * num_pieces
    num_replicas = len(gpu_indices)

    participants = []
    shards = []
    for replica_id, gpu_idx in enumerate(gpu_indices):
        assert 0 <= gpu_idx < len(node.devices), (
            f"GPU index {gpu_idx} out of range [0, {len(node.devices)})"
        )
        participant = Participant(node_id, node.devices[gpu_idx])
        participants.append(participant)

        dims = [
            TensorDimSpec("piece", outer, 0, outer),
            TensorDimSpec("data", piece_elements, 0, piece_elements),
        ]
        shard = TensorShardSpec(
            name=name,
            dims=dims,
            dtype=dtype,
            device=participant.device,
            replica_id=replica_id,
            num_replicas=num_replicas,
        )
        shards.append(shard)

    mesh = Mesh(participants, axis_names=("replicas",))
    per_shard_bytes = outer * piece_elements * dtype.itemsize

    return ReplicatedTensor(
        name=name,
        mesh=mesh,
        partition=None,
        dtype=dtype,
        _shards=shards,
        _shard_bytes=[per_shard_bytes] * num_replicas,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Benchmark replicated tensor lowering strategies"
)
parser.add_argument("--cluster-info", type=str, required=True)
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
    required=True,
    help="Comma-separated GPU indices for source replicas (e.g. '0,1' or '0,1,2,3').",
)
parser.add_argument(
    "--dst",
    type=str,
    required=True,
    help="Comma-separated GPU indices for destination replicas (e.g. '2,3' or '4,5,6,7').",
)
parser.add_argument(
    "--runs",
    type=int,
    default=1,
    help="Number of times to repeat the experiment (default: 1).",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="./replication",
    help="Base output directory for results (default: ./replication).",
)
parser.add_argument(
    "--n-copy-rounds",
    type=int,
    default=10,
    help="Number of measured copy rounds (default: 10).",
)
parser.add_argument(
    "--n-warmup-rounds",
    type=int,
    default=1,
    help="Number of warmup rounds (default: 1).",
)
parser.add_argument(
    "--num-pieces",
    type=int,
    default=1,
    help="Number of non-contiguous pieces per replica (default: 1). "
    "When > 1 the tensor is shaped (2*num_pieces, piece_elements) and the "
    "selection picks every other outer index, forcing the planner to see "
    "num_pieces disjoint runs per replica. Mirrors experiments/pack.",
)
args = parser.parse_args()

# Parse GPU indices
src_gpus = [int(g) for g in args.src.split(",")]
dst_gpus = [int(g) for g in args.dst.split(",")]
assert len(src_gpus) == len(dst_gpus), (
    f"src and dst must have the same number of GPUs: "
    f"got {len(src_gpus)} src vs {len(dst_gpus)} dst"
)

num_pieces = args.num_pieces
assert num_pieces >= 1, f"--num-pieces must be >= 1, got {num_pieces}"

# Connect to cluster
cluster_info = connect_prespawned(args.cluster_info)
print(cluster_info)

dtype = torch.float32
element_size = dtype.itemsize
tensor_bytes = _parse_size(args.size)
assert tensor_bytes % (num_pieces * element_size) == 0, (
    f"Size {args.size} ({tensor_bytes} bytes) must be divisible by "
    f"num_pieces * element_size ({num_pieces} * {element_size})"
)
piece_elements = tensor_bytes // (num_pieces * element_size)

# Selection: every other index on the outer "piece" dim -> num_pieces pieces.
piece_indices = list(range(0, 2 * num_pieces, 2))
selections = {"piece": piece_indices}

# Build replicated tensors
src = _build_replicated_tensor(
    "src_t", cluster_info, num_pieces, piece_elements, src_gpus, dtype
)
dst = _build_replicated_tensor(
    "dst_t", cluster_info, num_pieces, piece_elements, dst_gpus, dtype
)

print(f"Source GPUs: {src_gpus} ({len(src_gpus)} replicas)")
print(f"Dest GPUs:   {dst_gpus} ({len(dst_gpus)} replicas)")
print(f"Selected:    {tensor_bytes / (1 << 20):.0f} MiB per replica "
      f"({num_pieces} piece(s) of {piece_elements} elements)")
print(f"Allocated:   {2 * tensor_bytes / (1 << 20):.0f} MiB per replica "
      f"(tensor shape ({2 * num_pieces}, {piece_elements}))")
print()

for run_idx in range(args.runs):
    run_dir = os.path.join(args.output_dir, str(run_idx))
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n{'='*10} Run {run_idx} | src={src_gpus} dst={dst_gpus} {'='*10}")

    # --- Naive strategy ---
    schedule_naive = Schedule(
        hints=[ReplicationHint(dst_name="dst_t", strategy=ReplicationStrategy.Naive)]
    )
    result = run_experiment(
        cluster_info=cluster_info,
        src=src,
        dst=dst,
        copy_mode=CopyMode("pull"),
        init_value=10,
        selections=selections,
        n_copy_rounds=args.n_copy_rounds,
        n_warmup_rounds=args.n_warmup_rounds,
        blocking=False,

        hints=schedule_naive.hints,
    )
    print("=" * 10, "Naive Strategy", "=" * 10)
    print(result.pretty_print())
    result.dump_csv(os.path.join(run_dir, "naive"))

    # --- BatchedCopy strategy ---
    schedule_allgather = Schedule(
        hints=[ReplicationHint(dst_name="dst_t", strategy=ReplicationStrategy.BatchedCopy)]
    )
    result = run_experiment(
        cluster_info=cluster_info,
        src=src,
        dst=dst,
        copy_mode=CopyMode("pull"),
        init_value=10,
        selections=selections,
        n_copy_rounds=args.n_copy_rounds,
        n_warmup_rounds=args.n_warmup_rounds,
        blocking=False,

        hints=schedule_allgather.hints,
    )
    print("=" * 10, "BatchedCopy Strategy", "=" * 10)
    print(result.pretty_print())
    result.dump_csv(os.path.join(run_dir, "allgather"))

    # --- AllGather strategy ---
    schedule_allgather = Schedule(
        hints=[ReplicationHint(dst_name="dst_t", strategy=ReplicationStrategy.AllGather)]
    )
    result = run_experiment(
        cluster_info=cluster_info,
        src=src,
        dst=dst,
        copy_mode=CopyMode("pull"),
        init_value=10,
        selections=selections,
        n_copy_rounds=args.n_copy_rounds,
        n_warmup_rounds=args.n_warmup_rounds,
        blocking=False,

        hints=schedule_allgather.hints,
    )
    print("=" * 10, "AllGather Strategy", "=" * 10)
    print(result.pretty_print())
    result.dump_csv(os.path.join(run_dir, "allgather"))
