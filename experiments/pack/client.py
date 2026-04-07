"""Experiment client for Pack/Unpack pipelining.

Tests the interaction between PackUnpackCopies (grouping N copies into
Pack->Copy->Unpack) and Pipelining (chunking the packed transfer into
wavefront-ordered pipeline stages).

Uses a shape trick to generate N pack pieces from a single copy
operation: gives the tensor shape (2*N, piece_elements) and selects
every other index on the first dimension.  The planner sees N
non-contiguous blocks, creates N CopyOps between the same device pair,
and PackUnpackCopies groups them into a single Pack->Copy->Unpack chain.

Usage::

    # Boot cluster first:
    bash experiments/pack/boot.sh -c cluster.yaml

    # Run experiment:
    python experiments/pack/client.py \\
        --cluster-info cluster.yaml \\
        --size 256M \\
        --num-pieces 4 \\
        --pipeline-chunk-size 64M
"""

import argparse
import math
import os

import torch

from setu._commons.datatypes import TensorDim
from setu._coordinator import PipelineChunkSizeHint
from setu.bench.cluster_setup import connect_prespawned, resolve_device_specs
from setu.bench.helpers import ShardedTensor
from setu.bench.result import CopyMode
from setu.bench.runner import run_experiment
from setu.cluster.mesh import Mesh, P
from setu.schedule import Schedule
from setu.utils.parsing import parse_num_bytes as _parse_size


def build_packed_tensor(name, cluster_info, num_pieces, piece_elements, device_spec, dtype):
    """Build a 2D sharded tensor for the pack/unpack experiment.

    Shape: (2 * num_pieces, piece_elements).  The first dimension has
    2x the pieces so that selecting every other index creates
    non-contiguous blocks that force separate CopyOps.
    """
    participants = resolve_device_specs([device_spec], cluster_info)
    assert len(participants) == 1, f"Expected 1 device, got {len(participants)}"

    dims = [
        TensorDim("piece", 2 * num_pieces),
        TensorDim("data", piece_elements),
    ]
    mesh = Mesh(participants, axis_names=("devices",))
    return ShardedTensor(
        name=name,
        dims=dims,
        mesh=mesh,
        partition=P("devices", None),
        dtype=dtype,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pack/Unpack pipelining experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cluster-info", type=str, required=True)
    parser.add_argument(
        "--size",
        type=str,
        nargs="+",
        default=["256M"],
        help="Data size(s) to transfer, e.g. '256M', '1G'. "
        "Multiple values sweep over each size (default: 256M).",
    )
    parser.add_argument(
        "--num-pieces",
        type=int,
        required=True,
        help="Number of pack pieces (N). Must be >= 2.",
    )
    parser.add_argument(
        "--pipeline-chunk-size",
        type=str,
        default=None,
        help="Pipeline chunk size in bytes (e.g. '64M', '128M'). "
        "Passed as PipelineChunkSizeHint. If omitted, the pass default is used.",
    )
    parser.add_argument(
        "--src-device",
        type=str,
        default="0:0",
        help="Source device spec (default: 0:0).",
    )
    parser.add_argument(
        "--dst-device",
        type=str,
        default="0:1",
        help="Destination device spec (default: 0:1).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of times to repeat the experiment (default: 1).",
    )
    parser.add_argument(
        "--num-copy-rounds",
        type=int,
        default=20,
        help="Measured copy rounds per experiment (default: 20).",
    )
    parser.add_argument(
        "--num-warmup-rounds",
        type=int,
        default=1,
        help="Warmup rounds before measurement (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./pack",
        help="Base output directory for results (default: ./pack).",
    )
    args = parser.parse_args()

    cluster_info = connect_prespawned(args.cluster_info)
    print(cluster_info)

    num_pieces = args.num_pieces
    assert num_pieces >= 2, f"num_pieces must be >= 2, got {num_pieces}"

    dtype = torch.float32
    element_size = dtype.itemsize

    # Build schedules for baseline vs pipelined comparison
    pipeline_chunk_size = None
    pipeline_hints = []
    if args.pipeline_chunk_size is not None:
        pipeline_chunk_size = _parse_size(args.pipeline_chunk_size)
        pipeline_hints.append(PipelineChunkSizeHint(pipeline_chunk_size))

    # Baseline: no passes (ablation)
    schedule_baseline = Schedule(passes=[])
    # Optimized: full pass pipeline
    schedule_pipelined = Schedule(
        hints=pipeline_hints,
        passes=[
            "pack_unpack_copies",
            "pipelining",
            "register_tiling",
            "instruction_scheduler",
        ],
    )

    # Selection: every other index on the "piece" dimension → N pieces
    piece_indices = list(range(0, 2 * num_pieces, 2))
    selections = {"piece": piece_indices}

    for size_str in args.size:
        data_size = _parse_size(size_str)

        assert data_size % (num_pieces * element_size) == 0, (
            f"Data size {size_str} ({data_size} bytes) must be divisible by "
            f"num_pieces * element_size ({num_pieces} * {element_size})"
        )

        piece_size = data_size // num_pieces
        piece_elements = piece_size // element_size
        tensor_alloc = 2 * data_size

        # Derived values
        num_pipeline_chunks = (
            math.ceil(data_size / pipeline_chunk_size)
            if pipeline_chunk_size is not None
            else None
        )

        print(f"\n{'='*60}")
        print(f"  Data size:             {size_str} ({data_size} bytes)")
        print(f"  Num pieces:            {num_pieces}")
        print(f"  Piece size:            {piece_size} bytes ({piece_elements} elements)")
        print(f"  Tensor shape:          ({2 * num_pieces}, {piece_elements})")
        print(f"  Tensor alloc size:     {tensor_alloc} bytes (2x)")
        print(f"  Selection:             piece={piece_indices}")
        if pipeline_chunk_size is not None:
            print(f"  Pipeline chunk size:   {args.pipeline_chunk_size} ({pipeline_chunk_size} bytes)")
            print(f"  Pipeline chunks (est): {num_pipeline_chunks}")
        print(f"  Src device:            {args.src_device}")
        print(f"  Dst device:            {args.dst_device}")
        print(f"{'='*60}")

        src = build_packed_tensor(
            "src_t", cluster_info, num_pieces, piece_elements,
            args.src_device, dtype,
        )
        dst = build_packed_tensor(
            "dst_t", cluster_info, num_pieces, piece_elements,
            args.dst_device, dtype,
        )

        size_dir = os.path.join(args.output_dir, size_str)

        for run_idx in range(args.runs):
            run_dir = os.path.join(size_dir, str(run_idx))
            os.makedirs(run_dir, exist_ok=True)

            print(f"\n--- Size {size_str} | Run {run_idx} ---")

            # Baseline: no optimization passes
            result = run_experiment(
                cluster_info=cluster_info,
                src=src,
                dst=dst,
                copy_mode=CopyMode("pull"),
                init_value=10,
                selections=selections,
                n_copy_rounds=args.num_copy_rounds,
                n_warmup_rounds=args.num_warmup_rounds,
                blocking=False,

                hints=schedule_baseline.hints,
                pass_names=schedule_baseline.passes,
            )
            print("="*10, "Baseline (no passes)", "="*10)
            print(result.pretty_print())
            result.dump_csv(os.path.join(run_dir, "baseline"))

            # Pipelined: pack_unpack_copies + pipelining
            result = run_experiment(
                cluster_info=cluster_info,
                src=src,
                dst=dst,
                copy_mode=CopyMode("pull"),
                init_value=10,
                selections=selections,
                n_copy_rounds=args.num_copy_rounds,
                n_warmup_rounds=args.num_warmup_rounds,
                blocking=False,

                hints=schedule_pipelined.hints,
                pass_names=schedule_pipelined.passes,
            )
            print("="*10, "Pipelined", "="*10)
            print(result.pretty_print())
            result.dump_csv(os.path.join(run_dir, "pipelined"))


if __name__ == "__main__":
    main()
