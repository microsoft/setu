#!/usr/bin/env python3
"""
Sweep data sizes with rail-aligned bandwidth-aggregation hints on a
2-node cluster.

For each register size (outer loop) and data size (inner loop), runs:
  - baseline: bandwidth_aggregation pass only, no hints (planner
    aggregates over topology-discovered edge-disjoint paths)
  - rail_hints: bandwidth_aggregation only, with rail-aligned
    BandwidthHints forcing 0:src → 0:i → 1:i, one path per (src, dst),
    weight 1.0
  - rail_pipe_<chunk>: rail_hints plus the pipelining pass with a
    PipelineChunkSizeHint for each chunk size in --pipeline-chunks

Requires a 2-node Ray cluster (--ray-address).  Spawns its own Setu
Cluster with both bandwidth_aggregation and pipelining passes
registered; each variant selects which to run via Schedule.passes.

Usage::

    python experiments/sweep_rail_hints.py \
        --ray-address ray://10.0.0.1:10001 \
        --output-dir results/rail_hints \
        --pipeline-chunks 1M 4M 16M
"""

import argparse
import contextlib
import os
import sys
import uuid
from pathlib import Path
from typing import List

import ray

from setu._commons.datatypes import TensorDim
from setu._coordinator import (
    BandwidthHint,
    Link,
    Participant,
    Path as RoutePath,
    PipelineChunkSizeHint,
)
from setu.bench.cluster_setup import build_sharded_tensor, connect_prespawned
from setu.bench.result import CopyMode
from setu.bench.runner import run_experiment
from setu.cluster.ray.cluster import Cluster
from setu.schedule import Schedule
from setu.utils.parsing import parse_num_bytes


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep data sizes with rail-aligned routing hints"
    )
    parser.add_argument(
        "--ray-address",
        default=None,
        help="Ray head endpoint. Required unless --cluster-info is given, "
        "in which case Ray is reached via the YAML's ray_address.",
    )
    parser.add_argument(
        "--cluster-info",
        default=None,
        help="Path to a cluster.yaml dumped by setu.cluster.ray (or the "
        "boot_virtual_cluster helper). When set, skip the in-script Cluster() "
        "spawn and connect to that pre-spawned cluster. The outer register-"
        "size loop collapses to a single iteration since register size is "
        "fixed at boot time.",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for results"
    )
    parser.add_argument(
        "-b",
        default="32",
        help="Begin data size (default: 32). Suffixes: K, M, G.",
    )
    parser.add_argument(
        "-e",
        default="8G",
        help="End data size (default: 8G). Suffixes: K, M, G.",
    )
    parser.add_argument(
        "-f",
        type=int,
        default=2,
        help="Size step factor (default: 2).",
    )
    parser.add_argument(
        "--register-sizes",
        type=str,
        nargs="+",
        default=["1M"],
        help="Register sizes to sweep (default: 1M). Suffixes: K, M, G.",
    )
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
        help="Number of warmup rounds (default: 1).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pull", "copy"],
        default="pull",
        help="Copy mode (default: pull).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Timeout in seconds per experiment (default: 600).",
    )
    parser.add_argument(
        "--blocking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Block after each copy round (default: False). "
        "--blocking waits after each round, "
        "--no-blocking queues all rounds then syncs once.",
    )
    parser.add_argument(
        "--nccl-socket-ifname",
        type=str,
        default=None,
        help="Value for NCCL_SOCKET_IFNAME env var on actors.",
    )
    parser.add_argument(
        "--env",
        type=str,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional env vars to set on actors (repeatable).",
    )
    parser.add_argument(
        "--pipeline-chunks",
        type=str,
        nargs="*",
        default=[],
        help="Pipeline chunk sizes to sweep (e.g. '1M 4M 16M'). Each value "
        "produces a rail_pipe_<size> variant using the pipelining pass with "
        "a PipelineChunkSizeHint. Empty list (default) skips pipelining "
        "variants.",
    )
    return parser.parse_args()


def _human_label(nbytes: int) -> str:
    """Convert bytes to human-readable label like '32B', '1MB', '4GB'."""
    if nbytes >= (1 << 30):
        return f"{nbytes // (1 << 30)}GB"
    elif nbytes >= (1 << 20):
        return f"{nbytes // (1 << 20)}MB"
    elif nbytes >= (1 << 10):
        return f"{nbytes // (1 << 10)}KB"
    else:
        return f"{nbytes}B"


def _build_size_list(begin: int, end: int, factor: int) -> List[int]:
    """Build list of sizes from begin to end, stepping by factor."""
    sizes = []
    size = begin
    while size <= end:
        sizes.append(size)
        size *= factor
    return sizes


def _build_rail_hints(cluster_info):
    """Build rail-aligned BandwidthHints for src on node0:dev0 → all devs on node1.

    Each (src, dst) pair gets a single path forcing the rail
    (0:src → 0:i → 1:i) with weight 1.0, so BandwidthAggregation routes
    the full buffer along that path instead of discovering its own.
    """
    node0 = cluster_info.nodes[0]
    node1 = cluster_info.nodes[1]
    node0_id = uuid.UUID(node0.node_id)
    node1_id = uuid.UUID(node1.node_id)
    node0_devs = node0.devices
    node1_devs = node1.devices
    src_dev = node0_devs[0]

    hints = []
    for i, dst_dev in enumerate(node1_devs):
        path = RoutePath(
            hops=[
                Participant(node0_id, src_dev),
                Participant(node0_id, node0_devs[i]),
                Participant(node1_id, dst_dev),
            ],
            links=[Link(0.0, 1.0), Link(0.0, 1.0)],
        )
        hints.append(
            BandwidthHint(
                src=Participant(node0_id, src_dev),
                dst=Participant(node1_id, dst_dev),
                paths=[path],
                weights=[1.0],
            )
        )
    return hints


def main():
    args = parse_args()

    begin_bytes = parse_num_bytes(args.b)
    end_bytes = parse_num_bytes(args.e)
    sizes = _build_size_list(begin_bytes, end_bytes, args.f)
    register_sizes = [parse_num_bytes(s) for s in args.register_sizes]
    pipeline_chunks = [parse_num_bytes(s) for s in args.pipeline_chunks]
    copy_mode = CopyMode(args.mode)

    # Build env_vars for actors
    env_vars = {}
    if args.nccl_socket_ifname is not None:
        env_vars["NCCL_SOCKET_IFNAME"] = args.nccl_socket_ifname
    for entry in args.env:
        key, value = entry.split("=", 1)
        env_vars[key] = value
    env_vars = env_vars or None

    # Two cluster modes:
    #   --cluster-info: connect to a pre-spawned cluster (e.g. spawned by
    #     boot_virtual_cluster.py for single-host testing).  Register size
    #     is fixed at boot time, so the outer register-size loop collapses
    #     to a single iteration labeled "preset".
    #   --ray-address: spawn a fresh Setu Cluster() per register-size.
    if args.cluster_info is not None:
        # Skip ray.init: connect_prespawned reads ray_address from the YAML
        # and connects there.
        prespawned = connect_prespawned(args.cluster_info)
        register_iterations = [(None, "preset", prespawned)]
        if args.register_sizes != ["1M"]:  # non-default
            print(
                "Warning: --register-sizes is ignored when --cluster-info is "
                "set; register size is fixed by the spawned cluster."
            )
    else:
        if args.ray_address is None:
            raise SystemExit(
                "Either --cluster-info or --ray-address must be provided."
            )
        if not ray.is_initialized():
            ray.init(address=args.ray_address, log_to_driver=False)
        register_iterations = [
            (rs, f"reg_{_human_label(rs)}", None) for rs in register_sizes
        ]

    print(f"=== sweep_rail_hints ===")
    print(f"Output:          {args.output_dir}")
    if args.cluster_info is not None:
        print(f"Cluster:         pre-spawned ({args.cluster_info})")
    else:
        print(f"Register sizes:  {[_human_label(r) for r in register_sizes]}")
    print(f"Data sizes:      {_human_label(sizes[0])} .. {_human_label(sizes[-1])} ({len(sizes)} points)")
    print(f"Pipeline chunks: {[_human_label(c) for c in pipeline_chunks] if pipeline_chunks else 'none'}")
    print(f"Mode:            {copy_mode.value}")
    blocking_str = "blocking" if args.blocking else "non-blocking"
    print(f"Rounds:          {args.rounds} + {args.warmup_rounds} warmup, {blocking_str}")
    print()

    for register_size, reg_label, prespawned_info in register_iterations:
        reg_dir = os.path.join(args.output_dir, reg_label)
        os.makedirs(reg_dir, exist_ok=True)

        cluster = None
        if prespawned_info is not None:
            cluster_info = prespawned_info
            print(f"--- Pre-spawned cluster ---")
        else:
            print(f"--- Register size: {_human_label(register_size)} ---")
            # Start cluster with log capture
            cluster_log_path = os.path.join(reg_dir, "cluster.log")
            with open(cluster_log_path, "w") as log_f:
                with contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
                    cluster = Cluster(
                        passes=["bandwidth_aggregation", "pipelining"],
                        register_size=register_size,
                        env_vars=env_vars,
                    )
                    cluster_info = cluster.start()
            # Write cluster info YAML for reference
            Path(os.path.join(reg_dir, "cluster.yaml")).write_text(
                cluster_info.to_yaml()
            )

        assert len(cluster_info.nodes) == 2, (
            f"Expected 2 nodes, got {len(cluster_info.nodes)}"
        )

        print(f"  Cluster: {cluster_info.num_nodes} nodes, {cluster_info.total_gpus} GPUs")

        rail_hints = _build_rail_hints(cluster_info)

        failed = 0
        for data_size in sizes:
            size_label = _human_label(data_size)
            print(f"  {size_label}: ", end="", flush=True)

            # Source: 1 device on node 0
            src = build_sharded_tensor(
                "src_t", cluster_info, data_size, specs=["0:0"]
            )
            # Destination: all devices on node 1
            dst = build_sharded_tensor(
                "dst_t", cluster_info, data_size, specs=["1"]
            )

            # Each variant pins its own pass list so cluster-level pass
            # registration doesn't accidentally pull pipelining into the
            # baseline / rail_hints runs.
            variants = [
                ("baseline", Schedule(passes=["bandwidth_aggregation"])),
                (
                    "rail_hints",
                    Schedule(
                        hints=list(rail_hints),
                        passes=["bandwidth_aggregation"],
                    ),
                ),
            ]
            for chunk_bytes in pipeline_chunks:
                variants.append((
                    f"rail_pipe_{_human_label(chunk_bytes)}",
                    Schedule(
                        hints=list(rail_hints) + [
                            PipelineChunkSizeHint(chunk_bytes)
                        ],
                        passes=["bandwidth_aggregation", "pipelining"],
                    ),
                ))

            variant_results = []
            for variant, variant_schedule in variants:
                point_dir = os.path.join(reg_dir, size_label, variant)
                os.makedirs(point_dir, exist_ok=True)

                # The runner expects a callable ctx -> Schedule.  Bind the
                # per-variant Schedule via a default arg to avoid late-binding
                # surprises if `variants` is ever extended.
                variant_schedule_fn = lambda ctx, s=variant_schedule: s

                bench_log_path = os.path.join(point_dir, "bench.log")
                with open(bench_log_path, "w") as log_f:
                    with contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
                        result = run_experiment(
                            cluster_info=cluster_info,
                            src=src,
                            dst=dst,
                            copy_mode=copy_mode,
                            timeout=args.timeout,
                            n_copy_rounds=args.rounds,
                            n_warmup_rounds=args.warmup_rounds,
                            blocking=args.blocking,
                            schedule=variant_schedule_fn,
                        )

                result.dump_csv(point_dir)
                status = "PASS" if result.success else "FAIL"
                variant_results.append(f"{variant}={status}")
                if not result.success:
                    failed += 1

            print(", ".join(variant_results))

        print(f"  Done ({failed} failures)")
        if cluster is not None:
            cluster.stop()
        print()

    print("=== Sweep complete ===")


if __name__ == "__main__":
    main()
