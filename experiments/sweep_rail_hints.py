#!/usr/bin/env python3
"""
Sweep data sizes with rail-aligned routing hints on a 2-node cluster.

For each register size (outer loop) and data size (inner loop), runs
two variants:
  - baseline: no hints (planner chooses paths)
  - rail_hints: rail-aligned hints forcing 0:src → 0:i → 1:i

Requires a 2-node Ray cluster (--ray-address).  Spawns its own Setu
Cluster with shortest_path_routing pass enabled.

Usage::

    python experiments/sweep_rail_hints.py \
        --ray-address ray://10.0.0.1:10001 \
        --output-dir results/rail_hints
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
from setu._coordinator import Link, Participant, Path as RoutePath, RoutingHint
from setu.bench.__main__ import _build_sharded_tensor, _parse_size
from setu.bench.result import CopyMode
from setu.bench.runner import run_experiment
from setu.cluster.ray.cluster import Cluster


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep data sizes with rail-aligned routing hints"
    )
    parser.add_argument(
        "--ray-address", required=True, help="Ray head endpoint"
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
    """Build rail-aligned routing hints for src on node0:dev0 → all devs on node1."""
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
            RoutingHint(
                src=Participant(node0_id, src_dev),
                dst=Participant(node1_id, dst_dev),
                path=path,
            )
        )
    return hints


def main():
    args = parse_args()

    begin_bytes = _parse_size(args.b)
    end_bytes = _parse_size(args.e)
    sizes = _build_size_list(begin_bytes, end_bytes, args.f)
    register_sizes = [_parse_size(s) for s in args.register_sizes]
    copy_mode = CopyMode(args.mode)

    # Build env_vars for actors
    env_vars = {}
    if args.nccl_socket_ifname is not None:
        env_vars["NCCL_SOCKET_IFNAME"] = args.nccl_socket_ifname
    for entry in args.env:
        key, value = entry.split("=", 1)
        env_vars[key] = value
    env_vars = env_vars or None

    if not ray.is_initialized():
        ray.init(address=args.ray_address, log_to_driver=False)

    print(f"=== sweep_rail_hints ===")
    print(f"Output:          {args.output_dir}")
    print(f"Register sizes:  {[_human_label(r) for r in register_sizes]}")
    print(f"Data sizes:      {_human_label(sizes[0])} .. {_human_label(sizes[-1])} ({len(sizes)} points)")
    print(f"Mode:            {copy_mode.value}")
    blocking_str = "blocking" if args.blocking else "non-blocking"
    print(f"Rounds:          {args.rounds} + {args.warmup_rounds} warmup, {blocking_str}")
    print()

    for register_size in register_sizes:
        reg_label = f"reg_{_human_label(register_size)}"
        reg_dir = os.path.join(args.output_dir, reg_label)
        os.makedirs(reg_dir, exist_ok=True)

        print(f"--- Register size: {_human_label(register_size)} ---")

        # Start cluster with log capture
        cluster_log_path = os.path.join(reg_dir, "cluster.log")
        with open(cluster_log_path, "w") as log_f:
            with contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
                cluster = Cluster(
                    passes=["shortest_path_routing"],
                    register_size=register_size,
                    env_vars=env_vars,
                )
                cluster_info = cluster.start()

        # Write cluster info YAML for reference
        Path(os.path.join(reg_dir, "cluster.yaml")).write_text(cluster_info.to_yaml())

        assert len(cluster_info.nodes) == 2, (
            f"Expected 2 nodes, got {len(cluster_info.nodes)}"
        )

        print(f"  Cluster started: {cluster_info.num_nodes} nodes, {cluster_info.total_gpus} GPUs")

        hints = _build_rail_hints(cluster_info)

        failed = 0
        for data_size in sizes:
            size_label = _human_label(data_size)
            print(f"  {size_label}: ", end="", flush=True)

            # Source: 1 device on node 0
            src = _build_sharded_tensor(
                "src_t", cluster_info, data_size, specs=["0:0"]
            )
            # Destination: all devices on node 1
            dst = _build_sharded_tensor(
                "dst_t", cluster_info, data_size, specs=["1"]
            )

            variant_results = []
            for variant, variant_hints in [("baseline", None), ("rail_hints", hints)]:
                point_dir = os.path.join(reg_dir, size_label, variant)
                os.makedirs(point_dir, exist_ok=True)

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
                            hints=variant_hints,
                        )

                result.dump_csv(point_dir)
                status = "PASS" if result.success else "FAIL"
                variant_results.append(f"{variant}={status}")
                if not result.success:
                    failed += 1

            print(", ".join(variant_results))

        print(f"  Done ({failed} failures)")
        cluster.stop()
        print()

    print("=== Sweep complete ===")


if __name__ == "__main__":
    main()
