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

from setu.bench.cluster_setup import (
    build_sharded_tensor,
    connect_prespawned,
    spawn_local,
)
from setu.bench.result import CopyMode
from setu.bench.runner import run_experiment
from setu.utils.parsing import parse_num_bytes

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
    parser.add_argument(
        "--blocking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Block after each copy round (default: False). "
        "--blocking waits after each round, "
        "--no-blocking queues all rounds then syncs once, "
        "amortising control-plane overhead like nccl-test -C 0.",
    )
    parser.add_argument(
        "--enable-metrics",
        action="store_true",
        default=False,
        help="Enable telemetry metrics collection (starts MetricsServer).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write CSV result files (rounds, clients, telemetry).",
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default=None,
        help="Path to a schedule file (Python module with a 'schedule(ctx)' function). "
        "Overrides static hints with dynamic, per-copy schedule generation.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    setup_logging()

    tensor_bytes = parse_num_bytes(args.size)
    copy_mode = CopyMode(args.mode)

    # Generate metrics endpoint if metrics are enabled.
    metrics_endpoint = ""
    if args.enable_metrics:
        from setu.cluster.ray.actors import _find_free_port

        metrics_endpoint = f"tcp://*:{_find_free_port()}"

    # Load schedule if provided.
    schedule_spec = None
    if args.schedule:
        from setu.bench.schedule import ScheduleSpec

        schedule_spec = ScheduleSpec(args.schedule)

    cluster = None
    if args.cluster_info:
        cluster_info = connect_prespawned(args.cluster_info)
    else:
        cluster_info, cluster = spawn_local(
            args.gpus, args.passes, metrics_endpoint=metrics_endpoint
        )

    print("=== Setu copy benchmark ===")
    print(
        f"Nodes: {cluster_info.num_nodes}, "
        f"GPUs: {cluster_info.total_gpus}, "
        f"tensor: {tensor_bytes / (1 << 20):.0f} MiB"
    )
    print(f"Src: {args.src or 'all of node 0'}")
    print(f"Dst: {args.dst or 'all of node 0'}")
    blocking_str = "blocking" if args.blocking else "non-blocking"
    print(
        f"Mode: {copy_mode.value}, rounds: {args.rounds} + {args.warmup_rounds} warmup, {blocking_str}"
    )
    print()

    try:
        src = build_sharded_tensor("src_t", cluster_info, tensor_bytes, args.src)
        dst = build_sharded_tensor("dst_t", cluster_info, tensor_bytes, args.dst)

        result = run_experiment(
            cluster_info=cluster_info,
            src=src,
            dst=dst,
            copy_mode=copy_mode,
            init_value=args.init_value,
            timeout=args.timeout,
            n_copy_rounds=args.rounds,
            n_warmup_rounds=args.warmup_rounds,
            blocking=args.blocking,
            metrics_http_url=cluster_info.metrics_http_url,
            schedule=schedule_spec,
        )
        print(result.pretty_print())

        if args.output_dir:
            paths = result.dump_csv(args.output_dir)
            print(f"Results written to {args.output_dir}/")
            for p in paths:
                print(f"  {p}")

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
