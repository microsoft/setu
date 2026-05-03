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

    # YAML-driven (replays a setu_metrics.json record):
    python -m setu.bench --tensor-spec tensor_spec.yaml \\
                         --selections selections.yaml

    # pre-spawned cluster:
    # Terminal 1:
    python -m setu.cluster.ray --dump-info cluster.yaml
    # Terminal 2:
    python -m setu.bench --cluster-info cluster.yaml

Device spec format for --src / --dst (also used by tensor_spec.yaml):
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

    # YAML-driven path: tensor layout + selections.  Mutually exclusive with
    # --size / --src / --dst.
    parser.add_argument(
        "--tensor-spec",
        type=str,
        default=None,
        help="Path to tensor_spec.yaml describing src/dst tensors, dtype, "
        "dims, and per-side mesh+partition layout. Requires --selections.",
    )
    parser.add_argument(
        "--selections",
        type=str,
        default=None,
        help="Path to selections.yaml with one copy entry "
        "(src/dst per-dim selections). Requires --tensor-spec.",
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

    # --tensor-spec / --selections are mutually exclusive with the
    # --size / --src / --dst legacy path; both must be given together.
    yaml_mode = args.tensor_spec is not None or args.selections is not None
    if yaml_mode:
        if args.tensor_spec is None or args.selections is None:
            raise SystemExit("--tensor-spec and --selections must be given together")
        explicit_legacy = []
        if args.src is not None:
            explicit_legacy.append("--src")
        if args.dst is not None:
            explicit_legacy.append("--dst")
        # --size has a default ('256M'), so only flag it if the user passed it
        # explicitly.  Detect by checking sys.argv.
        if "--size" in sys.argv:
            explicit_legacy.append("--size")
        if explicit_legacy:
            raise SystemExit(
                f"--tensor-spec/--selections are mutually exclusive with "
                f"{', '.join(explicit_legacy)}"
            )

    copy_mode = CopyMode(args.mode)

    # Metrics are always enabled (required for bandwidth reporting).
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
    print(f"Nodes: {cluster_info.num_nodes}, GPUs: {cluster_info.total_gpus}")
    blocking_str = "blocking" if args.blocking else "non-blocking"

    src_selections = None
    dst_selections = None
    n_copy_rounds = args.rounds
    n_warmup_rounds = args.warmup_rounds

    try:
        if yaml_mode:
            from setu.bench.copy_spec import load_selections, load_tensor_spec

            src, dst = load_tensor_spec(args.tensor_spec, cluster_info)
            selections_list = load_selections(args.selections)
            # YAML mode runs every copy entry back-to-back inside a single
            # run_experiment (one cluster boot, one shard registration, one
            # warmup).  Per-copy telemetry only makes sense in blocking mode.
            assert args.blocking, (
                "yaml mode requires --blocking (per-copy telemetry needs "
                "blocking semantics)"
            )
            assert args.rounds == 1, (
                f"yaml mode requires --rounds 1 (got {args.rounds}); each "
                f"copies-entry is a single measured round"
            )
            assert args.warmup_rounds == 1, (
                f"yaml mode requires --warmup-rounds 1 "
                f"(got {args.warmup_rounds})"
            )
            src_selections = [s.src for s in selections_list]
            dst_selections = [s.dst for s in selections_list]
            n_copy_rounds = len(selections_list)
            print(f"Tensor spec: {args.tensor_spec}")
            print(f"Selections:  {args.selections} ({n_copy_rounds} copies)")
            print(
                f"Src tensor: {src.name}, mesh_shape={src.mesh.shape}, "
                f"axes={src.mesh.axis_names}, partition={src.partition}"
            )
            print(
                f"Dst tensor: {dst.name}, mesh_shape={dst.mesh.shape}, "
                f"axes={dst.mesh.axis_names}, partition={dst.partition}"
            )
        else:
            tensor_bytes = parse_num_bytes(args.size)
            print(f"Tensor: {tensor_bytes / (1 << 20):.0f} MiB")
            print(f"Src: {args.src or 'all of node 0'}")
            print(f"Dst: {args.dst or 'all of node 0'}")
            src = build_sharded_tensor("src_t", cluster_info, tensor_bytes, args.src)
            dst = build_sharded_tensor("dst_t", cluster_info, tensor_bytes, args.dst)

        print(
            f"Mode: {copy_mode.value}, rounds: {n_copy_rounds} + "
            f"{n_warmup_rounds} warmup, {blocking_str}"
        )
        print()

        result = run_experiment(
            cluster_info=cluster_info,
            src=src,
            dst=dst,
            copy_mode=copy_mode,
            init_value=args.init_value,
            src_selections=src_selections,
            dst_selections=dst_selections,
            timeout=args.timeout,
            n_copy_rounds=n_copy_rounds,
            n_warmup_rounds=n_warmup_rounds,
            blocking=args.blocking,
            schedule=schedule_spec,
            progress_dir=args.output_dir,
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
