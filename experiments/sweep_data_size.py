#!/usr/bin/env python3
"""Sweep data sizes on a Setu cluster and plot bandwidth vs. data size.

Usage:
    # Auto-spawn local cluster with defaults (4M..4G, GPUs 0:0 -> 0:1):
    python experiments/sweep_data_size.py --output-dir results/my_run

    # Pre-spawned cluster with custom range:
    python experiments/sweep_data_size.py \
        --cluster-info cluster.yaml \
        --src 0:0 --dst 0:1 \
        -b 1M -e 2G -f 4 \
        --output-dir results/my_run

    # Skip plotting:
    python experiments/sweep_data_size.py --output-dir results/my_run --no-plot
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from setu.bench.cluster_setup import (
    build_sharded_tensor,
    connect_prespawned,
    spawn_local,
)
from setu.bench.result import CopyMode, ExperimentResult
from setu.bench.runner import run_experiment
from setu.utils.parsing import parse_num_bytes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_logging():
    level = getattr(
        logging, os.getenv("SETU_LOG_LEVEL", "INFO").upper(), logging.INFO
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
    )
    handler.setFormatter(fmt)
    for name in ("setu", __name__):
        log = logging.getLogger(name)
        log.addHandler(handler)
        log.setLevel(level)


def _human_size(nbytes: int) -> str:
    """Return a short human-readable label like '4M' or '1G'."""
    if nbytes >= (1 << 30) and nbytes % (1 << 30) == 0:
        return f"{nbytes >> 30}G"
    if nbytes >= (1 << 20) and nbytes % (1 << 20) == 0:
        return f"{nbytes >> 20}M"
    if nbytes >= (1 << 10) and nbytes % (1 << 10) == 0:
        return f"{nbytes >> 10}K"
    return str(nbytes)


def _generate_sizes(begin: int, end: int, factor: int) -> List[int]:
    """Generate a geometric sequence from *begin* to *end* (inclusive)."""
    sizes = []
    s = begin
    while s <= end:
        sizes.append(s)
        s *= factor
    return sizes


# ---------------------------------------------------------------------------
# Data point
# ---------------------------------------------------------------------------


@dataclass
class SweepPoint:
    """Result of a single data-size experiment."""

    size_bytes: int
    size_label: str
    success: bool
    avg_elapsed_s: float  # mean of measured rounds
    bandwidth_gbps: float  # aggregate bandwidth in GB/s
    round_elapsed_s: List[float] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Sweep config (for metadata in plot / JSON)
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    src_specs: List[str]
    dst_specs: List[str]
    begin_size: str
    end_size: str
    factor: int
    rounds: int
    warmup_rounds: int
    copy_mode: str
    blocking: bool


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_sweep(
    points: List[SweepPoint],
    config: SweepConfig,
    output_path: str,
) -> None:
    """Generate a bandwidth-vs-data-size plot and save to *output_path*."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    passed = [p for p in points if p.success]
    if not passed:
        logger.warning("No successful data points to plot.")
        return

    sizes = [p.size_bytes for p in passed]
    bws = [p.bandwidth_gbps for p in passed]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sizes, bws, marker="o", linewidth=2, markersize=6)

    # Log-scale x-axis with human-readable labels.
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([_human_size(s) for s in sizes], rotation=45, ha="right")
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ax.set_xlabel("Data Size")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Setu Copy Bandwidth vs. Data Size")
    ax.grid(True, alpha=0.3)

    # Build metadata text box.
    meta_lines = [
        f"src: {', '.join(config.src_specs)}",
        f"dst: {', '.join(config.dst_specs)}",
        f"mode: {config.copy_mode}, {'blocking' if config.blocking else 'non-blocking'}",
        f"rounds: {config.rounds} + {config.warmup_rounds} warmup",
        f"range: {config.begin_size} .. {config.end_size} (x{config.factor})",
    ]
    meta_text = "\n".join(meta_lines)
    ax.text(
        0.02,
        0.98,
        meta_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Plot saved to %s", output_path)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep data sizes and plot bandwidth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Cluster
    p.add_argument(
        "--cluster-info",
        type=str,
        default=None,
        help="Path to ClusterInfo YAML. If omitted, auto-spawns a local cluster.",
    )
    p.add_argument("--gpus", type=int, default=None, help="GPUs for auto-spawn.")
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Results directory. If omitted, results are printed but not saved.",
    )

    # Device specs
    p.add_argument(
        "--src",
        type=str,
        nargs="+",
        default=["0:0"],
        help="Source device specs (default: 0:0).",
    )
    p.add_argument(
        "--dst",
        type=str,
        nargs="+",
        default=["0:1"],
        help="Dest device specs (default: 0:1).",
    )

    # Sweep range
    p.add_argument(
        "--begin",
        type=str,
        default="4M",
        help="Start size (default: 4M). Suffixes: K, M, G.",
    )
    p.add_argument(
        "--end",
        type=str,
        default="4G",
        help="End size (default: 4G). Suffixes: K, M, G.",
    )
    p.add_argument(
        "-f",
        "--factor",
        type=int,
        default=2,
        help="Geometric step multiplier (default: 2).",
    )

    # Experiment parameters
    p.add_argument("--rounds", type=int, default=10, help="Measured rounds (default: 10).")
    p.add_argument(
        "--warmup-rounds", type=int, default=1, help="Warmup rounds (default: 1)."
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["pull", "copy"],
        default="pull",
        help="Copy mode (default: pull).",
    )
    p.add_argument(
        "--blocking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Block after each round (default: non-blocking).",
    )
    p.add_argument(
        "--enable-metrics",
        action="store_true",
        default=False,
        help="Enable telemetry metrics collection.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-experiment timeout in seconds (default: 600).",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    _setup_logging()

    begin_bytes = parse_num_bytes(args.begin)
    end_bytes = parse_num_bytes(args.end)
    assert begin_bytes > 0 and end_bytes > 0, "Sizes must be positive"
    assert begin_bytes <= end_bytes, f"Begin ({args.begin}) > end ({args.end})"
    assert args.factor > 1, f"Factor must be > 1, got {args.factor}"

    sizes = _generate_sizes(begin_bytes, end_bytes, args.factor)
    copy_mode = CopyMode(args.mode)
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    sweep_config = SweepConfig(
        src_specs=args.src,
        dst_specs=args.dst,
        begin_size=args.begin,
        end_size=args.end,
        factor=args.factor,
        rounds=args.rounds,
        warmup_rounds=args.warmup_rounds,
        copy_mode=args.mode,
        blocking=args.blocking,
    )

    # -- Connect / spawn cluster --
    metrics_endpoint = ""
    if args.enable_metrics:
        from setu.cluster.ray.actors import _find_free_port

        metrics_endpoint = f"tcp://*:{_find_free_port()}"

    cluster = None
    if args.cluster_info:
        cluster_info = connect_prespawned(args.cluster_info)
    else:
        cluster_info, cluster = spawn_local(
            args.gpus, passes=None, metrics_endpoint=metrics_endpoint
        )

    def _cleanup():
        if cluster is not None:
            logger.info("Stopping cluster...")
            cluster.stop()

    signal.signal(signal.SIGINT, lambda *_: (_cleanup(), sys.exit(1)))
    signal.signal(signal.SIGTERM, lambda *_: (_cleanup(), sys.exit(1)))

    print("=== sweep_data_size ===")
    if args.output_dir:
        print(f"Output:  {args.output_dir}")
    print(f"Src:     {' '.join(args.src)}")
    print(f"Dst:     {' '.join(args.dst)}")
    print(f"Range:   {args.begin} .. {args.end} (x{args.factor})")
    print(f"Sizes:   {', '.join(_human_size(s) for s in sizes)}")
    print(f"Mode:    {copy_mode.value}, {'blocking' if args.blocking else 'non-blocking'}")
    print(f"Rounds:  {args.rounds} + {args.warmup_rounds} warmup")
    print()

    # -- Run sweep --
    points: List[SweepPoint] = []
    failed = 0

    try:
        for size_bytes in sizes:
            label = _human_size(size_bytes)
            point_dir = None
            if args.output_dir:
                point_dir = os.path.join(args.output_dir, label)
                os.makedirs(point_dir, exist_ok=True)

            print(f"--- {label} ({size_bytes} bytes) ---")

            try:
                src = build_sharded_tensor(
                    "src_t", cluster_info, size_bytes, args.src
                )
                dst = build_sharded_tensor(
                    "dst_t", cluster_info, size_bytes, args.dst
                )

                result = run_experiment(
                    cluster_info=cluster_info,
                    src=src,
                    dst=dst,
                    copy_mode=copy_mode,
                    init_value=7.0,
                    timeout=args.timeout,
                    n_copy_rounds=args.rounds,
                    n_warmup_rounds=args.warmup_rounds,
                    blocking=args.blocking,
                    metrics_http_url=cluster_info.metrics_http_url,
                )

                # Dump per-point CSVs.
                if point_dir:
                    result.dump_csv(point_dir)

                total_bytes = sum(result.shard_bytes) if result.shard_bytes else 0
                avg_elapsed = (
                    sum(result.round_elapsed_s) / len(result.round_elapsed_s)
                    if result.round_elapsed_s
                    else 0
                )
                bw_gbps = (
                    total_bytes / avg_elapsed / 1e9
                    if avg_elapsed > 0 and total_bytes > 0
                    else 0
                )

                pt = SweepPoint(
                    size_bytes=size_bytes,
                    size_label=label,
                    success=result.success,
                    avg_elapsed_s=avg_elapsed,
                    bandwidth_gbps=bw_gbps,
                    round_elapsed_s=list(result.round_elapsed_s),
                )
                points.append(pt)

                if result.success:
                    print(f"  PASS  {bw_gbps:.2f} GB/s  ({avg_elapsed * 1000:.2f} ms)")
                else:
                    print(f"  FAIL  {result.errors}")
                    failed += 1

            except Exception as e:
                logger.error("Experiment failed for %s: %s", label, e)
                points.append(
                    SweepPoint(
                        size_bytes=size_bytes,
                        size_label=label,
                        success=False,
                        avg_elapsed_s=0,
                        bandwidth_gbps=0,
                        error=str(e),
                    )
                )
                failed += 1

        # -- Dump summary JSON + plot --
        if args.output_dir:
            summary_path = os.path.join(args.output_dir, "sweep_summary.json")
            summary = {
                "config": asdict(sweep_config),
                "points": [asdict(p) for p in points],
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary: {summary_path}")

            plot_path = os.path.join(args.output_dir, "bandwidth.png")
            plot_sweep(points, sweep_config, plot_path)

        print()
        print("=== Sweep complete ===")
        if args.output_dir:
            print(f"Results: {args.output_dir}")
        print(f"Points:  {len(points)} total, {failed} failed")

    finally:
        _cleanup()


if __name__ == "__main__":
    main()
