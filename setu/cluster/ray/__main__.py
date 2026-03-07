"""
CLI entry point for starting a Setu Ray cluster.

Usage::

    setu-cluster
    setu-cluster --ray-address ray://10.0.0.1:10001
    python -m setu.cluster.ray
"""

import argparse
import signal
import threading
from pathlib import Path
from typing import Dict, Optional

import ray
from rich.console import Console
from rich.table import Table

from setu.cluster.info import ClusterInfo
from setu.cluster.ray.cluster import Cluster
from setu.logger import init_logger
from setu.utils.parsing import parse_num_bytes

logger = init_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the cluster CLI."""
    parser = argparse.ArgumentParser(
        description="Start a Setu cluster on an existing Ray cluster."
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default=None,
        help="Ray cluster address to connect to. If not given, starts a local Ray instance.",
    )
    parser.add_argument(
        "--nccl-socket-ifname",
        type=str,
        default=None,
        help="Value for NCCL_SOCKET_IFNAME env var on actors (default: not set).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="DEBUG",
        help='Value for SETU_LOG_LEVEL env var on actors (default: "DEBUG").',
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
        "--dump-info",
        type=str,
        default=None,
        metavar="PATH",
        help="Dump ClusterInfo YAML to this file path.",
    )
    parser.add_argument(
        "--passes",
        nargs="*",
        default=None,
        help="Planner passes to enable. Omit for default, pass none with "
        "'--passes' for ablation.",
    )
    parser.add_argument(
        "--enable-metrics",
        action="store_true",
        default=False,
        help="Enable telemetry metrics collection (starts MetricsServer).",
    )
    parser.add_argument(
        "--register-size",
        type=str,
        default="1M",
        metavar="SIZE",
        help="Size of each register (temporary buffer) per GPU (default: 1M). "
        "Accepts human-readable formats: '1M', '256K', '1G'. "
        "Plain integers are treated as bytes.",
    )
    return parser.parse_args()


def display_cluster_info(info: ClusterInfo) -> None:
    """Display cluster topology using a rich table."""
    console = Console()

    console.rule("Setu Cluster Info")
    console.print(f"  Coordinator: {info.coordinator_endpoint}")
    console.print(f"  Nodes: {info.num_nodes}  |  Total GPUs: {info.total_gpus}")
    if info.metrics_endpoint:
        console.print(f"  Metrics ZMQ: {info.metrics_endpoint}")
    if info.metrics_http_url:
        console.print(f"  Metrics HTTP: {info.metrics_http_url}")
    console.print()

    table = Table(title="Node Agents")
    table.add_column("Node ID", style="cyan")
    table.add_column("Endpoint", style="magenta")
    table.add_column("GPUs", justify="right", style="bold")

    for node in info.nodes:
        table.add_row(
            node.node_id,
            node.node_agent_endpoint,
            str(len(node.devices)),
        )

    console.print(table)
    console.print()


def _build_env_vars(args: argparse.Namespace) -> Optional[Dict[str, str]]:
    """Build env_vars dict from CLI arguments."""
    env_vars: Dict[str, str] = {}

    env_vars["SETU_LOG_LEVEL"] = args.log_level

    if args.nccl_socket_ifname is not None:
        env_vars["NCCL_SOCKET_IFNAME"] = args.nccl_socket_ifname

    for entry in args.env:
        if "=" not in entry:
            raise ValueError(f"Invalid --env format (expected KEY=VALUE): {entry!r}")
        key, value = entry.split("=", 1)
        env_vars[key] = value

    return env_vars if env_vars else None


def main() -> None:
    """Start a Setu cluster, display topology, and block until interrupted."""
    args = parse_args()

    if args.ray_address is not None:
        logger.info("Connecting to Ray at address=%s", args.ray_address)
        ray.init(address=args.ray_address, ignore_reinit_error=True)
    else:
        logger.info("Starting local Ray instance")
        ray.init(ignore_reinit_error=True)

    metrics_endpoint = ""
    if args.enable_metrics:
        from setu.cluster.ray.actors import _find_free_port

        metrics_endpoint = f"tcp://*:{_find_free_port()}"

    register_size = parse_num_bytes(args.register_size)

    env_vars = _build_env_vars(args)
    cluster = Cluster(
        env_vars=env_vars,
        passes=args.passes,
        metrics_endpoint=metrics_endpoint,
        register_size=register_size,
    )
    try:
        info = cluster.start()
        display_cluster_info(info)

        if args.dump_info:
            Path(args.dump_info).write_text(info.to_yaml())
            logger.info("ClusterInfo written to %s", args.dump_info)

        stop_event = threading.Event()

        def _signal_handler(signum: int, _frame: object) -> None:
            sig_name = signal.Signals(signum).name
            logger.info("Received %s, shutting down...", sig_name)
            stop_event.set()

        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)

        logger.info("Cluster is running. Press Ctrl+C to stop.")
        stop_event.wait()
    finally:
        logger.info("Stopping Setu cluster...")
        cluster.stop()
        logger.info("Setu cluster stopped.")


if __name__ == "__main__":
    main()
