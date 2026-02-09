"""
CLI entry point for starting a Setu Ray cluster.

Usage::

    setu-cluster
    setu-cluster --ray-address ray://10.0.0.1:10001
    python -m setu.ray
"""

import argparse
import signal
import threading

import ray
from rich.console import Console
from rich.table import Table

from setu.logger import init_logger
from setu.ray.cluster import ClusterInfo, SetuCluster

logger = init_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the cluster CLI."""
    parser = argparse.ArgumentParser(
        description="Start a Setu cluster on an existing Ray cluster."
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default="auto",
        help='Ray cluster address to connect to (default: "auto").',
    )
    return parser.parse_args()


def display_cluster_info(info: ClusterInfo) -> None:
    """Display cluster topology using a rich table."""
    console = Console()

    console.rule("Setu Cluster Info")
    console.print(f"  Coordinator: {info.coordinator_endpoint}")
    console.print(
        f"  Nodes: {info.num_nodes}  |  Total GPUs: {info.total_gpus}"
    )
    console.print()

    table = Table(title="Node Agents")
    table.add_column("Node ID", style="cyan")
    table.add_column("IP Address", style="green")
    table.add_column("Endpoint", style="magenta")
    table.add_column("GPUs", justify="right", style="bold")

    for na in info.node_agents:
        table.add_row(
            na.node_id,
            na.ip_address,
            na.node_agent_endpoint,
            str(na.num_gpus),
        )

    console.print(table)
    console.print()


def main() -> None:
    """Start a Setu cluster, display topology, and block until interrupted."""
    args = parse_args()

    logger.info("Connecting to Ray at address=%s", args.ray_address)
    ray.init(address=args.ray_address, ignore_reinit_error=True)

    cluster = SetuCluster()
    try:
        info = cluster.start()
        display_cluster_info(info)

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
