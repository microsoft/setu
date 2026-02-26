"""Standalone entry point: python -m setu.telemetry"""

import argparse
import signal
import sys

from setu.logger import init_logger
from setu.telemetry.server import MetricsServer
from setu.telemetry.sinks.csv_sink import CSVReportSink

logger = init_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Setu telemetry metrics server",
    )
    parser.add_argument(
        "--endpoint",
        default="tcp://*:9999",
        help="ZMQ PULL bind endpoint (default: tcp://*:9999)",
    )
    parser.add_argument(
        "--output-dir",
        default="./telemetry_results",
        help="Directory for CSV report output (default: ./telemetry_results)",
    )
    args = parser.parse_args()

    csv_sink = CSVReportSink(args.output_dir)
    server = MetricsServer(endpoint=args.endpoint, sinks=[csv_sink])

    def shutdown(signum, frame):
        logger.info("Received signal %d, shutting down...", signum)
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    server.start()
    logger.info(
        "Telemetry server running (endpoint=%s, output_dir=%s). Press Ctrl+C to stop.",
        args.endpoint,
        args.output_dir,
    )

    # Block until signal
    signal.pause()


if __name__ == "__main__":
    main()
