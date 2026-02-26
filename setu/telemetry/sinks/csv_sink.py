"""CSV report sink: writes telemetry reports as CSV files."""

import csv
import os
from typing import TYPE_CHECKING

from setu.logger import init_logger
from setu.telemetry.sinks.base import ReportSink

if TYPE_CHECKING:
    from setu.telemetry.server import CopySpecReport

logger = init_logger(__name__)


class CSVReportSink(ReportSink):
    """Writes CopySpec reports as CSV files in an output directory.

    Produces three files (appended to across multiple emit() calls):
    - summary.csv: one row per CopySpec.
    - worker_timings.csv: one row per worker per group.
    - stall_analysis.csv: one row per group.
    """

    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self._headers_written = {
            "summary": False,
            "worker_timings": False,
            "stall_analysis": False,
        }

    def emit(self, report: "CopySpecReport") -> None:
        self._write_summary(report)
        self._write_worker_timings(report)
        self._write_stall_analysis(report)
        logger.debug("CSVReportSink: wrote report for copy_op_id=%s", report.copy_op_id)

    def _write_summary(self, report: "CopySpecReport") -> None:
        path = os.path.join(self._output_dir, "summary.csv")
        write_header = not self._headers_written["summary"]
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "copy_op_id",
                        "compile_time_ms",
                        "e2e_time_ms",
                        "num_participants",
                    ]
                )
                self._headers_written["summary"] = True
            writer.writerow(
                [
                    str(report.copy_op_id),
                    report.compile_time_ms,
                    report.e2e_time_ms,
                    report.num_participants,
                ]
            )

    def _write_worker_timings(self, report: "CopySpecReport") -> None:
        path = os.path.join(self._output_dir, "worker_timings.csv")
        write_header = not self._headers_written["worker_timings"]
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "copy_op_id",
                        "node_id",
                        "device_rank",
                        "group_index",
                        "elapsed_ms",
                        "num_ops",
                    ]
                )
                self._headers_written["worker_timings"] = True
            for wm in report.worker_metrics:
                for gt in wm.group_timings:
                    writer.writerow(
                        [
                            str(report.copy_op_id),
                            str(wm.node_id),
                            wm.device_rank,
                            gt.group_index,
                            gt.elapsed_ms,
                            gt.num_ops,
                        ]
                    )

    def _write_stall_analysis(self, report: "CopySpecReport") -> None:
        if not report.stall_analysis:
            return
        path = os.path.join(self._output_dir, "stall_analysis.csv")
        write_header = not self._headers_written["stall_analysis"]
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(
                    [
                        "copy_op_id",
                        "group_index",
                        "max_elapsed_ms",
                        "min_elapsed_ms",
                        "spread_ms",
                    ]
                )
                self._headers_written["stall_analysis"] = True
            for group_idx, analysis in sorted(report.stall_analysis.items()):
                writer.writerow(
                    [
                        str(report.copy_op_id),
                        group_idx,
                        analysis["max_elapsed_ms"],
                        analysis["min_elapsed_ms"],
                        analysis["spread_ms"],
                    ]
                )

    def close(self) -> None:
        pass
