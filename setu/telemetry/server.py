"""Metrics server: ZMQ PULL listener with report aggregation and stall analysis."""

import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import zmq

from setu.logger import init_logger
from setu.telemetry.deserialize import (
    CompilationMetricsRecord,
    E2EMetricsRecord,
    NCCLWorkerMetricsRecord,
    deserialize_metrics_message,
)
from setu.telemetry.sinks.base import ReportSink

logger = init_logger(__name__)


@dataclass
class CopySpecReport:
    """Aggregated report for a single CopySpec execution."""

    copy_op_id: uuid.UUID
    compile_time_ms: Optional[float] = None
    e2e_time_ms: Optional[float] = None
    num_participants: Optional[int] = None
    participant_instruction_counts: Optional[List] = None
    pass_timings: Optional[List] = None
    worker_metrics: List[NCCLWorkerMetricsRecord] = field(default_factory=list)
    stall_analysis: Optional[Dict[int, Dict[str, Any]]] = None


def _compute_stall_analysis(
    worker_metrics: List[NCCLWorkerMetricsRecord],
) -> Dict[int, Dict[str, Any]]:
    """Compute per-group stall analysis from worker metrics.

    For each group index across all workers, compare group elapsed times.
    The spread (max - min) indicates stall time where faster workers waited
    at the next barrier for the slowest.
    """
    group_completions: Dict[int, list] = defaultdict(list)
    for wm in worker_metrics:
        for gt in wm.group_timings:
            group_completions[gt.group_index].append(
                {
                    "node_id": wm.node_id,
                    "device_rank": wm.device_rank,
                    "elapsed_ms": gt.elapsed_ms,
                }
            )

    stalls: Dict[int, Dict[str, Any]] = {}
    for group_idx, workers in sorted(group_completions.items()):
        if not workers:
            continue
        max_ms = max(w["elapsed_ms"] for w in workers)
        min_ms = min(w["elapsed_ms"] for w in workers)
        stalls[group_idx] = {
            "max_elapsed_ms": max_ms,
            "min_elapsed_ms": min_ms,
            "spread_ms": max_ms - min_ms,
            "slowest_workers": [w for w in workers if w["elapsed_ms"] == max_ms],
        }
    return stalls


class MetricsServer:
    """ZMQ PULL metrics server with report aggregation.

    Receives serialized MetricsMessage from C++ components, deserializes
    them, aggregates into per-CopySpec reports, and emits completed reports
    to pluggable sinks.

    Args:
        endpoint: ZMQ bind endpoint (e.g. "tcp://*:9999").
        sinks: List of ReportSink instances for output.
    """

    def __init__(
        self,
        endpoint: str = "tcp://*:9999",
        sinks: Optional[List[ReportSink]] = None,
    ) -> None:
        self._endpoint = endpoint
        self._sinks = sinks or []
        self._reports: Dict[uuid.UUID, CopySpecReport] = {}
        self._lock = threading.Lock()
        self._zmq_context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        """Start ZMQ PULL listener in a background thread."""
        self._zmq_context = zmq.Context()
        self._socket = self._zmq_context.socket(zmq.PULL)
        self._socket.bind(self._endpoint)
        self._socket.setsockopt(zmq.RCVTIMEO, 100)
        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="MetricsServerThread"
        )
        self._thread.start()
        logger.info("MetricsServer started on %s", self._endpoint)

    def stop(self) -> None:
        """Stop listener and close sockets."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._zmq_context is not None:
            self._zmq_context.term()
            self._zmq_context = None
        for sink in self._sinks:
            sink.close()
        logger.info("MetricsServer stopped")

    def get_report(self, copy_op_id: uuid.UUID) -> Optional[CopySpecReport]:
        """Get aggregated report for a single CopySpec execution."""
        with self._lock:
            return self._reports.get(copy_op_id)

    def get_all_reports(self) -> Dict[uuid.UUID, CopySpecReport]:
        """Get all collected reports."""
        with self._lock:
            return dict(self._reports)

    def _listen_loop(self) -> None:
        """Background thread: receive and process metrics messages."""
        while self._running:
            try:
                data = self._socket.recv()
            except zmq.Again:
                continue
            except zmq.ZMQError:
                if self._running:
                    logger.warning("MetricsServer: ZMQ error in recv", exc_info=True)
                break

            try:
                record = deserialize_metrics_message(data)
                self._ingest(record)
            except Exception:
                logger.warning(
                    "MetricsServer: failed to deserialize message", exc_info=True
                )

    def _ingest(self, record) -> None:
        """Process a deserialized metrics record."""
        with self._lock:
            if isinstance(record, NCCLWorkerMetricsRecord):
                report = self._get_or_create(record.copy_op_id)
                report.worker_metrics.append(record)

            elif isinstance(record, CompilationMetricsRecord):
                report = self._get_or_create(record.copy_op_id)
                report.compile_time_ms = record.total_compile_time_ms
                report.num_participants = record.num_participants
                report.participant_instruction_counts = (
                    record.participant_instruction_counts
                )
                report.pass_timings = [
                    {"pass_name": pt.pass_name, "elapsed_ms": pt.elapsed_ms}
                    for pt in record.pass_timings
                ]

            elif isinstance(record, E2EMetricsRecord):
                report = self._get_or_create(record.copy_op_id)
                report.e2e_time_ms = record.e2e_time_ms

                # E2E is the last metric received; finalize the report
                self._finalize_report(report)

    def _get_or_create(self, copy_op_id: uuid.UUID) -> CopySpecReport:
        """Get or create a CopySpecReport for the given copy_op_id."""
        if copy_op_id not in self._reports:
            self._reports[copy_op_id] = CopySpecReport(copy_op_id=copy_op_id)
        return self._reports[copy_op_id]

    def _finalize_report(self, report: CopySpecReport) -> None:
        """Compute stall analysis and emit to sinks."""
        if report.worker_metrics:
            report.stall_analysis = _compute_stall_analysis(report.worker_metrics)
        for sink in self._sinks:
            try:
                sink.emit(report)
            except Exception:
                logger.warning(
                    "MetricsServer: sink %s failed to emit", type(sink).__name__,
                    exc_info=True,
                )
