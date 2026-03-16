"""Metrics server: ZMQ PULL listener with report aggregation and stall analysis."""

import json
import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import zmq

from setu.logger import init_logger
from setu.telemetry.deserialize import (
    CompilationMetricsRecord,
    E2EMetricsRecord,
    NCCLWorkerMetricsRecord,
    TensorSelectionRecord,
    deserialize_metrics_message,
)
from setu.telemetry.sinks.base import ReportSink

logger = init_logger(__name__)


@dataclass
class CopySpecReport:
    """Aggregated report for a single CopySpec execution.

    Backend-agnostic: ``worker_metrics`` stores plain dicts whose shape
    depends on the backend (NCCL, future GDR, etc.).  The generic fields
    (compile, E2E, passes) are always present.
    """

    copy_op_id: uuid.UUID
    compile_time_ms: Optional[float] = None
    e2e_time_ms: Optional[float] = None
    total_bytes_transferred: Optional[int] = None
    src_name: Optional[str] = None
    dst_name: Optional[str] = None
    src_selection: Optional[Dict[str, Any]] = None
    dst_selection: Optional[Dict[str, Any]] = None
    num_participants: Optional[int] = None
    participant_instruction_counts: Optional[List] = None
    pass_timings: Optional[List] = None
    worker_metrics: List[Dict[str, Any]] = field(default_factory=list)
    stall_analysis: Optional[Dict[int, Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        d: Dict[str, Any] = {
            "copy_op_id": str(self.copy_op_id),
            "compile_time_ms": self.compile_time_ms,
            "e2e_time_ms": self.e2e_time_ms,
            "total_bytes_transferred": self.total_bytes_transferred,
            "src_name": self.src_name,
            "dst_name": self.dst_name,
            "src_selection": self.src_selection,
            "dst_selection": self.dst_selection,
            "num_participants": self.num_participants,
            "participant_instruction_counts": self.participant_instruction_counts,
            "pass_timings": self.pass_timings,
            "worker_metrics": self.worker_metrics,
        }
        if self.stall_analysis is not None:
            d["stall_analysis"] = {str(k): v for k, v in self.stall_analysis.items()}
        return d


def _selection_to_dict(
    sel: Optional[TensorSelectionRecord],
) -> Optional[Dict[str, Any]]:
    """Convert a TensorSelectionRecord to a JSON-serializable dict."""
    if sel is None:
        return None
    return {
        "name": sel.name,
        "indices": {
            dim_name: {
                "dim_size": irs.dim_size,
                "ranges": [{"start": r.start, "end": r.end} for r in irs.ranges],
            }
            for dim_name, irs in sel.indices.items()
        },
    }


def _compute_stall_analysis(
    worker_metrics: List[Dict[str, Any]],
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Compute per-group stall analysis from worker metrics.

    Only applicable to backends that provide ``group_timings`` in their
    worker metrics (e.g. NCCL).  Returns ``None`` if no group timing data
    is present.

    For each group index across all workers, compare group elapsed times.
    The spread (max - min) indicates stall time where faster workers waited
    at the next barrier for the slowest.
    """
    group_completions: Dict[int, list] = defaultdict(list)
    for wm in worker_metrics:
        for gt in wm.get("group_timings", []):
            group_completions[gt["group_index"]].append(
                {
                    "node_id": wm.get("node_id"),
                    "device_rank": wm.get("device_rank"),
                    "elapsed_ms": gt["elapsed_ms"],
                }
            )

    if not group_completions:
        return None

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


def _json_default(obj: Any) -> Any:
    """json.dumps fallback for non-serializable types."""
    if isinstance(obj, uuid.UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _make_http_handler(server_ref: "MetricsServer") -> type:
    """Create an HTTP request handler class bound to *server_ref*."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path.rstrip("/")

            if path == "/reports":
                with server_ref._lock:
                    payload = {
                        str(k): v.to_dict() for k, v in server_ref._reports.items()
                    }
                self._json_response(200, payload)

            elif path.startswith("/reports/"):
                copy_op_id_str = path[len("/reports/") :]
                try:
                    key = uuid.UUID(copy_op_id_str)
                except ValueError:
                    self._json_response(400, {"error": "invalid UUID"})
                    return
                with server_ref._lock:
                    report = server_ref._reports.get(key)
                if report is None:
                    self._json_response(404, {"error": "not found"})
                else:
                    self._json_response(200, report.to_dict())

            else:
                self._json_response(404, {"error": "not found"})

        def do_POST(self):
            path = urlparse(self.path).path.rstrip("/")

            if path == "/reset":
                server_ref.reset_reports()
                self._json_response(200, {"status": "ok"})
            else:
                self._json_response(404, {"error": "not found"})

        def _json_response(self, status: int, body: Any) -> None:
            data = json.dumps(body, default=_json_default).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            # Silence default stderr logging from BaseHTTPRequestHandler.
            pass

    return _Handler


class MetricsServer:
    """ZMQ PULL metrics server with report aggregation.

    Receives serialized MetricsMessage from C++ components, deserializes
    them, aggregates into per-CopySpec reports, and emits completed reports
    to pluggable sinks.

    Args:
        endpoint: ZMQ bind endpoint (e.g. "tcp://*:9999").
        sinks: List of ReportSink instances for output.
        http_port: Port for the HTTP query API (0 = disabled).
    """

    def __init__(
        self,
        endpoint: str = "tcp://*:9999",
        sinks: Optional[List[ReportSink]] = None,
        http_port: int = 0,
    ) -> None:
        self._endpoint = endpoint
        self._sinks = sinks or []
        self._http_port = http_port
        self._reports: Dict[uuid.UUID, CopySpecReport] = {}
        self._lock = threading.Lock()
        self._zmq_context: Optional[zmq.Context] = None
        self._socket: Optional[zmq.Socket] = None
        self._thread: Optional[threading.Thread] = None
        self._http_server: Optional[ThreadingHTTPServer] = None
        self._http_thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def http_port(self) -> int:
        """Actual HTTP port (may differ from requested port when using port 0)."""
        if self._http_server is not None:
            return self._http_server.server_address[1]
        return self._http_port

    def start(self) -> None:
        """Start ZMQ PULL listener and optional HTTP server."""
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

        if self._http_port > 0:
            handler_cls = _make_http_handler(self)
            self._http_server = ThreadingHTTPServer(
                ("0.0.0.0", self._http_port), handler_cls
            )
            self._http_thread = threading.Thread(
                target=self._http_server.serve_forever,
                daemon=True,
                name="MetricsHTTPThread",
            )
            self._http_thread.start()
            logger.info("MetricsServer HTTP API started on port %d", self.http_port)

    def stop(self) -> None:
        """Stop listener, HTTP server, and close sockets."""
        self._running = False
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=2.0)
            self._http_thread = None
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

    def reset_reports(self) -> None:
        """Clear all collected reports."""
        with self._lock:
            self._reports.clear()

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
                report.worker_metrics.append(
                    {
                        "backend": "nccl",
                        "copy_op_id": str(record.copy_op_id),
                        "node_id": str(record.node_id),
                        "device_rank": record.device_rank,
                        "group_timings": [
                            {
                                "group_index": gt.group_index,
                                "elapsed_ms": gt.elapsed_ms,
                                "num_ops": gt.num_ops,
                            }
                            for gt in record.group_timings
                        ],
                        "total_execute_ms": record.total_execute_ms,
                    }
                )

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
                report.total_bytes_transferred = record.total_bytes_transferred
                report.src_name = record.src_name
                report.dst_name = record.dst_name
                report.src_selection = _selection_to_dict(record.src_selection)
                report.dst_selection = _selection_to_dict(record.dst_selection)

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
                    "MetricsServer: sink %s failed to emit",
                    type(sink).__name__,
                    exc_info=True,
                )


class MetricsClient:
    """HTTP client for querying MetricsServer reports.

    Args:
        base_url: Base URL of the MetricsServer HTTP API (e.g. "http://localhost:8080").
    """

    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def get_all_reports(self) -> Dict[str, Any]:
        """Fetch all reports from the server."""
        url = f"{self._base_url}/reports"
        with urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def reset_reports(self) -> None:
        """Clear all reports on the server."""
        req = Request(f"{self._base_url}/reset", method="POST")
        with urlopen(req, timeout=10):
            pass

    def get_report(self, copy_op_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single report by copy_op_id. Returns None if not found."""
        url = f"{self._base_url}/reports/{copy_op_id}"
        try:
            with urlopen(url, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None
