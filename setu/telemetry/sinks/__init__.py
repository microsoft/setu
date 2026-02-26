"""Report sinks for telemetry output."""

from setu.telemetry.sinks.base import ReportSink
from setu.telemetry.sinks.csv_sink import CSVReportSink

__all__ = ["ReportSink", "CSVReportSink"]
