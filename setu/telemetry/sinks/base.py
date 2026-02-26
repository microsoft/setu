"""Abstract base class for telemetry report sinks."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from setu.telemetry.server import CopySpecReport


class ReportSink(ABC):
    """Pluggable output destination for completed CopySpec reports."""

    @abstractmethod
    def emit(self, report: "CopySpecReport") -> None:
        """Write a completed report to the output destination."""

    def close(self) -> None:
        """Clean up resources (optional)."""
