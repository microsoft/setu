"""Schedule loading for bench: CopyContext, ScheduleSpec, and module loader.

A *schedule file* is a Python module exposing a ``schedule(ctx)`` function
that returns a :class:`~setu.schedule.Schedule` (or a plain list of hints
for convenience).  ``ScheduleSpec`` is a picklable wrapper: it stores only
the absolute path so Ray workers can re-load the module independently.
"""

import importlib.util
import os
from dataclasses import dataclass
from typing import Callable, Optional

from setu.cluster.info import ClusterInfo
from setu.schedule import Schedule


@dataclass(frozen=True)
class CopyContext:
    """Context available to a schedule factory for each copy submission."""

    src_name: str
    dst_name: str
    cluster_info: ClusterInfo
    total_bytes: int
    round_index: int  # 0-based measured round; -1 for warmup


def _load_module_fn(path: str) -> Callable:
    """Load a schedule module and return its ``schedule`` function."""
    abs_path = os.path.abspath(path)
    assert os.path.isfile(abs_path), f"Schedule file not found: {abs_path}"

    spec = importlib.util.spec_from_file_location("_setu_schedule_module", abs_path)
    assert spec is not None, f"Failed to create module spec from {abs_path}"
    assert spec.loader is not None, f"Module spec has no loader: {abs_path}"

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert hasattr(
        mod, "schedule"
    ), f"Schedule module {abs_path} must define a 'schedule' function"
    fn = getattr(mod, "schedule")
    assert callable(fn), f"'schedule' attribute in {abs_path} is not callable"
    return fn


class ScheduleSpec:
    """Picklable wrapper around a schedule file path.

    Stores only the absolute path.  Lazily loads the module on first
    ``__call__``.  Pickle serializes just the path; workers re-load
    independently.
    """

    def __init__(self, path: str):
        self._path = os.path.abspath(path)
        self._fn: Optional[Callable] = None

    def __call__(self, ctx: CopyContext) -> Schedule:
        if self._fn is None:
            self._fn = _load_module_fn(self._path)
        result = self._fn(ctx)
        # Allow schedule functions to return a plain list of hints
        if isinstance(result, list):
            return Schedule(hints=result)
        assert isinstance(
            result, Schedule
        ), f"Schedule function must return Schedule or list, got {type(result)}"
        return result

    def __getstate__(self):
        return {"_path": self._path}

    def __setstate__(self, state):
        self._path = state["_path"]
        self._fn = None

    def __repr__(self):
        return f"ScheduleSpec({self._path!r})"
