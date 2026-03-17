"""Core Schedule type for Setu copy operations.

A Schedule describes the optimization strategy for a single copy operation.
v1 wraps a hints list; future versions will control pass selection and
pass parameters.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class Schedule:
    """Optimization strategy for a single copy operation.

    v1: hints only.  Trivially picklable since hint objects have pickle support.
    Future: passes, pass parameters, etc.
    """

    hints: List = field(default_factory=list)
    # v2: passes: Optional[List[str]] = None
