"""Core Schedule type for Setu copy operations.

A Schedule describes the optimization strategy for a single copy operation.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Schedule:
    """Optimization strategy for a single copy operation.

    Args:
        hints: Compiler hints (e.g., PipelineChunkSizeHint).
        passes: Pass selection.  ``None`` runs all registered passes
            (default).  An empty list ``[]`` runs no passes (ablation).
            A list of pass names runs only those passes, in registered
            order.  Names use the same snake_case convention as
            ``--passes`` CLI args (e.g., ``"pipelining"``).
    """

    hints: List = field(default_factory=list)
    passes: Optional[List[str]] = None
