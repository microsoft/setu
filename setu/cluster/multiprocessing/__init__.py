"""Multiprocessing-based Setu cluster implementation.

Provides a cluster that spawns coordinator and node agents as child
processes using ``torch.multiprocessing``.
"""

from setu.cluster.multiprocessing.cluster import Cluster

__all__ = [
    "Cluster",
]
