"""
Cluster abstraction for E2E tests.

Re-exports the multiprocessing Cluster as SetuTestCluster with a
convenience ``mp_context`` property used by test harnesses.
"""

from setu.cluster.multiprocessing import Cluster


class SetuTestCluster(Cluster):
    """Thin wrapper that exposes the spawn multiprocessing context."""

    @property
    def mp_context(self):
        """Return the ``spawn`` multiprocessing context used by the cluster."""
        return self._ctx
