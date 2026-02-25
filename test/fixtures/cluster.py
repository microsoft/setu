"""
Cluster abstraction for E2E tests.

Re-exports the multiprocessing Cluster as SetuTestCluster.
"""

from setu.cluster.multiprocessing import Cluster as SetuTestCluster
