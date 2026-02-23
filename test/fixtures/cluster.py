"""
Cluster abstraction for E2E tests.

Re-exports SingleNodeCluster from setu.cluster as SetuTestCluster
for backwards compatibility.
"""

from setu.cluster import SingleNodeCluster

SetuTestCluster = SingleNodeCluster
