import ray
from setu.ray import ClusterInfo, NodeAgentInfo, SetuCluster
import time

# connect to ray cluster
ray.init()

# setup setu cluster
setu_cluster = SetuCluster()
info = setu_cluster.start()

print(info)

while True:
    time.sleep(1)