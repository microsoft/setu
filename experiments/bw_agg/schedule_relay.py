"""Schedule file for bandwidth aggregation relay routing.

Constructs BandwidthHint with a direct path and a relay path through
intermediate devices.  Configurable via environment variables:

    SETU_BW_AGG_GROUP  — GPU group: 0 (devices 0-3) or 1 (devices 4-7).
                         Default: 1.
    SETU_BW_AGG_PIPELINE_CHUNK_SIZE — Pipeline chunk size in bytes.
                                      Optional; omit for pass default.

Usage::

    python -m setu.bench --cluster-info cluster.yaml \\
        --schedule experiments/bw_agg/schedule_relay.py \\
        --src 0:4 --dst 0:7
"""

import os
import uuid

from setu._coordinator import BandwidthHint, Link, Participant, Path as RoutePath, PipelineChunkSizeHint
from setu.schedule import Schedule


GROUP = int(os.environ.get("SETU_BW_AGG_GROUP", "1"))
PIPELINE_CHUNK_SIZE = os.environ.get("SETU_BW_AGG_PIPELINE_CHUNK_SIZE", None)


def schedule(ctx):
    """Build relay routing hints for the configured GPU group."""
    cluster_info = ctx.cluster_info
    node_id = uuid.UUID(cluster_info.nodes[0].node_id)
    devices = cluster_info.nodes[0].devices

    base = GROUP * 4

    hints = [
        BandwidthHint(
            src=Participant(node_id, devices[base]),
            dst=Participant(node_id, devices[base + 3]),
            paths=[
                RoutePath(
                    hops=[
                        Participant(node_id, devices[base]),
                        Participant(node_id, devices[base + 3]),
                    ],
                    links=[Link(0.0, 1.0)],
                ),
                RoutePath(
                    hops=[
                        Participant(node_id, devices[base]),
                        Participant(node_id, devices[base + 1]),
                        Participant(node_id, devices[base + 2]),
                        Participant(node_id, devices[base + 3]),
                    ],
                    links=[Link(0.0, 1.0), Link(0.0, 1.0)],
                ),
            ],
            weights=[0.5, 0.5],
        )
    ]

    if PIPELINE_CHUNK_SIZE is not None:
        from setu.utils.parsing import parse_num_bytes

        hints.append(PipelineChunkSizeHint(parse_num_bytes(PIPELINE_CHUNK_SIZE)))

    return Schedule(hints=hints)
