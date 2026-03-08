"""Shared pass resolution for all cluster backends.

Maps string pass names to C++ Pass objects via a registry, allowing
cluster specs and CLI tools to refer to passes by name.
"""

from typing import List, Optional

from setu._coordinator import Topology

_PASS_REGISTRY = {
    "shortest_path_routing": lambda topo, **_: _make_shortest_path_routing(topo),
    "bandwidth_aggregation": lambda topo, **_: _make_bandwidth_aggregation(topo),
    "pipelining": lambda topo, **kw: _make_pipelining(
        kw.get("chunk_size_elements", 33554432)
    ),
    "register_tiling": lambda topo, **_: _make_register_tiling(),
    "instruction_scheduler": lambda topo, **_: _make_instruction_scheduler(),
}

AVAILABLE_PASSES: list = list(_PASS_REGISTRY.keys())


def _make_shortest_path_routing(topology: Optional[Topology]):
    from setu._coordinator import ShortestPathRouting

    return ShortestPathRouting(topology)


def _make_bandwidth_aggregation(topology: Optional[Topology]):
    from setu._coordinator import BandwidthAggregation

    return BandwidthAggregation(topology)


def _make_register_tiling():
    from setu._coordinator import RegisterTiling

    return RegisterTiling()


def _make_pipelining(chunk_size_elements: int):
    from setu._coordinator import Pipelining

    return Pipelining(chunk_size_elements)


def _make_instruction_scheduler():
    from setu._coordinator import InstructionScheduler

    return InstructionScheduler()


def resolve_passes(
    passes: Optional[List[str]],
    topology: Optional[Topology] = None,
) -> list:
    """Build Pass objects from a list of pass names.

    Args:
        passes: Pass names to resolve.
            ``None`` → default (ShortestPathRouting when topology is set).
            ``[]``   → no passes (useful for ablation).
        topology: Optional topology for passes that need it.

    Returns:
        List of constructed Pass objects.

    Raises:
        ValueError: If a pass name is unknown or a required dependency
            (e.g. topology) is missing.
    """
    if passes is None:
        # Default: add ShortestPathRouting when topology is available
        if topology is not None:
            from setu._coordinator import ShortestPathRouting

            return [ShortestPathRouting(topology)]
        return []

    result = []
    for name in passes:
        factory = _PASS_REGISTRY.get(name)
        if factory is None:
            raise ValueError(f"Unknown pass: {name!r}. Known: {list(_PASS_REGISTRY)}")
        result.append(factory(topology))
    return result
