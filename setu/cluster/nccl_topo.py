"""Load a Setu Topology from an NCCL topology XML dump.

Usage::

    from setu.cluster.nccl_topo import load_nccl_topo

    topo = load_nccl_topo("nccl_topo.xml", node_id=my_node_id)
"""

import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import torch

from setu._commons.datatypes import Device
from setu._coordinator import Link, Participant, Topology
from setu.logger import init_logger

logger = init_logger(__name__)

# Per-NVLink unidirectional bandwidth by SM version (Gbps).
_NVLINK_BW_GBPS_PER_LINK: Dict[int, float] = {
    70: 200.0,  # V100  — NVLink2, 25 GB/s per link
    80: 200.0,  # A100  — NVLink3, 25 GB/s per link
    86: 200.0,  # A30   — NVLink3, 25 GB/s per link
    90: 400.0,  # H100  — NVLink4, 50 GB/s per link
}
_DEFAULT_NVLINK_BW_GBPS_PER_LINK = 200.0

# Latency estimates (μs) — order-of-magnitude for routing decisions.
_NVLINK_LATENCY_US = 1.0
_PCIE_INTRA_NUMA_LATENCY_US = 2.0
_PCIE_INTER_NUMA_LATENCY_US = 5.0


@dataclass
class _GpuInfo:
    dev: int  # CUDA device index
    busid: str  # PCI bus ID
    sm: int  # SM version
    numa_id: int  # NUMA node
    pcie_bw_gbps: float  # PCIe bandwidth (Gbps)
    nvlink_targets: List[Tuple[str, int]] = field(
        default_factory=list
    )  # (target_busid, count)


def _parse_pcie_bw_gbps(link_speed: str, link_width: int) -> float:
    """Convert PCIe link_speed + width to effective bandwidth in Gbps.

    Example: "16.0 GT/s PCIe", width=16 → ~252 Gbps (Gen4 x16).
    """
    gts = float(link_speed.split()[0])
    # 128b/130b encoding for PCIe Gen3+
    return gts * link_width * 128.0 / 130.0


def _parse_gpus(xml_path: str) -> List[_GpuInfo]:
    """Parse GPU info from an NCCL topology XML dump."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gpus: List[_GpuInfo] = []
    for cpu_elem in root.iter("cpu"):
        numa_id = int(cpu_elem.get("numaid", "0"))

        for pci_elem in cpu_elem.iter("pci"):
            gpu_elem = pci_elem.find("gpu")
            if gpu_elem is None:
                continue

            busid = pci_elem.get("busid", "")
            link_speed = pci_elem.get("link_speed", "16.0 GT/s PCIe")
            link_width = int(pci_elem.get("link_width", "16"))

            dev = int(gpu_elem.get("dev", "-1"))
            sm = int(gpu_elem.get("sm", "0"))

            nvlink_targets = []
            for nvlink_elem in gpu_elem.iter("nvlink"):
                target = nvlink_elem.get("target", "")
                count = int(nvlink_elem.get("count", "1"))
                nvlink_targets.append((target, count))

            gpus.append(
                _GpuInfo(
                    dev=dev,
                    busid=busid,
                    sm=sm,
                    numa_id=numa_id,
                    pcie_bw_gbps=_parse_pcie_bw_gbps(link_speed, link_width),
                    nvlink_targets=nvlink_targets,
                )
            )

    gpus.sort(key=lambda g: g.dev)
    return gpus


def load_nccl_topo(
    xml_path: str,
    node_id: uuid.UUID,
) -> Topology:
    """Build a Setu Topology from an NCCL topology XML dump.

    Parses the XML produced by ``NCCL_TOPO_DUMP_FILE`` and creates a
    :class:`Topology` with:

    - **NVLink** edges for directly connected GPU pairs (high BW, low latency).
    - **PCIe** edges for all other GPU pairs, with different latencies for
      intra-NUMA vs cross-NUMA communication.

    Args:
        xml_path: Path to the NCCL topology XML file.
        node_id: UUID identifying the node these GPUs belong to.

    Returns:
        A populated Topology.
    """
    gpus = _parse_gpus(xml_path)
    assert len(gpus) > 0, f"No GPUs found in {xml_path}"

    logger.info(
        "Parsed %d GPUs from %s (NUMA nodes: %s)",
        len(gpus),
        xml_path,
        sorted({g.numa_id for g in gpus}),
    )

    # Map PCI bus ID → GPU info for NVLink target resolution.
    busid_to_gpu: Dict[str, _GpuInfo] = {g.busid: g for g in gpus}

    # Build participants.
    participants: Dict[int, Participant] = {}
    for gpu in gpus:
        device = Device(torch_device=torch.device(f"cuda:{gpu.dev}"))
        participants[gpu.dev] = Participant(node_id, device)

    topo = Topology()

    # Track which pairs have NVLink so we can skip PCIe for those.
    nvlink_pairs: Set[Tuple[int, int]] = set()

    # Add NVLink edges.
    for gpu in gpus:
        bw_per_link = _NVLINK_BW_GBPS_PER_LINK.get(
            gpu.sm, _DEFAULT_NVLINK_BW_GBPS_PER_LINK
        )
        for target_busid, count in gpu.nvlink_targets:
            target_gpu = busid_to_gpu.get(target_busid)
            if target_gpu is None:
                continue

            pair = (min(gpu.dev, target_gpu.dev), max(gpu.dev, target_gpu.dev))
            if pair in nvlink_pairs:
                continue  # Already added bidirectionally.
            nvlink_pairs.add(pair)

            total_bw = bw_per_link * count
            link = Link(
                _NVLINK_LATENCY_US,
                total_bw,
                tag=f"nvlink_x{count}",
            )
            topo.add_bidirectional_link(
                participants[gpu.dev],
                participants[target_gpu.dev],
                link,
            )
            logger.debug(
                "NVLink: cuda:%d <-> cuda:%d  %d links, %.0f Gbps",
                gpu.dev,
                target_gpu.dev,
                count,
                total_bw,
            )

    # Add PCIe edges for pairs without NVLink.
    for i, g1 in enumerate(gpus):
        for g2 in gpus[i + 1 :]:
            pair = (g1.dev, g2.dev)
            if pair in nvlink_pairs:
                continue

            same_numa = g1.numa_id == g2.numa_id
            latency = (
                _PCIE_INTRA_NUMA_LATENCY_US
                if same_numa
                else _PCIE_INTER_NUMA_LATENCY_US
            )
            bw = min(g1.pcie_bw_gbps, g2.pcie_bw_gbps)
            tag = "pcie" if same_numa else "pcie_cross_numa"

            link = Link(latency, bw, tag=tag)
            topo.add_bidirectional_link(
                participants[g1.dev],
                participants[g2.dev],
                link,
            )
            logger.debug(
                "PCIe%s: cuda:%d <-> cuda:%d  %.0f Gbps",
                "" if same_numa else " (cross-NUMA)",
                g1.dev,
                g2.dev,
                bw,
            )

    edges = topo.get_edges()
    logger.info(
        "Topology: %d GPUs, %d NVLink pairs, %d total directed edges",
        len(gpus),
        len(nvlink_pairs),
        len(edges),
    )

    return topo
