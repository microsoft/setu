"""Experiment runner: cluster-agnostic orchestration for copy experiments."""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union

import torch

from setu._coordinator import Participant
from setu.cluster.handle import ClientHandle
from setu.cluster.protocol import Cluster
from setu.experiment.helpers import ShardedTensor
from setu.logger import init_logger

logger = init_logger(__name__)


class CopyMode(Enum):
    """How destination clients transfer data from sources."""

    PULL = "pull"  # one-sided: dest calls submit_pull
    COPY = "copy"  # two-sided: dest calls submit_copy


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    success: bool
    elapsed_s: float
    source_results: List[Dict] = field(default_factory=list)
    dest_results: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module-level body functions (must be picklable for subprocess backends)
# ---------------------------------------------------------------------------


def _source_body(client, participant, shard_spec, init_value):
    """Register a source shard, fill with init_value, return result dict."""
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None, f"Failed to register source shard {shard_spec.name}"
    client.wait_for_shard_allocation(shard_ref)

    with client.write(shard_ref) as tensor:
        tensor.fill_(init_value)
        if shard_spec.device.torch_device.type == "cuda":
            torch.cuda.synchronize()

    return {
        "success": True,
        "shard_name": shard_spec.name,
        "device": str(shard_spec.device),
    }


def _dest_body(
    client, participant, shard_spec, src_name, copy_mode, init_value, selections
):
    """Register a dest shard, perform copy/pull, verify, return result dict."""
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None, f"Failed to register dest shard {shard_spec.name}"
    client.wait_for_shard_allocation(shard_ref)

    src_selection = client.select(src_name)
    dst_selection = client.select(shard_spec.name)

    if selections is not None:
        for dim_name, indices in selections.items():
            src_selection = src_selection.where(dim_name, indices)
            dst_selection = dst_selection.where(dim_name, indices)

    if copy_mode == CopyMode.PULL:
        copy_op_id = client.pull(src_selection, dst_selection)
    else:
        copy_op_id = client.copy(src_selection, dst_selection)

    assert copy_op_id is not None, "Copy operation returned None"
    client.wait(copy_op_id)

    # Read back and verify
    with client.read(shard_ref) as tensor:
        actual_value = tensor.mean().item()
        values_match = abs(actual_value - init_value) < 1e-5

    return {
        "success": True,
        "shard_name": shard_spec.name,
        "device": str(shard_spec.device),
        "expected_value": init_value,
        "actual_value": actual_value,
        "values_match": values_match,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_experiment(
    cluster: Cluster,
    src: ShardedTensor,
    dst: ShardedTensor,
    copy_mode: CopyMode = CopyMode.PULL,
    init_value: float = 42.0,
    selections: Optional[Dict[str, Union[Set[int], list]]] = None,
    timeout: float = 60.0,
) -> ExperimentResult:
    """Run a copy experiment on a Setu cluster.

    Args:
        cluster: A started Cluster instance (any backend).
        src: ShardedTensor describing the source.
        dst: ShardedTensor describing the destination.
        copy_mode: PULL (one-sided) or COPY (two-sided).
        init_value: Value to fill source tensors with.
        selections: Optional dim name -> index set for partial copies.
        timeout: Timeout in seconds for the entire experiment.

    Returns:
        ExperimentResult with timing and per-shard results.
    """
    cluster_info = cluster.cluster_info
    assert cluster_info is not None, "Cluster has not been started"

    # Normalise selections to sets
    norm_selections: Optional[Dict[str, Set[int]]] = None
    if selections is not None:
        norm_selections = {k: set(v) for k, v in selections.items()}

    src_shards = src.shards
    dst_shards = dst.shards
    errors: List[str] = []
    source_results: List[Dict] = []
    dest_results: List[Dict] = []
    src_handles: List[ClientHandle] = []
    dst_handles: List[ClientHandle] = []

    t0 = time.monotonic()

    try:
        # --- Spawn source clients ---
        for shard in src_shards:
            node = cluster_info.node_for_device(shard.device)
            participant = Participant(uuid.UUID(node.node_id), shard.device)
            handle = cluster.spawn_client(
                participant,
                _source_body,
                shard,
                init_value,
            )
            src_handles.append(handle)

        # Wait for all sources to be ready
        source_results = [h.result(timeout=timeout) for h in src_handles]
        logger.info("All %d source shard(s) registered and filled", len(src_shards))

        # --- Spawn dest clients ---
        for shard in dst_shards:
            node = cluster_info.node_for_device(shard.device)
            participant = Participant(uuid.UUID(node.node_id), shard.device)
            handle = cluster.spawn_client(
                participant,
                _dest_body,
                shard,
                src.name,
                copy_mode,
                init_value,
                norm_selections,
            )
            dst_handles.append(handle)

        # Collect dest results
        dest_results = [h.result(timeout=timeout) for h in dst_handles]
        logger.info("All %d dest task(s) completed", len(dst_shards))

        # Check dest results for value mismatches
        for result in dest_results:
            if not result.get("success"):
                errors.append(
                    f"Dest shard {result.get('shard_name')} failed: "
                    f"{result.get('error')}"
                )
            elif not result.get("values_match", True):
                errors.append(
                    f"Dest shard {result['shard_name']}: value mismatch "
                    f"expected={result['expected_value']} "
                    f"actual={result['actual_value']}"
                )

    except Exception as e:
        errors.append(str(e))
        logger.error("Experiment failed: %s", e)

    finally:
        for h in dst_handles + src_handles:
            try:
                h.stop()
            except Exception:
                pass

    elapsed = time.monotonic() - t0
    return ExperimentResult(
        success=len(errors) == 0,
        elapsed_s=elapsed,
        source_results=source_results,
        dest_results=dest_results,
        errors=errors,
    )
