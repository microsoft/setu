"""Experiment runner: cluster-agnostic orchestration for copy experiments."""

import functools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union

import torch

from setu.cluster.handle import ClientHandle
from setu.cluster.protocol import Cluster
from setu.experiment.helpers import ShardedTensor
from setu.logger import init_logger

logger = init_logger(__name__)

# A per-dimension selection accepted by TensorSelection.where().
DimSelection = Union[int, slice, List[int], Set[int]]


class CopyMode(Enum):
    """How destination clients transfer data from sources."""

    PULL = "pull"  # one-sided: dest calls submit_pull
    COPY = "copy"  # two-sided: dest calls submit_copy


@dataclass
class ExperimentResult:
    """Result of a single experiment run.

    Attributes:
        success: True if all shards were copied and values matched, False on
            any error or mismatch.
        elapsed_s: Wall-clock time for the entire experiment (seconds).
        source_results: Per-source-shard dicts, each containing:
            - ``success`` (bool): whether the shard was registered and filled.
            - ``shard_name`` (str): tensor shard name.
            - ``device`` (str): device string the shard was placed on.
        dest_results: Per-destination-shard dicts, each containing:
            - ``success`` (bool): whether the copy completed.
            - ``shard_name`` (str): tensor shard name.
            - ``device`` (str): device string the shard was placed on.
            - ``expected_value`` (float): the init value written to the source.
            - ``actual_value`` (float): the mean value read back after copy.
            - ``values_match`` (bool): True if expected and actual are within
              1e-5 tolerance.
        errors: Human-readable error strings collected during the experiment.
    """

    success: bool
    elapsed_s: float
    source_results: List[Dict] = field(default_factory=list)
    dest_results: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module-level body functions (must be picklable for subprocess backends)
# ---------------------------------------------------------------------------


def _source_body(
    client, participant, shard_spec, init_value, copy_mode, dst_name, selections
):
    """Register a source shard, fill with init_value, participate in two-sided copy if needed.

    This is a generator: it yields once after the shard is allocated and
    filled so the orchestrator can wait for all sources to be ready before
    spawning destination clients.
    """
    t_body = time.monotonic()
    tag = f"source_body[{shard_spec.name}@{shard_spec.device}]"

    t0 = time.monotonic()
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None, f"Failed to register source shard {shard_spec.name}"
    logger.debug("%s: register_tensor_shard took %.3fs", tag, time.monotonic() - t0)

    t0 = time.monotonic()
    client.wait_for_shard_allocation(shard_ref)
    logger.debug("%s: wait_for_shard_allocation took %.3fs", tag, time.monotonic() - t0)

    t0 = time.monotonic()
    with client.write(shard_ref) as tensor:
        logger.debug(
            "%s: tensor shape=%s dtype=%s device=%s",
            tag, tensor.shape, tensor.dtype, tensor.device,
        )
        tensor.fill_(init_value)
        if shard_spec.device.torch_device.type == "cuda":
            torch.cuda.synchronize()
    logger.debug("%s: write+fill took %.3fs", tag, time.monotonic() - t0)

    logger.debug(
        "%s: ready, yielding (total setup=%.3fs)", tag, time.monotonic() - t_body
    )

    yield {
        "success": True,
        "shard_name": shard_spec.name,
        "device": str(shard_spec.device),
    }

    if copy_mode == CopyMode.COPY:
        # Two-sided: source must also call copy()
        t0 = time.monotonic()
        src_selection = client.select(shard_spec.name)
        dst_selection = client.select(dst_name)

        if selections is not None:
            for dim_name, indices in selections.items():
                src_selection = src_selection.where(dim_name, indices)
                dst_selection = dst_selection.where(dim_name, indices)

        copy_op_id = client.copy(src_selection, dst_selection)
        assert copy_op_id is not None, "Copy operation returned None"
        logger.debug("%s: submitted copy op %s, waiting...", tag, copy_op_id)
        client.wait(copy_op_id)
        logger.debug("%s: copy complete in %.3fs", tag, time.monotonic() - t0)

    logger.debug("%s: total body time=%.3fs", tag, time.monotonic() - t_body)


def _dest_body(
    client, participant, shard_spec, src_name, copy_mode, value_to_match, selections
):
    """Register a dest shard, perform copy/pull, verify, return result dict."""
    t_body = time.monotonic()
    tag = f"dest_body[{shard_spec.name}@{shard_spec.device}]"

    t0 = time.monotonic()
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None, f"Failed to register dest shard {shard_spec.name}"
    logger.debug("%s: register_tensor_shard took %.3fs", tag, time.monotonic() - t0)

    t0 = time.monotonic()
    client.wait_for_shard_allocation(shard_ref)
    logger.debug("%s: wait_for_shard_allocation took %.3fs", tag, time.monotonic() - t0)

    t0 = time.monotonic()
    src_selection = client.select(src_name)
    dst_selection = client.select(shard_spec.name)

    if selections is not None:
        for dim_name, indices in selections.items():
            src_selection = src_selection.where(dim_name, indices)
            dst_selection = dst_selection.where(dim_name, indices)
    logger.debug("%s: select+where took %.3fs", tag, time.monotonic() - t0)

    t0 = time.monotonic()
    if copy_mode == CopyMode.PULL:
        copy_op_id = client.pull(src_selection, dst_selection)
    else:
        copy_op_id = client.copy(src_selection, dst_selection)

    assert copy_op_id is not None, "Copy operation returned None"
    logger.debug(
        "%s: submit %s op %s took %.3fs, waiting...",
        tag, copy_mode.value, copy_op_id, time.monotonic() - t0,
    )

    t0 = time.monotonic()
    client.wait(copy_op_id)
    logger.debug("%s: wait(copy_op) took %.3fs", tag, time.monotonic() - t0)

    # Read back and verify
    t0 = time.monotonic()
    with client.read(shard_ref) as tensor:
        actual_value = tensor.mean().item()
        values_match = abs(actual_value - value_to_match) < 1e-5
    logger.debug(
        "%s: readback took %.3fs — expected=%s actual=%s match=%s",
        tag, time.monotonic() - t0, value_to_match, actual_value, values_match,
    )

    logger.debug("%s: total body time=%.3fs", tag, time.monotonic() - t_body)

    return {
        "success": True,
        "shard_name": shard_spec.name,
        "device": str(shard_spec.device),
        "expected_value": value_to_match,
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
    selections: Optional[Dict[str, DimSelection]] = None,
    timeout: float = 60.0,
) -> ExperimentResult:
    """Run a copy experiment on a Setu cluster.

    Args:
        cluster: A started Cluster instance (any backend).
        src: ShardedTensor describing the source.
        dst: ShardedTensor describing the destination.
        copy_mode: PULL (one-sided) or COPY (two-sided).
        init_value: Value to fill source tensors with.
        selections: Optional dim name -> selection for partial copies.
            Each value is either a ``set`` of integer indices or a ``slice``.
        timeout: Timeout in seconds for the entire experiment.

    Returns:
        ExperimentResult with timing and per-shard results.
    """
    cluster_info = cluster.cluster_info
    assert cluster_info is not None, "Cluster has not been started"

    src_shards = src.shards
    dst_shards = dst.shards
    errors: List[str] = []
    source_results: List[Dict] = []
    dest_results: List[Dict] = []
    src_handles: List[ClientHandle] = []
    dst_handles: List[ClientHandle] = []

    logger.debug(
        "run_experiment: mode=%s init_value=%s selections=%s timeout=%s",
        copy_mode,
        init_value,
        selections,
        timeout,
    )
    logger.debug(
        "run_experiment: src tensor=%s shards=%d mesh_shape=%s partition=%s",
        src.name,
        len(src_shards),
        src.mesh.shape,
        src.partition,
    )
    logger.debug(
        "run_experiment: dst tensor=%s shards=%d mesh_shape=%s partition=%s",
        dst.name,
        len(dst_shards),
        dst.mesh.shape,
        dst.partition,
    )
    for i, (p, s) in enumerate(zip(src.mesh.participants, src_shards)):
        logger.debug(
            "run_experiment: src_shard[%d] name=%s participant=%s dims=%s",
            i,
            s.name,
            p,
            [(d.name, d.size, d.start, d.end) for d in s.dims],
        )
    for i, (p, s) in enumerate(zip(dst.mesh.participants, dst_shards)):
        logger.debug(
            "run_experiment: dst_shard[%d] name=%s participant=%s dims=%s",
            i,
            s.name,
            p,
            [(d.name, d.size, d.start, d.end) for d in s.dims],
        )

    t0 = time.monotonic()

    try:
        # --- Spawn source clients ---
        t_phase = time.monotonic()
        logger.debug("run_experiment: spawning %d source clients", len(src_shards))
        for participant, shard in zip(src.mesh.participants, src_shards):
            body = functools.partial(
                _source_body,
                shard_spec=shard,
                init_value=init_value,
                copy_mode=copy_mode,
                dst_name=dst.name,
                selections=selections,
            )
            handle = cluster.spawn_client(participant, body)
            src_handles.append(handle)
        logger.debug(
            "run_experiment: spawned %d source clients in %.3fs",
            len(src_shards), time.monotonic() - t_phase,
        )

        # Wait for every source to signal that its shard is ready.
        t_phase = time.monotonic()
        logger.debug("run_experiment: waiting for source shards to be ready")
        source_results = []
        for i, h in enumerate(src_handles):
            t_wait = time.monotonic()
            r = h.next_result(timeout=timeout)
            logger.debug(
                "run_experiment: source[%d] ready in %.3fs", i, time.monotonic() - t_wait
            )
            source_results.append(r)
        logger.debug(
            "run_experiment: all %d sources ready in %.3fs",
            len(src_handles), time.monotonic() - t_phase,
        )

        # --- Spawn dest clients (sources are guaranteed ready) ---
        t_phase = time.monotonic()
        logger.debug("run_experiment: spawning %d dest clients", len(dst_shards))
        for participant, shard in zip(dst.mesh.participants, dst_shards):
            body = functools.partial(
                _dest_body,
                shard_spec=shard,
                src_name=src.name,
                copy_mode=copy_mode,
                value_to_match=init_value,
                selections=selections,
            )
            handle = cluster.spawn_client(participant, body)
            dst_handles.append(handle)
        logger.debug(
            "run_experiment: spawned %d dest clients in %.3fs",
            len(dst_shards), time.monotonic() - t_phase,
        )

        # Drain remaining source values (two-sided copy completion).
        t_phase = time.monotonic()
        logger.debug("run_experiment: draining %d source handles", len(src_handles))
        for i, h in enumerate(src_handles):
            t_drain = time.monotonic()
            try:
                while True:
                    extra = h.next_result(timeout=timeout)
                    logger.debug(
                        "run_experiment: source handle[%d] extra value: %s", i, extra
                    )
            except StopIteration:
                pass
            logger.debug(
                "run_experiment: source handle[%d] drained in %.3fs",
                i, time.monotonic() - t_drain,
            )
        logger.debug(
            "run_experiment: all sources drained in %.3fs",
            time.monotonic() - t_phase,
        )

        t_phase = time.monotonic()
        logger.debug("run_experiment: collecting %d dest results", len(dst_handles))
        dest_results = []
        for i, h in enumerate(dst_handles):
            t_wait = time.monotonic()
            r = h.result(timeout=timeout)
            logger.debug(
                "run_experiment: dest[%d] result in %.3fs: %s",
                i, time.monotonic() - t_wait, r,
            )
            dest_results.append(r)
        logger.debug(
            "run_experiment: all %d dest results collected in %.3fs",
            len(dst_handles), time.monotonic() - t_phase,
        )

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
        logger.error("Experiment failed: %s", e, exc_info=True)

    finally:
        logger.debug(
            "run_experiment: stopping %d handles", len(dst_handles) + len(src_handles)
        )
        for h in dst_handles + src_handles:
            try:
                h.stop()
            except Exception:
                pass

    elapsed = time.monotonic() - t0
    logger.debug(
        "run_experiment: finished in %.3fs, success=%s, errors=%s",
        elapsed,
        len(errors) == 0,
        errors,
    )
    return ExperimentResult(
        success=len(errors) == 0,
        elapsed_s=elapsed,
        source_results=source_results,
        dest_results=dest_results,
        errors=errors,
    )
