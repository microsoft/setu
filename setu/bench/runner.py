"""Benchmark runner: orchestration for copy experiments on a Setu cluster.

Uses an SPMD barrier-based approach inspired by nccl-test: all processes
run the same code path autonomously, coordinating via barriers.  The
parent just spawns them and collects final results.

Backend-agnostic — takes a ``ClusterInfo`` and an ``ExperimentBackend``
for client spawning and barrier synchronization.  Defaults to
:class:`~setu.bench.backends.RayBackend` when no backend is given.
"""

import functools
import time
from typing import Dict, List, Optional, Set, Union

import torch

from setu.bench.helpers import ShardedTensor
from setu.bench.result import CopyMode, ExperimentResult
from setu.cluster.info import ClusterInfo
from setu.logger import init_logger

logger = init_logger(__name__)

# A per-dimension selection accepted by TensorSelection.where().
DimSelection = Union[int, slice, List[int], Set[int]]


# ---------------------------------------------------------------------------
# Module-level body functions (must be picklable for Ray actor serialization)
# ---------------------------------------------------------------------------


def _source_body(
    client,
    participant,
    barrier,
    shard_spec,
    init_value,
    copy_mode,
    dst_name,
    selections,
    n_warmup_rounds,
    n_copy_rounds,
    blocking,
    hints=None,
):
    """Register a source shard, fill with init_value, then run N copy rounds.

    Plain function (no generator).  Uses *barrier* for SPMD coordination:
        1. Register shard, fill with init_value.
        2. barrier.wait()   -- all shards registered before any copies.
        3. For each round:
             barrier.wait()   -- all ready before round starts.
             <timed copy work>
             barrier.wait()   -- all done before next round.

    Returns a dict with results.
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
            tag,
            tensor.shape,
            tensor.dtype,
            tensor.device,
        )
        tensor.fill_(init_value)
        if shard_spec.device.torch_device.type == "cuda":
            torch.cuda.synchronize()
    logger.debug("%s: write+fill took %.3fs", tag, time.monotonic() - t0)

    logger.debug(
        "%s: ready, entering barrier (total setup=%.3fs)",
        tag,
        time.monotonic() - t_body,
    )

    barrier.wait()  # all registered

    def _submit_source_copy():
        src_selection = client.select(shard_spec.name)
        dst_selection = client.select(dst_name)
        if selections is not None:
            for dim_name, indices in selections.items():
                src_selection = src_selection.where(dim_name, indices)
                dst_selection = dst_selection.where(dim_name, indices)
        op_id = client.copy(src_selection, dst_selection, hints=hints)
        assert op_id is not None, "Copy operation returned None"
        return op_id

    # Warmup rounds — always blocking so NCCL communicators are fully initialised
    # before the timed section.
    warmup_times = []
    for round_i in range(n_warmup_rounds):
        barrier.wait()  # round start
        t_round = time.monotonic()
        if copy_mode == CopyMode.COPY:
            client.wait(_submit_source_copy())
        elapsed = time.monotonic() - t_round
        barrier.wait()  # round end
        warmup_times.append(elapsed)
        logger.debug("%s: warmup %d complete in %.3fs", tag, round_i, elapsed)

    # Measured rounds.
    round_times = []
    if blocking:
        for round_i in range(n_copy_rounds):
            barrier.wait()  # round start
            t_round = time.monotonic()
            if copy_mode == CopyMode.COPY:
                op_id = _submit_source_copy()
                logger.debug(
                    "%s: round %d, submitted copy op %s, waiting...",
                    tag,
                    round_i,
                    op_id,
                )
                client.wait(op_id)
            elapsed = time.monotonic() - t_round
            barrier.wait()  # round end
            round_times.append(elapsed)
            logger.debug("%s: round %d complete in %.3fs", tag, round_i, elapsed)
    else:
        # Non-blocking: submit all copies, then wait for all.  Mirrors nccl-test
        # -C 0 behaviour where ops are queued and synced once at the end.
        barrier.wait()  # all ranks start together
        t_all = time.monotonic()
        op_ids = []
        if copy_mode == CopyMode.COPY:
            for round_i in range(n_copy_rounds):
                op_ids.append(_submit_source_copy())
                logger.debug("%s: round %d, submitted (non-blocking)", tag, round_i)
        for op_id in op_ids:
            client.wait(op_id)
        total_elapsed = time.monotonic() - t_all
        barrier.wait()  # all ranks done
        per_round = total_elapsed / n_copy_rounds if n_copy_rounds > 0 else 0.0
        round_times = [per_round] * n_copy_rounds
        logger.debug(
            "%s: non-blocking batch complete in %.3fs (%.3f/round)",
            tag,
            total_elapsed,
            per_round,
        )

    logger.debug("%s: total body time=%.3fs", tag, time.monotonic() - t_body)

    return {
        "role": "source",
        "success": True,
        "shard_name": shard_spec.name,
        "device": str(shard_spec.device.torch_device),
        "round_elapsed_s": warmup_times + round_times,
    }


def _dest_body(
    client,
    participant,
    barrier,
    shard_spec,
    src_name,
    copy_mode,
    value_to_match,
    selections,
    n_warmup_rounds,
    n_copy_rounds,
    blocking,
    hints=None,
):
    """Register a dest shard, then run N copy rounds with verification on the last.

    Plain function (no generator).  Uses *barrier* for SPMD coordination
    with the same barrier call pattern as ``_source_body``.

    Returns a dict with results including value verification.
    """
    t_body = time.monotonic()
    tag = f"dest_body[{shard_spec.name}@{shard_spec.device}]"

    t0 = time.monotonic()
    shard_ref = client.register_tensor_shard(shard_spec)
    assert shard_ref is not None, f"Failed to register dest shard {shard_spec.name}"
    logger.debug("%s: register_tensor_shard took %.3fs", tag, time.monotonic() - t0)

    t0 = time.monotonic()
    client.wait_for_shard_allocation(shard_ref)
    logger.debug("%s: wait_for_shard_allocation took %.3fs", tag, time.monotonic() - t0)

    logger.debug(
        "%s: ready, entering barrier (total setup=%.3fs)",
        tag,
        time.monotonic() - t_body,
    )

    barrier.wait()  # all registered

    def _submit_dest_copy():
        src_selection = client.select(src_name)
        dst_selection = client.select(shard_spec.name)
        if selections is not None:
            for dim_name, indices in selections.items():
                src_selection = src_selection.where(dim_name, indices)
                dst_selection = dst_selection.where(dim_name, indices)
        if copy_mode == CopyMode.PULL:
            op_id = client.pull(src_selection, dst_selection, hints=hints)
        else:
            op_id = client.copy(src_selection, dst_selection, hints=hints)
        assert op_id is not None, "Copy operation returned None"
        return op_id

    # Warmup rounds — always blocking.
    warmup_times = []
    for round_i in range(n_warmup_rounds):
        barrier.wait()  # round start
        t_round = time.monotonic()
        client.wait(_submit_dest_copy())
        elapsed = time.monotonic() - t_round
        barrier.wait()  # round end
        warmup_times.append(elapsed)
        logger.debug("%s: warmup %d complete in %.3fs", tag, round_i, elapsed)

    # Measured rounds.
    round_times = []
    if blocking:
        for round_i in range(n_copy_rounds):
            barrier.wait()  # round start
            t_round = time.monotonic()
            op_id = _submit_dest_copy()
            logger.debug(
                "%s: round %d, submit %s op %s, waiting...",
                tag,
                round_i,
                copy_mode.value,
                op_id,
            )
            client.wait(op_id)
            elapsed = time.monotonic() - t_round
            barrier.wait()  # round end
            round_times.append(elapsed)
            logger.debug("%s: round %d complete in %.3fs", tag, round_i, elapsed)
    else:
        # Non-blocking: queue all copies, sync once at the end.
        barrier.wait()  # all ranks start together
        t_all = time.monotonic()
        op_ids = [_submit_dest_copy() for round_i in range(n_copy_rounds)]
        logger.debug(
            "%s: submitted %d %s ops (non-blocking)",
            tag,
            n_copy_rounds,
            copy_mode.value,
        )
        for op_id in op_ids:
            client.wait(op_id)
        total_elapsed = time.monotonic() - t_all
        barrier.wait()  # all ranks done
        per_round = total_elapsed / n_copy_rounds if n_copy_rounds > 0 else 0.0
        round_times = [per_round] * n_copy_rounds
        logger.debug(
            "%s: non-blocking batch complete in %.3fs (%.3f/round)",
            tag,
            total_elapsed,
            per_round,
        )

    # Verify values after all rounds complete
    t_read = time.monotonic()
    with client.read(shard_ref) as tensor:
        actual_value = tensor.mean().item()
        values_match = abs(actual_value - value_to_match) < 1e-5
    logger.debug(
        "%s: readback took %.3fs — expected=%s actual=%s match=%s",
        tag,
        time.monotonic() - t_read,
        value_to_match,
        actual_value,
        values_match,
    )

    logger.debug("%s: total body time=%.3fs", tag, time.monotonic() - t_body)

    return {
        "role": "dest",
        "success": True,
        "shard_name": shard_spec.name,
        "device": str(shard_spec.device.torch_device),
        "round_elapsed_s": warmup_times + round_times,
        "expected_value": value_to_match,
        "actual_value": actual_value,
        "values_match": values_match,
    }


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------


def _collect_results(
    handles,
    timeout: float,
    poll_interval: float,
):
    """Collect results from all handles, polling with *poll_interval*."""
    import queue as _queue

    results = [None] * len(handles)
    remaining_indices = list(range(len(handles)))
    t0 = time.monotonic()
    n_done = 0

    while remaining_indices:
        deadline = t0 + timeout
        still_pending = []

        for idx in remaining_indices:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Experiment timed out after {timeout}s "
                    f"({n_done}/{len(handles)} clients finished)"
                )
            try:
                results[idx] = handles[idx].result(
                    timeout=min(poll_interval, remaining)
                )
                n_done += 1
                logger.debug(
                    "run_experiment: client %d/%d finished", n_done, len(handles)
                )
            except _queue.Empty:
                still_pending.append(idx)

        remaining_indices = still_pending

    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_experiment(
    cluster_info: ClusterInfo,
    src: ShardedTensor,
    dst: ShardedTensor,
    copy_mode: CopyMode = CopyMode.PULL,
    init_value: float = 42.0,
    selections: Optional[Dict[str, DimSelection]] = None,
    timeout: float = 60.0,
    n_copy_rounds: int = 1,
    n_warmup_rounds: int = 0,
    blocking: bool = True,
    metrics_http_url: str = "",
    hints: Optional[List] = None,
) -> ExperimentResult:
    """Run a copy experiment on a Setu cluster.

    Spawns all source and destination processes which self-coordinate via
    a barrier.  The parent just waits for final results and aggregates.

    Args:
        cluster_info: A ClusterInfo describing the running cluster.
        src: ShardedTensor describing the source.
        dst: ShardedTensor describing the destination.
        copy_mode: PULL (one-sided) or COPY (two-sided).
        init_value: Value to fill source tensors with.
        selections: Optional dim name -> selection for partial copies.
            Each value is either a ``set`` of integer indices or a ``slice``.
        timeout: Timeout in seconds for the entire experiment.
        n_copy_rounds: Number of *measured* copy rounds to execute.
        n_warmup_rounds: Number of warmup rounds executed before the measured
            rounds.  Warmup rounds run the same copy logic but their timings
            are excluded from the reported results.  Use this to absorb
            one-time costs like NCCL communicator initialisation.

    Returns:
        ExperimentResult with timing and per-shard results.
    """
    from setu.bench.backends import backend_for

    backend = backend_for(cluster_info)

    src_shards = src.shards
    dst_shards = dst.shards
    errors: List[str] = []
    handles = []

    n_total = len(src_shards) + len(dst_shards)
    shard_bytes = src.shard_bytes

    logger.debug(
        "run_experiment: mode=%s init_value=%s selections=%s timeout=%s "
        "n_copy_rounds=%d n_warmup_rounds=%d n_total_clients=%d",
        copy_mode,
        init_value,
        selections,
        timeout,
        n_copy_rounds,
        n_warmup_rounds,
        n_total,
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

    # Reset telemetry so only this experiment's reports are collected.
    if metrics_http_url:
        try:
            from setu.telemetry.server import MetricsClient

            MetricsClient(metrics_http_url).reset_reports()
        except Exception:
            logger.warning("run_experiment: failed to reset metrics", exc_info=True)

    t0 = time.monotonic()

    try:
        # Create barriers via backend
        barriers = backend.create_barrier(n_total)

        # Spawn ALL processes (sources + dests) — they self-coordinate via barrier
        rank = 0
        t_spawn = time.monotonic()
        logger.debug("run_experiment: spawning %d source clients", len(src_shards))
        for participant, shard in zip(src.mesh.participants, src_shards):
            body = functools.partial(
                _source_body,
                barrier=barriers[rank],
                shard_spec=shard,
                init_value=init_value,
                copy_mode=copy_mode,
                dst_name=dst.name,
                selections=selections,
                n_warmup_rounds=n_warmup_rounds,
                n_copy_rounds=n_copy_rounds,
                blocking=blocking,
                hints=hints,
            )
            handles.append(backend.spawn_client(cluster_info, participant, body))
            rank += 1

        logger.debug("run_experiment: spawning %d dest clients", len(dst_shards))
        for participant, shard in zip(dst.mesh.participants, dst_shards):
            body = functools.partial(
                _dest_body,
                barrier=barriers[rank],
                shard_spec=shard,
                src_name=src.name,
                copy_mode=copy_mode,
                value_to_match=init_value,
                selections=selections,
                n_warmup_rounds=n_warmup_rounds,
                n_copy_rounds=n_copy_rounds,
                blocking=blocking,
                hints=hints,
            )
            handles.append(backend.spawn_client(cluster_info, participant, body))
            rank += 1

        logger.debug(
            "run_experiment: spawned %d clients in %.3fs, waiting for results",
            n_total,
            time.monotonic() - t_spawn,
        )

        # Wait for all results, periodically checking cluster health.
        _HEALTH_POLL_S = 2.0
        results = _collect_results(
            handles,
            timeout,
            _HEALTH_POLL_S,
        )

        # Aggregate — strip warmup round timings from per-client results
        source_results = [r for r in results if r["role"] == "source"]
        dest_results = [r for r in results if r["role"] == "dest"]

        warmup_round_elapsed_s = []
        if n_warmup_rounds > 0:
            all_client_results = source_results + dest_results
            warmup_round_elapsed_s = [
                max(r["round_elapsed_s"][i] for r in all_client_results)
                for i in range(n_warmup_rounds)
            ]
            for r in all_client_results:
                r["round_elapsed_s"] = r["round_elapsed_s"][n_warmup_rounds:]

        round_elapsed_s = [
            max(r["round_elapsed_s"][i] for r in source_results + dest_results)
            for i in range(n_copy_rounds)
        ]

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
        source_results = []
        dest_results = []
        round_elapsed_s = []
        warmup_round_elapsed_s = []

    finally:
        logger.debug("run_experiment: stopping %d handles", len(handles))
        for h in handles:
            try:
                h.stop()
            except Exception:
                pass

    # Collect telemetry metrics if HTTP endpoint is available.
    metrics_reports = None
    if metrics_http_url:
        time.sleep(0.2)  # Let trailing metrics arrive at the server.
        try:
            from setu.telemetry.server import MetricsClient

            client = MetricsClient(metrics_http_url)
            metrics_reports = client.get_all_reports()
            logger.info(
                "run_experiment: collected %d metrics reports",
                len(metrics_reports) if metrics_reports else 0,
            )
        except Exception:
            logger.warning("run_experiment: failed to collect metrics", exc_info=True)

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
        round_elapsed_s=round_elapsed_s,
        source_results=source_results,
        dest_results=dest_results,
        errors=errors,
        copy_mode=copy_mode,
        shard_bytes=shard_bytes,
        n_warmup_rounds=n_warmup_rounds,
        warmup_round_elapsed_s=warmup_round_elapsed_s,
        blocking=blocking,
        metrics_reports=metrics_reports,
    )
