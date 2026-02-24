"""Experiment runner: cluster-agnostic orchestration for copy experiments.

Uses an SPMD barrier-based approach inspired by nccl-test: all processes
run the same code path autonomously, coordinating via barriers.  The
parent just spawns them and collects final results.
"""

import functools
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union

import torch

from setu.cluster.barrier import Barrier
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


def _format_size(nbytes: float) -> str:
    """Format a byte count as a human-readable string."""
    if nbytes >= 1 << 30:
        return f"{nbytes / (1 << 30):.2f} GiB"
    if nbytes >= 1 << 20:
        return f"{nbytes / (1 << 20):.2f} MiB"
    if nbytes >= 1 << 10:
        return f"{nbytes / (1 << 10):.2f} KiB"
    return f"{nbytes:.0f} B"


def _format_bw(nbytes: float, elapsed_s: float) -> str:
    """Format bandwidth as GB/s (base-10, like networking convention)."""
    if elapsed_s <= 1e-9 or nbytes <= 0:
        return "--"
    gbps = nbytes / elapsed_s / 1e9
    if gbps >= 1.0:
        return f"{gbps:.2f} GB/s"
    return f"{gbps * 1000:.1f} MB/s"


def _format_time(seconds: float) -> str:
    """Format time with appropriate unit."""
    if seconds >= 1.0:
        return f"{seconds:.3f} s"
    return f"{seconds * 1000:.2f} ms"


@dataclass
class ExperimentResult:
    """Result of a single experiment run.

    Attributes:
        success: True if all shards were copied and values matched, False on
            any error or mismatch.
        elapsed_s: Wall-clock time for the entire experiment (seconds).
        round_elapsed_s: Per-round wall-clock timings (seconds).  One entry
            per copy round.
        source_results: Per-source-shard dicts, each containing:
            - ``success`` (bool): whether the shard was registered and filled.
            - ``shard_name`` (str): tensor shard name.
            - ``device`` (str): device string the shard was placed on.
            - ``round_elapsed_s`` (List[float]): per-round timing from this client.
        dest_results: Per-destination-shard dicts from the final round, each
            containing:
            - ``success`` (bool): whether the copy completed.
            - ``shard_name`` (str): tensor shard name.
            - ``device`` (str): device string the shard was placed on.
            - ``round_elapsed_s`` (List[float]): per-round timing from this client.
            - ``expected_value`` (float): the init value written to the source.
            - ``actual_value`` (float): the mean value read back after copy.
            - ``values_match`` (bool): True if expected and actual are within
              1e-5 tolerance.
        errors: Human-readable error strings collected during the experiment.
        copy_mode: The copy mode used for the experiment.
        shard_bytes: Per-shard data size in bytes (one per source + dest, in
            spawn order).  Populated by :func:`run_experiment`.
    """

    success: bool
    elapsed_s: float
    round_elapsed_s: List[float] = field(default_factory=list)
    source_results: List[Dict] = field(default_factory=list)
    dest_results: List[Dict] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    copy_mode: Optional[CopyMode] = None
    shard_bytes: List[int] = field(default_factory=list)
    n_warmup_rounds: int = 0
    warmup_round_elapsed_s: List[float] = field(default_factory=list)

    def pretty_print(self) -> str:
        """Render a formatted summary of the experiment results.

        Uses ``rich`` for table rendering.  Returns a plain string
        (ANSI escape codes included when the console supports colour).
        """
        import io

        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

        console = Console(file=io.StringIO(), force_terminal=False)
        all_results = self.source_results + self.dest_results
        total_bytes = sum(self.shard_bytes) if self.shard_bytes else 0
        n_rounds = len(self.round_elapsed_s)

        # -- Header --
        status_style = "bold green" if self.success else "bold red"
        status_text = "PASS" if self.success else "FAIL"

        header = Table.grid(padding=(0, 2))
        header.add_column(style="bold")
        header.add_column()
        header.add_row("Result", Text(status_text, style=status_style))
        if self.copy_mode is not None:
            header.add_row("Copy mode", self.copy_mode.value.upper())
        header.add_row("E2E wall time", _format_time(self.elapsed_s))
        if total_bytes:
            header.add_row("Total data", _format_size(total_bytes))
        rounds_str = f"{n_rounds} measured"
        if self.n_warmup_rounds > 0:
            rounds_str += f" + {self.n_warmup_rounds} warmup"
        header.add_row("Rounds", rounds_str)
        header.add_row(
            "Clients",
            f"{len(self.source_results)} src + "
            f"{len(self.dest_results)} dst = {len(all_results)}",
        )
        console.print(header)

        # -- Warmup round summary --
        if self.warmup_round_elapsed_s:
            wt = Table(
                title="Warmup Rounds (barrier-to-barrier, max across clients)",
                show_lines=False,
            )
            wt.add_column("Round", justify="right")
            wt.add_column("Latency", justify="right")
            wt.add_column("Agg BW", justify="right")

            for i, elapsed in enumerate(self.warmup_round_elapsed_s):
                bw = _format_bw(total_bytes, elapsed) if total_bytes else "--"
                wt.add_row(str(i), _format_time(elapsed), bw)
            if len(self.warmup_round_elapsed_s) > 1:
                avg_warmup = (
                    sum(self.warmup_round_elapsed_s)
                    / len(self.warmup_round_elapsed_s)
                )
                bw = _format_bw(total_bytes, avg_warmup) if total_bytes else "--"
                wt.add_row(
                    Text("avg", style="bold"),
                    Text(_format_time(avg_warmup), style="bold"),
                    Text(bw, style="bold"),
                )
            console.print(wt)

        # -- Per-round summary --
        if self.round_elapsed_s:
            rt = Table(
                title="Round Summary (barrier-to-barrier, max across clients)",
                show_lines=False,
            )
            rt.add_column("Round", justify="right")
            rt.add_column("Latency", justify="right")
            rt.add_column("Agg BW", justify="right")

            for i, elapsed in enumerate(self.round_elapsed_s):
                bw = _format_bw(total_bytes, elapsed) if total_bytes else "--"
                rt.add_row(str(i), _format_time(elapsed), bw)
            if n_rounds > 1:
                avg_round = sum(self.round_elapsed_s) / n_rounds
                bw = _format_bw(total_bytes, avg_round) if total_bytes else "--"
                rt.add_row(
                    Text("avg", style="bold"),
                    Text(_format_time(avg_round), style="bold"),
                    Text(bw, style="bold"),
                )
            console.print(rt)

        # -- Per-client table --
        if all_results:
            ct = Table(title="Per-Client Breakdown", show_lines=False)
            ct.add_column("Role", justify="right")
            ct.add_column("Device", justify="left")
            ct.add_column("Shard", justify="left")
            ct.add_column("Size", justify="right")
            ct.add_column("Avg Lat", justify="right")
            ct.add_column("Avg BW", justify="right")
            ct.add_column("OK", justify="center")

            for idx, r in enumerate(all_results):
                role = r.get("role", "?")
                device = r.get("device", "?")
                shard_name = r.get("shard_name", "?")
                round_times = r.get("round_elapsed_s", [])

                nbytes = self.shard_bytes[idx] if idx < len(self.shard_bytes) else 0
                size_str = _format_size(nbytes) if nbytes else "--"

                # In PULL mode sources don't initiate copies — skip lat/BW.
                is_idle = (role == "source" and self.copy_mode == CopyMode.PULL)
                if round_times and not is_idle:
                    avg_lat = sum(round_times) / len(round_times)
                    lat_str = _format_time(avg_lat)
                    bw_str = _format_bw(nbytes, avg_lat) if nbytes else "--"
                else:
                    lat_str = "--"
                    bw_str = "--"

                match_str = ""
                match_style = ""
                if "values_match" in r:
                    if r["values_match"]:
                        match_str = "yes"
                        match_style = "green"
                    else:
                        match_str = "FAIL"
                        match_style = "bold red"

                ct.add_row(
                    role, device, shard_name, size_str,
                    lat_str, bw_str,
                    Text(match_str, style=match_style),
                )
            console.print(ct)

        # -- Errors --
        if self.errors:
            console.print("[bold red]Errors:[/bold red]")
            for err in self.errors:
                console.print(f"  - {err}")

        return console.file.getvalue()


# ---------------------------------------------------------------------------
# Module-level body functions (must be picklable for subprocess backends)
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
    n_copy_rounds,
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

    round_times = []
    for round_i in range(n_copy_rounds):
        barrier.wait()  # round start

        t_round = time.monotonic()

        if copy_mode == CopyMode.COPY:
            src_selection = client.select(shard_spec.name)
            dst_selection = client.select(dst_name)

            if selections is not None:
                for dim_name, indices in selections.items():
                    src_selection = src_selection.where(dim_name, indices)
                    dst_selection = dst_selection.where(dim_name, indices)

            copy_op_id = client.copy(src_selection, dst_selection)
            assert copy_op_id is not None, "Copy operation returned None"
            logger.debug(
                "%s: round %d, submitted copy op %s, waiting...",
                tag,
                round_i,
                copy_op_id,
            )
            client.wait(copy_op_id)

        elapsed = time.monotonic() - t_round

        barrier.wait()  # round end

        round_times.append(elapsed)
        logger.debug("%s: round %d complete in %.3fs", tag, round_i, elapsed)

    logger.debug("%s: total body time=%.3fs", tag, time.monotonic() - t_body)

    return {
        "role": "source",
        "success": True,
        "shard_name": shard_spec.name,
        "device": str(shard_spec.device.torch_device),
        "round_elapsed_s": round_times,
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
    n_copy_rounds,
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

    round_times = []
    for round_i in range(n_copy_rounds):
        barrier.wait()  # round start

        t_round = time.monotonic()

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
        logger.debug(
            "%s: round %d, submit %s op %s, waiting...",
            tag,
            round_i,
            copy_mode.value,
            copy_op_id,
        )
        client.wait(copy_op_id)

        elapsed = time.monotonic() - t_round

        barrier.wait()  # round end

        round_times.append(elapsed)
        logger.debug("%s: round %d complete in %.3fs", tag, round_i, elapsed)

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
        "round_elapsed_s": round_times,
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
    n_copy_rounds: int = 1,
    n_warmup_rounds: int = 0,
) -> ExperimentResult:
    """Run a copy experiment on a Setu cluster.

    Spawns all source and destination processes which self-coordinate via
    a barrier.  The parent just waits for final results and aggregates.

    Args:
        cluster: A started Cluster instance (any backend).
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
    cluster_info = cluster.cluster_info
    assert cluster_info is not None, "Cluster has not been started"

    # Bodies execute warmup + measured rounds; we strip warmup from results.
    n_total_rounds = n_warmup_rounds + n_copy_rounds

    src_shards = src.shards
    dst_shards = dst.shards
    errors: List[str] = []
    handles = []

    n_total = len(src_shards) + len(dst_shards)

    # Compute per-shard byte sizes (source shards first, then dest shards)
    element_size = src.dtype.itemsize
    shard_bytes: List[int] = []
    for shard in src_shards:
        n_elements = 1
        for d in shard.dims:
            n_elements *= d.get_owned_size()
        shard_bytes.append(n_elements * element_size)
    element_size_dst = dst.dtype.itemsize
    for shard in dst_shards:
        n_elements = 1
        for d in shard.dims:
            n_elements *= d.get_owned_size()
        shard_bytes.append(n_elements * element_size_dst)

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

    t0 = time.monotonic()

    try:
        # Create barriers — the cluster picks the right type
        barriers = cluster.create_barrier(n_total)

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
                n_copy_rounds=n_total_rounds,
            )
            handles.append(cluster.spawn_client(participant, body))
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
                n_copy_rounds=n_total_rounds,
            )
            handles.append(cluster.spawn_client(participant, body))
            rank += 1

        logger.debug(
            "run_experiment: spawned %d clients in %.3fs, waiting for results",
            n_total,
            time.monotonic() - t_spawn,
        )

        # Wait for all results (bodies run autonomously)
        results = [h.result(timeout=timeout) for h in handles]

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
    )
