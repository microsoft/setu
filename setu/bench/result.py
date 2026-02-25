"""Experiment result types and formatting utilities."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


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
        shard_bytes: Per-shard data size in bytes (source shards only).
            Populated by :func:`run_experiment`.
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
                avg_warmup = sum(self.warmup_round_elapsed_s) / len(
                    self.warmup_round_elapsed_s
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

                # shard_bytes covers source shards only; dest shards mirror them.
                n_src = len(self.source_results)
                shard_idx = idx if idx < n_src else idx - n_src
                nbytes = (
                    self.shard_bytes[shard_idx]
                    if shard_idx < len(self.shard_bytes)
                    else 0
                )
                size_str = _format_size(nbytes) if nbytes else "--"

                # In PULL mode sources don't initiate copies — skip lat/BW.
                is_idle = role == "source" and self.copy_mode == CopyMode.PULL
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
                    role,
                    device,
                    shard_name,
                    size_str,
                    lat_str,
                    bw_str,
                    Text(match_str, style=match_style),
                )
            console.print(ct)

        # -- Errors --
        if self.errors:
            console.print("[bold red]Errors:[/bold red]")
            for err in self.errors:
                console.print(f"  - {err}")

        return console.file.getvalue()
