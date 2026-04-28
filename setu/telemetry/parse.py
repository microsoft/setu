"""Extract bench-friendly selections.yaml entries from a setu_metrics.json.

Each record in ``setu_metrics.json`` carries the full src/dst selection used
for one ``CopySpec`` (per-dim ``ranges``).  This tool looks up one or more
records by ``copy_op_id`` and emits a YAML file consumable by
``python -m setu.bench --selections``.

Usage::

    python -m setu.telemetry.parse setu_metrics.json --op-id <ID> [--op-id ...]

Per-dim selections are emitted in the most compact form that round-trips:
  * single contiguous range  -> ``{start: a, end: b}`` (slice)
  * full-range dim           -> omitted (defaults to "all")
  * multiple ranges          -> ``[i, j, k, ...]`` (explicit indices)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml


def _per_dim_yaml(dim_size: int, ranges: List[Dict[str, int]]):
    """Convert a metrics-JSON per-dim entry to the most compact YAML form.

    Returns ``None`` if the dim spans its full size (caller should skip).
    """
    if len(ranges) == 1:
        r = ranges[0]
        start = int(r["start"])
        end = int(r["end"])
        if start == 0 and end == dim_size:
            return None  # full range; caller omits the key
        return {"start": start, "end": end}
    indices: List[int] = []
    for r in ranges:
        indices.extend(range(int(r["start"]), int(r["end"])))
    return indices


def _selection_to_yaml(selection: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a metrics-JSON selection block to the YAML per-dim mapping."""
    out: Dict[str, Any] = {}
    for dim_name, dim_entry in selection["indices"].items():
        compact = _per_dim_yaml(int(dim_entry["dim_size"]), dim_entry["ranges"])
        if compact is not None:
            out[dim_name] = compact
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract selections.yaml entries from setu_metrics.json"
    )
    parser.add_argument(
        "metrics_json",
        type=str,
        help="Path to setu_metrics.json (a mapping of copy_op_id -> record)",
    )
    parser.add_argument(
        "--op-id",
        action="append",
        required=True,
        dest="op_ids",
        help="copy_op_id to extract; pass multiple times for multiple records.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    metrics = json.loads(Path(args.metrics_json).read_text())
    if not isinstance(metrics, dict):
        print(
            f"error: {args.metrics_json}: expected top-level mapping of "
            f"copy_op_id -> record",
            file=sys.stderr,
        )
        return 1

    copies: List[Dict[str, Any]] = []
    for op_id in args.op_ids:
        if op_id not in metrics:
            print(
                f"error: copy_op_id {op_id!r} not found in {args.metrics_json}",
                file=sys.stderr,
            )
            return 1
        record = metrics[op_id]
        copies.append(
            {
                "src": _selection_to_yaml(record["src_selection"]),
                "dst": _selection_to_yaml(record["dst_selection"]),
            }
        )

    yaml.safe_dump(
        {"copies": copies},
        sys.stdout,
        sort_keys=False,
        default_flow_style=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
