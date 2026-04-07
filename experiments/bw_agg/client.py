import argparse
import os
import uuid

from setu._coordinator import BandwidthHint, Participant, Link, Path as RoutePath, PipelineChunkSizeHint
from setu.bench.cluster_setup import connect_prespawned, build_sharded_tensor
from setu.bench.result import CopyMode
from setu.bench.runner import run_experiment
from setu.schedule import Schedule
from setu.utils.parsing import parse_num_bytes as _parse_size

parser = argparse.ArgumentParser()
parser.add_argument("--cluster-info", type=str, required=True)
parser.add_argument(
    "--size",
    type=str,
    nargs="+",
    default=["256M"],
    help="Total tensor size(s) in bytes, e.g. '256M', '1G', '512K'. "
    "Suffix: K=KiB, M=MiB, G=GiB. Multiple values sweep over each size "
    "(default: 256M).",
)
parser.add_argument(
    "--group",
    type=int,
    choices=[0, 1],
    default=1,
    help="GPU group to test: 0 = devices 0-3, 1 = devices 4-7 (default: 1).",
)
parser.add_argument(
    "--runs",
    type=int,
    default=1,
    help="Number of times to repeat the experiment (default: 1).",
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="./bw_agg",
    help="Base output directory for results (default: ./bw_agg).",
)
parser.add_argument(
    "--num-copy-rounds",
    type=int,
    default=20,
)
parser.add_argument(
    "--pipeline-chunk-size",
    type=str,
    default=None,
    help="Pipeline chunk size in bytes (e.g. '64M', '128M'). "
    "Passed as a PipelineChunkSizeHint. If omitted, the pass default is used.",
)
args = parser.parse_args()

cluster_info = connect_prespawned(args.cluster_info)
print(cluster_info)

# Select GPU group: 0 = devices 0-3, 1 = devices 4-7
base = args.group * 4

node_id = uuid.UUID(cluster_info.nodes[0].node_id)
devices = cluster_info.nodes[0].devices

pipeline_chunk_size_hints = (
    [PipelineChunkSizeHint(_parse_size(args.pipeline_chunk_size))]
    if args.pipeline_chunk_size is not None
    else []
)

schedule_baseline = Schedule(hints=pipeline_chunk_size_hints)
schedule_relay = Schedule(
    hints=[
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
    + pipeline_chunk_size_hints
)

for size_str in args.size:
    tensor_bytes = _parse_size(size_str)
    size_dir = os.path.join(args.output_dir, size_str)

    src_spec = [f"0:{base}"]
    dst_spec = [f"0:{base + 3}"]
    src = build_sharded_tensor("src_t", cluster_info, tensor_bytes, src_spec)
    dst = build_sharded_tensor("dst_t", cluster_info, tensor_bytes, dst_spec)

    for run_idx in range(args.runs):
        run_dir = os.path.join(size_dir, str(run_idx))
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n{'='*10} Size {size_str} | Run {run_idx} | Group {args.group} (devices {base}-{base+3}) {'='*10}")

        # without hints (baseline)
        result = run_experiment(
            cluster_info=cluster_info,
            src=src,
            dst=dst,
            copy_mode=CopyMode("pull"),
            init_value=10,
            n_copy_rounds=args.num_copy_rounds,
            n_warmup_rounds=1,
            blocking=False,

            hints=schedule_baseline.hints,
        )
        print("="*10, "Without Hints", "="*10)
        print(result.pretty_print())
        result.dump_csv(os.path.join(run_dir, "wo_hints"))

        # with hints (baseline + relay)
        result = run_experiment(
            cluster_info=cluster_info,
            src=src,
            dst=dst,
            copy_mode=CopyMode("pull"),
            init_value=10,
            n_copy_rounds=args.num_copy_rounds,
            n_warmup_rounds=1,
            blocking=False,

            hints=schedule_relay.hints,
        )
        print("="*10, "With Hints", "="*10)
        print(result.pretty_print())
        result.dump_csv(os.path.join(run_dir, "w_hints"))
