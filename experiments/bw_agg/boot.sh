#!/usr/bin/env bash
set -euo pipefail

# Defaults
OUTPUT_DIR="bw_agg"
NSYS_ENABLED=false
REGISTER_SIZE="2G"
PASSES=("bandwidth_aggregation" "register_tiling" "instruction_scheduler")
CLUSTER_YAML="cluster.yaml"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Boot the cluster for bandwidth aggregation performance testing.

Options:
  -o, --output-dir DIR       Root output directory (default: $OUTPUT_DIR)
  -n, --nsys                 Enable nsys profiling
  -r, --register-size SIZE   Register size (default: $REGISTER_SIZE)
  -p, --passes PASSES        Comma-separated list of passes
                             (default: bandwidth_aggregation,register_tiling,instruction_scheduler)
  -c, --cluster YAML         Cluster config file (default: $CLUSTER_YAML)
  -h, --help                 Show this help message
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        -n|--nsys)
            NSYS_ENABLED=true; shift ;;
        -r|--register-size)
            REGISTER_SIZE="$2"; shift 2 ;;
        -p|--passes)
            IFS=',' read -ra PASSES <<< "$2"; shift 2 ;;
        -c|--cluster)
            CLUSTER_YAML="$2"; shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "Unknown option: $1"; usage ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# Build passes args
PASSES_ARGS=()
for pass in "${PASSES[@]}"; do
    PASSES_ARGS+=("\"$pass\"")
done

# Build the base command
CMD=(python -m setu.cluster.ray
    --dump-info "$CLUSTER_YAML"
    --enable-metrics
    --passes "${PASSES[@]}"
    --register-size "$REGISTER_SIZE"
    --env "NCCL_DEBUG=TRACE"
    --env "NCCL_DEBUG_SUBSYS=ALL"
    --env "NCCL_DEBUG_FILE=${OUTPUT_DIR}/nccl.log"
    --env "NCCL_TOPO_DUMP_FILE=${OUTPUT_DIR}/topo.xml"
)

# Wrap with nsys if enabled
if [[ "$NSYS_ENABLED" == true ]]; then
    CMD=(nsys profile
        --trace cuda,nvtx
        -o "${OUTPUT_DIR}/setu_perf_debugging"
        --force-overwrite=true
        "${CMD[@]}"
    )
fi

echo "=== Boot BW Agg ==="
echo "Output dir:    $OUTPUT_DIR"
echo "Nsys:          $NSYS_ENABLED"
echo "Register size: $REGISTER_SIZE"
echo "Passes:        ${PASSES[*]}"
echo "Cluster:       $CLUSTER_YAML"
echo "==================="
echo "Running: ${CMD[*]}"
echo "==================="

"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/cluster.log"
