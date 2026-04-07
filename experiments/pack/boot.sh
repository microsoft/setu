#!/usr/bin/env bash
set -euo pipefail

# Defaults
OUTPUT_DIR="pack"
NSYS_ENABLED=false
REGISTER_SIZE="2G"
PASSES=("pack_unpack_copies" "pipelining" "register_tiling" "instruction_scheduler")
CLUSTER_YAML="cluster.yaml"
EXTRA_ENVS=()

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Boot the cluster for Pack/Unpack pipelining performance testing.

Options:
  -o, --output-dir DIR       Root output directory (default: $OUTPUT_DIR)
  -n, --nsys                 Enable nsys profiling
  -r, --register-size SIZE   Register size (default: $REGISTER_SIZE)
  -p, --passes PASSES        Comma-separated list of passes
                             (default: pack_unpack_copies,pipelining,register_tiling,instruction_scheduler)
  -c, --cluster YAML         Cluster config file (default: $CLUSTER_YAML)
  -e, --env KEY=VALUE        Extra environment variable (repeatable)
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
        -e|--env)
            EXTRA_ENVS+=("$2"); shift 2 ;;
        -h|--help)
            usage ;;
        *)
            echo "Unknown option: $1"; usage ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

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

# Append extra env vars
for env_var in "${EXTRA_ENVS[@]}"; do
    CMD+=(--env "$env_var")
done

# Wrap with nsys if enabled
if [[ "$NSYS_ENABLED" == true ]]; then
    CMD=(nsys profile
        --trace cuda,nvtx
        -o "${OUTPUT_DIR}/setu_perf_debugging"
        --force-overwrite=true
        "${CMD[@]}"
    )
fi

echo "=== Boot Pack/Unpack ==="
echo "Output dir:    $OUTPUT_DIR"
echo "Nsys:          $NSYS_ENABLED"
echo "Register size: $REGISTER_SIZE"
echo "Passes:        ${PASSES[*]}"
echo "Cluster:       $CLUSTER_YAML"
echo "======================="
echo "Running: ${CMD[*]}"
echo "======================="

"${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/cluster.log"
