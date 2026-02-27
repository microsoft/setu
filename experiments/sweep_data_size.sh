#!/usr/bin/env bash
#
# Sweep data sizes on a pre-spawned Setu cluster.
#
# Usage:
#   ./experiments/sweep_data_size.sh \
#       --ray-address ray://10.0.0.1:10001 \
#       --output-dir results/my_run \
#       --src "0:0-1" --dst "0:0-3"
#
#   # Custom range and step:
#   ./experiments/sweep_data_size.sh \
#       --ray-address ray://10.0.0.1:10001 \
#       --output-dir results/my_run \
#       --src "0:0-1" --dst "0:0-3" \
#       -b 1K -e 1G -f 4
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Defaults ────────────────────────────────────────────────────────
RAY_ADDRESS=""
OUTPUT_DIR=""
SRC_SPECS=()
DST_SPECS=()
BEGIN_SIZE="32"
END_SIZE="8G"
FACTOR=2
NCCL_SOCKET_IFNAME=""
ENV_ARGS=()

# ── Parse args ──────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ray-address)  RAY_ADDRESS="$2"; shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2";  shift 2 ;;
    --src)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        SRC_SPECS+=("$1"); shift
      done
      ;;
    --dst)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        DST_SPECS+=("$1"); shift
      done
      ;;
    -b)  BEGIN_SIZE="$2"; shift 2 ;;
    -e)  END_SIZE="$2";   shift 2 ;;
    -f)  FACTOR="$2";     shift 2 ;;
    --nccl-socket-ifname) NCCL_SOCKET_IFNAME="$2"; shift 2 ;;
    --env) ENV_ARGS+=("--env" "$2"); shift 2 ;;
    -h|--help)
      cat <<'EOF'
Usage: sweep_data_size.sh --ray-address ADDR --output-dir DIR --src SPECS --dst SPECS [OPTIONS]

Required:
  --ray-address ADDR   Ray head endpoint (e.g. ray://10.0.0.1:10001)
  --output-dir  DIR    Directory for results
  --src         SPECS  Source device specs (same format as setu.bench --src)
  --dst         SPECS  Dest device specs   (same format as setu.bench --dst)

Options:
  -b SIZE   Begin size (default: 32).  Suffixes: K, M, G.
  -e SIZE   End size   (default: 8G).  Suffixes: K, M, G.
  -f N      Step multiplier (default: 2)
  --nccl-socket-ifname NAME  NCCL socket interface name (e.g. enp1s0f0)
  --env KEY=VALUE            Extra env vars for actors (repeatable)
  -h        Show this help
EOF
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Validate required args ──────────────────────────────────────────
USAGE="Usage: $0 --ray-address ADDR --output-dir DIR --src SPECS --dst SPECS"
err=0
if [[ -z "$RAY_ADDRESS" ]]; then
  echo "ERROR: --ray-address is required"; err=1
fi
if [[ -z "$OUTPUT_DIR" ]]; then
  echo "ERROR: --output-dir is required"; err=1
fi
if [[ ${#SRC_SPECS[@]} -eq 0 ]]; then
  echo "ERROR: --src is required"; err=1
fi
if [[ ${#DST_SPECS[@]} -eq 0 ]]; then
  echo "ERROR: --dst is required"; err=1
fi
if [[ $err -ne 0 ]]; then
  echo "$USAGE"
  exit 1
fi

# ── Parse human-readable sizes to bytes ─────────────────────────────
parse_size() {
  local s="${1^^}"  # uppercase
  case "${s: -1}" in
    K) echo $(( ${s%K} * 1024 )) ;;
    M) echo $(( ${s%M} * 1024 * 1024 )) ;;
    G) echo $(( ${s%G} * 1024 * 1024 * 1024 )) ;;
    *) echo "$s" ;;
  esac
}

BEGIN_BYTES=$(parse_size "$BEGIN_SIZE")
END_BYTES=$(parse_size "$END_SIZE")

if [[ $BEGIN_BYTES -le 0 || $END_BYTES -le 0 || $FACTOR -le 1 ]]; then
  echo "ERROR: invalid range: begin=$BEGIN_SIZE end=$END_SIZE factor=$FACTOR"
  exit 1
fi
if [[ $BEGIN_BYTES -gt $END_BYTES ]]; then
  echo "ERROR: begin ($BEGIN_SIZE = ${BEGIN_BYTES}B) > end ($END_SIZE = ${END_BYTES}B)"
  exit 1
fi

# ── Setup ───────────────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
echo "=== sweep_data_size ==="
echo "Output:    $OUTPUT_DIR"
echo "Src:       ${SRC_SPECS[*]}"
echo "Dst:       ${DST_SPECS[*]}"
echo "Range:     $BEGIN_SIZE .. $END_SIZE (x$FACTOR)"

# ── Spawn cluster ──────────────────────────────────────────────────
CLUSTER_INFO="$OUTPUT_DIR/cluster.yaml"
CLUSTER_LOG="$OUTPUT_DIR/cluster.log"

CLUSTER_CMD=(python -m setu.cluster.ray
  --enable-metrics
  --dump-info "$CLUSTER_INFO"
  --ray-address "$RAY_ADDRESS"
)
if [[ -n "$NCCL_SOCKET_IFNAME" ]]; then
  CLUSTER_CMD+=(--nccl-socket-ifname "$NCCL_SOCKET_IFNAME")
fi
if [[ ${#ENV_ARGS[@]} -gt 0 ]]; then
  CLUSTER_CMD+=("${ENV_ARGS[@]}")
fi

echo "Starting cluster..."
"${CLUSTER_CMD[@]}" > "$CLUSTER_LOG" 2>&1 &
CLUSTER_PID=$!

cleanup() {
  echo ""
  echo "Stopping cluster (pid $CLUSTER_PID)..."
  kill "$CLUSTER_PID" 2>/dev/null || true
  wait "$CLUSTER_PID" 2>/dev/null || true
  echo "Done."
}
trap cleanup EXIT

# Wait for cluster info file to appear.
WAIT_TIMEOUT=60
WAITED=0
while [[ ! -f "$CLUSTER_INFO" ]]; do
  sleep 1
  WAITED=$((WAITED + 1))
  if [[ $WAITED -ge $WAIT_TIMEOUT ]]; then
    echo "ERROR: cluster did not produce $CLUSTER_INFO within ${WAIT_TIMEOUT}s"
    echo "Cluster log:"
    cat "$CLUSTER_LOG"
    exit 1
  fi
done
echo "Cluster ready (took ${WAITED}s)"

# ── Build size list ────────────────────────────────────────────────
SIZES=()
SIZE=$BEGIN_BYTES
while [[ $SIZE -le $END_BYTES ]]; do
  SIZES+=("$SIZE")
  SIZE=$((SIZE * FACTOR))
done

# ── Human-readable label ───────────────────────────────────────────
human_label() {
  local b=$1
  if   [[ $b -ge $((1 << 30)) ]]; then
    echo "$(echo "$b" | awk '{printf "%.0fG", $1/2^30}')B"
  elif [[ $b -ge $((1 << 20)) ]]; then
    echo "$(echo "$b" | awk '{printf "%.0fM", $1/2^20}')B"
  elif [[ $b -ge $((1 << 10)) ]]; then
    echo "$(echo "$b" | awk '{printf "%.0fK", $1/2^10}')B"
  else
    echo "${b}B"
  fi
}

echo "Sweeping ${#SIZES[@]} data sizes: $(human_label "${SIZES[0]}") .. $(human_label "${SIZES[-1]}")"
echo ""

# ── Run sweep ──────────────────────────────────────────────────────
FAILED=0
for SIZE_BYTES in "${SIZES[@]}"; do
  LABEL=$(human_label "$SIZE_BYTES")
  POINT_DIR="$OUTPUT_DIR/$LABEL"
  mkdir -p "$POINT_DIR"

  echo "--- $LABEL ($SIZE_BYTES bytes) ---"

  if python -m setu.bench \
      --cluster-info "$CLUSTER_INFO" \
      --size "$SIZE_BYTES" \
      --src "${SRC_SPECS[@]}" \
      --dst "${DST_SPECS[@]}" \
      --enable-metrics \
      --output-dir "$POINT_DIR" \
      > "$POINT_DIR/bench.log" 2>&1; then
    echo "  PASS"
  else
    echo "  FAIL (see $POINT_DIR/bench.log)"
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "=== Sweep complete ==="
echo "Results: $OUTPUT_DIR"
echo "Points: ${#SIZES[@]} total, $FAILED failed"
