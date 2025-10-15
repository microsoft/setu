#!/bin/bash
set -e

# Configuration
SPEC_FILE="${1}"
GITHUB_REPO="${2}"
# GITHUB_ORG="${3}" # ignored because we are using repo-scoped runners
# Accept token from multiple possible sources with credential directory being highest priority
GITHUB_TOKEN="${4}"
DOCKER_IMAGE="myoung34/github-runner:latest"
LOG_FILE="/var/log/github-runners.log"

log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Validate token
if [ -z "$GITHUB_TOKEN" ]; then
    log_with_timestamp "ERROR: No GitHub token provided. Set token via systemd credential, environment variable, or command argument."
    exit 1
fi

# Create log file if it doesn't exist
touch "$LOG_FILE"

# Token validation and obscuring in logs
TOKEN_LENGTH=${#GITHUB_TOKEN}
MASKED_TOKEN="${GITHUB_TOKEN:0:4}...${GITHUB_TOKEN: -4}"
log_with_timestamp "Using GitHub token: $MASKED_TOKEN (length: $TOKEN_LENGTH)"

# Trap signals for clean shutdown
trap 'log_with_timestamp "Received shutdown signal. Exiting..."; exit 0' SIGTERM SIGINT

log_with_timestamp "Starting GitHub Actions Runner service"

# Check dependencies
command -v docker >/dev/null 2>&1 || {
    log_with_timestamp "Docker is required but not installed. Aborting." >&2
    exit 1
}
command -v jq >/dev/null 2>&1 || {
    log_with_timestamp "jq is required but not installed. Aborting." >&2
    exit 1
}
command -v nvidia-smi >/dev/null 2>&1 || {
    log_with_timestamp "nvidia-smi is required but not installed. Aborting." >&2
    exit 1
}

# Check if spec file exists
if [ ! -f "$SPEC_FILE" ]; then
    echo "Spec file not found: $SPEC_FILE" >&2
    exit 1
fi

# Check if spec file is valid JSON
if ! jq empty "$SPEC_FILE" 2>/dev/null; then
    echo "Invalid JSON in spec file: $SPEC_FILE" >&2
    exit 1
fi

# Get total number of GPUs in system
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $GPU_COUNT GPUs in the system"

echo "Reading runner specification from $SPEC_FILE"
readarray -t specs < <(jq -c '.[]' "$SPEC_FILE")

# Initialize GPU allocation tracking
declare -A GPU_ALLOCATION
for ((i = 0; i < GPU_COUNT; i++)); do
    GPU_ALLOCATION[$i]=0
done

for spec in "${specs[@]}"; do
    num_gpus=$(echo "$spec" | jq -r '.num_gpus')
    num_instances=$(echo "$spec" | jq -r '.num_instances')
    runner_name_prefix=$(echo "$spec" | jq -r '.runner_name_prefix')
    runner_tags=$(echo "$spec" | jq -r '.tags | join(",")')

    echo "Configuring runner group: $runner_name_prefix with $num_instances instances, each with $num_gpus GPUs"

    for ((i = 1; i <= num_instances; i++)); do
        runner_name="${runner_name_prefix}-${i}"

        allocated_gpus=""
        allocated_gpu_count=0
        for gpu_id in "${!GPU_ALLOCATION[@]}"; do
            if [ "${GPU_ALLOCATION[$gpu_id]}" -eq 0 ]; then
                if [ -z "$allocated_gpus" ]; then
                    allocated_gpus="$gpu_id"
                else
                    allocated_gpus="$allocated_gpus,$gpu_id"
                fi
                GPU_ALLOCATION[$gpu_id]=1
                allocated_gpu_count=$((allocated_gpu_count + 1))

                if [ "$allocated_gpu_count" -eq "$num_gpus" ]; then
                    break
                fi
            fi
        done

        if [ "$allocated_gpu_count" -ne "$num_gpus" ]; then
            log_with_timestamp "ERROR: Insufficient GPUs to allocate for runner: $runner_name"
            exit 1
        fi

        # Container name
        container_name="gh-runner-${runner_name}"

        # Check if container already exists
        if docker ps -a --filter "name=$container_name" --format '{{.Names}}' | grep -q "^$container_name$"; then
            log_with_timestamp "WARNING: Container $container_name already exists. Removing it first..."
            docker rm -f "$container_name" || log_with_timestamp "Failed to remove existing container $container_name"
        fi

        # Prepare Docker run command
        docker_cmd="docker run -d --restart always --name $container_name"

        # Add GPU configuration if GPUs are allocated
        if [ -n "$allocated_gpus" ]; then
            gpu_env="--env NVIDIA_VISIBLE_DEVICES=$allocated_gpus"
        else
            gpu_env="--env NVIDIA_VISIBLE_DEVICES="
        fi

        # Add GitHub Actions runner configuration
        docker_cmd="$docker_cmd $gpu_env \
            --env REPO_URL=$GITHUB_REPO \
            --env RUNNER_SCOPE=repo \
            --env RUNNER_NAME=$runner_name \
            --env ACCESS_TOKEN=$GITHUB_TOKEN \
            --env LABELS="self-hosted,$runner_tags" \
            --env RUNNER_WORKDIR=/tmp/github-runner-$runner_name \
            --volume /var/run/docker.sock:/var/run/docker.sock \
            --volume /tmp/github-runner-$runner_name:/tmp/github-runner-$runner_name \
            $DOCKER_IMAGE"

        log_with_timestamp "Launching container for runner: $runner_name with GPUs: $allocated_gpus"
        eval "$docker_cmd" || {
            log_with_timestamp "ERROR: Failed to start container $container_name"
            # Release allocated GPUs on failure
            if [ -n "$allocated_gpus" ]; then
                for gpu_id in $(echo "$allocated_gpus" | tr ',' ' '); do
                    GPU_ALLOCATION[$gpu_id]=0
                done
            fi
        }

        echo "Runner $runner_name started successfully"
    done
done

log_with_timestamp "All runner containers have been started"
log_with_timestamp "GPU Allocation Summary:"
for gpu_id in "${!GPU_ALLOCATION[@]}"; do
    if [ "${GPU_ALLOCATION[$gpu_id]}" -eq 1 ]; then
        status="Allocated"
    else
        status="Available"
    fi
    log_with_timestamp "GPU $gpu_id: $status"
done

# Add health monitoring loop
log_with_timestamp "Entering monitoring mode to ensure containers remain running..."
while true; do
    sleep 300 # Check every 5 minutes

    # Get list of expected runner container names
    expected_containers=$(docker ps --filter "name=gh-runner-" --format "{{.Names}}" | sort)
    expected_count=$(echo "$expected_containers" | wc -l)

    # Check if any containers stopped
    running_containers=$(docker ps --filter "name=gh-runner-" --format "{{.Names}}" | sort)
    running_count=$(echo "$running_containers" | wc -l)

    if [ "$running_count" -lt "$expected_count" ]; then
        log_with_timestamp "Warning: Some containers are not running. Expected $expected_count, found $running_count"

        # Restart any stopped containers
        for container in $expected_containers; do
            if ! docker ps --filter "name=$container" --format "{{.Names}}" | grep -q "$container"; then
                log_with_timestamp "Restarting container: $container"
                docker start "$container" || log_with_timestamp "Failed to restart container: $container"
            fi
        done
    else
        log_with_timestamp "Health check: All $running_count containers are running"
    fi
done