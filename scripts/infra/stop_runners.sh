#!/bin/bash
set -e

LOG_FILE="/var/log/github-runners.log"

# Function to log messages with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "Beginning GitHub Action runner cleanup process..."

# Get a list of all gh-runner containers (both running and stopped)
CONTAINERS=$(docker ps -a --filter "name=gh-runner-" -q)

if [ -z "$CONTAINERS" ]; then
    log "No GitHub runner containers found to clean up"
else
    NUM_CONTAINERS=$(echo "$CONTAINERS" | wc -l)
    log "Found $NUM_CONTAINERS GitHub runner containers to clean up"

    # Stop all running containers first
    RUNNING_CONTAINERS=$(docker ps --filter "name=gh-runner-" -q)
    if [ -n "$RUNNING_CONTAINERS" ]; then
        log "Stopping running containers..."
        # shellcheck disable=SC2086
        docker stop $RUNNING_CONTAINERS
        log "Waiting for containers to stop gracefully..."
        sleep 5
    else
        log "No running containers found"
    fi

    # Force remove all containers (both running and stopped)
    log "Removing all GitHub runner containers..."
    for container in $CONTAINERS; do
        CONTAINER_NAME=$(docker inspect --format='{{.Name}}' "$container" | sed 's/\///')
        log "Removing container: $CONTAINER_NAME (ID: $container)"
        docker rm -f "$container" || log "Warning: Failed to remove container $CONTAINER_NAME"
    done
fi

# Check if any containers still exist after cleanup
REMAINING=$(docker ps -a --filter "name=gh-runner-" -q)
if [ -n "$REMAINING" ]; then
    NUM_REMAINING=$(echo "$REMAINING" | wc -l)
    log "WARNING: $NUM_REMAINING containers still remain after cleanup attempt"
    log "Attempting force removal with different approach..."

    # Try an alternative approach - this is more aggressive
    docker ps -a --filter "name=gh-runner-" --format "{{.Names}}" | xargs -r docker rm -f

    # Final check
    FINAL_CHECK=$(docker ps -a --filter "name=gh-runner-" -q)
    if [ -n "$FINAL_CHECK" ]; then
        log "ERROR: Unable to remove all containers. Manual intervention may be required."
    else
        log "All containers successfully removed after second attempt"
    fi
else
    log "All GitHub Action runner containers have been successfully removed"
fi

# Clean up any dangling Docker volumes related to runners
log "Cleaning up dangling volumes..."
docker volume ls -qf dangling=true | xargs -r docker volume rm

log "Cleanup process completed"