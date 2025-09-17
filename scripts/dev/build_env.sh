#!/bin/bash
set -e
set -o pipefail

# Build environment configuration
# Should be sourced from other scripts, not executed directly

_script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${_script_dir}/logging.sh"

# Environment variable handling
get_cuda_version() {
    if [[ -n "${SETU_CI_CUDA_VERSION:-}" ]]; then
        local cuda_major cuda_minor
        cuda_major=$(echo "${SETU_CI_CUDA_VERSION}" | cut -d. -f1)
        cuda_minor=$(echo "${SETU_CI_CUDA_VERSION}" | cut -d. -f2)
        echo "cu${cuda_major}${cuda_minor}"
    else
        echo "cu129"
        echo "cu129"
    fi
}

get_torch_version() {
    echo "${SETU_CI_TORCH_VERSION:-2.8}"
}

get_build_type() {
    echo "${BUILD_TYPE:-Debug}"
}

# Project hash generation
get_project_hash() {
    echo "$(get_project_root)" | sha256sum | cut -c1-16
}

# Get project root directory
get_project_root() {
    git rev-parse --show-toplevel 2>/dev/null || pwd
}

# Get timestamp for logging
get_timestamp() {
    date +"%Y%m%d_%H%M%S"
}

# RAM disk configuration
get_ramdisk_config() {
    local project_hash
    project_hash=$(get_project_hash)
    echo "/dev/shm/setu-build-${project_hash}"
}

get_ccache_dir() {
    # ccache directory configuration
    local project_hash
    project_hash=$(get_project_hash)
    echo "/dev/shm/setu-ccache-${project_hash}"
}

use_ramdisk() {
    [[ "${USE_RAMDISK:-0}" == "1" ]] && [[ -d "/dev/shm" ]] && [[ "${SETU_IS_CI_CONTEXT:-0}" == "0" ]] && [[ "$(get_total_memory_gb)" -gt 100 ]]
}

# Memory detection
get_total_memory_gb() {
    if command -v free >/dev/null 2>&1; then
        free -g | awk '/^Mem:/ {print $2}'
    else
        echo "32" # Default fallback
    fi
}

get_ccache_size_gb() {
    local total_memory
    total_memory=$(get_total_memory_gb)
    local ccache_size

    if [[ $total_memory -lt 128 ]]; then
        ccache_size=$((total_memory * 5 / 100))
    else
        ccache_size=$((total_memory * 2 / 100))
    fi

    # Ensure minimum size of 1GB and maximum of 20GB
    if [[ $ccache_size -lt 1 ]]; then
        ccache_size=1
    elif [[ $ccache_size -gt 20 ]]; then
        ccache_size=20
    fi

    echo "$ccache_size"
}

# Setup build directories
create_build_dirs() {
    # Create build subdirectory for build type
    local build_type build_subdir
    build_type=$(get_build_type)
    build_subdir="build/$(echo "$build_type" | tr '[:upper:]' '[:lower:]')"

    # Use ramdisk for build directory if enabled
    if use_ramdisk; then
        local ramdisk_dir
        ramdisk_dir=$(get_ramdisk_config)
        log_info "Using RAM disk for build: ${YELLOW}${ramdisk_dir}${RESET}"
        mkdir -p "$ramdisk_dir"
        # Create parent build directory first, then link subdirectory to ramdisk
        mkdir -p "$(dirname "$build_subdir")"
        rm -rf "$build_subdir"
        ln -sf "$ramdisk_dir" "$build_subdir"
    else
        mkdir -p "$build_subdir"
    fi

    # Always create these directories
    mkdir -p test_reports logs

    # Setup ccache if available
    if command -v ccache >/dev/null 2>&1; then
        setup_ccache
    fi

    log_info "Created directories: ${build_subdir}/, test_reports/, logs/"
}

# ccache configuration
setup_ccache() {
    local ccache_size ccache_dir
    ccache_size=$(get_ccache_size_gb)
    ccache_dir=$(get_ccache_dir)

    # Ensure ccache directory exists
    mkdir -p "$ccache_dir"
    export CCACHE_DIR="$ccache_dir"

    # Configure ccache settings
    ccache --set-config "cache_dir=${ccache_dir}"
    ccache --set-config "max_size=${ccache_size}G"
    ccache --set-config compression=true
    ccache --set-config compression_level=6
    ccache --set-config sloppiness=pch_defines,time_macros

    local total_memory cache_info
    total_memory=$(get_total_memory_gb)
    cache_info=$(ccache --show-config | grep max_size || echo "max_size = ${ccache_size}G")
    log_info "ccache configured - Dir: ${ccache_dir}, Size: ${cache_info} (Total RAM: ${total_memory}GB)"
}
