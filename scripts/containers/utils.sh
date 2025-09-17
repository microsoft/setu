#!/bin/bash

# Container utility functions - pure function library
# Should be sourced from other scripts, not executed directly

# Prevent multiple sourcing
if [[ "${SETU_CONTAINER_UTILS_LOADED:-}" == "true" ]]; then
    return 0
fi
readonly SETU_CONTAINER_UTILS_LOADED="true"

# Basic assertions
assert_env() {
    local var_name="$1"
    if [ -z "${!var_name}" ]; then
        echo "Error: Environment variable '$var_name' is not set."
        exit 1
    fi
}

# GitHub CI logging helpers
log_group() {
    echo "::group::$1"
}

log_endgroup() {
    echo "::endgroup::"
}

# Conda initialization
init_conda() {
    export PATH="${HOME}/conda/bin:${PATH}"
    mamba shell init --shell=bash
    mamba shell init --shell=zsh
    source /root/conda/etc/profile.d/conda.sh
}

# HuggingFace login (optional - only if token is provided)
login_huggingface() {
    if [[ -n "${HUGGINGFACE_TOKEN:-}" ]]; then
        echo "Logging into HuggingFace..."
        huggingface-cli login --token "$HUGGINGFACE_TOKEN"
    else
        echo "No HuggingFace token provided - skipping login"
    fi
}

# Setu environment functions
create_setu_conda_env() {
    local root_dir="$1"
    log_group "Create conda environment"
    pushd "$root_dir" || exit 1
    mamba env create -f environment-dev.yml -n setu
    popd || exit 1
    log_endgroup
}

activate_setu_conda_env() {
    conda activate setu
    echo "Activated setu conda environment with Python location: $(which python)"
}

# System utilities
install_git_lfs() {
    # MLC-LLM requires git-lfs to download models
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
    apt-get update
    apt-get install git-lfs
    git lfs install
}
