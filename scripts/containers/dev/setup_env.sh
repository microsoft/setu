#!/bin/bash
set -ex

_setup_env_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
_setup_root_dir=$(dirname "$(dirname "$(dirname "$_setup_env_dir")")")

# Source container utilities
source "${_setup_env_dir}/../utils.sh"

log_group "Setup base system dependencies"
bash "${_setup_env_dir}/setup/install_base.sh"
log_endgroup

log_group "Setup mamba package manager"
bash "${_setup_env_dir}/setup/install_mamba.sh"
log_endgroup

# Initialize conda
init_conda

# Create and activate setu environment
create_setu_conda_env "$_setup_root_dir"
activate_setu_conda_env

# Install pip dependencies using proper setup function
pushd "$_setup_root_dir" || exit 1

# Source dev utilities
source "scripts/dev/utils.sh"

# Clean build artifacts
"scripts/dev/build.sh" clean-force

# Run setup_dependencies
"scripts/dev/setup.sh" dependencies

popd || exit 1
