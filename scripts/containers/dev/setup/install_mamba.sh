#!/bin/bash
set -ex

CONDA_DIR=${HOME}/conda
echo "::group::Installing Miniforge to $CONDA_DIR..."

# Download and install Miniforge
wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3.sh -b -p "${HOME}/conda"
rm Miniforge3.sh

"$CONDA_DIR/bin/mamba" shell init --shell=bash
"$CONDA_DIR/bin/mamba" shell init --shell=zsh

echo "::endgroup::"
