# Setu Development Container

This container provides a consistent development environment with CUDA and PyTorch support, along with useful developer tools. It's designed to match the CI environment for building and testing Setu.

## Features

- CUDA support with cuDNN
- PyTorch with CUDA support
- Miniforge (faster Conda alternative)
- ZSH with Oh My Zsh and Powerlevel10k theme
- FZF (fuzzy finder) for enhanced command-line navigation
- Development tools (git, vim, nano, etc.)
- Similar environment to CI pipelines for consistent development and testing

## Available Images and Tags

Images are named with CUDA and PyTorch versions in the image name:

- `ghcr.io/project-vajra/setu-dev-cuda12.9.0-torch2.8:latest`
- `ghcr.io/project-vajra/setu-dev-cuda12.9.0-torch2.8:YYYYMMDD` (date-based)
- `ghcr.io/project-vajra/setu-dev-cuda12.9.0-torch2.8:sha-<commit>` (commit-based)

The image name itself contains the version information for clarity, and tags provide different versioning options

## Building the Container

### Using Make (Recommended)

```bash
# From this directory
make build

# With specific versions
make build CUDA_VERSION=12.8 PYTORCH_VERSION=2.8
```

## Running the Container

```bash
# Run with default versions
make run

# Run with specific versions
make run CUDA_VERSION=11.8 PYTORCH_VERSION=2.0
```

