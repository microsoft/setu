# Setu

[![Publish Nightly Build to PyPI](https://github.com/project-vajra/setu/actions/workflows/publish_nightly.yml/badge.svg)](https://github.com/project-vajra/setu/actions/workflows/publish_nightly.yml) [![Publish Release to PyPI](https://github.com/project-vajra/setu/actions/workflows/publish_release.yml/badge.svg)](https://github.com/project-vajra/setu/actions/workflows/publish_release.yml) [![Deploy Documentation](https://github.com/project-vajra/setu/actions/workflows/deploy_docs.yml/badge.svg)](https://github.com/project-vajra/setu/actions/workflows/deploy_docs.yml) [![Test Suite](https://github.com/project-vajra/setu/actions/workflows/test_suite.yml/badge.svg)](https://github.com/project-vajra/setu/actions/workflows/test_suite.yml) [![Run Linters](https://github.com/project-vajra/setu/actions/workflows/lint.yml/badge.svg)](https://github.com/project-vajra/setu/actions/workflows/lint.yml)

Global Tensor Exchange for Distributed Deep Learning Workloads

## Setup

### Option 1: Using VS Code Devcontainer

Setu now supports development using VS Code devcontainers, which provides a consistent, pre-configured development environment:

1. Install [Docker](https://www.docker.com/products/docker-desktop) and [VS Code](https://code.visualstudio.com/)
2. Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code
3. Clone the repository:
   ```sh
   git clone https://github.com/project-vajra/setu
   ```
4. Open the repository in VS Code
5. VS Code will detect the devcontainer configuration and prompt you to reopen the project in a container. Click "Reopen in Container".
6. If you want to use a subset of GPUs, update the `--gpus` flag in `.devcontainer/devcontainer.json`.
7. The devcontainer will set up the environment with all dependencies automatically
8. Use VS Code's built-in build tasks (Terminal > Run Build Task...) to easily run common Setu commands like build, test, and lint directly from the IDE

### Option 2: Using Conda/Mamba

#### Prerequisites

- **CUDA**: Setu has been tested with CUDA 12.6 on A100 and H100 GPUs
- **Git**: For cloning the repository

#### Setup

First, ensure you have mamba installed,
```sh
# Install mamba (if not already available)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

Next clone the repository,
```sh
# Clone repository
git clone https://github.com/project-vajra/setu
cd setu
```

Now create a mamba environment,
```sh
# Create environment
make setup/environment
```

Now activate the environment,
```sh
# Activate environment
mamba activate ./env
```

Finally, build the project:
```
# Install dependencies and build
make build
```

## Development

### Building

```sh
# Intelligent build (automatically selects optimal strategy)
make build

# Advanced build options (for specific needs)
make build/native              # Native extension only (full build)
make build/native_incremental  # Incremental native build (fastest)
make build/editable            # Install editable package (legacy)
make build/wheel               # Build Python wheel
```

### Testing

```sh
# Build tests intelligently (automatically detects what needs rebuilding)
make build/test

# Run all tests
make test

# Run specific test types
make test/unit           # All unit tests (Python + C++)
make test/integration    # Integration tests
make test/ctest          # C++ tests only

# Rerun only failed tests (for faster iteration)
make test/unit-failed-only     # Failed unit tests (Python + C++)
make test/pyunit-failed-only   # Failed Python unit tests only
make test/ctest-failed-only    # Failed C++ tests only
```

### Code Quality

```sh
# Lint code
make lint

# Format code
make format

# Check system dependencies
make setup/check
```

### Environment Management

```sh
# Create environment (conda)
make setup/environment

# Update environment
make setup/update-environment
```

### Utilities

```sh
# Clean build artifacts
make clean

# Show ccache statistics
make ccache/stats

# Show all available targets
make help
```

### Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft’s Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
