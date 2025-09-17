# Setu Docker Containers

This directory contains Docker containers used for various Setu development, testing, and deployment scenarios. Each subdirectory represents a specific container environment with its own Dockerfile, scripts, and configuration.

## Available Containers

| Container | Description | Tags | Path |
|-----------|-------------|------|------|
| `setu-dev` | Development environment with CUDA and PyTorch | `cuda12.9.0-torch2.8` | [dev](./dev/) |
| *Add more containers as they are created* | | | |

## Common Usage Patterns

All containers follow a consistent structure and provide similar command interfaces through their Makefiles.

### Building Containers

From the container's directory:

```bash
# Build using default settings
make build

# Build with specific versions or parameters
make build PARAM1=value1 PARAM2=value2
```

### Running Containers

```bash
# Run using default settings (mounts repository root to /app)
make run

# Run with specific versions or parameters
make run PARAM1=value1 PARAM2=value2
```

## Container Structure

Each container follows this general structure:

```
containers/
├── container-type/
│   ├── Dockerfile       # Container definition
│   ├── README.md        # Container-specific documentation
│   └── Makefile         # Build and run commands
```

## Creating a New Container

To add a new container:

1. Create a new directory under `containers/` with an appropriate name
2. Copy the structure from an existing container (e.g., `dev`)
3. Modify the Dockerfile and scripts for your specific needs
4. Update the Makefile with appropriate build and run parameters
5. Add container-specific documentation in the README.md
6. Add the container to the list in this README

## Common Environment Variables

These environment variables are recognized by most container Makefiles:

| Variable | Description | Example |
|----------|-------------|---------|
| `ORGANIZATION` | GitHub organization name | `project-vajra` |
| `REGISTRY` | Container registry | `ghcr.io` |
| `IMAGE_NAME` | Name of the container image | `setu-dev` |
| `TAG` | Container tag | `latest`, `cuda12.9.0-torch2.8` |

## CI/CD Integration

Containers are automatically built and pushed to the GitHub Container Registry on changes to their Dockerfile or scripts. See the relevant GitHub Actions workflows in `.github/workflows/` for details.

## Adding Container to CI/CD Pipeline

To add a new container to the CI/CD pipeline:

1. Create a new workflow file in `.github/workflows/`
2. Configure the workflow to trigger on changes to the container files
3. Use the standard GitHub Actions for building and pushing Docker images

## Common Issues and Solutions

### GPU Access

Ensure your Docker installation supports GPU access:

```bash
# Install NVIDIA Container Toolkit
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker
sudo systemctl restart docker
```

### Permission Issues

If you encounter permission errors when running Docker:

```bash
# Add your user to the docker group
sudo usermod -aG docker $USER

# Apply changes in current terminal
newgrp docker
```

## Contributing

When contributing new containers or modifying existing ones:

1. Follow the established directory structure
2. Include comprehensive documentation in the README.md
3. Ensure the Makefile provides a consistent interface
4. Add appropriate CI/CD workflows
5. Test the container thoroughly before submitting a PR
