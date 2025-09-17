# Default values - can be overridden when calling make
CUDA_VERSION ?= 12.9.0
PYTORCH_VERSION ?= 2.8
ORGANIZATION ?= $(shell git config --get remote.origin.url | sed -E 's/.*[:/]([^/]+)\/[^/]+\.git/\1/')
# Image name now includes versions for clarity
IMAGE_NAME ?= setu-dev-cuda$(CUDA_VERSION)-torch$(PYTORCH_VERSION)
REGISTRY ?= ghcr.io

# Simple tag - just use latest, or can be overridden
TAG ?= latest

USERNAME := $(if $(USERNAME),$(USERNAME),$(error USERNAME is required. Usage: make <target> USERNAME=<your-username>))
CONTAINER_NAME ?= setu-dev-container-$(USERNAME)
GPU_FLAG ?= all
SHM_SIZE ?= 12gb

# Path configurations
REPO_ROOT := $(shell git rev-parse --show-toplevel)
HOME := $(shell echo ~)
DOCKERFILE_PATH := $(REPO_ROOT)/docker/containers/dev/Dockerfile

.PHONY: help build run clean

help:
	@echo "Setu Development Container Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  help                 - Show this help message"
	@echo "  build                - Build container with specific versions"
	@echo "                         Usage: make build USERNAME=<username> [CUDA_VERSION=11.8] [PYTORCH_VERSION=2.0]"
	@echo "  start                - Start the container with specific versions"
	@echo "                         Usage: make start USERNAME=<username> [CUDA_VERSION=11.8] [PYTORCH_VERSION=2.0]"
	@echo "  stop                 - Stop the running container"
	@echo "                         Usage: make stop USERNAME=<username>"
	@echo "  run                  - Run the container with repo root mounted"
	@echo "                         Usage: make run USERNAME=<username> [CUDA_VERSION=11.8] [PYTORCH_VERSION=2.0]"
	@echo "  clean                - Remove the container and image"
	@echo "                         Usage: make clean USERNAME=<username>"
	@echo ""
	@echo "Environment variables:"
	@echo "  USERNAME             - Your username (REQUIRED)"
	@echo "  CUDA_VERSION         - CUDA version (default: 12.9.0)"
	@echo "  PYTORCH_VERSION      - PyTorch version (default: 2.8)"
	@echo "  ORGANIZATION         - GitHub organization name (default: derived from git)"
	@echo "  IMAGE_NAME           - Image name (default: setu-dev-cuda<version>-torch<version>)"
	@echo "  REGISTRY             - Container registry (default: ghcr.io)"
	@echo "  TAG                  - Image tag (default: latest)"
	@echo "  CONTAINER_NAME       - Running container name (default: setu-dev-container-<username>)"
	@echo "  GPU_FLAG             - GPU flag for docker run (default: all)"
	@echo "  SHM_SIZE             - Shared memory size (default: 12gb)"

# Build with specific CUDA and PyTorch versions
build:
	@echo "Building container with CUDA $(CUDA_VERSION) and PyTorch $(PYTORCH_VERSION)..."
	docker build \
		--build-arg CUDA_VERSION=$(CUDA_VERSION) \
		--build-arg PYTORCH_VERSION=$(PYTORCH_VERSION) \
		-t $(REGISTRY)/$(ORGANIZATION)/$(IMAGE_NAME):$(TAG) \
		-f $(DOCKERFILE_PATH) \
		$(REPO_ROOT)

# Run the container with the repository root mounted
run:
	@echo "Running container with CUDA $(CUDA_VERSION) and PyTorch $(PYTORCH_VERSION)..."
	@echo "Using container name: $(CONTAINER_NAME)"
	@echo "Mounting repository root to /repo in the container"
	docker run -it --rm \
		--name $(CONTAINER_NAME) \
		--gpus $(GPU_FLAG) \
		--shm-size=$(SHM_SIZE) \
		-v $(REPO_ROOT):/repo \
		-w /repo \
		--tmpfs/dev/shm:rw,exec,nosuid,nodev,mode=1777,size=16g \
		$(REGISTRY)/$(ORGANIZATION)/$(IMAGE_NAME):$(TAG)

# Start the container with specific versions
start:
	@echo "Starting long-running container with CUDA $(CUDA_VERSION) and PyTorch $(PYTORCH_VERSION)..."
	@echo "Using container name: $(CONTAINER_NAME)"
	docker run -d \
		--name $(CONTAINER_NAME) \
		--gpus $(GPU_FLAG) \
		--shm-size=$(SHM_SIZE) \
		-v $(REPO_ROOT):/repo \
		-w /repo \
		--tmpfs "/dev/shm:rw,exec,nosuid,nodev,mode=1777,size=16g" \
		$(REGISTRY)/$(ORGANIZATION)/$(IMAGE_NAME):$(TAG) \
		sleep infinity

attach:
	@echo "Attaching to running container..."
	@echo "Using container name: $(CONTAINER_NAME)"
	-docker exec -it $(CONTAINER_NAME) zsh

# Stop the running container
stop:
	@echo "Stopping container..."
	@echo "Using container name: $(CONTAINER_NAME)"
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true

# Clean up container and image
clean:
	@echo "Cleaning up container and image..."
	@echo "Using container name: $(CONTAINER_NAME)"
	-docker stop $(CONTAINER_NAME) 2>/dev/null || true
	-docker rm $(CONTAINER_NAME) 2>/dev/null || true
	-docker rmi $(REGISTRY)/$(ORGANIZATION)/$(IMAGE_NAME):$(TAG) 2>/dev/null || true
	-docker rmi $(IMAGE_NAME):latest 2>/dev/null || true
