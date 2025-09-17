ARG CUDA_VERSION=12.9.0

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04

ARG PYTORCH_VERSION=2.8

# Set environment variables
ENV SETU_CI_CUDA_VERSION=${CUDA_VERSION%.*}
ENV SETU_CI_TORCH_VERSION=${PYTORCH_VERSION}

# Set up working directory
WORKDIR /repo

# Copy everything from the repo home directory
COPY . /tmp/repo

# Make scripts executable
RUN chmod +x /tmp/repo/scripts/containers/dev/setup/*.sh

# Setup development environment (includes base deps, mamba, conda env, and pip deps)
RUN /tmp/repo/scripts/containers/dev/setup_env.sh

# Install ZSH and related components
RUN /tmp/repo/scripts/containers/dev/setup/install_zsh.sh

# Clean up
RUN rm -rf /tmp/repo

# Set default shell to zsh
SHELL ["/bin/zsh", "-c"]
CMD ["zsh"]

# Add labels
LABEL org.opencontainers.image.title="Setu Development Container"
LABEL org.opencontainers.image.description="CUDA ${CUDA_VERSION} with PyTorch ${PYTORCH_VERSION} for Setu development"
LABEL org.opencontainers.image.vendor="Vajra Team"
LABEL org.opencontainers.image.cuda.version="${CUDA_VERSION}"
LABEL org.opencontainers.image.pytorch.version="${PYTORCH_VERSION}"

