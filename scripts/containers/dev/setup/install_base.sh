#!/bin/bash
set -ex

echo "::group::Installing base system dependencies..."

# Update and install required packages
apt-get update
apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    zsh \
    vim \
    nano \
    less \
    openssh-client \
    ca-certificates \
    python3-pip \
    python3-dev

# Clean up
apt-get clean
rm -rf /var/lib/apt/lists/*

# Set up git global config
git config --global credential.helper "cache --timeout=604800"

echo "::endgroup::"
