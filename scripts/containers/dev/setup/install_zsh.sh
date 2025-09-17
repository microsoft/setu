#!/bin/bash
set -ex

_script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ZSH_DIR=${ZSH_DIR:-/root/.oh-my-zsh}

echo "::group::Setting up ZSH with Oh My Zsh to $ZSH_DIR..."

# Install zsh
apt-get update
apt-get install -y zsh

# Set zsh as default shell
chsh -s "$(which zsh)"

# Install Oh My Zsh
git clone --depth=1 https://github.com/ohmyzsh/ohmyzsh.git "$ZSH_DIR"

# Install Powerlevel10k theme
git clone --depth=1 https://github.com/romkatv/powerlevel10k.git "$ZSH_DIR/custom/themes/powerlevel10k"

# Create p10k.zsh config file
cp "$_script_dir/dotconfigs/p10k.zsh" ~/.p10k.zsh

# Copy zshrc config
cp "$_script_dir/dotconfigs/zshrc" ~/.zshrc

# Install fzf
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install --all

echo "::endgroup::"
