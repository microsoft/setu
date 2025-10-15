#!/bin/bash
set -e

_script_dir=$(dirname "$(readlink -f "$0")")

# Check if running as root
if [ "$(id -u)" -ne 0 ]; then
    echo "This script must be run as root. Use sudo."
    exit 1
fi

# cannot use the user's $GITHUB_TOKEN env var because script is run as root
# so take it as an argument instead
GITHUB_TOKEN="${1}" 

if [ -z "$GITHUB_TOKEN" ]; then
    echo "No GitHub token provided as argument. https://docs.github.com/en/rest/actions/self-hosted-runners?apiVersion=2022-11-28#create-a-registration-token-for-a-repository"
    exit 1
fi

sudo mkdir -p /opt/github-runners

# Create environment file with the token
echo "Creating environment file with secure permissions..."
echo "GITHUB_TOKEN=$GITHUB_TOKEN" >"/opt/github-runners/github-token.env"

# Set secure permissions
chmod 600 "/opt/github-runners/github-token.env"
chown root:root "/opt/github-runners/github-token.env"

pushd "$_script_dir"

sudo cp launch_runners.sh /opt/github-runners/
sudo cp stop_runners.sh /opt/github-runners/
sudo cp runner_spec.json /opt/github-runners/
sudo cp github-runner.service /etc/systemd/system/

chmod 744 /opt/github-runners/launch_runners.sh
chmod 744 /opt/github-runners/stop_runners.sh

popd

# Reload systemd to recognize the new service
sudo systemctl daemon-reload

# Enable the service to start on boot
sudo systemctl enable github-runner

echo "Start the service with: sudo systemctl start github-runner"