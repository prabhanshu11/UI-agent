#!/bin/bash
# Stop NVIDIA MPS (Multi-Process Service) daemon.
#
# Usage:
#   ./scripts/stop-mps.sh

set -euo pipefail

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps

if ! pgrep -x nvidia-cuda-mps > /dev/null 2>&1; then
    echo "MPS daemon is not running."
    exit 0
fi

echo "Stopping MPS daemon..."
echo quit | nvidia-cuda-mps-control
echo "MPS daemon stopped."
