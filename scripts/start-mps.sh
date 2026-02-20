#!/bin/bash
# Start NVIDIA MPS (Multi-Process Service) for concurrent GPU access.
#
# MPS allows multiple CUDA processes to share the GPU simultaneously
# with controlled resource allocation. This enables:
#   - YOLO inference at 10% GPU threads (always-on cursor detection)
#   - YOLO training at 80% GPU threads (when retraining model)
#   - Both running concurrently without context-switch overhead
#
# Usage:
#   ./scripts/start-mps.sh          # Start MPS daemon
#   CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=10 python dashboard.py  # 10% GPU
#   CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 yolo train ...        # 80% GPU

set -euo pipefail

echo "Starting NVIDIA MPS daemon..."

# Check if MPS is already running
if pgrep -x nvidia-cuda-mps > /dev/null 2>&1; then
    echo "MPS daemon is already running."
    echo "  PID: $(pgrep -x nvidia-cuda-mps)"
    exit 0
fi

# Set default active thread percentage for new processes
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log

mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

# Start the MPS control daemon
nvidia-cuda-mps-control -d

# Set default thread percentage (10% for inference processes)
echo "set_default_active_thread_percentage 10" | nvidia-cuda-mps-control

echo "MPS daemon started."
echo ""
echo "GPU resource allocation:"
echo "  Default: 10% threads (inference)"
echo "  Override: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 for training"
echo ""
echo "Usage:"
echo "  # Dashboard with YOLO inference (10% GPU)"
echo "  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=10 uv run python -m src.web.kvm_dashboard"
echo ""
echo "  # YOLO training (80% GPU)"
echo "  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=80 uv run yolo detect train ..."
echo ""
echo "Stop with: ./scripts/stop-mps.sh"
