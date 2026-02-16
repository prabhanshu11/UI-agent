#!/bin/bash
# Video viewer: Display Windows HDMI capture feed.
#
# Runs on: DESKTOP (where HDMI capture card is /dev/video0)
#
# Usage:
#   ./scripts/video_viewer.sh              # Default 1920x1080@30
#   ./scripts/video_viewer.sh 1280x720 60  # Custom resolution/fps

DEVICE="${VIDEO_DEVICE:-/dev/video0}"
RESOLUTION="${1:-1920x1080}"
FPS="${2:-30}"

if [ ! -e "$DEVICE" ]; then
    echo "Error: $DEVICE not found"
    echo "Available video devices:"
    ls /dev/video* 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "Starting video viewer..."
echo "  Device: $DEVICE"
echo "  Resolution: $RESOLUTION"
echo "  FPS: $FPS"
echo "  Press 'q' to quit"

exec ffplay -f v4l2 \
    -input_format mjpeg \
    -video_size "$RESOLUTION" \
    -framerate "$FPS" \
    -i "$DEVICE" \
    -window_title "Windows KVM" \
    -loglevel warning
