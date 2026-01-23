#!/bin/bash
# Start L3 worker on desktop GPU
# Run this from laptop: ./scripts/start_desktop_worker.sh

SERVER_URL="http://100.103.8.87:8780"

echo "Starting worker on desktop..."
ssh desktop "cd ~/Programs/UI-agent && nohup uv run python src/agents/job_server.py --worker --server-url $SERVER_URL --device gpu > /tmp/worker.log 2>&1 &"

echo "Worker started. Check logs with: ssh desktop 'tail -f /tmp/worker.log'"
