#!/bin/bash
# Wait for full-frame training to finish, then run crop mode
echo "[$(date)] Waiting for full-frame training to finish..."
while pgrep -f "train_phase1_model.py.*--mode full" > /dev/null 2>&1; do
    sleep 10
done
echo "[$(date)] Full-frame training done. Starting crop mode..."
cd /home/prabhanshu/Programs/UI-agent
uv run python scripts/train_phase1_model.py --mode crop --epochs 100 2>&1
echo "[$(date)] Crop training complete."
