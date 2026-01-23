"""Star Trek Computer Dashboard - Real-time agent monitoring.

Shows:
- Live/recent video frames from capture
- Keyframes being identified
- Agent activity and summaries
- Multi-level storyline narratives
"""

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import base64

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# Base paths
UI_AGENT_DIR = Path(__file__).parent.parent.parent
RECORDINGS_DIR = UI_AGENT_DIR / "recordings"
EXPERIENCES_DIR = UI_AGENT_DIR / "experiences"
AGENTS_DIR = UI_AGENT_DIR / "agents"

app = FastAPI(title="Star Trek Computer Dashboard")


class AgentStatus(BaseModel):
    name: str
    status: str
    current_task: Optional[str] = None
    last_output: Optional[str] = None


class StorylineEvent(BaseModel):
    timestamp: str
    storyline: str
    event_type: str
    content: str
    agent: Optional[str] = None


LIVE_FRAME_PATH = Path("/tmp/live_frame.png")


def get_live_frame() -> tuple[Optional[Path], Optional[str]]:
    """Get the live frame being recorded (updated every second by ffmpeg).

    Returns:
        (frame_path, frame_timestamp) tuple
    """
    if LIVE_FRAME_PATH.exists():
        mtime = datetime.fromtimestamp(LIVE_FRAME_PATH.stat().st_mtime)
        age_seconds = (datetime.now() - mtime).total_seconds()

        # Only use if fresh (less than 5 seconds old)
        if age_seconds < 5:
            return LIVE_FRAME_PATH, mtime.strftime("%H:%M:%S")

    return None, None


def get_latest_frame() -> tuple[Optional[Path], Optional[str]]:
    """Get the most recent frame - prefer live, fallback to video segment.

    Returns:
        (frame_path, frame_timestamp) tuple
    """
    # Try live frame first
    live_frame, live_time = get_live_frame()
    if live_frame:
        return live_frame, live_time

    # Fallback to video segment extraction
    today = datetime.now().strftime("%Y-%m-%d")
    today_dir = RECORDINGS_DIR / today

    if not today_dir.exists():
        return None, None

    videos = sorted(today_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not videos:
        return None, None

    temp_frame = Path("/tmp/dashboard_frame.png")

    # Try videos in order - skip ones that are still being written
    for video in videos:
        try:
            # Get duration - if this fails, file is incomplete
            probe_result = subprocess.run([
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video)
            ], capture_output=True, text=True, timeout=3)

            if not probe_result.stdout.strip():
                continue  # Incomplete file, try next

            duration = float(probe_result.stdout.strip())
            if duration < 1:
                continue  # Too short

            # Extract frame from end of this video
            seek_time = max(0, duration - 0.5)

            result = subprocess.run([
                "ffmpeg", "-ss", str(seek_time), "-i", str(video),
                "-frames:v", "1", "-y", str(temp_frame)
            ], capture_output=True, timeout=5)

            if result.returncode == 0 and temp_frame.exists() and temp_frame.stat().st_size > 1000:
                # Calculate frame timestamp
                video_mtime = datetime.fromtimestamp(video.stat().st_mtime)
                frame_time = video_mtime - timedelta(seconds=(duration - seek_time))

                return temp_frame, frame_time.strftime("%H:%M:%S")

        except Exception:
            continue  # Try next video

    return None, None


def get_keyframes(limit: int = 20) -> list[dict]:
    """Get recent keyframes from experiences."""
    keyframes = []

    for exp_dir in sorted(EXPERIENCES_DIR.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue

        kf_dir = exp_dir / "keyframes"
        if kf_dir.exists():
            for img in sorted(kf_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True):
                keyframes.append({
                    "path": str(img),
                    "name": img.name,
                    "experience": exp_dir.name,
                    "timestamp": datetime.fromtimestamp(img.stat().st_mtime).isoformat()
                })
                if len(keyframes) >= limit:
                    break

        if len(keyframes) >= limit:
            break

    return keyframes


def get_storyline_events(limit: int = 50) -> list[dict]:
    """Get recent storyline events across all experiences."""
    events = []

    for exp_dir in sorted(EXPERIENCES_DIR.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue

        storylines_dir = exp_dir / "storylines"
        if storylines_dir.exists():
            timeline_file = storylines_dir / "timeline.jsonl"
            if timeline_file.exists():
                try:
                    lines = timeline_file.read_text().strip().split('\n')
                    for line in reversed(lines[-limit:]):
                        if line:
                            event = json.loads(line)
                            event["experience"] = exp_dir.name
                            events.append(event)
                except Exception:
                    pass

        if len(events) >= limit:
            break

    # Sort by timestamp descending
    events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return events[:limit]


def get_agent_statuses() -> list[dict]:
    """Get status of all agents."""
    statuses = []

    for agent_dir in AGENTS_DIR.iterdir():
        if not agent_dir.is_dir():
            continue

        agent_md = agent_dir / "AGENT.md"
        output_md = agent_dir / "output.md"
        task_md = agent_dir / "task.md"

        status = {
            "name": agent_dir.name,
            "status": "idle",
            "task": None,
            "last_output": None
        }

        if task_md.exists():
            content = task_md.read_text()
            if "status: pending" in content:
                status["status"] = "pending"
            elif "status: in_progress" in content:
                status["status"] = "running"
            # Extract task description
            for line in content.split('\n'):
                if line.startswith("description:"):
                    status["task"] = line.split(":", 1)[1].strip().strip('"')
                    break

        if output_md.exists():
            # Get last few lines of output
            content = output_md.read_text()
            lines = [l for l in content.split('\n') if l.strip()][-5:]
            status["last_output"] = '\n'.join(lines)

        statuses.append(status)

    return statuses


def image_to_base64(path: Path) -> str:
    """Convert image to base64 data URL."""
    if not path.exists():
        return ""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode()
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def get_agent_analysis_state() -> dict:
    """Get the current agent analysis state from state files."""
    state = {
        "status": "Observing",
        "last_keyframe": None,
        "change_percent": None,
        "keyframe_count": 0
    }

    # Read agent state from keyframe-identifier
    state_file = AGENTS_DIR / "keyframe-identifier" / "state.json"
    if state_file.exists():
        try:
            agent_state = json.loads(state_file.read_text())
            state.update(agent_state)
        except Exception:
            pass

    # Get latest keyframe info
    keyframes = get_keyframes(limit=1)
    if keyframes:
        kf = keyframes[0]
        state["last_keyframe"] = f"{kf['name']} ({kf['timestamp'].split('T')[1].split('.')[0]})"

    # Count total keyframes today
    today = datetime.now().strftime("%Y-%m-%d")
    today_exp = EXPERIENCES_DIR / today
    if today_exp.exists():
        kf_dir = today_exp / "keyframes"
        if kf_dir.exists():
            state["keyframe_count"] = len(list(kf_dir.glob("*.png")))

    # Check if screenshot extractor is active
    extractor_task = AGENTS_DIR / "screenshot-extractor" / "task.md"
    if extractor_task.exists():
        content = extractor_task.read_text()
        if "status: in_progress" in content:
            state["status"] = "Extracting screenshots"
        elif "status: pending" in content:
            state["status"] = "Pending extraction"

    return state


# API Endpoints

@app.get("/")
async def dashboard():
    """Serve the main dashboard HTML."""
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/api/frame")
async def get_frame():
    """Get the latest frame as base64 with timestamp."""
    frame, frame_time = get_latest_frame()
    if frame:
        return {
            "image": image_to_base64(frame),
            "path": str(frame),
            "frame_time": frame_time,
            "current_time": datetime.now().strftime("%H:%M:%S"),
            "lag_info": f"Frame from {frame_time}" if frame_time else "Unknown"
        }
    return {"image": None, "error": "No frames available"}


@app.get("/api/frame/agent")
async def get_agent_frame():
    """Get the frame as the agent sees it, with analysis data."""
    frame, frame_time = get_latest_frame()
    if not frame:
        return {"image": None, "error": "No frames available"}

    # Get analysis data from agent state files
    analysis = get_agent_analysis_state()

    return {
        "image": image_to_base64(frame),
        "path": str(frame),
        "frame_time": frame_time,
        "current_time": datetime.now().strftime("%H:%M:%S"),
        "analysis": analysis
    }


@app.get("/api/keyframes")
async def api_keyframes(limit: int = 20):
    """Get recent keyframes."""
    keyframes = get_keyframes(limit)
    # Add base64 images
    for kf in keyframes:
        path = Path(kf["path"])
        if path.exists():
            kf["image"] = image_to_base64(path)
    return {"keyframes": keyframes}


@app.get("/api/storyline")
async def api_storyline(limit: int = 50):
    """Get storyline events."""
    events = get_storyline_events(limit)
    return {"events": events}


@app.get("/api/agents")
async def api_agents():
    """Get agent statuses."""
    return {"agents": get_agent_statuses()}


@app.get("/api/knowledge")
async def api_knowledge(limit: int = 20):
    """Get knowledge storyline entries - prioritizing real insights."""
    knowledge_items = []
    timeline_summaries = []

    # Read from knowledge.jsonl
    for exp_dir in sorted(EXPERIENCES_DIR.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue

        storylines_dir = exp_dir / "storylines"
        knowledge_file = storylines_dir / "knowledge.jsonl"

        if knowledge_file.exists():
            try:
                lines = knowledge_file.read_text().strip().split('\n')
                for line in reversed(lines[-200:]):  # Read more to find summaries
                    if line:
                        entry = json.loads(line)
                        entry["experience"] = exp_dir.name
                        # Separate timeline_summaries (real L2 insights) from basic captures
                        if entry.get("event_type") == "timeline_summary":
                            timeline_summaries.append(entry)
                        else:
                            knowledge_items.append(entry)
            except Exception:
                pass

        if len(timeline_summaries) >= limit:
            break

    # Sort by timestamp descending
    timeline_summaries.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    knowledge_items.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    # Get counts
    today = datetime.now().strftime("%Y-%m-%d")
    today_exp = EXPERIENCES_DIR / today
    keyframe_count = 0
    analyzed_count = 0

    if today_exp.exists():
        kf_dir = today_exp / "keyframes"
        if kf_dir.exists():
            keyframe_count = len(list(kf_dir.glob("*.png")))

    # Count L3 analyses
    analyzers_dir = UI_AGENT_DIR / "agents" / "analyzers"
    if analyzers_dir.exists():
        for batch_file in analyzers_dir.glob("*.jsonl"):
            try:
                analyzed_count += len(batch_file.read_text().strip().split('\n'))
            except Exception:
                pass

    # Get hierarchical agent status
    agent_hierarchy = get_agent_hierarchy_status()

    return {
        "timeline_summaries": timeline_summaries[:limit],  # Real L2 insights
        "items": knowledge_items[:limit],  # Basic captures
        "insight_count": len(timeline_summaries),
        "keyframe_count": keyframe_count,
        "analyzed_count": analyzed_count,
        "agent_hierarchy": agent_hierarchy
    }


def get_agent_hierarchy_status() -> dict:
    """Get L1/L2/L3 agent hierarchy status."""
    hierarchy = {
        "L3": {"status": "Idle", "processed": 0, "mode": "Parallel Vision Analysis"},
        "L2": {"status": "Idle", "processed": 0, "mode": "Timeline Summarization"},
        "L1": {"status": "Ready", "mode": "Strategic Orchestrator"}
    }

    # L3 state - keyframe-identifier
    l3_state = AGENTS_DIR / "keyframe-identifier" / "state.json"
    if l3_state.exists():
        try:
            state = json.loads(l3_state.read_text())
            hierarchy["L3"]["status"] = state.get("status", "Idle")
            hierarchy["L3"]["processed"] = state.get("processed_count", 0)
            hierarchy["L3"]["mode"] = state.get("mode", "Parallel Vision Analysis")
        except Exception:
            pass

    # L2 state - summarizer
    l2_state = AGENTS_DIR / "summarizer" / "state.json"
    if l2_state.exists():
        try:
            state = json.loads(l2_state.read_text())
            hierarchy["L2"]["status"] = state.get("status", "Idle")
            hierarchy["L2"]["processed"] = state.get("windows_processed", 0)
            hierarchy["L2"]["mode"] = state.get("mode", "Timeline Summarization")
        except Exception:
            pass

    return hierarchy


@app.get("/api/hierarchy")
async def api_hierarchy():
    """Get hierarchical agent status."""
    return get_agent_hierarchy_status()


@app.get("/api/status")
async def api_status():
    """Get overall system status."""
    # Check if recording is active
    result = subprocess.run(["pgrep", "-f", "ffmpeg.*video4"], capture_output=True)
    recording_active = result.returncode == 0

    # Get recording info
    today = datetime.now().strftime("%Y-%m-%d")
    today_dir = RECORDINGS_DIR / today
    recordings = []
    if today_dir.exists():
        for vid in today_dir.glob("*.mp4"):
            recordings.append({
                "name": vid.name,
                "size_mb": vid.stat().st_size / 1024 / 1024,
                "modified": datetime.fromtimestamp(vid.stat().st_mtime).isoformat()
            })

    return {
        "recording_active": recording_active,
        "recordings_today": len(recordings),
        "recordings": recordings[-5:],  # Last 5
        "timestamp": datetime.now().isoformat()
    }


# Dashboard HTML
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Star Trek Computer - Agent Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            padding: 1rem 2rem;
            border-bottom: 2px solid #ff6b35;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            color: #ff6b35;
            font-size: 1.5rem;
        }
        .status-badge {
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
        }
        .status-recording { background: #2ecc71; color: #000; }
        .status-idle { background: #95a5a6; color: #000; }

        .container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: auto auto auto;
            gap: 1rem;
            padding: 1rem;
            max-width: 1800px;
            margin: 0 auto;
        }

        .panel {
            background: #12121a;
            border-radius: 8px;
            border: 1px solid #2a2a3a;
            overflow: hidden;
        }
        .panel-header {
            background: #1a1a2e;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #2a2a3a;
            font-weight: 600;
            color: #ff6b35;
        }
        .panel-content {
            padding: 1rem;
        }

        /* Live Feed Panel */
        .live-feed {
            grid-column: 1;
            grid-row: 1;
        }
        .live-feed img {
            width: 100%;
            height: auto;
            border-radius: 4px;
            background: #000;
        }
        .live-feed .placeholder {
            width: 100%;
            aspect-ratio: 16/9;
            background: #1a1a2e;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #666;
            border-radius: 4px;
        }

        /* Knowledge Panel */
        .knowledge {
            grid-column: 2;
            grid-row: 1;
        }
        .knowledge-list {
            max-height: 200px;
            overflow-y: auto;
        }
        .knowledge-item {
            padding: 0.5rem;
            border-bottom: 1px solid #2a2a3a;
            font-size: 0.85rem;
        }
        .knowledge-item:last-child { border-bottom: none; }
        .knowledge-item .time {
            color: #666;
            font-family: monospace;
            font-size: 0.75rem;
            margin-right: 0.5rem;
        }
        .knowledge-item .tag {
            display: inline-block;
            padding: 0.1rem 0.4rem;
            border-radius: 3px;
            font-size: 0.7rem;
            margin-right: 0.5rem;
        }
        .knowledge-item .tag.insight { background: #9b59b6; color: #fff; }
        .knowledge-item .tag.workflow { background: #3498db; color: #fff; }
        .knowledge-item .tag.technical { background: #e74c3c; color: #fff; }

        /* Keyframes Panel */
        .keyframes {
            grid-column: 2;
            grid-row: 2;
        }
        .keyframes-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.5rem;
            max-height: 600px;
            overflow-y: auto;
        }
        .keyframe-item {
            background: #1a1a2e;
            border-radius: 4px;
            overflow: hidden;
        }
        .keyframe-item img {
            width: 100%;
            aspect-ratio: 16/9;
            object-fit: cover;
        }
        .keyframe-item .info {
            padding: 0.25rem 0.5rem;
            font-size: 0.7rem;
            color: #888;
        }

        /* Agents Panel - Hierarchical */
        .agents {
            grid-column: 1;
            grid-row: 2 / 3;
        }
        .hierarchy-card {
            background: linear-gradient(135deg, #1a1a2e 0%, #12121a 100%);
            padding: 0.75rem;
            border-radius: 6px;
            margin-bottom: 0.75rem;
            border-left: 4px solid #666;
            position: relative;
        }
        .hierarchy-card.L1 { border-left-color: #ff6b35; background: linear-gradient(135deg, #2a1a1e 0%, #1a1a2e 100%); }
        .hierarchy-card.L2 { border-left-color: #9b59b6; }
        .hierarchy-card.L3 { border-left-color: #3498db; }
        .hierarchy-card.running { box-shadow: 0 0 10px rgba(46, 204, 113, 0.3); }
        .hierarchy-level {
            position: absolute;
            top: 0.5rem;
            right: 0.5rem;
            font-size: 0.7rem;
            padding: 0.15rem 0.4rem;
            border-radius: 3px;
            font-weight: bold;
        }
        .hierarchy-level.L1 { background: #ff6b35; color: #000; }
        .hierarchy-level.L2 { background: #9b59b6; color: #fff; }
        .hierarchy-level.L3 { background: #3498db; color: #fff; }
        .hierarchy-name {
            font-weight: 600;
            color: #fff;
            margin-bottom: 0.25rem;
            font-size: 0.95rem;
        }
        .hierarchy-mode {
            font-size: 0.75rem;
            color: #888;
            margin-bottom: 0.25rem;
        }
        .hierarchy-stats {
            display: flex;
            gap: 1rem;
            font-size: 0.8rem;
            margin-top: 0.5rem;
        }
        .hierarchy-stat {
            display: flex;
            align-items: center;
            gap: 0.25rem;
        }
        .hierarchy-stat .label { color: #666; }
        .hierarchy-stat .value { color: #2ecc71; font-weight: bold; }
        .hierarchy-stat .value.running { color: #2ecc71; }
        .hierarchy-stat .value.idle { color: #f39c12; }
        .agent-card {
            background: #1a1a2e;
            padding: 0.75rem;
            border-radius: 4px;
            margin-bottom: 0.5rem;
            border-left: 3px solid #666;
        }
        .agent-card.running { border-left-color: #2ecc71; }
        .agent-card.pending { border-left-color: #f39c12; }
        .agent-card.idle { border-left-color: #666; }
        .agent-name {
            font-weight: 600;
            color: #fff;
            margin-bottom: 0.25rem;
        }
        .agent-status {
            font-size: 0.8rem;
            color: #888;
        }
        .agent-output {
            font-size: 0.75rem;
            color: #666;
            margin-top: 0.5rem;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 60px;
            overflow: hidden;
        }

        /* Timeline Summary Insight */
        .insight-card {
            background: linear-gradient(135deg, #1f1a2e 0%, #12121a 100%);
            padding: 0.75rem;
            border-radius: 6px;
            margin-bottom: 0.5rem;
            border-left: 4px solid #9b59b6;
        }
        .insight-card .time-window {
            font-size: 0.75rem;
            color: #9b59b6;
            font-weight: bold;
            margin-bottom: 0.25rem;
        }
        .insight-card .content {
            font-size: 0.85rem;
            color: #e0e0e0;
            line-height: 1.4;
        }

        /* Storyline Panel */
        .storyline {
            grid-column: 1 / 3;
            grid-row: 3;
        }
        .storyline-events {
            max-height: 300px;
            overflow-y: auto;
        }
        .event {
            padding: 0.5rem;
            border-bottom: 1px solid #2a2a3a;
            display: grid;
            grid-template-columns: 100px 80px 1fr auto;
            gap: 0.5rem;
            font-size: 0.85rem;
        }
        .event:hover { background: #1a1a2e; }
        .event-time { color: #666; font-family: monospace; font-size: 0.75rem; }
        .event-type {
            padding: 0.1rem 0.4rem;
            border-radius: 3px;
            font-size: 0.7rem;
            text-align: center;
        }
        .event-type.main { background: #3498db; color: #fff; }
        .event-type.knowledge { background: #9b59b6; color: #fff; }
        .event-type.technical { background: #e74c3c; color: #fff; }
        .event-type.agent { background: #2ecc71; color: #000; }
        .event-type.perception { background: #f39c12; color: #000; }
        .event-content { color: #e0e0e0; }
        .event-agent { color: #888; font-size: 0.75rem; }

        /* Mode Toggle Buttons */
        .mode-btn {
            padding: 0.3rem 0.6rem;
            border: 1px solid #3a3a4a;
            background: #1a1a2e;
            color: #888;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.75rem;
            transition: all 0.2s;
        }
        .mode-btn:hover {
            border-color: #ff6b35;
            color: #fff;
        }
        .mode-btn.active {
            background: #ff6b35;
            border-color: #ff6b35;
            color: #000;
            font-weight: 600;
        }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1a1a2e; }
        ::-webkit-scrollbar-thumb { background: #3a3a4a; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #4a4a5a; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🖖 Star Trek Computer - Agent Dashboard</h1>
        <div>
            <span id="recording-status" class="status-badge status-idle">Checking...</span>
            <span id="timestamp" style="margin-left: 1rem; color: #666; font-size: 0.85rem;"></span>
        </div>
    </div>

    <div class="container">
        <!-- Live Feed -->
        <div class="panel live-feed">
            <div class="panel-header">
                📹 Live Feed
                <div class="mode-toggle" style="float: right; display: flex; gap: 0.5rem; align-items: center;">
                    <button id="mode-recording" class="mode-btn active" onclick="setViewMode('recording')">Recording</button>
                    <button id="mode-agent" class="mode-btn" onclick="setViewMode('agent')">Agent View</button>
                    <span id="frame-time" style="margin-left: 0.5rem; font-weight: normal; font-size: 0.85rem; color: #888;"></span>
                </div>
            </div>
            <div class="panel-content">
                <div id="live-frame" class="placeholder">Loading...</div>
                <div id="agent-analysis" style="display: none; margin-top: 0.5rem; padding: 0.75rem; background: #1a1a2e; border-radius: 4px; border: 1px solid #2a2a3a;">
                    <div style="color: #ff6b35; font-size: 0.8rem; margin-bottom: 0.5rem;">🔍 Agent Analysis</div>
                    <div id="analysis-content" style="font-size: 0.85rem; color: #aaa;"></div>
                </div>
                <div id="lag-warning" style="margin-top: 0.5rem; padding: 0.5rem; background: #2a2a3a; border-radius: 4px; font-size: 0.8rem; display: none;">
                    <span style="color: #f39c12;">⚠️</span> <span id="lag-text"></span>
                </div>
            </div>
        </div>

        <!-- Knowledge Building -->
        <div class="panel knowledge">
            <div class="panel-header">🧠 Knowledge Building</div>
            <div class="panel-content">
                <div id="knowledge-stats" style="display: flex; gap: 1rem; margin-bottom: 1rem; font-size: 0.85rem;">
                    <div style="background: #1a1a2e; padding: 0.5rem 1rem; border-radius: 4px;">
                        <span style="color: #888;">Keyframes:</span>
                        <span id="kf-count" style="color: #2ecc71; font-weight: bold;">0</span>
                    </div>
                    <div style="background: #1a1a2e; padding: 0.5rem 1rem; border-radius: 4px;">
                        <span style="color: #888;">Insights:</span>
                        <span id="insight-count" style="color: #9b59b6; font-weight: bold;">0</span>
                    </div>
                    <div style="background: #1a1a2e; padding: 0.5rem 1rem; border-radius: 4px;">
                        <span style="color: #888;">Status:</span>
                        <span id="agent-status-text" style="color: #f39c12;">Idle</span>
                    </div>
                </div>
                <div id="knowledge-list" class="knowledge-list">
                    <div class="placeholder">Waiting for knowledge...</div>
                </div>
            </div>
        </div>

        <!-- Keyframes Grid (now smaller) -->
        <div class="panel keyframes">
            <div class="panel-header">🎯 Keyframes (<span id="keyframes-count">0</span>)</div>
            <div class="panel-content">
                <div id="keyframes-grid" class="keyframes-grid">
                    <div class="placeholder">Loading keyframes...</div>
                </div>
            </div>
        </div>

        <!-- Hierarchical Agents -->
        <div class="panel agents">
            <div class="panel-header">🤖 Agent Hierarchy (L1/L2/L3)</div>
            <div class="panel-content" id="agents-container">
                <div id="hierarchy-view">Loading hierarchy...</div>
                <div id="legacy-agents" style="margin-top: 1rem; border-top: 1px solid #2a2a3a; padding-top: 1rem; display: none;">
                    <div style="color: #666; font-size: 0.75rem; margin-bottom: 0.5rem;">Other Agents</div>
                    <div id="legacy-agents-list"></div>
                </div>
            </div>
        </div>

        <!-- Storyline -->
        <div class="panel storyline">
            <div class="panel-header">📜 Storyline Events</div>
            <div class="panel-content">
                <div id="storyline-events" class="storyline-events">
                    Loading events...
                </div>
            </div>
        </div>
    </div>

    <script>
        // View mode state
        let currentViewMode = 'recording';

        function setViewMode(mode) {
            currentViewMode = mode;

            // Update button states
            document.getElementById('mode-recording').classList.toggle('active', mode === 'recording');
            document.getElementById('mode-agent').classList.toggle('active', mode === 'agent');

            // Show/hide analysis panel
            document.getElementById('agent-analysis').style.display = mode === 'agent' ? 'block' : 'none';

            // Trigger immediate frame update
            updateFrame();
        }

        // Update functions
        async function updateFrame() {
            try {
                const endpoint = currentViewMode === 'agent' ? '/api/frame/agent' : '/api/frame';
                const res = await fetch(endpoint);
                const data = await res.json();
                const container = document.getElementById('live-frame');
                const frameTimeEl = document.getElementById('frame-time');
                const lagWarning = document.getElementById('lag-warning');
                const lagText = document.getElementById('lag-text');
                const analysisContent = document.getElementById('analysis-content');

                if (data.image) {
                    container.innerHTML = `<img src="${data.image}" alt="${currentViewMode === 'agent' ? 'Agent view' : 'Live feed'}">`;

                    // Show frame time
                    if (data.frame_time) {
                        const modeLabel = currentViewMode === 'agent' ? '🔍 Agent' : '📹 Recording';
                        frameTimeEl.textContent = `${modeLabel} | Frame: ${data.frame_time} | Now: ${data.current_time}`;

                        // Calculate lag
                        if (data.frame_time && data.current_time) {
                            const frameParts = data.frame_time.split(':').map(Number);
                            const nowParts = data.current_time.split(':').map(Number);
                            const frameSeconds = frameParts[0] * 3600 + frameParts[1] * 60 + frameParts[2];
                            const nowSeconds = nowParts[0] * 3600 + nowParts[1] * 60 + nowParts[2];
                            const lagSeconds = nowSeconds - frameSeconds;

                            if (lagSeconds > 30) {
                                lagWarning.style.display = 'block';
                                lagText.textContent = `Frame is ${lagSeconds} seconds behind (${Math.floor(lagSeconds/60)}m ${lagSeconds%60}s lag)`;
                            } else {
                                lagWarning.style.display = 'none';
                            }
                        }
                    }

                    // Update agent analysis if in agent mode
                    if (currentViewMode === 'agent' && data.analysis) {
                        analysisContent.innerHTML = `
                            <div style="margin-bottom: 0.5rem;">
                                <strong>Last keyframe:</strong> ${data.analysis.last_keyframe || 'None detected'}
                            </div>
                            <div style="margin-bottom: 0.5rem;">
                                <strong>Change detected:</strong> ${data.analysis.change_percent ? data.analysis.change_percent.toFixed(1) + '%' : 'N/A'}
                            </div>
                            <div>
                                <strong>Status:</strong> ${data.analysis.status || 'Observing'}
                            </div>
                        `;
                    }
                } else {
                    container.innerHTML = '<div class="placeholder">No frame available</div>';
                    frameTimeEl.textContent = '';
                    lagWarning.style.display = 'none';
                    if (currentViewMode === 'agent') {
                        analysisContent.innerHTML = '<em>Waiting for frames...</em>';
                    }
                }
            } catch (e) {
                console.error('Frame update failed:', e);
            }
        }

        async function updateKeyframes() {
            try {
                const res = await fetch('/api/keyframes?limit=9');
                const data = await res.json();
                const container = document.getElementById('keyframes-grid');
                const countEl = document.getElementById('keyframes-count');

                if (data.keyframes && data.keyframes.length > 0) {
                    countEl.textContent = data.keyframes.length;
                    container.innerHTML = data.keyframes.map(kf => `
                        <div class="keyframe-item">
                            <img src="${kf.image || ''}" alt="${kf.name}">
                            <div class="info">${kf.name}</div>
                        </div>
                    `).join('');
                } else {
                    countEl.textContent = '0';
                    container.innerHTML = '<div class="placeholder">No keyframes yet</div>';
                }
            } catch (e) {
                console.error('Keyframes update failed:', e);
            }
        }

        async function updateKnowledge() {
            try {
                const res = await fetch('/api/knowledge?limit=15');
                const data = await res.json();

                // Update stats - now with analyzed count
                document.getElementById('kf-count').textContent = data.keyframe_count || 0;
                document.getElementById('insight-count').textContent = data.insight_count || 0;

                // Update agent hierarchy status text
                const statusEl = document.getElementById('agent-status-text');
                const hierarchy = data.agent_hierarchy;
                if (hierarchy) {
                    const l3Running = hierarchy.L3?.status === 'Running';
                    const l2Running = hierarchy.L2?.status === 'Running';
                    if (l3Running && l2Running) {
                        statusEl.textContent = 'L3+L2 Active';
                        statusEl.style.color = '#2ecc71';
                    } else if (l3Running) {
                        statusEl.textContent = 'L3 Active';
                        statusEl.style.color = '#3498db';
                    } else if (l2Running) {
                        statusEl.textContent = 'L2 Active';
                        statusEl.style.color = '#9b59b6';
                    } else {
                        statusEl.textContent = 'Idle';
                        statusEl.style.color = '#f39c12';
                    }
                }

                // Update knowledge list - prioritize timeline_summaries (REAL insights)
                const container = document.getElementById('knowledge-list');
                let html = '';

                // First show real L2 timeline summaries
                if (data.timeline_summaries && data.timeline_summaries.length > 0) {
                    html += data.timeline_summaries.slice(0, 5).map(item => {
                        const time = item.timestamp ? item.timestamp.split('T')[1].split('.')[0] : '';
                        const window = item.metadata?.time_window || '';
                        return `
                            <div class="insight-card">
                                <div class="time-window">🕐 ${window} Window | ${time}</div>
                                <div class="content">${item.content}</div>
                            </div>
                        `;
                    }).join('');
                }

                // Then show basic captures (limited)
                if (data.items && data.items.length > 0) {
                    const basicItems = data.items.slice(0, 3);
                    html += basicItems.map(item => {
                        const time = item.timestamp ? item.timestamp.split('T')[1].split('.')[0] : '';
                        return `
                            <div class="knowledge-item" style="opacity: 0.6;">
                                <span class="time">${time}</span>
                                <span style="font-size: 0.8rem;">${item.content.substring(0, 80)}...</span>
                            </div>
                        `;
                    }).join('');
                }

                if (html) {
                    container.innerHTML = html;
                } else {
                    container.innerHTML = '<div class="placeholder">Collecting knowledge... L3 analyzing keyframes</div>';
                }

                // Also update hierarchy view
                updateHierarchyView(data.agent_hierarchy, data.analyzed_count);

            } catch (e) {
                console.error('Knowledge update failed:', e);
            }
        }

        function updateHierarchyView(hierarchy, analyzedCount) {
            if (!hierarchy) return;

            const container = document.getElementById('hierarchy-view');
            const l1 = hierarchy.L1 || {};
            const l2 = hierarchy.L2 || {};
            const l3 = hierarchy.L3 || {};

            container.innerHTML = `
                <div class="hierarchy-card L1">
                    <span class="hierarchy-level L1">L1</span>
                    <div class="hierarchy-name">Strategic Orchestrator</div>
                    <div class="hierarchy-mode">${l1.mode || 'Task Context & Directives'}</div>
                    <div class="hierarchy-stats">
                        <div class="hierarchy-stat">
                            <span class="label">Status:</span>
                            <span class="value">${l1.status || 'Ready for user'}</span>
                        </div>
                    </div>
                </div>

                <div class="hierarchy-card L2 ${l2.status === 'Running' ? 'running' : ''}">
                    <span class="hierarchy-level L2">L2</span>
                    <div class="hierarchy-name">Timeline Summarizer</div>
                    <div class="hierarchy-mode">${l2.mode || '5-minute window aggregation'}</div>
                    <div class="hierarchy-stats">
                        <div class="hierarchy-stat">
                            <span class="label">Status:</span>
                            <span class="value ${l2.status === 'Running' ? 'running' : 'idle'}">${l2.status || 'Idle'}</span>
                        </div>
                        <div class="hierarchy-stat">
                            <span class="label">Windows:</span>
                            <span class="value">${l2.processed || 0}</span>
                        </div>
                    </div>
                </div>

                <div class="hierarchy-card L3 ${l3.status === 'Running' ? 'running' : ''}">
                    <span class="hierarchy-level L3">L3</span>
                    <div class="hierarchy-name">Parallel Vision Analyzers</div>
                    <div class="hierarchy-mode">${l3.mode || 'Claude SDK + Vision'}</div>
                    <div class="hierarchy-stats">
                        <div class="hierarchy-stat">
                            <span class="label">Status:</span>
                            <span class="value ${l3.status === 'Running' ? 'running' : 'idle'}">${l3.status || 'Idle'}</span>
                        </div>
                        <div class="hierarchy-stat">
                            <span class="label">Analyzed:</span>
                            <span class="value">${analyzedCount || l3.processed || 0}</span>
                        </div>
                    </div>
                </div>
            `;
        }

        async function updateAgents() {
            try {
                const res = await fetch('/api/agents');
                const data = await res.json();
                const container = document.getElementById('agents-container');

                if (data.agents && data.agents.length > 0) {
                    container.innerHTML = data.agents.map(agent => `
                        <div class="agent-card ${agent.status}">
                            <div class="agent-name">${agent.name}</div>
                            <div class="agent-status">
                                Status: ${agent.status}
                                ${agent.task ? ` | Task: ${agent.task}` : ''}
                            </div>
                            ${agent.last_output ? `<div class="agent-output">${agent.last_output}</div>` : ''}
                        </div>
                    `).join('');
                } else {
                    container.innerHTML = '<div>No agents found</div>';
                }
            } catch (e) {
                console.error('Agents update failed:', e);
            }
        }

        async function updateStoryline() {
            try {
                const res = await fetch('/api/storyline?limit=30');
                const data = await res.json();
                const container = document.getElementById('storyline-events');

                if (data.events && data.events.length > 0) {
                    container.innerHTML = data.events.map(event => {
                        const time = event.timestamp ? event.timestamp.split('T')[1].split('.')[0] : '';
                        return `
                            <div class="event">
                                <span class="event-time">${time}</span>
                                <span class="event-type ${event.storyline}">${event.storyline}</span>
                                <span class="event-content">${event.content}</span>
                                <span class="event-agent">${event.agent_name || ''}</span>
                            </div>
                        `;
                    }).join('');
                } else {
                    container.innerHTML = '<div class="placeholder">No events yet</div>';
                }
            } catch (e) {
                console.error('Storyline update failed:', e);
            }
        }

        async function updateStatus() {
            try {
                const res = await fetch('/api/status');
                const data = await res.json();

                const badge = document.getElementById('recording-status');
                if (data.recording_active) {
                    badge.textContent = '● Recording';
                    badge.className = 'status-badge status-recording';
                } else {
                    badge.textContent = '○ Idle';
                    badge.className = 'status-badge status-idle';
                }

                const timestamp = document.getElementById('timestamp');
                timestamp.textContent = new Date().toLocaleTimeString();
            } catch (e) {
                console.error('Status update failed:', e);
            }
        }

        // Initial load
        updateFrame();
        updateKeyframes();
        updateKnowledge();
        updateAgents();
        updateStoryline();
        updateStatus();

        // Refresh intervals
        setInterval(updateFrame, 2000);      // Every 2 seconds
        setInterval(updateKeyframes, 5000);  // Every 5 seconds
        setInterval(updateKnowledge, 3000);  // Every 3 seconds
        setInterval(updateAgents, 3000);     // Every 3 seconds
        setInterval(updateStoryline, 2000);  // Every 2 seconds
        setInterval(updateStatus, 1000);     // Every second
    </script>
</body>
</html>
"""


def run_dashboard(host: str = "0.0.0.0", port: int = 8765):
    """Run the dashboard server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_dashboard()
