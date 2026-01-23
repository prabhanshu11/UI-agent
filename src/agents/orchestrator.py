"""Star Trek Computer Orchestrator - Coordinates all agents and talks to user.

This is the main "brain" that:
- Monitors the control file for commands
- Dispatches tasks to sub-agents
- Tracks experiences and learns from them
- Provides conversational interface for user
"""

import os
import subprocess
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass


@dataclass
class AgentStatus:
    """Status of a sub-agent."""
    name: str
    status: str  # idle, running, completed, error
    current_task: Optional[str] = None
    last_update: Optional[str] = None


class StarTrekOrchestrator:
    """Main orchestrator for the Star Trek computer agent system."""

    MODES = ["doing", "meeting", "learning"]

    def __init__(self, base_dir: Path):
        """Initialize orchestrator.

        Args:
            base_dir: Root of UI-agent repo
        """
        self.base_dir = Path(base_dir)
        self.agents_dir = self.base_dir / "agents"
        self.experiences_dir = self.base_dir / "experiences"
        self.modes_dir = self.base_dir / "modes"
        self.recordings_dir = self.base_dir / "recordings"

        self.control_file = self.agents_dir / "CONTROL.md"
        self.current_mode = "doing"
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.agents = {
            "screenshot-extractor": AgentStatus("screenshot-extractor", "idle"),
            "keyframe-identifier": AgentStatus("keyframe-identifier", "idle"),
            "summarizer": AgentStatus("summarizer", "idle"),
        }

    def set_mode(self, mode: str) -> bool:
        """Change operating mode.

        Args:
            mode: One of 'doing', 'meeting', 'learning'

        Returns:
            True if mode changed successfully
        """
        if mode not in self.MODES:
            return False

        self.current_mode = mode
        self._update_control_file()
        return True

    def dispatch_task(self, agent_name: str, task: dict) -> bool:
        """Send a task to a sub-agent.

        Args:
            agent_name: Name of agent to dispatch to
            task: Task configuration dict

        Returns:
            True if task was dispatched
        """
        if agent_name not in self.agents:
            return False

        agent_dir = self.agents_dir / agent_name
        task_file = agent_dir / "task.md"

        # Write task file
        task_yaml = yaml.dump(task, default_flow_style=False)
        task_content = f"""# Task for {agent_name}

```yaml
{task_yaml}```

## Instructions
Process the task above and write results to output.md.
"""

        task_file.write_text(task_content)

        # Update agent status
        self.agents[agent_name].status = "pending"
        self.agents[agent_name].current_task = task.get('description', 'unnamed task')
        self.agents[agent_name].last_update = datetime.now().isoformat()

        self._update_control_file()
        self._log_event(f"task_dispatched", agent_name, task.get('description', ''))

        return True

    def check_agent_outputs(self) -> dict[str, Optional[str]]:
        """Check for completed agent outputs.

        Returns:
            Dict of agent_name -> output content (or None if not complete)
        """
        outputs = {}

        for agent_name in self.agents:
            agent_dir = self.agents_dir / agent_name
            output_file = agent_dir / "output.md"

            if output_file.exists():
                content = output_file.read_text()
                if "Status**: completed" in content:
                    outputs[agent_name] = content
                    self.agents[agent_name].status = "completed"
                else:
                    outputs[agent_name] = None
            else:
                outputs[agent_name] = None

        return outputs

    def start_recording(
        self,
        fps: int = 20,
        segment_minutes: int = 5,
        output_dir: Optional[Path] = None
    ) -> tuple[bool, Optional[int], Optional[str]]:
        """Start video recording from capture card.

        Args:
            fps: Frames per second
            segment_minutes: Duration of each segment file
            output_dir: Where to save recordings

        Returns:
            (success, pid, filename) tuple
        """
        if output_dir is None:
            date_dir = datetime.now().strftime("%Y-%m-%d")
            output_dir = self.recordings_dir / date_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%H%M%S")
        output_pattern = output_dir / f"session_{timestamp}_%03d.mp4"

        cmd = [
            "ffmpeg",
            "-f", "v4l2",
            "-input_format", "mjpeg",
            "-video_size", "1920x1080",
            "-framerate", str(fps),
            "-i", "/dev/video4",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-f", "segment",
            "-segment_time", str(segment_minutes * 60),
            "-reset_timestamps", "1",
            str(output_pattern)
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            self._log_event("recording_started", "system", str(output_pattern))

            return True, proc.pid, str(output_pattern)

        except Exception as e:
            return False, None, str(e)

    def create_experience(self, task_id: str, description: str) -> Path:
        """Create a new experience folder for a task.

        Args:
            task_id: Unique task identifier
            description: Task description

        Returns:
            Path to experience directory
        """
        exp_dir = self.experiences_dir / task_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (exp_dir / "screenshots").mkdir(exist_ok=True)
        (exp_dir / "keyframes").mkdir(exist_ok=True)
        (exp_dir / "summaries").mkdir(exist_ok=True)

        # Create task.md
        task_content = f"""# Task: {description}

```yaml
task_id: "{task_id}"
created: "{datetime.now().isoformat()}"
mode: {self.current_mode}
description: "{description}"
status: in_progress
```
"""
        (exp_dir / "task.md").write_text(task_content)

        # Create empty progress.md
        (exp_dir / "progress.md").write_text(f"# Progress: {task_id}\n\n")

        self._log_event("experience_created", "system", task_id)

        return exp_dir

    def complete_experience(self, task_id: str, learnings: list[dict]) -> bool:
        """Mark experience as complete and record learnings.

        Args:
            task_id: Task identifier
            learnings: List of learning dicts with 'insight' and 'applies_to' keys

        Returns:
            True if experience was completed
        """
        exp_dir = self.experiences_dir / task_id
        if not exp_dir.exists():
            return False

        # Create learnings.md
        learnings_yaml = yaml.dump(learnings, default_flow_style=False)
        learnings_content = f"""# Learnings: {task_id}

**Completed**: {datetime.now().isoformat()}

```yaml
{learnings_yaml}```
"""
        (exp_dir / "learnings.md").write_text(learnings_content)

        # Update task status
        task_file = exp_dir / "task.md"
        if task_file.exists():
            content = task_file.read_text()
            content = content.replace("status: in_progress", "status: completed")
            task_file.write_text(content)

        self._log_event("experience_completed", "system", task_id)

        return True

    def find_similar_experiences(self, description: str, limit: int = 5) -> list[Path]:
        """Find past experiences similar to a description.

        Args:
            description: Task description to match
            limit: Maximum experiences to return

        Returns:
            List of experience directory paths
        """
        # Simple keyword matching for now
        # TODO: Use embeddings for semantic search
        keywords = set(description.lower().split())
        matches = []

        for exp_dir in self.experiences_dir.iterdir():
            if not exp_dir.is_dir() or exp_dir.name == "TEMPLATE.md":
                continue

            task_file = exp_dir / "task.md"
            if task_file.exists():
                content = task_file.read_text().lower()
                score = sum(1 for kw in keywords if kw in content)
                if score > 0:
                    matches.append((score, exp_dir))

        # Sort by score descending
        matches.sort(key=lambda x: x[0], reverse=True)

        return [exp for _, exp in matches[:limit]]

    def _update_control_file(self):
        """Update the control file with current state."""
        timestamp = datetime.now().isoformat()

        agent_rows = ""
        for name, agent in self.agents.items():
            agent_rows += f"| {name} | {agent.status} | {agent.current_task or '-'} | {agent.last_update or '-'} |\n"

        content = f"""# Star Trek Computer - Agent Control Center

**Status**: ACTIVE
**Mode**: {self.current_mode}
**Session**: {self.session_id}
**Updated**: {timestamp}

## Current Agents

| Agent | Status | Task | Last Update |
|-------|--------|------|-------------|
{agent_rows}
## Operating Modes

- **doing**: Active task execution
- **meeting**: Passive observation
- **learning**: High-detail capture

Current: **{self.current_mode}**

## Commands

To dispatch a task, create a task.md in the agent's directory.
Agents will write results to output.md.

## Event Log

See recordings and experiences directories for full history.
"""

        self.control_file.write_text(content)

    def _log_event(self, event_type: str, agent: str, details: str):
        """Log an event to the control file.

        Args:
            event_type: Type of event
            agent: Agent or system that triggered it
            details: Event details
        """
        # For now just update control file
        # TODO: Append to persistent event log
        self._update_control_file()


def extract_screenshots_from_recording(orchestrator: StarTrekOrchestrator):
    """Helper to extract screenshots from current recording.

    Args:
        orchestrator: The orchestrator instance
    """
    from .screenshot_extractor import ScreenshotExtractor

    # Find latest recording
    today = datetime.now().strftime("%Y-%m-%d")
    recordings = list((orchestrator.recordings_dir / today).glob("*.mp4"))

    if not recordings:
        print("No recordings found")
        return

    latest = max(recordings, key=lambda p: p.stat().st_mtime)
    print(f"Processing: {latest}")

    # Create experience
    exp_dir = orchestrator.create_experience(
        f"{today}_screenshot_extraction",
        f"Extract screenshots from {latest.name}"
    )

    # Run extraction
    extractor = ScreenshotExtractor(
        orchestrator.agents_dir / "screenshot-extractor"
    )

    frames = extractor.extract_change_frames(
        latest,
        exp_dir / "keyframes",
        threshold=0.15,
        min_interval=5.0
    )

    print(f"Extracted {len(frames)} keyframes")

    return frames


if __name__ == "__main__":
    # Test the orchestrator
    base_dir = Path(__file__).parent.parent.parent
    orchestrator = StarTrekOrchestrator(base_dir)

    print(f"Orchestrator initialized in {orchestrator.current_mode} mode")
    print(f"Agents: {list(orchestrator.agents.keys())}")

    # Update control file
    orchestrator._update_control_file()
    print(f"Control file updated: {orchestrator.control_file}")
