"""Multi-Storyline Logger - Agent Zero Compatible.

Tracks multiple parallel narratives of agent activity:
- Main storyline: Primary task flow
- Sub-storylines: Parallel agent activities
- Knowledge storyline: Learnings and insights
- Technical storyline: Code, commands, errors

Based on Agent Zero's hierarchical agent communication patterns.
https://github.com/agent0ai/agent-zero
"""

import json
import os
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from enum import Enum


class StorylineType(str, Enum):
    """Types of storylines to track."""
    MAIN = "main"           # Primary task narrative
    KNOWLEDGE = "knowledge" # Learnings and insights
    TECHNICAL = "technical" # Commands, code, errors
    AGENT = "agent"         # Sub-agent activities
    USER = "user"           # User interactions
    PERCEPTION = "perception"  # What was seen/heard


@dataclass
class StorylineEntry:
    """Single entry in a storyline."""
    timestamp: str
    storyline: str
    event_type: str
    content: str
    metadata: dict = field(default_factory=dict)
    parent_entry_id: Optional[str] = None
    agent_name: Optional[str] = None

    @property
    def entry_id(self) -> str:
        """Generate unique entry ID."""
        return f"{self.storyline}_{self.timestamp.replace(':', '').replace('-', '').replace('.', '_')}"

    def to_dict(self) -> dict:
        return {
            "entry_id": self.entry_id,
            **asdict(self)
        }


class StorylineLogger:
    """Multi-storyline logger compatible with Agent Zero patterns.

    Maintains parallel narratives:
    - Main storyline tracks primary goal progress
    - Knowledge storyline captures learnings for future use
    - Technical storyline records commands, outputs, errors
    - Agent storyline tracks sub-agent dispatches and returns
    """

    def __init__(
        self,
        experience_dir: Path,
        session_id: Optional[str] = None
    ):
        """Initialize logger.

        Args:
            experience_dir: Directory to store logs
            session_id: Unique session identifier
        """
        self.experience_dir = Path(experience_dir)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = self.experience_dir / "storylines"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize storyline files
        self.storylines: dict[str, list[StorylineEntry]] = {
            st.value: [] for st in StorylineType
        }

        # Track active context
        self.current_task: Optional[str] = None
        self.active_agents: dict[str, str] = {}  # agent_name -> current_task

    def log(
        self,
        storyline: StorylineType,
        event_type: str,
        content: str,
        metadata: Optional[dict] = None,
        agent_name: Optional[str] = None,
        parent_entry: Optional[StorylineEntry] = None
    ) -> StorylineEntry:
        """Log an entry to a storyline.

        Args:
            storyline: Which storyline to log to
            event_type: Type of event (start, progress, complete, error, insight)
            content: Human-readable description
            metadata: Additional structured data
            agent_name: Which agent generated this entry
            parent_entry: Link to parent entry if hierarchical

        Returns:
            The created entry
        """
        entry = StorylineEntry(
            timestamp=datetime.now().isoformat(),
            storyline=storyline.value,
            event_type=event_type,
            content=content,
            metadata=metadata or {},
            parent_entry_id=parent_entry.entry_id if parent_entry else None,
            agent_name=agent_name
        )

        self.storylines[storyline.value].append(entry)
        self._write_entry(entry)

        return entry

    def log_main(self, event_type: str, content: str, **kwargs) -> StorylineEntry:
        """Log to main storyline."""
        return self.log(StorylineType.MAIN, event_type, content, **kwargs)

    def log_knowledge(
        self,
        insight: str,
        applies_to: list[str],
        confidence: float = 0.8,
        **kwargs
    ) -> StorylineEntry:
        """Log a learning/insight to knowledge storyline.

        Args:
            insight: What was learned
            applies_to: Categories this applies to
            confidence: How confident in this insight (0-1)
        """
        return self.log(
            StorylineType.KNOWLEDGE,
            "insight",
            insight,
            metadata={"applies_to": applies_to, "confidence": confidence},
            **kwargs
        )

    def log_technical(
        self,
        event_type: str,
        content: str,
        command: Optional[str] = None,
        output: Optional[str] = None,
        error: Optional[str] = None,
        **kwargs
    ) -> StorylineEntry:
        """Log technical details.

        Args:
            event_type: command, output, error, code
            content: Description
            command: The command run
            output: Command output
            error: Error message if any
        """
        metadata = {}
        if command:
            metadata["command"] = command
        if output:
            metadata["output"] = output[:1000]  # Truncate long outputs
        if error:
            metadata["error"] = error

        return self.log(
            StorylineType.TECHNICAL,
            event_type,
            content,
            metadata=metadata,
            **kwargs
        )

    def log_agent(
        self,
        agent_name: str,
        event_type: str,
        content: str,
        task: Optional[str] = None,
        result: Optional[str] = None,
        **kwargs
    ) -> StorylineEntry:
        """Log sub-agent activity.

        Args:
            agent_name: Name of the sub-agent
            event_type: dispatched, working, completed, error
            content: Description
            task: Task assigned to agent
            result: Result from agent
        """
        metadata = {"agent": agent_name}
        if task:
            metadata["task"] = task
            self.active_agents[agent_name] = task
        if result:
            metadata["result"] = result

        if event_type == "completed":
            self.active_agents.pop(agent_name, None)

        return self.log(
            StorylineType.AGENT,
            event_type,
            content,
            metadata=metadata,
            agent_name=agent_name,
            **kwargs
        )

    def log_perception(
        self,
        perception_type: str,
        content: str,
        screenshot_path: Optional[str] = None,
        detected_elements: Optional[list[dict]] = None,
        **kwargs
    ) -> StorylineEntry:
        """Log perception events (screenshots, UI detection).

        Args:
            perception_type: screenshot, ui_change, text_detected
            content: Description of what was perceived
            screenshot_path: Path to screenshot if applicable
            detected_elements: UI elements detected
        """
        metadata = {}
        if screenshot_path:
            metadata["screenshot"] = screenshot_path
        if detected_elements:
            metadata["elements"] = detected_elements

        return self.log(
            StorylineType.PERCEPTION,
            perception_type,
            content,
            metadata=metadata,
            **kwargs
        )

    def log_user(
        self,
        event_type: str,
        content: str,
        **kwargs
    ) -> StorylineEntry:
        """Log user interaction."""
        return self.log(StorylineType.USER, event_type, content, **kwargs)

    def _write_entry(self, entry: StorylineEntry):
        """Write entry to disk."""
        # Append to storyline file
        storyline_file = self.log_dir / f"{entry.storyline}.jsonl"
        with open(storyline_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

        # Also write to combined timeline
        timeline_file = self.log_dir / "timeline.jsonl"
        with open(timeline_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")

    def get_storyline(self, storyline: StorylineType) -> list[StorylineEntry]:
        """Get all entries from a storyline."""
        return self.storylines[storyline.value]

    def get_recent(self, count: int = 10) -> list[StorylineEntry]:
        """Get most recent entries across all storylines."""
        all_entries = []
        for entries in self.storylines.values():
            all_entries.extend(entries)
        all_entries.sort(key=lambda e: e.timestamp, reverse=True)
        return all_entries[:count]

    def get_knowledge_summary(self) -> list[dict]:
        """Get all knowledge entries as a summary."""
        return [
            {
                "insight": e.content,
                "applies_to": e.metadata.get("applies_to", []),
                "confidence": e.metadata.get("confidence", 0.5),
                "timestamp": e.timestamp
            }
            for e in self.storylines[StorylineType.KNOWLEDGE.value]
        ]

    def export_narrative(self) -> str:
        """Export storylines as human-readable narrative.

        Returns:
            Markdown-formatted narrative of the session
        """
        output = f"# Session Narrative: {self.session_id}\n\n"

        for storyline_type in StorylineType:
            entries = self.storylines[storyline_type.value]
            if entries:
                output += f"## {storyline_type.value.title()} Storyline\n\n"
                for entry in entries:
                    output += f"- **{entry.timestamp}** [{entry.event_type}]: {entry.content}\n"
                    if entry.metadata:
                        for k, v in entry.metadata.items():
                            if isinstance(v, str) and len(v) > 100:
                                v = v[:100] + "..."
                            output += f"  - {k}: {v}\n"
                output += "\n"

        return output

    def save_narrative(self) -> Path:
        """Save narrative to file."""
        narrative = self.export_narrative()
        narrative_file = self.experience_dir / f"narrative_{self.session_id}.md"
        narrative_file.write_text(narrative)
        return narrative_file


if __name__ == "__main__":
    # Demo usage
    from pathlib import Path

    logger = StorylineLogger(
        experience_dir=Path("/tmp/test_storylines"),
        session_id="demo_001"
    )

    # Log some events
    main_entry = logger.log_main("start", "Beginning screenshot extraction task")

    logger.log_technical(
        "command",
        "Starting ffmpeg recording",
        command="ffmpeg -f v4l2 -i /dev/video4 ...",
    )

    logger.log_agent(
        "screenshot-extractor",
        "dispatched",
        "Dispatching screenshot extraction",
        task="Extract keyframes from recording"
    )

    logger.log_perception(
        "screenshot",
        "Captured Windows 11 desktop - blue wallpaper visible",
        screenshot_path="/tmp/frame_001.png"
    )

    logger.log_knowledge(
        "USB capture card works reliably at 1920x1080 20fps",
        applies_to=["video_capture", "usb_hdmi"],
        confidence=0.95
    )

    logger.log_agent(
        "screenshot-extractor",
        "completed",
        "Extracted 5 keyframes",
        result="5 frames extracted to keyframes/"
    )

    logger.log_main("complete", "Screenshot extraction task completed")

    # Save narrative
    narrative_path = logger.save_narrative()
    print(f"Narrative saved to: {narrative_path}")
    print("\n" + logger.export_narrative())
