"""Experience session logger — structured event log per session.

Each session is an "experience": a timestamped sequence of perception (what was
seen), action (what was done), and outcome (what happened). Future sessions can
query past experiences to make better decisions.

Delegates to StorylineLogger when one is provided, mapping hardware events
to appropriate storyline tracks:
    cursor_*   → PERCEPTION (cursor position is a perception of screen state)
    ble_*      → TECHNICAL  (BLE connection is a technical detail)
    screenshot → PERCEPTION
    click      → TECHNICAL
    session_*  → MAIN       (session lifecycle is main narrative)

When no StorylineLogger is provided, falls back to standalone JSONL logging
(original behavior, fully backwards-compatible).

Log format: JSON Lines (.jsonl), one event per line.
Location: logs/experiences/YYYY-MM-DD_HHMMSS.jsonl
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.storyline_logger import StorylineLogger

# Event → StorylineType mapping
_EVENT_STORYLINE_MAP = {
    # Perception events
    "cursor_detected": "perception",
    "cursor_moved": "perception",
    "cursor_lost": "perception",
    "screenshot": "perception",
    "lissajous_start": "perception",
    "lissajous_failed": "perception",
    # Technical events
    "click": "technical",
    "ble_reset": "technical",
    "ble_stale": "technical",
    "heartbeat_start": "technical",
    "heartbeat_stop": "technical",
    # Main narrative events
    "session_start": "main",
    "session_end": "main",
    "session_error": "main",
}


class ExperienceLogger:
    """Session-scoped event logger writing JSON Lines.

    Optionally delegates to a StorylineLogger for unified multi-track logging.

    Usage (standalone):
        logger = ExperienceLogger()
        logger.log("cursor_detected", position=(620, 88), details={"method": "lissajous"})

    Usage (with StorylineLogger):
        from src.agents.storyline_logger import StorylineLogger
        storyline = StorylineLogger(experience_dir=Path("logs"))
        logger = ExperienceLogger(storyline_logger=storyline)
        logger.log("cursor_detected", ...)  # goes to PERCEPTION track
    """

    def __init__(
        self,
        log_dir: str = "logs/experiences",
        storyline_logger: Optional["StorylineLogger"] = None,
    ):
        """Start a new experience session.

        Args:
            log_dir: Directory for session log files. Created if missing.
            storyline_logger: If provided, hardware events are delegated
                to the appropriate storyline track. The standalone JSONL
                file is still written for backwards compatibility.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.session_start = datetime.now()
        filename = self.session_start.strftime("%Y-%m-%d_%H%M%S.jsonl")
        self.log_path = self.log_dir / filename
        self._file = open(self.log_path, "a")
        self._storyline = storyline_logger

        # Write session header
        self._write_event("session_start", details={
            "pid": os.getpid(),
            "log_path": str(self.log_path),
        })

    def log(
        self,
        event: str,
        position: Optional[tuple[int, int]] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """Log an event to the session file (and storyline if configured).

        Args:
            event: Event type (cursor_detected, click, ble_reset, etc.).
            position: Optional (x, y) cursor position at time of event.
            details: Optional dict of event-specific metadata.
        """
        self._write_event(event, position=position, details=details)
        self._delegate_to_storyline(event, position, details)

    def _delegate_to_storyline(
        self,
        event: str,
        position: Optional[tuple[int, int]],
        details: Optional[dict[str, Any]],
    ):
        """Forward hardware event to StorylineLogger on the right track."""
        if self._storyline is None:
            return

        from src.agents.storyline_logger import StorylineType

        storyline_name = _EVENT_STORYLINE_MAP.get(event, "technical")
        storyline_type = StorylineType(storyline_name)

        metadata = dict(details) if details else {}
        if position is not None:
            metadata["position"] = {"x": position[0], "y": position[1]}

        content = event.replace("_", " ").capitalize()
        if position:
            content += f" at ({position[0]}, {position[1]})"

        self._storyline.log(
            storyline_type,
            event,
            content,
            metadata=metadata,
            agent_name="hardware",
        )

    def _write_event(
        self,
        event: str,
        position: Optional[tuple[int, int]] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        """Write a single JSON line to the log file."""
        record = {
            "timestamp": time.time(),
            "iso": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
        }
        if position is not None:
            record["position"] = {"x": position[0], "y": position[1]}
        if details:
            record["details"] = details
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self):
        """Finalize the session log."""
        self._write_event("session_end", details={
            "duration_s": round(time.time() - self.session_start.timestamp(), 1),
        })
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self._write_event("session_error", details={
                "error": str(exc_val),
                "type": exc_type.__name__,
            })
        self.close()
