"""Multi-Storyline Logger - Agent Zero Compatible + OpenTelemetry.

Tracks multiple parallel narratives of agent activity:
- Main storyline: Primary task flow
- Sub-storylines: Parallel agent activities
- Knowledge storyline: Learnings and insights
- Technical storyline: Code, commands, errors

OTel integration adds trace/span semantics on top — each "experience"
is an OTel trace, each "step" is a span. Spans nest within storylines,
and timeline.jsonl entries include trace_id/span_id for correlation.

Based on Agent Zero's hierarchical agent communication patterns.
https://github.com/agent0ai/agent-zero
"""

import json
import os
import socket
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from enum import Enum

from opentelemetry import trace, context
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import StatusCode


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


class _FileSpanExporter(SpanExporter):
    """Exports OTel spans as JSON lines to a file alongside storyline entries."""

    def __init__(self, log_dir: Path):
        self._log_dir = log_dir

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        spans_file = self._log_dir / "spans.jsonl"
        with open(spans_file, "a") as f:
            for span in spans:
                ctx = span.get_span_context()
                record = {
                    "type": "otel_span",
                    "trace_id": format(ctx.trace_id, "032x"),
                    "span_id": format(ctx.span_id, "016x"),
                    "parent_span_id": (
                        format(span.parent.span_id, "016x")
                        if span.parent else None
                    ),
                    "name": span.name,
                    "start_time": span.start_time,
                    "end_time": span.end_time,
                    "duration_ms": (
                        (span.end_time - span.start_time) / 1_000_000
                        if span.end_time and span.start_time else None
                    ),
                    "status": span.status.status_code.name,
                    "attributes": dict(span.attributes) if span.attributes else {},
                }
                f.write(json.dumps(record) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


class StorylineLogger:
    """Multi-storyline logger with OpenTelemetry trace/span integration.

    Maintains parallel narratives:
    - Main storyline tracks primary goal progress
    - Knowledge storyline captures learnings for future use
    - Technical storyline records commands, outputs, errors
    - Agent storyline tracks sub-agent dispatches and returns

    OTel integration:
    - start_experience() creates a root trace span
    - start_step() creates child spans within the experience
    - end_step() closes spans with status and attributes
    - All storyline entries include trace_id/span_id when a span is active
    """

    def __init__(
        self,
        experience_dir: Path,
        session_id: Optional[str] = None
    ):
        """Initialize logger with OTel tracing.

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

        # Initialize OpenTelemetry
        resource = Resource.create({
            "service.name": "ui-agent",
            "host.name": socket.gethostname(),
            "session.id": self.session_id,
        })
        self._tracer_provider = TracerProvider(resource=resource)
        self._tracer_provider.add_span_processor(
            SimpleSpanProcessor(_FileSpanExporter(self.log_dir))
        )
        self._tracer = self._tracer_provider.get_tracer("ui.experience")

        # Track current OTel context for automatic span parenting
        self._otel_context: Optional[context.Context] = None

    def start_experience(self, name: str, attributes: dict | None = None) -> trace.Span:
        """Start a new experience trace (root span).

        Args:
            name: Experience name (e.g., "pair_k380")
            attributes: Optional span attributes

        Returns:
            OTel Span — caller should store this and pass to end_step()
        """
        span = self._tracer.start_span(
            name=name,
            attributes=attributes or {},
        )
        # Store the context so child spans auto-parent
        self._otel_context = trace.set_span_in_context(span)

        # Also log to main storyline
        self.log_main("start", f"Starting experience: {name}",
                       agent_name="navigator")
        return span

    def start_step(
        self,
        name: str,
        parent: trace.Span | None = None,
        attributes: dict | None = None,
    ) -> trace.Span:
        """Start a step span within an experience.

        Args:
            name: Step name (e.g., "Open Start Menu")
            parent: Explicit parent span. If None, parents to current context.
            attributes: Optional span attributes

        Returns:
            OTel Span for this step
        """
        ctx = self._otel_context
        if parent is not None:
            ctx = trace.set_span_in_context(parent)

        span = self._tracer.start_span(
            name=name,
            context=ctx,
            attributes=attributes or {},
        )
        # Update context so nested calls auto-parent to this step
        self._otel_context = trace.set_span_in_context(span)
        return span

    def end_step(
        self,
        span: trace.Span,
        status: str = "OK",
        attributes: dict | None = None,
    ):
        """End a step span with status and final attributes.

        Args:
            span: The span to end
            status: "OK", "ERROR", or "UNSET"
            attributes: Final attributes to add before closing
        """
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)

        status_code = {
            "OK": StatusCode.OK,
            "ERROR": StatusCode.ERROR,
        }.get(status, StatusCode.UNSET)
        span.set_status(status_code)
        span.end()

        # Restore parent context if possible
        parent_ctx = span.parent
        if parent_ctx:
            self._otel_context = trace.set_span_in_context(
                trace.NonRecordingSpan(parent_ctx)
            )

    def shutdown(self):
        """Flush and shut down the OTel tracer provider."""
        self._tracer_provider.shutdown()

    # ── Existing storyline API (unchanged signatures) ─────────────

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

        # Inject OTel trace/span IDs into metadata when a span is active
        if self._otel_context is not None:
            current_span = trace.get_current_span(self._otel_context)
            span_ctx = current_span.get_span_context()
            if span_ctx.is_valid:
                entry.metadata["trace_id"] = format(span_ctx.trace_id, "032x")
                entry.metadata["span_id"] = format(span_ctx.span_id, "016x")

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
                    # Include span info if present
                    span_info = ""
                    if "trace_id" in entry.metadata:
                        span_info = f" [span:{entry.metadata['span_id'][:8]}]"
                    output += f"- **{entry.timestamp}** [{entry.event_type}]{span_info}: {entry.content}\n"
                    if entry.metadata:
                        for k, v in entry.metadata.items():
                            if k in ("trace_id", "span_id"):
                                continue  # Already shown in span_info
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

    # Demo OTel integration
    experience_span = logger.start_experience("demo_experience", {
        "experience.type": "test",
    })

    step1 = logger.start_step("Step 1: Capture Screen")
    logger.log_perception("screenshot", "Captured desktop screenshot",
                          screenshot_path="/tmp/frame_001.png")
    logger.end_step(step1, status="OK")

    step2 = logger.start_step("Step 2: Find Element")
    logger.log_perception("element_search", "Looking for Settings button")
    logger.end_step(step2, status="OK", attributes={"element.found": True})

    logger.end_step(experience_span, status="OK")
    logger.shutdown()

    # Save narrative
    narrative_path = logger.save_narrative()
    print(f"Narrative saved to: {narrative_path}")
    print("\n" + logger.export_narrative())
