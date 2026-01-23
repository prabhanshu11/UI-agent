"""Star Trek Computer Agent System.

Agents communicate via markdown files:
- CONTROL.md: Central orchestration
- task.md: Input tasks for each agent
- output.md: Results from each agent

Agents:
- screenshot-extractor: Extracts frames from video
- keyframe-identifier: Identifies significant moments
- summarizer: Describes screenshot content
- storyline-logger: Multi-narrative logging (Agent Zero compatible)

Modes:
- doing: Active task execution
- meeting: Passive observation
- learning: High-detail capture with maximum logging
"""

from .orchestrator import StarTrekOrchestrator
from .screenshot_extractor import ScreenshotExtractor
from .summarizer import ScreenshotSummarizer
from .storyline_logger import StorylineLogger, StorylineType, StorylineEntry

__all__ = [
    "StarTrekOrchestrator",
    "ScreenshotExtractor",
    "ScreenshotSummarizer",
    "StorylineLogger",
    "StorylineType",
    "StorylineEntry",
]
