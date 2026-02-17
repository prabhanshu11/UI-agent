"""Navigation types for declarative UI step sequences.

Defines the vocabulary for describing navigation steps, their outcomes,
and recovery hypotheses. Used by WindowsNavigator to execute and log
perceive→act→verify loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any


class StepAction(Enum):
    """What to do at a navigation step."""
    CLICK = "click"
    RIGHT_CLICK = "right_click"
    DOUBLE_CLICK = "double_click"
    WAIT = "wait"  # No action — just wait and verify


class StepOutcome(Enum):
    """Result of executing a navigation step."""
    SUCCESS = "success"
    ELEMENT_NOT_FOUND = "element_not_found"
    CLICK_NO_EFFECT = "click_no_effect"
    UNEXPECTED_SCREEN = "unexpected_screen"
    TIMEOUT = "timeout"
    YELLOW_ALERT = "yellow_alert"  # Recovered via hypothesis system


@dataclass
class NavigationStep:
    """Declarative description of a single UI navigation action.

    The navigator will:
    1. Capture screen (PERCEIVE)
    2. Find `target` via LLM vision (FIND)
    3. Perform `action` at the found element (ACT)
    4. Capture screen again and check `verify` (VERIFY)

    Args:
        name: Human-readable name, appears in span name & MAIN storyline
        target: LLM vision target description (e.g., "Windows Start button")
        action: What to do (CLICK, RIGHT_CLICK, DOUBLE_CLICK, WAIT)
        verify: Expected screen state after action (LLM query), optional
        timeout: Max seconds to wait for element / verification
        retries: Max attempts to find element before giving up
        pre_delay: Seconds to wait before starting this step
        post_delay: Seconds to wait after action before verification
        optional: If True, failure doesn't abort the sequence
    """
    name: str
    target: str
    action: StepAction = StepAction.CLICK
    verify: str = ""
    timeout: float = 5.0
    retries: int = 3
    pre_delay: float = 0.5
    post_delay: float = 1.0
    optional: bool = False


@dataclass
class StepResult:
    """Outcome of executing a single NavigationStep.

    Captures everything that happened during the step for logging
    and post-mortem analysis.
    """
    step: NavigationStep
    outcome: StepOutcome
    element_found: Optional[Any] = None  # UIElement from LLMVision
    position_clicked: Optional[tuple[int, int]] = None
    screenshot_before: Optional[str] = None
    screenshot_after: Optional[str] = None
    attempts: int = 0
    elapsed_s: float = 0.0
    hypotheses: Optional[list[dict]] = None  # If recovery was attempted
    error: Optional[str] = None


@dataclass
class Hypothesis:
    """A recovery hypothesis dispatched to a Claude Agent SDK instance.

    When a step fails verification, the navigator generates hypotheses
    about what went wrong. Each is dispatched as a parallel Claude CLI
    subprocess with a screenshot to analyze.
    """
    name: str               # e.g., "click_missed", "dialog_appeared"
    prompt_file: str        # Path to .md template in hypotheses/
    screenshot: str         # Path to screenshot to analyze
    result: Optional[str] = None
    confidence: float = 0.0
    recommended_action: Optional[str] = None
