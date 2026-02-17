"""Navigation module — declarative UI step sequences with recovery.

Provides WindowsNavigator for executing perceive→act→verify loops
on a Windows machine controlled via ESP32 BLE HID mouse + HDMI capture.
"""

from .types import (
    Hypothesis,
    NavigationStep,
    StepAction,
    StepOutcome,
    StepResult,
)
from .navigator import WindowsNavigator

__all__ = [
    "WindowsNavigator",
    "NavigationStep",
    "StepAction",
    "StepOutcome",
    "StepResult",
    "Hypothesis",
]
