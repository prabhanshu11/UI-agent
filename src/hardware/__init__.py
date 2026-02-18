"""Hardware integration module for UI-Agent.

Provides interfaces for USB devices, capture cards, and other hardware peripherals.
"""

from .usb_capture import USBCaptureCard
from .esp32_mouse import ESP32Mouse
from .hdmi_sensor import HDMISensor
from .cursor_detector import (
    LissajousCurve,
    CurveMatch,
    MotionBlob,
    extract_motion_blobs,
    match_curve_segment,
)
from .cursor_locator import CursorLocator, CursorPosition
from .experience_logger import ExperienceLogger
from .cursor_tracker import CursorTracker, TrackedPosition
from .cursor_sample_collector import CursorSampleCollector, extract_gray_patch
from .cursor_recognizer import CursorRecognizer
from .daemon_client import DaemonClient, DaemonState

__all__ = [
    "USBCaptureCard",
    "ESP32Mouse",
    "HDMISensor",
    "LissajousCurve",
    "CurveMatch",
    "MotionBlob",
    "CursorLocator",
    "CursorPosition",
    "ExperienceLogger",
    "CursorTracker",
    "TrackedPosition",
    "CursorSampleCollector",
    "CursorRecognizer",
    "extract_motion_blobs",
    "extract_gray_patch",
    "match_curve_segment",
    "DaemonClient",
    "DaemonState",
]
