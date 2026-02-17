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
    "extract_motion_blobs",
    "match_curve_segment",
]
