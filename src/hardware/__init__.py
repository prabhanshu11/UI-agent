"""Hardware integration module for UI-Agent.

Provides interfaces for USB devices, capture cards, and other hardware peripherals.
"""

from .usb_capture import USBCaptureCard
from .esp32_mouse import ESP32Mouse
from .hdmi_sensor import HDMISensor
from .cursor_detector import CursorDetection, detect_cursor_movement
from .cursor_locator import CursorLocator, CursorPosition

__all__ = [
    "USBCaptureCard",
    "ESP32Mouse",
    "HDMISensor",
    "CursorDetection",
    "CursorLocator",
    "CursorPosition",
    "detect_cursor_movement",
]
