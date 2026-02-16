"""Hardware integration module for UI-Agent.

Provides interfaces for USB devices, capture cards, and other hardware peripherals.
"""

from .usb_capture import USBCaptureCard
from .esp32_mouse import ESP32Mouse
from .pi_keyboard import PiKeyboardClient

__all__ = ["USBCaptureCard", "ESP32Mouse", "PiKeyboardClient"]
