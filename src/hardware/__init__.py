"""Hardware integration module for UI-Agent.

Provides interfaces for USB devices, capture cards, and other hardware peripherals.
"""

from .usb_capture import USBCaptureCard

__all__ = ["USBCaptureCard"]
