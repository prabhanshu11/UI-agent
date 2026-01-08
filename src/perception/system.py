import time
import logging
from typing import Optional
from ..core.types import UIState
from .screen import ScreenCapturer
from .input import InputMonitor

logger = logging.getLogger(__name__)

class PerceptionSystem:
    """
    Coordinator for sensory input. Refactored to use modular components.
    """
    def __init__(self, screenshot_interval: float = 1.0, inactivity_period: float = 20.0, save_dir: str = "logs/screenshots"):
        self.screenshot_interval = screenshot_interval
        self.inactivity_period = inactivity_period
        
        self.screen_capturer = ScreenCapturer(save_dir=save_dir)
        self.input_monitor = InputMonitor()
        
        self.last_screenshot_time = 0.0
        self.is_active = True
        
        # Start monitoring immediately
        self.input_monitor.start()

    def capture(self) -> UIState:
        """
        Determine if a snapshot is needed based on state, and capture if so.
        """
        now = time.time()
        time_since_last_shot = now - self.last_screenshot_time
        time_since_activity = self.input_monitor.get_time_since_activity()
        
        # 1. Update Inactivity State
        if self.is_active and time_since_activity > self.inactivity_period:
            self.is_active = False
            logger.info("Inactivity detected: Pausing auto-capture")
        elif not self.is_active and time_since_activity < 1.0: # Recent activity
            self.is_active = True
            logger.info("Activity resumed: Waking up")

        # 2. Decide if we capture
        should_capture = False
        is_key_moment = self.input_monitor.check_and_reset_key_moment()
        
        if is_key_moment:
            should_capture = True
            logger.info("Capturing: Key Moment triggered")
        elif self.is_active and time_since_last_shot >= self.screenshot_interval:
            should_capture = True
        
        # 3. Execute Capture
        screenshot_path = None
        if should_capture:
            screenshot_path = self.screen_capturer.capture()
            self.last_screenshot_time = now
            
        return UIState(
            timestamp=now,
            screenshot=screenshot_path,
            metadata={
                "is_active": self.is_active,
                "triggered_by": "key_moment" if is_key_moment else "interval"
            }
        )

    def shutdown(self):
        """Stop listeners."""
        self.input_monitor.stop()
