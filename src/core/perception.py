import time
import os
import threading
import logging
from typing import Optional
from datetime import datetime
import mss
from pynput import mouse, keyboard
from .types import UIState

logger = logging.getLogger(__name__)

class PerceptionSystem:
    """
    Handles sensory input for the agent.
    - Visual: Screenshots via mss
    - Activity: Input monitoring via pynput to determine 'active' vs 'inactive' states.
    """
    def __init__(self, screenshot_interval: float = 1.0, inactivity_period: float = 20.0, save_dir: str = "logs/screenshots"):
        self.screenshot_interval = screenshot_interval
        self.inactivity_period = inactivity_period
        self.save_dir = save_dir
        
        # State tracking
        self.last_activity_time = time.time()
        self.last_screenshot_time = 0.0
        self.is_active = True
        self.key_moment_flag = False
        
        # Setup storage
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize Input Listeners
        self.mouse_listener = mouse.Listener(
            on_move=self._on_input,
            on_click=self._on_input,
            on_scroll=self._on_input
        )
        self.keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_input
        )
        
        self.mouse_listener.start()
        self.keyboard_listener.start()
        
        logger.info("PerceptionSystem initialized with pynput hooks")

    def _on_input(self, *args):
        """Generic input handler so update activity timestamp."""
        self.last_activity_time = time.time()
        
        # If we were inactive and input happens, trigger "Wake Up" key moment
        if not self.is_active:
            self.is_active = True
            self.key_moment_flag = True
            logger.info("Activity resumed: Waking up")

    def _on_key_press(self, key):
        """Specific key handler to detect combinations."""
        self._on_input()
        
        # Example Key Moment: Alt+Tab (simplified check, might need robust combo handling)
        # For now, treat ANY key press as strictly activity. 
        # Advanced combo detection can be added here.

    def capture(self) -> UIState:
        """
        Determine if a snapshot is needed based on state, and capture if so.
        """
        now = time.time()
        time_since_last_shot = now - self.last_screenshot_time
        time_since_activity = now - self.last_activity_time
        
        # 1. Update Inactivity State
        if self.is_active and time_since_activity > self.inactivity_period:
            self.is_active = False
            logger.info("Inactivity detected: Pausing auto-capture")
        
        # 2. Decide if we capture
        should_capture = False
        
        if self.key_moment_flag:
            should_capture = True
            logger.info("Capturing: Key Moment triggered")
        elif self.is_active and time_since_last_shot >= self.screenshot_interval:
            should_capture = True
        
        # 3. Execute Capture
        screenshot_path = None
        if should_capture:
            screenshot_path = self._take_screenshot()
            self.last_screenshot_time = now
            self.key_moment_flag = False # Reset flag
            
        return UIState(
            timestamp=now,
            screenshot=screenshot_path, # Returns path for now, can return PIL Image later
            metadata={
                "is_active": self.is_active,
                "triggered_by": "key_moment" if self.key_moment_flag else "interval"
            }
        )

    def _take_screenshot(self) -> str:
        """Captures screen using mss and saves to file."""
        try:
            with mss.mss() as sct:
                # Capture primary monitor (monitor 1 usually, 0 is all combined)
                monitor = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                sct_img = sct.grab(monitor)
                
                # Setup filename
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"screenshot_{timestamp_str}.png"
                filepath = os.path.join(self.save_dir, filename)
                
                mss.tools.to_png(sct_img.rgb, sct_img.size, output=filepath)
                return filepath
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None

    def shutdown(self):
        """Stop listeners."""
        self.mouse_listener.stop()
        self.keyboard_listener.stop()
