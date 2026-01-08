import time
import threading
import logging
from pynput import mouse, keyboard

logger = logging.getLogger(__name__)

class InputMonitor:
    """
    Monitors mouse and keyboard input to detect activity levels.
    """
    def __init__(self):
        self.last_activity_time = time.time()
        self.key_moment_flag = False
        self._is_running = False
        
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

    def start(self):
        """Starts the input listeners."""
        if not self._is_running:
            self.mouse_listener.start()
            self.keyboard_listener.start()
            self._is_running = True
            logger.info("InputMonitor started")

    def stop(self):
        """Stops the input listeners."""
        if self._is_running:
            self.mouse_listener.stop()
            self.keyboard_listener.stop()
            self._is_running = False
            logger.info("InputMonitor stopped")

    def _on_input(self, *args):
        """Generic input handler to update activity timestamp."""
        self.last_activity_time = time.time()

    def _on_key_press(self, key):
        """Specific key handler."""
        self._on_input()
        # Future: Check for specific key combos here to set self.key_moment_flag

    def get_time_since_activity(self) -> float:
        """Returns seconds since last detected input."""
        return time.time() - self.last_activity_time

    def check_and_reset_key_moment(self) -> bool:
        """Checks if a key moment occurred and resets the flag."""
        if self.key_moment_flag:
            self.key_moment_flag = False
            return True
        return False
