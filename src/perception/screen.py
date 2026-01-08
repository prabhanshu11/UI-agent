import os
import logging
from datetime import datetime
import mss
import mss.tools

logger = logging.getLogger(__name__)

class ScreenCapturer:
    """
    Handles screen capturing using mss.
    """
    def __init__(self, save_dir: str = "logs/screenshots"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
    def capture(self, monitor_index: int = 1) -> str:
        """
        Captures the screen and saves it to a file.
        :param monitor_index: Index of the monitor to capture (1-based usually).
        :return: Path to the saved screenshot, or None if failed.
        """
        try:
            with mss.mss() as sct:
                # Handle monitor selection safely
                if monitor_index < len(sct.monitors):
                    monitor = sct.monitors[monitor_index]
                else:
                    logger.warning(f"Monitor index {monitor_index} out of range, defaulting to 0 (all)")
                    monitor = sct.monitors[0]
                
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
