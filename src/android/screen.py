"""Android screen capture and analysis."""

import tempfile
from pathlib import Path
from PIL import Image

from .controller import AndroidController
from ..vision import LLMVision, ElementDetector, UIElement


class AndroidScreen:
    """Screen capture and LLM-based UI analysis for Android."""
    
    def __init__(
        self,
        controller: AndroidController,
        llm_vision: LLMVision | None = None,
    ):
        self.controller = controller
        self.llm_vision = llm_vision
        self.detector = ElementDetector(llm_vision) if llm_vision else None
        self._screen_size: tuple[int, int] | None = None
    
    @property
    def screen_size(self) -> tuple[int, int]:
        if self._screen_size is None:
            self._screen_size = self.controller.get_screen_size()
        return self._screen_size
    
    def capture(self) -> Image.Image:
        """Capture current screen as PIL Image."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            self.controller.screenshot(temp_path)
            return Image.open(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)
    
    def find_element(self, description: str) -> UIElement | None:
        """Find UI element by description using LLM vision.
        
        Args:
            description: What to find (e.g., "settings button", "camera preview")
            
        Returns:
            UIElement if found
        """
        if self.detector is None:
            raise RuntimeError("LLM vision not configured")
        
        image = self.capture()
        return self.detector.find(image, description)
    
    def tap_element(self, description: str) -> bool:
        """Find and tap an element.
        
        Args:
            description: Element to find and tap
            
        Returns:
            True if element found and tapped
        """
        element = self.find_element(description)
        if element:
            self.controller.tap(element.center_x, element.center_y)
            return True
        return False
    
    def get_all_elements(self) -> list[UIElement]:
        """Detect all UI elements on screen."""
        if self.detector is None:
            raise RuntimeError("LLM vision not configured")
        
        image = self.capture()
        return self.detector.detect_all(image)
    
    def wait_for_element(
        self,
        description: str,
        timeout_seconds: float = 10.0,
        poll_interval: float = 1.0,
    ) -> UIElement | None:
        """Wait for element to appear.
        
        Args:
            description: Element to wait for
            timeout_seconds: Max wait time
            poll_interval: Time between checks
            
        Returns:
            Element if found within timeout
        """
        import time
        start = time.time()
        
        while time.time() - start < timeout_seconds:
            element = self.find_element(description)
            if element:
                return element
            time.sleep(poll_interval)
        
        return None
    
    def tap_and_wait(
        self,
        tap_description: str,
        wait_for: str,
        timeout: float = 5.0,
    ) -> bool:
        """Tap element and wait for result.
        
        Args:
            tap_description: Element to tap
            wait_for: Element to wait for after tap
            timeout: Wait timeout
            
        Returns:
            True if tap succeeded and wait element appeared
        """
        if not self.tap_element(tap_description):
            return False
        
        import time
        time.sleep(0.5)  # Brief wait for UI to react
        
        return self.wait_for_element(wait_for, timeout) is not None
