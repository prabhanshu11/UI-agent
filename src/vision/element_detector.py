"""High-level element detection combining LLM vision with caching."""

import hashlib
import json
from pathlib import Path
from typing import Optional
from PIL import Image

from .llm_vision import LLMVision, UIElement, VisionResult


class ElementDetector:
    """Cached element detection with LLM vision fallback."""
    
    def __init__(
        self,
        llm_vision: LLMVision | None = None,
        cache_dir: str = ".ui_cache",
        use_cache: bool = True,
    ):
        self.llm_vision = llm_vision
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self._element_cache: dict[str, list[UIElement]] = {}
        
        if use_cache:
            self.cache_dir.mkdir(exist_ok=True)
    
    def _image_hash(self, image: Image.Image) -> str:
        """Generate hash of image for caching."""
        # Simple hash based on resized thumbnail
        thumb = image.resize((64, 64)).convert("L")
        return hashlib.md5(thumb.tobytes()).hexdigest()[:16]
    
    def detect_all(
        self,
        image: Image.Image | str | Path,
        force_refresh: bool = False,
    ) -> list[UIElement]:
        """Detect all UI elements in image.
        
        Args:
            image: Screenshot image
            force_refresh: Skip cache and re-analyze
            
        Returns:
            List of detected UI elements
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Check cache first
        img_hash = self._image_hash(image)
        if not force_refresh and img_hash in self._element_cache:
            return self._element_cache[img_hash]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{img_hash}.json"
        if not force_refresh and self.use_cache and cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                elements = [UIElement(**e) for e in data]
                self._element_cache[img_hash] = elements
                return elements
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Use LLM vision
        if self.llm_vision is None:
            raise RuntimeError("LLM vision not configured")
        
        result = self.llm_vision.analyze_screen(image)
        elements = result.elements
        
        # Cache results
        self._element_cache[img_hash] = elements
        if self.use_cache:
            cache_file.write_text(json.dumps([
                {
                    "name": e.name,
                    "element_type": e.element_type,
                    "x": e.x,
                    "y": e.y,
                    "width": e.width,
                    "height": e.height,
                    "confidence": e.confidence,
                    "description": e.description,
                }
                for e in elements
            ]))
        
        return elements
    
    def find(
        self,
        image: Image.Image | str | Path,
        target: str,
        use_cached_elements: bool = True,
    ) -> UIElement | None:
        """Find specific element by description.
        
        Args:
            image: Screenshot
            target: Element description
            use_cached_elements: Check cached elements first
            
        Returns:
            Matching element or None
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Try to find in cached elements first
        if use_cached_elements:
            img_hash = self._image_hash(image)
            if img_hash in self._element_cache:
                for elem in self._element_cache[img_hash]:
                    if target.lower() in elem.name.lower() or target.lower() in elem.description.lower():
                        return elem
        
        # Use LLM to find specific element
        if self.llm_vision:
            return self.llm_vision.find_element(image, target)
        
        return None
    
    def find_by_type(
        self,
        image: Image.Image | str | Path,
        element_type: str,
    ) -> list[UIElement]:
        """Find all elements of a specific type.
        
        Args:
            image: Screenshot
            element_type: Type to filter by (button, text_field, etc.)
            
        Returns:
            List of matching elements
        """
        elements = self.detect_all(image)
        return [e for e in elements if e.element_type == element_type]
    
    def clear_cache(self):
        """Clear all cached elements."""
        self._element_cache.clear()
        if self.use_cache:
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
