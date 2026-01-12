"""LLM-based vision for UI element inference using OpenRouter API."""

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import httpx
from PIL import Image


@dataclass
class UIElement:
    """Represents a detected UI element."""
    name: str
    element_type: str  # button, text_field, image, icon, etc.
    x: int
    y: int
    width: int
    height: int
    confidence: float
    description: str = ""
    
    @property
    def center_x(self) -> int:
        return self.x + self.width // 2
    
    @property
    def center_y(self) -> int:
        return self.y + self.height // 2


@dataclass 
class VisionResult:
    """Result from LLM vision analysis."""
    elements: list[UIElement]
    screen_description: str
    raw_response: dict
    

class LLMVision:
    """Use LLM vision models via OpenRouter for UI understanding."""
    
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    
    # Models with vision capabilities
    VISION_MODELS = {
        "claude-sonnet": "anthropic/claude-sonnet-4",
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gemini-flash": "google/gemini-flash-1.5",
    }
    
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        site_url: str = "https://github.com/prabhanshu11/ui-agent",
        site_name: str = "UI-Agent",
    ):
        """Initialize LLM Vision.
        
        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            model: Model to use (key from VISION_MODELS or full model ID)
            site_url: Your site URL for OpenRouter rankings
            site_name: Your app name for OpenRouter rankings
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key required")
        
        self.model = self.VISION_MODELS.get(model, model)
        self.site_url = site_url
        self.site_name = site_name
        self.client = httpx.Client(timeout=60.0)
    
    def _encode_image(self, image: Image.Image | str | Path) -> str:
        """Encode image to base64 data URL."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)
        
        # Convert to RGB if necessary
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        # Resize if too large (most APIs have limits)
        max_dim = 2048
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"
    
    def analyze_screen(
        self,
        image: Image.Image | str | Path,
        task: str = "Identify all interactive UI elements",
    ) -> VisionResult:
        """Analyze screen image and identify UI elements.
        
        Args:
            image: Screenshot as PIL Image or path
            task: Specific task or question about the UI
            
        Returns:
            VisionResult with detected elements
        """
        image_url = self._encode_image(image)
        
        prompt = f"""Analyze this UI screenshot and {task}.

Return a JSON object with:
1. "screen_description": Brief description of what screen/app this is
2. "elements": Array of UI elements, each with:
   - "name": Element label or purpose
   - "type": One of: button, text_field, icon, image, text, toggle, slider, dropdown, tab, menu_item
   - "x": Approximate X coordinate (pixels from left)
   - "y": Approximate Y coordinate (pixels from top)  
   - "width": Approximate width in pixels
   - "height": Approximate height in pixels
   - "description": What this element does

Focus on interactive elements that can be clicked/tapped.
Respond ONLY with valid JSON, no markdown or explanation."""

        response = self.client.post(
            self.OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "max_tokens": 2000,
            },
        )
        response.raise_for_status()
        result = response.json()
        
        return self._parse_response(result)
    
    def find_element(
        self,
        image: Image.Image | str | Path,
        target: str,
    ) -> UIElement | None:
        """Find a specific UI element by description.
        
        Args:
            image: Screenshot
            target: Description of element to find (e.g., "settings button", "login field")
            
        Returns:
            UIElement if found, None otherwise
        """
        image_url = self._encode_image(image)
        
        prompt = f"""Find the UI element that matches: "{target}"

Return a JSON object with:
- "found": true/false
- "element": If found, object with name, type, x, y, width, height, description
- "confidence": 0.0 to 1.0 confidence score

Respond ONLY with valid JSON."""

        response = self.client.post(
            self.OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "max_tokens": 500,
            },
        )
        response.raise_for_status()
        result = response.json()
        
        try:
            content = result["choices"][0]["message"]["content"]
            # Clean up response
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            data = json.loads(content)
            if data.get("found") and data.get("element"):
                elem = data["element"]
                return UIElement(
                    name=elem.get("name", target),
                    element_type=elem.get("type", "unknown"),
                    x=elem.get("x", 0),
                    y=elem.get("y", 0),
                    width=elem.get("width", 50),
                    height=elem.get("height", 50),
                    confidence=data.get("confidence", 0.5),
                    description=elem.get("description", ""),
                )
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
        
        return None
    
    def _parse_response(self, result: dict) -> VisionResult:
        """Parse LLM response into VisionResult."""
        elements = []
        screen_description = ""
        
        try:
            content = result["choices"][0]["message"]["content"]
            # Clean up markdown code blocks if present
            content = content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            data = json.loads(content)
            screen_description = data.get("screen_description", "")
            
            for elem in data.get("elements", []):
                elements.append(UIElement(
                    name=elem.get("name", "unknown"),
                    element_type=elem.get("type", "unknown"),
                    x=elem.get("x", 0),
                    y=elem.get("y", 0),
                    width=elem.get("width", 50),
                    height=elem.get("height", 50),
                    confidence=elem.get("confidence", 0.8),
                    description=elem.get("description", ""),
                ))
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Failed to parse LLM response: {e}")
        
        return VisionResult(
            elements=elements,
            screen_description=screen_description,
            raw_response=result,
        )
    
    def close(self):
        """Close HTTP client."""
        self.client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
