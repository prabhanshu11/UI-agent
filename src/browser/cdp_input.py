"""
CDP-based input controller for VDI automation.

Connects to an existing browser via Chrome DevTools Protocol and sends
input events directly to the browser - no system mouse hijacking.

Usage:
    # Launch Edge with CDP:
    # microsoft-edge-stable --remote-debugging-port=9222

    controller = CDPInputController()
    await controller.connect("http://localhost:9222")
    await controller.click(500, 300)
    await controller.type_text("hello")
    screenshot = await controller.screenshot()
"""

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import Any

from playwright.async_api import async_playwright, Browser, CDPSession, Page

logger = logging.getLogger(__name__)


@dataclass
class CDPConfig:
    """Configuration for CDP input controller."""
    cdp_endpoint: str = "http://localhost:9222"
    click_delay_ms: int = 50  # Delay between mouse down/up
    type_delay_ms: int = 30   # Delay between keystrokes
    default_timeout_ms: int = 30000


class CDPInputController:
    """
    Send input directly to browser via CDP.

    This controller connects to an existing browser (Edge, Chrome) that was
    launched with --remote-debugging-port and sends mouse/keyboard events
    via the CDP Input domain.

    Key benefit: Input goes directly to the browser tab, NOT the system mouse.
    Your real mouse cursor stays where it is.
    """

    def __init__(self, config: CDPConfig | None = None):
        self.config = config or CDPConfig()
        self._playwright = None
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._cdp: CDPSession | None = None
        self._connected = False

    async def connect(self, cdp_endpoint: str | None = None) -> None:
        """
        Connect to an existing browser via CDP.

        Args:
            cdp_endpoint: CDP endpoint URL (e.g., "http://localhost:9222")
        """
        endpoint = cdp_endpoint or self.config.cdp_endpoint

        if self._connected:
            logger.warning("Already connected, disconnecting first")
            await self.disconnect()

        self._playwright = await async_playwright().start()

        try:
            self._browser = await self._playwright.chromium.connect_over_cdp(endpoint)
            contexts = self._browser.contexts
            if contexts:
                pages = contexts[0].pages
                if pages:
                    self._page = pages[0]
                else:
                    self._page = await contexts[0].new_page()
            else:
                context = await self._browser.new_context()
                self._page = await context.new_page()

            # Create CDP session for low-level input
            self._cdp = await self._page.context.new_cdp_session(self._page)
            self._connected = True

            logger.info("Connected to browser via CDP at %s", endpoint)

        except Exception as e:
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
            raise ConnectionError(f"Failed to connect to CDP: {e}") from e

    async def disconnect(self) -> None:
        """Disconnect from the browser."""
        if self._cdp:
            await self._cdp.detach()
            self._cdp = None

        # Don't close browser/pages - we connected to existing browser
        self._page = None
        self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        self._connected = False
        logger.info("Disconnected from browser")

    def _ensure_connected(self) -> None:
        """Raise if not connected."""
        if not self._connected or not self._cdp:
            raise RuntimeError("Not connected. Call connect() first.")

    @property
    def page(self) -> Page:
        """Get the current page."""
        self._ensure_connected()
        return self._page

    async def click(self, x: int, y: int, button: str = "left") -> None:
        """
        Click at pixel coordinates.

        Args:
            x: X coordinate (pixels from left of viewport)
            y: Y coordinate (pixels from top of viewport)
            button: "left", "right", or "middle"
        """
        self._ensure_connected()

        button_map = {"left": "left", "right": "right", "middle": "middle"}
        btn = button_map.get(button, "left")

        # Mouse move
        await self._cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseMoved",
            "x": x,
            "y": y,
        })

        # Mouse down
        await self._cdp.send("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": x,
            "y": y,
            "button": btn,
            "clickCount": 1,
        })

        await asyncio.sleep(self.config.click_delay_ms / 1000)

        # Mouse up
        await self._cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": x,
            "y": y,
            "button": btn,
            "clickCount": 1,
        })

        logger.debug("Clicked at (%d, %d) with %s button", x, y, button)

    async def double_click(self, x: int, y: int) -> None:
        """Double-click at coordinates."""
        self._ensure_connected()

        await self._cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseMoved",
            "x": x,
            "y": y,
        })

        for click_count in [1, 2]:
            await self._cdp.send("Input.dispatchMouseEvent", {
                "type": "mousePressed",
                "x": x,
                "y": y,
                "button": "left",
                "clickCount": click_count,
            })
            await asyncio.sleep(self.config.click_delay_ms / 1000)
            await self._cdp.send("Input.dispatchMouseEvent", {
                "type": "mouseReleased",
                "x": x,
                "y": y,
                "button": "left",
                "clickCount": click_count,
            })
            if click_count == 1:
                await asyncio.sleep(0.05)  # Brief pause between clicks

        logger.debug("Double-clicked at (%d, %d)", x, y)

    async def right_click(self, x: int, y: int) -> None:
        """Right-click at coordinates."""
        await self.click(x, y, button="right")

    async def move_mouse(self, x: int, y: int) -> None:
        """Move mouse to coordinates without clicking."""
        self._ensure_connected()

        await self._cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseMoved",
            "x": x,
            "y": y,
        })
        logger.debug("Moved mouse to (%d, %d)", x, y)

    async def drag(self, start_x: int, start_y: int, end_x: int, end_y: int,
                   steps: int = 10) -> None:
        """
        Drag from start to end coordinates.

        Args:
            start_x, start_y: Starting position
            end_x, end_y: Ending position
            steps: Number of intermediate steps for smooth drag
        """
        self._ensure_connected()

        # Move to start
        await self._cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseMoved",
            "x": start_x,
            "y": start_y,
        })

        # Press
        await self._cdp.send("Input.dispatchMouseEvent", {
            "type": "mousePressed",
            "x": start_x,
            "y": start_y,
            "button": "left",
            "clickCount": 1,
        })

        # Move in steps
        for i in range(1, steps + 1):
            progress = i / steps
            x = int(start_x + (end_x - start_x) * progress)
            y = int(start_y + (end_y - start_y) * progress)
            await self._cdp.send("Input.dispatchMouseEvent", {
                "type": "mouseMoved",
                "x": x,
                "y": y,
            })
            await asyncio.sleep(0.01)

        # Release
        await self._cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseReleased",
            "x": end_x,
            "y": end_y,
            "button": "left",
            "clickCount": 1,
        })

        logger.debug("Dragged from (%d, %d) to (%d, %d)", start_x, start_y, end_x, end_y)

    async def scroll(self, x: int, y: int, delta_x: int = 0, delta_y: int = -100) -> None:
        """
        Scroll at coordinates.

        Args:
            x, y: Position to scroll at
            delta_x: Horizontal scroll (positive = right)
            delta_y: Vertical scroll (negative = down, positive = up)
        """
        self._ensure_connected()

        await self._cdp.send("Input.dispatchMouseEvent", {
            "type": "mouseWheel",
            "x": x,
            "y": y,
            "deltaX": delta_x,
            "deltaY": delta_y,
        })
        logger.debug("Scrolled at (%d, %d) by (%d, %d)", x, y, delta_x, delta_y)

    async def type_text(self, text: str) -> None:
        """
        Type text character by character.

        Args:
            text: Text to type
        """
        self._ensure_connected()

        for char in text:
            await self._cdp.send("Input.dispatchKeyEvent", {
                "type": "char",
                "text": char,
            })
            await asyncio.sleep(self.config.type_delay_ms / 1000)

        logger.debug("Typed %d characters", len(text))

    async def press_key(self, key: str, modifiers: list[str] | None = None) -> None:
        """
        Press a key.

        Args:
            key: Key name (e.g., "Enter", "Tab", "Escape", "Backspace", "a", "A")
            modifiers: List of modifiers ["Ctrl", "Shift", "Alt", "Meta"]
        """
        self._ensure_connected()

        # Key name to code mapping (common keys)
        key_definitions = {
            "Enter": {"key": "Enter", "code": "Enter", "keyCode": 13},
            "Tab": {"key": "Tab", "code": "Tab", "keyCode": 9},
            "Escape": {"key": "Escape", "code": "Escape", "keyCode": 27},
            "Backspace": {"key": "Backspace", "code": "Backspace", "keyCode": 8},
            "Delete": {"key": "Delete", "code": "Delete", "keyCode": 46},
            "ArrowUp": {"key": "ArrowUp", "code": "ArrowUp", "keyCode": 38},
            "ArrowDown": {"key": "ArrowDown", "code": "ArrowDown", "keyCode": 40},
            "ArrowLeft": {"key": "ArrowLeft", "code": "ArrowLeft", "keyCode": 37},
            "ArrowRight": {"key": "ArrowRight", "code": "ArrowRight", "keyCode": 39},
            "Home": {"key": "Home", "code": "Home", "keyCode": 36},
            "End": {"key": "End", "code": "End", "keyCode": 35},
            "PageUp": {"key": "PageUp", "code": "PageUp", "keyCode": 33},
            "PageDown": {"key": "PageDown", "code": "PageDown", "keyCode": 34},
            "Space": {"key": " ", "code": "Space", "keyCode": 32},
            "F1": {"key": "F1", "code": "F1", "keyCode": 112},
            "F2": {"key": "F2", "code": "F2", "keyCode": 113},
            "F5": {"key": "F5", "code": "F5", "keyCode": 116},
            "F11": {"key": "F11", "code": "F11", "keyCode": 122},
        }

        # Modifier flags
        modifier_flags = 0
        if modifiers:
            if "Alt" in modifiers:
                modifier_flags |= 1
            if "Ctrl" in modifiers:
                modifier_flags |= 2
            if "Meta" in modifiers:
                modifier_flags |= 4
            if "Shift" in modifiers:
                modifier_flags |= 8

        key_def = key_definitions.get(key)
        if key_def:
            event = {
                "type": "keyDown",
                "key": key_def["key"],
                "code": key_def["code"],
                "windowsVirtualKeyCode": key_def["keyCode"],
                "modifiers": modifier_flags,
            }
        else:
            # Single character key
            event = {
                "type": "keyDown",
                "key": key,
                "text": key,
                "modifiers": modifier_flags,
            }

        await self._cdp.send("Input.dispatchKeyEvent", event)
        await asyncio.sleep(self.config.click_delay_ms / 1000)

        event["type"] = "keyUp"
        await self._cdp.send("Input.dispatchKeyEvent", event)

        logger.debug("Pressed key: %s (modifiers: %s)", key, modifiers)

    async def key_combination(self, *keys: str) -> None:
        """
        Press a key combination (e.g., Ctrl+A, Ctrl+Shift+V).

        Args:
            keys: Keys to press together (e.g., "Ctrl", "a")
        """
        modifiers = []
        main_key = None

        for key in keys:
            if key in ("Ctrl", "Shift", "Alt", "Meta"):
                modifiers.append(key)
            else:
                main_key = key

        if main_key:
            await self.press_key(main_key, modifiers)
        logger.debug("Key combination: %s", "+".join(keys))

    async def screenshot(self, format: str = "png", quality: int = 80) -> bytes:
        """
        Take a screenshot of the viewport.

        Args:
            format: "png" or "jpeg"
            quality: JPEG quality (1-100), ignored for PNG

        Returns:
            Screenshot image bytes
        """
        self._ensure_connected()

        params = {"format": format}
        if format == "jpeg":
            params["quality"] = quality

        result = await self._cdp.send("Page.captureScreenshot", params)
        image_data = base64.b64decode(result["data"])

        logger.debug("Screenshot taken (%s, %d bytes)", format, len(image_data))
        return image_data

    async def screenshot_to_file(self, path: str, format: str = "png") -> str:
        """
        Take a screenshot and save to file.

        Args:
            path: File path to save to
            format: "png" or "jpeg"

        Returns:
            Path to saved file
        """
        image_data = await self.screenshot(format=format)
        with open(path, "wb") as f:
            f.write(image_data)
        logger.info("Screenshot saved to %s", path)
        return path

    async def navigate(self, url: str) -> dict[str, Any]:
        """
        Navigate to a URL.

        Args:
            url: URL to navigate to

        Returns:
            Navigation result
        """
        self._ensure_connected()

        result = await self._cdp.send("Page.navigate", {"url": url})
        # Wait for page load
        await asyncio.sleep(0.5)

        logger.info("Navigated to %s", url)
        return result

    async def get_viewport_size(self) -> tuple[int, int]:
        """Get the current viewport size."""
        self._ensure_connected()

        result = await self._cdp.send("Page.getLayoutMetrics")
        viewport = result.get("cssLayoutViewport", {})
        return (
            int(viewport.get("clientWidth", 0)),
            int(viewport.get("clientHeight", 0))
        )

    async def wait(self, seconds: float) -> None:
        """Wait for specified seconds."""
        await asyncio.sleep(seconds)


# Convenience functions for quick usage
async def quick_connect(port: int = 9222) -> CDPInputController:
    """Quick connect to browser on localhost."""
    controller = CDPInputController()
    await controller.connect(f"http://localhost:{port}")
    return controller
