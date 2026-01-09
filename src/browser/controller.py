"""Browser controller using Playwright for headless browser automation."""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

logger = logging.getLogger(__name__)


@dataclass
class BrowserConfig:
    """Configuration for browser controller."""
    headless: bool = True
    user_data_dir: Path | None = None
    viewport_width: int = 1280
    viewport_height: int = 720
    timeout_ms: int = 30000


class BrowserController:
    """
    Headless browser controller using Playwright.

    Provides methods for browser automation including navigation,
    DOM extraction, form filling, and element interaction.
    """

    def __init__(self, config: BrowserConfig | None = None):
        self.config = config or BrowserConfig()
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def start(self) -> None:
        """Start the browser."""
        if self._playwright is not None:
            logger.warning("Browser already started")
            return

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless,
        )

        context_options = {
            "viewport": {
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
        }

        if self.config.user_data_dir:
            context_options["storage_state"] = str(self.config.user_data_dir / "state.json")

        self._context = await self._browser.new_context(**context_options)
        self._page = await self._context.new_page()
        self._page.set_default_timeout(self.config.timeout_ms)

        logger.info("Browser started (headless=%s)", self.config.headless)

    async def stop(self) -> None:
        """Stop the browser and cleanup resources."""
        if self._context and self.config.user_data_dir:
            await self._context.storage_state(
                path=str(self.config.user_data_dir / "state.json")
            )

        if self._page:
            await self._page.close()
            self._page = None

        if self._context:
            await self._context.close()
            self._context = None

        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        logger.info("Browser stopped")

    @property
    def page(self) -> Page:
        """Get the current page, raising if not started."""
        if self._page is None:
            raise RuntimeError("Browser not started. Call start() first.")
        return self._page

    async def navigate(self, url: str, wait_until: str = "domcontentloaded") -> dict[str, Any]:
        """
        Navigate to a URL.

        Args:
            url: The URL to navigate to
            wait_until: When to consider navigation complete
                       ("load", "domcontentloaded", "networkidle")

        Returns:
            Navigation result with URL and title
        """
        response = await self.page.goto(url, wait_until=wait_until)
        title = await self.page.title()

        logger.info("Navigated to: %s (title: %s)", url, title)

        return {
            "url": self.page.url,
            "title": title,
            "status": response.status if response else None,
        }

    async def click(self, selector: str) -> dict[str, Any]:
        """
        Click an element.

        Args:
            selector: CSS selector or XPath for the element

        Returns:
            Click result
        """
        await self.page.click(selector)
        logger.info("Clicked: %s", selector)
        return {"selector": selector, "success": True}

    async def fill(self, selector: str, value: str) -> dict[str, Any]:
        """
        Fill a form field.

        Args:
            selector: CSS selector for the input field
            value: Value to fill

        Returns:
            Fill result
        """
        await self.page.fill(selector, value)
        logger.info("Filled %s with value", selector)
        return {"selector": selector, "success": True}

    async def select_option(self, selector: str, value: str) -> dict[str, Any]:
        """
        Select an option from a dropdown.

        Args:
            selector: CSS selector for the select element
            value: Value to select

        Returns:
            Select result
        """
        await self.page.select_option(selector, value)
        logger.info("Selected %s in %s", value, selector)
        return {"selector": selector, "value": value, "success": True}

    async def type_text(self, selector: str, text: str, delay: int = 50) -> dict[str, Any]:
        """
        Type text into an element with realistic delays.

        Args:
            selector: CSS selector for the input
            text: Text to type
            delay: Delay between keystrokes in ms

        Returns:
            Type result
        """
        await self.page.type(selector, text, delay=delay)
        logger.info("Typed text into %s", selector)
        return {"selector": selector, "success": True}

    async def screenshot(self, path: str | None = None, full_page: bool = False) -> bytes:
        """
        Take a screenshot.

        Args:
            path: Optional path to save the screenshot
            full_page: Whether to capture full scrollable page

        Returns:
            Screenshot bytes
        """
        screenshot_bytes = await self.page.screenshot(path=path, full_page=full_page)
        logger.info("Screenshot taken (full_page=%s)", full_page)
        return screenshot_bytes

    async def extract_text(self, selector: str) -> str:
        """
        Extract text content from an element.

        Args:
            selector: CSS selector for the element

        Returns:
            Text content
        """
        element = await self.page.query_selector(selector)
        if element:
            return await element.text_content() or ""
        return ""

    async def extract_attribute(self, selector: str, attribute: str) -> str | None:
        """
        Extract an attribute from an element.

        Args:
            selector: CSS selector for the element
            attribute: Attribute name

        Returns:
            Attribute value or None
        """
        element = await self.page.query_selector(selector)
        if element:
            return await element.get_attribute(attribute)
        return None

    async def get_page_content(self) -> dict[str, Any]:
        """
        Get page content including URL, title, and HTML.

        Returns:
            Page content dictionary
        """
        return {
            "url": self.page.url,
            "title": await self.page.title(),
            "html": await self.page.content(),
        }

    async def wait_for_selector(self, selector: str, timeout: int | None = None) -> bool:
        """
        Wait for an element to appear.

        Args:
            selector: CSS selector to wait for
            timeout: Optional timeout in ms

        Returns:
            True if element found
        """
        try:
            await self.page.wait_for_selector(selector, timeout=timeout)
            return True
        except Exception:
            return False

    async def wait_for_navigation(self, timeout: int | None = None) -> None:
        """Wait for navigation to complete."""
        await self.page.wait_for_load_state("domcontentloaded", timeout=timeout)

    async def evaluate(self, script: str) -> Any:
        """
        Evaluate JavaScript in the page context.

        Args:
            script: JavaScript to evaluate

        Returns:
            Result of evaluation
        """
        return await self.page.evaluate(script)

    async def is_element_visible(self, selector: str) -> bool:
        """Check if an element is visible."""
        element = await self.page.query_selector(selector)
        if element:
            return await element.is_visible()
        return False

    async def get_element_count(self, selector: str) -> int:
        """Count elements matching selector."""
        elements = await self.page.query_selector_all(selector)
        return len(elements)
