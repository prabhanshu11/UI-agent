"""
Selenium-based browser controller for UI testing.

Provides browser automation methods that can be exposed as Claude Agent SDK tools.
"""

import base64
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import shutil

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
)

logger = logging.getLogger(__name__)


@dataclass
class BrowserConfig:
    """Configuration for Selenium browser."""
    headless: bool = False  # Visible for testing
    window_width: int = 1400
    window_height: int = 900
    timeout: int = 10
    screenshot_dir: Path = Path("logs/screenshots")


class SeleniumBrowser:
    """
    Selenium browser controller with methods designed for AI agent use.

    Each method returns structured data suitable for Claude to understand
    the result of actions taken.
    """

    def __init__(self, config: BrowserConfig | None = None):
        self.config = config or BrowserConfig()
        self.driver: webdriver.Chrome | None = None
        self.config.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._screenshot_counter = 0

    def start(self) -> dict[str, Any]:
        """Start the browser."""
        if self.driver:
            return {"status": "already_running", "message": "Browser already started"}

        options = Options()
        if self.config.headless:
            options.add_argument("--headless=new")
        options.add_argument(f"--window-size={self.config.window_width},{self.config.window_height}")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        # Use system chromedriver if available, otherwise fall back to webdriver-manager
        chromedriver_path = shutil.which("chromedriver")
        if chromedriver_path:
            service = Service(chromedriver_path)
        else:
            from webdriver_manager.chrome import ChromeDriverManager
            service = Service(ChromeDriverManager().install())

        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.implicitly_wait(self.config.timeout)

        logger.info("Browser started (headless=%s)", self.config.headless)
        return {"status": "started", "window_size": f"{self.config.window_width}x{self.config.window_height}"}

    def stop(self) -> dict[str, Any]:
        """Stop the browser."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("Browser stopped")
            return {"status": "stopped"}
        return {"status": "not_running"}

    def navigate(self, url: str) -> dict[str, Any]:
        """
        Navigate to a URL.

        Returns current URL and page title after navigation.
        """
        if not self.driver:
            return {"error": "Browser not started"}

        self.driver.get(url)
        time.sleep(0.5)  # Brief wait for page load

        return {
            "url": self.driver.current_url,
            "title": self.driver.title,
            "status": "navigated"
        }

    def screenshot(self, name: str = "screenshot") -> dict[str, Any]:
        """
        Take a screenshot and return it as base64.

        Also saves to disk for debugging.
        """
        if not self.driver:
            return {"error": "Browser not started"}

        self._screenshot_counter += 1
        filename = f"{self._screenshot_counter:03d}_{name}.png"
        filepath = self.config.screenshot_dir / filename

        self.driver.save_screenshot(str(filepath))

        # Also get base64 for Claude to analyze
        screenshot_b64 = self.driver.get_screenshot_as_base64()

        logger.info("Screenshot saved: %s", filepath)
        return {
            "filepath": str(filepath),
            "base64": screenshot_b64,
            "width": self.driver.get_window_size()["width"],
            "height": self.driver.get_window_size()["height"],
        }

    def get_page_structure(self) -> dict[str, Any]:
        """
        Get the page structure - headings, links, buttons, forms.

        Returns a summary suitable for AI analysis.
        """
        if not self.driver:
            return {"error": "Browser not started"}

        structure = {
            "url": self.driver.current_url,
            "title": self.driver.title,
            "headings": [],
            "links": [],
            "buttons": [],
            "inputs": [],
            "tables": [],
            "text_blocks": [],
        }

        # Get headings
        for level in range(1, 5):
            headings = self.driver.find_elements(By.TAG_NAME, f"h{level}")
            for h in headings:
                text = h.text.strip()
                if text:
                    structure["headings"].append({"level": level, "text": text})

        # Get links (first 20)
        links = self.driver.find_elements(By.TAG_NAME, "a")[:20]
        for link in links:
            text = link.text.strip()
            href = link.get_attribute("href")
            if text and href:
                structure["links"].append({"text": text, "href": href})

        # Get buttons
        buttons = self.driver.find_elements(By.TAG_NAME, "button")
        buttons += self.driver.find_elements(By.CSS_SELECTOR, "input[type='button'], input[type='submit']")
        for btn in buttons[:10]:
            text = btn.text.strip() or btn.get_attribute("value") or ""
            if text:
                structure["buttons"].append({"text": text})

        # Get input fields
        inputs = self.driver.find_elements(By.TAG_NAME, "input")
        inputs += self.driver.find_elements(By.TAG_NAME, "textarea")
        for inp in inputs[:10]:
            inp_type = inp.get_attribute("type") or "text"
            placeholder = inp.get_attribute("placeholder") or ""
            name = inp.get_attribute("name") or ""
            structure["inputs"].append({
                "type": inp_type,
                "name": name,
                "placeholder": placeholder,
            })

        # Get tables info
        tables = self.driver.find_elements(By.TAG_NAME, "table")
        for i, table in enumerate(tables[:5]):
            rows = len(table.find_elements(By.TAG_NAME, "tr"))
            cols = len(table.find_elements(By.TAG_NAME, "th"))
            structure["tables"].append({"index": i, "rows": rows, "cols": cols})

        return structure

    def scroll_down(self, pixels: int = 500) -> dict[str, Any]:
        """Scroll down by specified pixels."""
        if not self.driver:
            return {"error": "Browser not started"}

        self.driver.execute_script(f"window.scrollBy(0, {pixels})")
        time.sleep(0.3)

        scroll_y = self.driver.execute_script("return window.pageYOffset")
        page_height = self.driver.execute_script("return document.body.scrollHeight")

        return {
            "scrolled": pixels,
            "current_scroll_y": scroll_y,
            "page_height": page_height,
        }

    def scroll_to_top(self) -> dict[str, Any]:
        """Scroll to top of page."""
        if not self.driver:
            return {"error": "Browser not started"}

        self.driver.execute_script("window.scrollTo(0, 0)")
        time.sleep(0.3)
        return {"scrolled_to": "top"}

    def click_element(self, selector: str, selector_type: str = "css") -> dict[str, Any]:
        """
        Click an element by CSS selector or XPath.

        Args:
            selector: CSS selector or XPath
            selector_type: "css" or "xpath"
        """
        if not self.driver:
            return {"error": "Browser not started"}

        by = By.CSS_SELECTOR if selector_type == "css" else By.XPATH

        try:
            element = WebDriverWait(self.driver, self.config.timeout).until(
                EC.element_to_be_clickable((by, selector))
            )
            element.click()
            time.sleep(0.3)
            return {"status": "clicked", "selector": selector}
        except TimeoutException:
            return {"error": f"Element not found or not clickable: {selector}"}
        except ElementClickInterceptedException:
            # Try JavaScript click
            try:
                element = self.driver.find_element(by, selector)
                self.driver.execute_script("arguments[0].click();", element)
                return {"status": "clicked_via_js", "selector": selector}
            except Exception as e:
                return {"error": f"Click failed: {str(e)}"}

    def click_text(self, text: str) -> dict[str, Any]:
        """Click an element containing specific text."""
        if not self.driver:
            return {"error": "Browser not started"}

        xpath = f"//*[contains(text(), '{text}')]"
        try:
            element = WebDriverWait(self.driver, self.config.timeout).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            element.click()
            time.sleep(0.3)
            return {"status": "clicked", "text": text}
        except TimeoutException:
            return {"error": f"Element with text '{text}' not found"}

    def get_element_text(self, selector: str, selector_type: str = "css") -> dict[str, Any]:
        """Get text content of an element."""
        if not self.driver:
            return {"error": "Browser not started"}

        by = By.CSS_SELECTOR if selector_type == "css" else By.XPATH

        try:
            element = self.driver.find_element(by, selector)
            return {"text": element.text, "selector": selector}
        except NoSuchElementException:
            return {"error": f"Element not found: {selector}"}

    def count_elements(self, selector: str, selector_type: str = "css") -> dict[str, Any]:
        """Count elements matching selector."""
        if not self.driver:
            return {"error": "Browser not started"}

        by = By.CSS_SELECTOR if selector_type == "css" else By.XPATH
        elements = self.driver.find_elements(by, selector)
        return {"count": len(elements), "selector": selector}

    def check_element_visible(self, selector: str, selector_type: str = "css") -> dict[str, Any]:
        """Check if an element is visible."""
        if not self.driver:
            return {"error": "Browser not started"}

        by = By.CSS_SELECTOR if selector_type == "css" else By.XPATH

        try:
            element = self.driver.find_element(by, selector)
            visible = element.is_displayed()
            return {"visible": visible, "selector": selector}
        except NoSuchElementException:
            return {"visible": False, "error": f"Element not found: {selector}"}

    def get_console_errors(self) -> dict[str, Any]:
        """Get browser console errors (Chrome only)."""
        if not self.driver:
            return {"error": "Browser not started"}

        logs = self.driver.get_log("browser")
        errors = [log for log in logs if log["level"] in ("SEVERE", "ERROR")]
        warnings = [log for log in logs if log["level"] == "WARNING"]

        return {
            "errors": errors,
            "warnings": warnings,
            "total_log_entries": len(logs),
        }

    def get_current_state(self) -> dict[str, Any]:
        """Get current browser state - URL, title, scroll position."""
        if not self.driver:
            return {"error": "Browser not started"}

        return {
            "url": self.driver.current_url,
            "title": self.driver.title,
            "scroll_y": self.driver.execute_script("return window.pageYOffset"),
            "page_height": self.driver.execute_script("return document.body.scrollHeight"),
            "viewport_height": self.driver.execute_script("return window.innerHeight"),
        }
