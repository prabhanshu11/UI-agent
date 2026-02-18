"""
UI Testing Agent - Simplified version.

Provides clean Selenium tools for Claude to test web UIs autonomously.
Claude handles the intelligence; this provides the browser interface.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic

from .browser import SeleniumBrowser, BrowserConfig

logger = logging.getLogger(__name__)


@dataclass
class UITestReport:
    """Structured UI testing report."""
    url: str
    tested_at: str = field(default_factory=lambda: datetime.now().isoformat())
    issues: list[dict] = field(default_factory=list)
    screenshots: list[str] = field(default_factory=list)
    summary: str = ""
    raw_analysis: str = ""


class UITestingAgent:
    """
    Simple UI testing agent using Claude + Selenium.

    Provides browser tools to Claude and lets it intelligently
    test the UI with minimal constraints.
    """

    def __init__(self, headless: bool = False):
        self.browser = SeleniumBrowser(BrowserConfig(
            headless=headless,
            screenshot_dir=Path("logs/screenshots"),
        ))
        self.client = anthropic.Anthropic()
        self.screenshots: list[str] = []

    # Define tools for Claude
    TOOLS = [
        {
            "name": "navigate",
            "description": "Navigate browser to URL. Returns page title and URL.",
            "input_schema": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"]
            }
        },
        {
            "name": "screenshot",
            "description": "Take screenshot. Returns filepath. Use to see current page state.",
            "input_schema": {
                "type": "object",
                "properties": {"name": {"type": "string", "default": "screenshot"}},
            }
        },
        {
            "name": "get_page_structure",
            "description": "Get page structure: headings, links, buttons, inputs, tables.",
            "input_schema": {"type": "object", "properties": {}}
        },
        {
            "name": "scroll",
            "description": "Scroll page. direction: 'down', 'up', or 'top'. pixels: amount (default 500).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["down", "up", "top"]},
                    "pixels": {"type": "integer", "default": 500}
                },
                "required": ["direction"]
            }
        },
        {
            "name": "click",
            "description": "Click element by CSS selector or by visible text.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "text": {"type": "string"}
                }
            }
        },
        {
            "name": "get_text",
            "description": "Get text content of element by CSS selector.",
            "input_schema": {
                "type": "object",
                "properties": {"selector": {"type": "string"}},
                "required": ["selector"]
            }
        },
        {
            "name": "check_element",
            "description": "Check if element exists and is visible. Returns count and visibility.",
            "input_schema": {
                "type": "object",
                "properties": {"selector": {"type": "string"}},
                "required": ["selector"]
            }
        },
        {
            "name": "get_console_errors",
            "description": "Get JavaScript console errors from browser.",
            "input_schema": {"type": "object", "properties": {}}
        },
        {
            "name": "get_state",
            "description": "Get current browser state: URL, title, scroll position, page height.",
            "input_schema": {"type": "object", "properties": {}}
        },
    ]

    def _execute_tool(self, name: str, args: dict) -> str:
        """Execute a browser tool and return result as JSON string."""
        if name == "navigate":
            result = self.browser.navigate(args["url"])
        elif name == "screenshot":
            result = self.browser.screenshot(args.get("name", "screenshot"))
            if "filepath" in result:
                self.screenshots.append(result["filepath"])
            # Remove base64 to keep response small
            result = {k: v for k, v in result.items() if k != "base64"}
        elif name == "get_page_structure":
            result = self.browser.get_page_structure()
        elif name == "scroll":
            direction = args.get("direction", "down")
            pixels = args.get("pixels", 500)
            if direction == "top":
                result = self.browser.scroll_to_top()
            elif direction == "up":
                result = self.browser.scroll_down(-pixels)
            else:
                result = self.browser.scroll_down(pixels)
        elif name == "click":
            if "text" in args:
                result = self.browser.click_text(args["text"])
            elif "selector" in args:
                result = self.browser.click_element(args["selector"])
            else:
                result = {"error": "Need selector or text"}
        elif name == "get_text":
            result = self.browser.get_element_text(args["selector"])
        elif name == "check_element":
            count = self.browser.count_elements(args["selector"])
            visible = self.browser.check_element_visible(args["selector"])
            result = {**count, **visible}
        elif name == "get_console_errors":
            result = self.browser.get_console_errors()
        elif name == "get_state":
            result = self.browser.get_current_state()
        else:
            result = {"error": f"Unknown tool: {name}"}

        return json.dumps(result, indent=2)

    def test_url(self, url: str, instructions: str = "") -> UITestReport:
        """
        Test a URL with Claude as the intelligent tester.

        Args:
            url: URL to test
            instructions: Additional testing instructions

        Returns:
            UITestReport with findings
        """
        self.browser.start()
        self.screenshots = []

        default_instructions = f"""Test the web page at {url} thoroughly.

Your task:
1. Navigate to the URL
2. Take screenshots as you explore
3. Examine page structure
4. Scroll through entire page
5. Test interactive elements (buttons, expandable sections)
6. Check for console errors
7. Note any UI issues you find

Report issues in these categories:
- CRITICAL: Page broken, major features don't work
- HIGH: Significant usability problems
- MEDIUM: Noticeable issues affecting experience
- LOW: Minor polish items

Issue types: visual, functional, accessibility, usability, performance

{instructions}

After testing, provide a structured report with:
1. Summary of what you tested
2. List of issues found (categorized by severity)
3. Recommendations for fixes
"""

        messages = [{"role": "user", "content": default_instructions}]

        try:
            # Agentic loop - let Claude work through the testing
            max_turns = 25
            for turn in range(max_turns):
                response = self.client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    tools=self.TOOLS,
                    messages=messages,
                )

                # Process response
                assistant_content = []
                tool_results = []

                for block in response.content:
                    if block.type == "text":
                        print(f"\n[Claude]: {block.text[:200]}..." if len(block.text) > 200 else f"\n[Claude]: {block.text}")
                        assistant_content.append(block)
                    elif block.type == "tool_use":
                        print(f"\n[Tool]: {block.name}({json.dumps(block.input)})")
                        assistant_content.append(block)

                        result = self._execute_tool(block.name, block.input)
                        print(f"[Result]: {result[:100]}..." if len(result) > 100 else f"[Result]: {result}")

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })

                # Add assistant message
                messages.append({"role": "assistant", "content": assistant_content})

                # If there were tool uses, add results and continue
                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                else:
                    # No more tool calls - Claude is done
                    break

                if response.stop_reason == "end_turn" and not tool_results:
                    break

            # Extract final analysis from last assistant message
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text

            return UITestReport(
                url=url,
                screenshots=self.screenshots,
                summary=final_text,
                raw_analysis=final_text,
            )

        finally:
            self.browser.stop()


def format_report(report: UITestReport) -> str:
    """Format report as markdown."""
    return f"""# UI Test Report

**URL:** {report.url}
**Tested:** {report.tested_at}
**Screenshots:** {len(report.screenshots)}

---

## Analysis

{report.summary}

---

## Screenshots

{chr(10).join(f'- `{s}`' for s in report.screenshots)}
"""
