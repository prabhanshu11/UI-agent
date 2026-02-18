"""
UI Tester - Selenium-based intelligent UI testing agent.

Uses Claude Agent SDK with custom Selenium tools to autonomously
browse, analyze, and report UI issues.
"""

from .browser import SeleniumBrowser
from .agent import UITestingAgent

__all__ = ["SeleniumBrowser", "UITestingAgent"]
