#!/usr/bin/env python3
"""
UI Testing Runner - Simple CLI for the UI testing agent.

Usage:
    cd ~/Programs/UI-agent
    uv run python -m src.ui_tester.runner http://127.0.0.1:5050/session/xxx
"""

import argparse
import sys
from pathlib import Path

from .agent import UITestingAgent, format_report


def main():
    parser = argparse.ArgumentParser(description="Test a web UI with Claude")
    parser.add_argument("url", help="URL to test")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--instructions", "-i", default="", help="Extra instructions")
    parser.add_argument("--output", "-o", default="logs/ui_report.md", help="Output file")

    args = parser.parse_args()

    print(f"Testing: {args.url}")
    print(f"Headless: {args.headless}")
    print("-" * 50)

    agent = UITestingAgent(headless=args.headless)
    report = agent.test_url(args.url, args.instructions)

    # Save report
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_report(report))

    print("\n" + "=" * 50)
    print(f"Report saved: {output}")
    print(f"Screenshots: {len(report.screenshots)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
