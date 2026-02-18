#!/usr/bin/env python3
"""Test CDP click - should NOT move system mouse."""

import asyncio
import sys
sys.path.insert(0, "/home/prabhanshu/Programs/UI-agent")

from src.browser.cdp_input import CDPInputController

async def main():
    controller = CDPInputController()

    print("Connecting to Edge via CDP...")
    await controller.connect("http://localhost:9222")
    print("Connected!")

    # Navigate to example.com first
    print("Navigating to example.com...")
    await controller.navigate("https://example.com")
    await controller.wait(1)

    # Take before screenshot
    await controller.screenshot_to_file("/tmp/cdp-before-click.png")
    print("Before screenshot saved")

    # The "More information..." link is approximately at:
    # Looking at the screenshot, it's around x=314, y=174 in the original image
    # (multiplied by 1.28 from displayed coords)
    # Let's click it
    print("Clicking 'More information...' link at (314, 174)...")
    print(">>> YOUR SYSTEM MOUSE SHOULD NOT MOVE <<<")
    await controller.click(314, 174)

    # Wait for navigation
    await controller.wait(2)

    # Take after screenshot
    await controller.screenshot_to_file("/tmp/cdp-after-click.png")
    print("After screenshot saved")

    await controller.disconnect()
    print("Done! Check /tmp/cdp-after-click.png to verify the click worked")

if __name__ == "__main__":
    asyncio.run(main())
