#!/usr/bin/env python3
"""Quick test of CDP input controller."""

import asyncio
import sys
sys.path.insert(0, "/home/prabhanshu/Programs/UI-agent")

from src.browser.cdp_input import CDPInputController

async def main():
    controller = CDPInputController()

    print("Connecting to Edge via CDP...")
    await controller.connect("http://localhost:9222")
    print("Connected!")

    # Get viewport size
    width, height = await controller.get_viewport_size()
    print(f"Viewport size: {width}x{height}")

    # Navigate to a test page
    print("Navigating to example.com...")
    await controller.navigate("https://example.com")
    await controller.wait(1)

    # Take screenshot
    print("Taking screenshot...")
    await controller.screenshot_to_file("/tmp/cdp-test-screenshot.png")
    print("Screenshot saved to /tmp/cdp-test-screenshot.png")

    # Disconnect (but don't close browser)
    await controller.disconnect()
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
