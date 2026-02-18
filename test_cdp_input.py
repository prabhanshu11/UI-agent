#!/usr/bin/env python3
"""Test CDP input more thoroughly."""

import asyncio
import sys
sys.path.insert(0, "/home/prabhanshu/Programs/UI-agent")

from src.browser.cdp_input import CDPInputController

async def main():
    controller = CDPInputController()

    print("Connecting to Edge via CDP...")
    await controller.connect("http://localhost:9222")
    print("Connected!")

    # Navigate to example.com
    print("Navigating to example.com...")
    await controller.navigate("https://example.com")
    await controller.wait(1)

    # Test 1: Use keyboard to navigate - Tab to focus the link, then Enter
    print("\nTest 1: Keyboard navigation")
    print("Pressing Tab to focus the link...")
    await controller.press_key("Tab")
    await controller.wait(0.5)

    print("Pressing Enter to click the focused link...")
    await controller.press_key("Enter")
    await controller.wait(2)

    # Take screenshot to see result
    await controller.screenshot_to_file("/tmp/cdp-keyboard-nav.png")
    print("Screenshot saved to /tmp/cdp-keyboard-nav.png")

    # Navigate back to example.com for click test
    print("\nNavigating back to example.com...")
    await controller.navigate("https://example.com")
    await controller.wait(1)

    # Test 2: Click directly on the link using Playwright's page object
    # The CDPInputController has access to the page
    print("\nTest 2: Getting link position from DOM then clicking via CDP")

    # Use page.evaluate to get the link's bounding box
    page = controller.page
    link_box = await page.evaluate('''() => {
        const link = document.querySelector('a');
        if (link) {
            const rect = link.getBoundingClientRect();
            return {x: rect.x + rect.width/2, y: rect.y + rect.height/2};
        }
        return null;
    }''')

    if link_box:
        print(f"Link center at: ({link_box['x']}, {link_box['y']})")
        print("Clicking via CDP...")
        await controller.click(int(link_box['x']), int(link_box['y']))
        await controller.wait(2)
        await controller.screenshot_to_file("/tmp/cdp-dom-guided-click.png")
        print("Screenshot saved to /tmp/cdp-dom-guided-click.png")
    else:
        print("Could not find link")

    await controller.disconnect()
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())
