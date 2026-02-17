You are analyzing a screenshot from an HDMI capture of a Windows 11 machine.

The agent was trying to navigate to: "{target}"
Expected to see: "{expected}"
But the screen doesn't match what was expected.

Analyze what IS currently on screen:
1. What application or settings page is visible?
2. What is the window title?
3. Is this a different page in the same application, or a completely different app?
4. If this is Windows Settings, which section/page is currently showing?
5. How would you navigate from the current screen to the intended target?

Provide step-by-step instructions to recover:
- If it's the wrong Settings page, describe which sidebar items to click
- If it's a different app, describe how to get back (Alt+Tab, click taskbar, etc.)
- If a menu or dropdown is covering the target, describe how to dismiss it

Respond with JSON:
{{"current_screen": "Windows Settings > System > Display", "target_screen": "Windows Settings > Bluetooth & devices", "same_application": true, "recovery_steps": ["Click 'Bluetooth & devices' in the left sidebar"], "recommended_action": "navigate_sidebar", "confidence": 0.85}}
