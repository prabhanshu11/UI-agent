You are analyzing a screenshot from an HDMI capture of a Windows 11 machine.

The agent attempted to click on: "{target}"
The click was at pixel coordinates: ({x}, {y})
Expected outcome: "{expected}"
Actual outcome: The screen did not change as expected.

Analyze the screenshot:
1. Is the cursor visible? If so, where exactly (pixel coordinates)?
2. Is the target element visible on screen?
3. If both are visible, what is the approximate pixel distance between cursor and target?
4. What likely went wrong? (cursor didn't reach target, target moved, wrong element clicked)

Respond with JSON:
{{"cursor_position": [x, y], "target_visible": true, "target_position": [x, y], "diagnosis": "The cursor appears to be 50px to the left of the target button", "recommended_action": "retry_click_with_offset", "confidence": 0.85}}

If cursor is not visible, set cursor_position to null.
If target is not visible, set target_visible to false and target_position to null.
