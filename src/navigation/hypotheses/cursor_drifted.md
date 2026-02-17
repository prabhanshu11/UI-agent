You are analyzing a screenshot from an HDMI capture of a Windows 11 machine.

The agent believes the cursor should be near ({expected_x}, {expected_y}) but the last
action had no visible effect, suggesting the cursor has drifted from its estimated position.

This happens because we control the cursor via relative HID mouse reports through an ESP32
Bluetooth device. Accumulated rounding errors or missed reports cause position drift.

Analyze the screenshot:
1. Find the mouse cursor arrow anywhere on the screen
2. Report its exact pixel position
3. Compare to where we expected it ({expected_x}, {expected_y})
4. Estimate the drift magnitude and direction

The cursor is a standard Windows white arrow pointer, possibly with a shadow.
It may be anywhere on the 1920x1080 screen.

Respond with JSON:
{{"cursor_found": true, "cursor_position": [x, y], "expected_position": [{expected_x}, {expected_y}], "drift_pixels": 150, "drift_direction": "cursor is 150px below and 30px right of expected", "recommended_action": "recalibrate_position", "confidence": 0.8}}

If cursor is not visible (hidden, in a fullscreen app, off-screen), set cursor_found to false.
