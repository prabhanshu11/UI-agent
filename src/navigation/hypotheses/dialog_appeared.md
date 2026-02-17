You are analyzing a screenshot from an HDMI capture of a Windows 11 machine.

The agent was navigating to: "{target}"
Expected screen state: "{expected}"
Instead, the screen shows something unexpected.

Look for:
1. Modal dialogs (OK/Cancel, Yes/No, confirmation prompts)
2. UAC (User Account Control) elevation prompts
3. Error message boxes
4. Windows Security prompts
5. Notification popups or toast messages
6. "Do you want to save" or similar interruption dialogs

For each dialog found, identify:
- The dialog title and message text
- Available buttons and their labels
- Whether it's blocking the target action

Respond with JSON:
{{"dialog_found": true, "dialog_type": "confirmation", "dialog_title": "Bluetooth Settings", "dialog_message": "Remove this device?", "buttons": ["Remove", "Cancel"], "recommended_action": "click_button:Remove", "blocks_target": true, "confidence": 0.9}}

If no dialog is visible, set dialog_found to false and describe what IS on screen.
