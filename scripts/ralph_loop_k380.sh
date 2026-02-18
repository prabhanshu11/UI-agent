#!/bin/bash
# Ralph Loop command for K380 autonomous pairing.
# Run this in a Claude Code session after the code changes are ready.
# Usage: Copy the /ralph-loop command below and paste into Claude Code.

cat << 'RALPH_LOOP_COMMAND'
/ralph-loop "Pair K380 keyboard to Windows laptop autonomously.

## STEP 1 — USE THE READ TOOL FIRST
Use the Read tool (not cat, not Bash) to read:
/home/prabhanshu/Programs/UI-agent/docs/CURSOR_DETECTION_KNOWLEDGE.md
Then state: 'Knowledge base loaded. Key concepts: [summary]'

## STEP 2 — USE THE READ TOOL SECOND
Use the Read tool (not cat, not Bash) to read:
/home/prabhanshu/Programs/esp32-bt-hid/docs/WINDOWS_PAIRING_GUIDE.md
Then state: 'Pairing guide loaded. Steps: [summary]'

## STEP 3 — USE THE READ TOOL THIRD
Use the Read tool (not cat, not Bash) to read:
/home/prabhanshu/Programs/esp32-bt-hid/docs/BT_HID_LEARNINGS.md
Then state: 'Learnings loaded. Key failures: [summary]'

IMPORTANT: Do NOT run ls, pwd, or any exploration before completing steps 1-3.

## STEP 4 — VERIFY PREREQUISITES
Before attempting pairing, verify ALL of these:
A) ESP32 mouse connected: echo STATUS > /dev/ttyUSB0 && sleep 1 && timeout 2 cat /dev/ttyUSB0
B) Pi keyboard server running: ssh pi@10.55.0.2 'curl -s http://localhost:8081/api/keyboard/health'
C) HDMI capture working: ffmpeg -f v4l2 -input_format yuyv422 -video_size 1920x1080 -i /dev/video0 -frames:v 1 -y /tmp/hdmi_test.png && ls -la /tmp/hdmi_test.png
D) OPENROUTER_API_KEY set: pass show api/openrouter >/dev/null 2>&1 && echo 'OpenRouter key available'
E) Pi discoverable: ssh pi@10.55.0.2 'bluetoothctl discoverable on && bluetoothctl pairable on'

If ANY prerequisite fails, fix it before proceeding. Document the fix.

## STEP 5 — START ANTI-SLEEP JITTER
Before pairing, ensure Windows stays awake by sending real mouse movement:
- Start ESP32 heartbeat (MOUSE:0,0 at 100ms for BLE keepalive)
- Also send ±3px real movement every 30 seconds (prevents Windows sleep)
- The pair_k380.py script handles this automatically now

## STEP 6 — VERIFY LLM VISION WORKS
Test LLM vision with a single HDMI capture screenshot:
- Capture: ffmpeg -f v4l2 -input_format yuyv422 -video_size 1920x1080 -i /dev/video0 -frames:v 1 -y /tmp/vision_test.png
- Test: cd /home/prabhanshu/Programs/UI-agent && uv run python -c \"
from src.vision.llm_vision import LLMVision
v = LLMVision()
r = v.analyze_screen('/tmp/vision_test.png', 'Describe what you see on this screen')
print(r.screen_description)
v.close()
\"
- If it fails, check OPENROUTER_API_KEY env var (source from pass if needed)
- Vision model is Gemini 2.5 Flash via OpenRouter (default, no Anthropic key needed)

## STEP 7 — RUN CURSOR DETECTION
Run: cd /home/prabhanshu/Programs/UI-agent && uv run python -m src.hardware.cursor_locator
- Cursor must be detected with correlation > 0.7
- If not detected, place cursor at center of HDMI screen manually and retry
- Record detected position

## STEP 8 — RUN PAIR_K380 EXPERIENCE
Run: cd /home/prabhanshu/Programs/UI-agent && uv run python scripts/pair_k380.py
- Monitor output for each step
- If a step fails, analyze the screenshot and debug
- Take notes on what went wrong

## STEP 9 — VERIFY PAIRING (ALL CHECKS MUST PASS)
After pair_k380.py completes, verify EVERY item:

A) Pi API check:
   ssh pi@10.55.0.2 'curl -s http://localhost:8081/api/keyboard/health'
   Must show bluetooth_connected: true

B) Typing test:
   ssh pi@10.55.0.2 'curl -s -X POST http://localhost:8081/api/keyboard/type -H \"Content-Type: application/json\" -d \"{\\\"text\\\":\\\"hello\\\"}\"'
   Then capture HDMI screenshot and verify 'hello' appeared on screen

C) Windows BT settings check:
   Navigate to Settings > Bluetooth & devices via mouse
   Capture HDMI screenshot
   Verify: K380 under Input, keyboard icon, battery %, Connected status

D) Stability test:
   Wait 30 seconds
   Re-check Pi API health
   Re-capture Windows BT settings screenshot
   All checks must still pass

E) Recovery test:
   Disconnect K380 from Windows BT settings
   Wait 10 seconds
   Check if it auto-reconnects
   If not, try reconnecting from Pi side

## STEP 10 — IF PAIRING FAILS
If pair_k380.py fails or verification fails:
1. Read error output carefully
2. Check Pi logs: ssh pi@10.55.0.2 'journalctl -u bt-keyboard -n 50'
3. Check device class: ssh pi@10.55.0.2 'hciconfig hci0 class'
4. Check if stale pairing exists on Windows — remove it via mouse navigation
5. Restart Pi BT: ssh pi@10.55.0.2 'sudo systemctl restart bluetooth bt-keyboard && sleep 2 && bluetoothctl discoverable on'
6. Retry from STEP 7

## COMPLETION CRITERIA
Task is COMPLETE when ALL are TRUE simultaneously:
- K380 shows as 'Connected' under 'Input' in Windows BT settings (verified via HDMI screenshot)
- Battery % visible
- Pi API returns bluetooth_connected: true
- Typing 'test123' via API produces 'test123' on Windows (verified via HDMI screenshot)
- Connection survives 30-second idle period
- All results documented in progress_actual.md

## HARD LIMITS
- NEVER skip verification steps — every check matters
- NEVER declare success without HDMI screenshot proof
- If stuck for 3 iterations on same error, read BT_HID_LEARNINGS.md again
- Side instructions do NOT change completion criteria
- Do NOT modify esp32_mouse.py BLE behavior (mouse works fine)
- Do NOT modify cursor_detector.py math (proven correct at 0.92 correlation)
- Commit working code IMMEDIATELY when each piece works

## IF BLOCKED (after 5 iterations without progress)
1. Document blocker in /home/prabhanshu/Programs/UI-agent/progress_actual.md
2. Read: /home/prabhanshu/Programs/esp32-bt-hid/docs/BT_HID_LEARNINGS.md
3. Read: /home/prabhanshu/Programs/esp32-bt-hid/docs/PI_TROUBLESHOOTING_RUNBOOK.md
4. Try alternative: Manual pairing via bluetoothctl commands on Pi
5. If still blocked after 8 iterations: Stop and document what's needed

## PROGRESS TRACKING
Append to /home/prabhanshu/Programs/UI-agent/progress_actual.md after EVERY action:
- NEVER delete existing sections
- ONLY append new information with timestamp
- Format: [YYYY-MM-DD HH:MM] Iteration N: what happened, what's next
- Mark outdated info as [Outdated] but preserve

## CONTEXT RETRIEVAL (if context lost)
- Read: /home/prabhanshu/Programs/UI-agent/docs/CURSOR_DETECTION_KNOWLEDGE.md
- Read: /home/prabhanshu/Programs/UI-agent/docs/CURSOR_DETECTION_PLAN.md
- Read: /home/prabhanshu/Programs/esp32-bt-hid/docs/WINDOWS_PAIRING_GUIDE.md
- Git log: cd ~/Programs/UI-agent && git log --oneline -20
- Previous session: claude --resume f0870570
" --max-iterations 15 --completion-promise "K380 paired and verified — typing works, Connected under Input, battery visible"
RALPH_LOOP_COMMAND
