# Desktop KVM Setup — Claude Code Prompt

**Copy everything below the line and paste into Claude Code on the desktop:**

---

Continue the KVM system setup. The laptop session created all the scripts and pushed them. Read `~/Programs/UI-agent/DESKTOP_HANDOVER.md` for the full context, then execute these steps in order:

1. **Pull and sync**: `cd ~/Programs/UI-agent && git pull origin master && uv sync`

2. **Fix Pi networking**: The Pi Zero 2W is on USB interface `enp8s0f3u3u4`. Assign IP `10.55.0.1/24`, bring it up, ping Pi at `10.55.0.2`, then verify keyboard API: `curl http://10.55.0.2:8081/api/keyboard/health`. If Pi doesn't respond, power cycle it (unplug/replug USB), wait 45s, retry.

3. **Verify HDMI capture**: Check `/dev/video0` exists, run `v4l2-ctl --list-formats-ext -d /dev/video0`, then test with `./scripts/video_viewer.sh` — Windows screen should be visible.

4. **Test keyboard passthrough**: Run `uv run python scripts/keyboard_passthrough.py --list` to find the keyboard, then start it. Press Scroll Lock to toggle KVM mode, type characters, verify they appear on Windows.

5. **Report status** of each step so I can coordinate the laptop side (mouse passthrough).

The handover document has troubleshooting steps for common issues. The keyboard passthrough talks to the Pi at `10.55.0.2:8081` via HTTP which forwards via Bluetooth HID to Windows.
