# Session Log: Cursor Locate + K380 Keyboard Pairing
**Date:** 2026-02-17
**Goal:** Locate ESP32 mouse cursor on HDMI-captured Windows screen, then drive Bluetooth UI to fix K380 keyboard pairing

## Hardware Setup
- **HDMI capture:** MacroSilicon MS2109 at `/dev/video0`, MJPEG 1920x1080@30fps
- **ESP32 mouse:** `/dev/ttyUSB0`, 115200 baud, status=CONNECTED
- **Machine:** Laptop (ThinkPad T14)
- **Target:** Windows laptop with multi-monitor setup

## Timeline

### 14:42 — HDMI Capture: Black Screen
- Initial capture → 8.6KB black PNG (no signal)
- **Root cause:** USB autosuspend set to `auto` at `/sys/bus/usb/devices/5-4/power/control`
- **Fix:** `echo "on" | sudo tee /sys/bus/usb/devices/5-4/power/control`
- After fix: 1.08MB capture showing Windows desktop with VS Code/IDE

### 14:45 — Windows Screen State
- VS Code or dark IDE with terminal open
- Browser dev tools visible
- Cursor visible in lower-center area (white arrow)
- User confirmed they will move cursor to OTHER monitor for Lissajous test

### 14:4x — ESP32 Mouse Not Moving Cursor
- ESP32 reports CONNECTED, returns OK for MOUSE commands
- But cursor doesn't move on Windows — stale BLE HID connection
- **Fix:** DTR serial toggle resets ESP32 → forces BLE reconnect
  ```python
  mouse.ser.dtr = False; time.sleep(0.1); mouse.ser.dtr = True
  ```
- After reset: reconnected in ~1s, "INFO: Mouse connected"
- Cursor still on other monitor (not on HDMI screen)

### 14:5x — Lissajous Cursor Detection: Attempt 1 (Live Pipeline)
- Parameters: default LissajousCurve (amp_x=3000, amp_y=1500, omega ratio=√2)
- max_duration=10s, video_fps=30
- Result: **TIMEOUT at 85.6s** — only 111 frames, 1 analysis pass
- Issue: `load_video_frames()` too slow on growing file (entire video decoded each check)

### 15:0x — Lissajous Cursor Detection: Attempt 2 (Offline Analysis)
- Changed approach: record full 8s video first, then analyze offline
- 347 frames loaded in 4.7s
- Blob extraction: 56.2s for 346 frame pairs (160ms/pair — scipy.ndimage.label bottleneck)
- 75/346 frames had motion blobs
- **CURSOR FOUND at (1842, 15)** — correlation 0.92, t=[2.80, 3.77]s
- 30 matched blobs in the detection sequence
- Moved cursor to screen center (960, 540), shake confirmed visible
- Position tracking working: ESP32 mouse now accurately controlled

### ESP32 BLE Reconnect (New Primitive)
When ESP32 BLE HID goes stale (CONNECTED but no cursor movement):
1. Toggle DTR on serial port → hardware reset
2. Wait 3s for boot + BLE reconnect
3. Verify with STATUS command
This should be added as a `reconnect()` method on ESP32Mouse.

## Observations & Learnings

### HDMI Autosuspend (Recurring Issue)
Every session starts with black screen. Checklist documented at `docs/HDMI_CAPTURE_CHECKLIST.md`.
The udev rule in local-bootstrapping should prevent this but isn't applied on this machine.

### Lissajous Performance
- **Recording:** 347 frames in 10.2s (ESP32 20ms/report = 10.2s for 401 reports, correct)
- **Frame loading:** 4.7s for 347 frames via raw RGB pipe — fast
- **Blob extraction:** 56.2s for 346 pairs — SLOW (scipy.ndimage.label on full-res)
  - Optimization: downsample frames to 960×540 before diff/label
  - Or use OpenCV connectedComponents (faster than scipy)
- **Curve matching:** <0.01s — very fast, the math is cheap
- **Total pipeline:** ~70s offline, needs to be <10s for live detection

### Live vs Offline Approach
The live pipeline (check every 0.5s) doesn't work because load_video_frames decodes the entire video each time. Better approaches:
1. **Offline:** Record full sweep, analyze after (current working approach)
2. **Streaming:** Use ffmpeg pipe to deliver frames in real-time (requires rewrite)
3. **Downsampled live:** Only analyze every Nth frame at reduced resolution
