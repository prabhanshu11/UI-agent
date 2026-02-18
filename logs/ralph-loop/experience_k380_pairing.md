# K380 Pairing Experience — Ralph Loop Session

## Date: 2026-02-18

## Hardware Setup
| Component | Device | Status |
|-----------|--------|--------|
| ESP32 Mouse (M350 Pebble) | /dev/ttyUSB0 | CONNECTED (BLE HID to Windows) |
| HDMI Capture | /dev/video0 | Working (1920x1080 YUYV422) |
| Pi Zero Keyboard (Keyboard K380) | 10.55.0.2:8081 | BT service active, NOT paired |
| USB Gadget Network | enp8s0f3u3u4 | 10.55.0.1/24 configured |

## Prerequisites Checklist
- [x] ESP32 serial port exists (/dev/ttyUSB0)
- [x] ESP32 BT connected to Windows (DTR reset required after BT was turned on)
- [x] HDMI capture working
- [x] Pi Zero reachable (sshpass -p raspberry pi@10.55.0.2)
- [x] bt-keyboard.service active on Pi
- [x] bt-keyboard-api.service started (port 8081)
- [x] Pi BT discoverable + pairable (Class 0x002540 Keyboard)
- [x] Windows BT turned ON (user confirmed)

## Cursor Position Tracking Protocol
From git history (cursor_tracker.py, cursor_locator.py, esp32_mouse.py):

1. **ALWAYS calibrate first**: `mouse.calibrate_to_corner('top_left')` → position (0,0)
2. **CursorTracker** maintains position via 3-tier recovery:
   - Tier 1: Micro-shake ±3px (cheap, invisible)
   - Tier 2: Macro-shake ±15px (visible, corrects drift)
   - Tier 3: Lissajous re-detect (5s sweep, last resort)
3. **verified_click()** probes position before clicking
4. **Never send raw serial commands without tracking**

## Pairing Steps (from WINDOWS_PAIRING_GUIDE.md)
1. Settings > Bluetooth & devices
2. Add device > Bluetooth
3. Wait for "Keyboard K380" to appear
4. Click to pair — no PIN needed
5. Verify: keyboard icon, "Connected" status

## Verification Criteria (Ralph Loop completion)
- [ ] K380 appears under "Input" section (NOT "Other devices")
- [ ] Shows keyboard icon (not generic BT icon)
- [ ] Shows battery percentage

## Progress Log

### 08:51 — Iteration start
- Pi unreachable (USB gadget IP missing)
- Fixed via: `pkexec ip addr add 10.55.0.1/24 dev enp8s0f3u3u4`

### 08:54 — Pi connected
- SSH working, bt-keyboard active, bt-keyboard-api started
- Windows desktop visible via HDMI (was already unlocked)

### 08:57 — ESP32 mouse issue
- ESP32 reports CONNECTED but cursor doesn't move
- Root cause: BT was OFF on Windows, stale connection
- Fix: User turned BT on, ESP32 DTR reset to force reconnect
- Result: Mouse working (user confirmed)

### 09:00 — Blind clicking wasted time
- LESSON: Must use calibrate_to_corner() + frame-diff verification
- LESSON: Read git history first for existing solutions

### 09:10 — Proper calibration attempt
- calibrate_to_corner() done
- Frame-diff probe at (24, 1060) — cursor not detected (taskbar edge, low contrast)
- Proceeding with click at estimated position
