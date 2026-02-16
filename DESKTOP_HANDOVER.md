# Desktop KVM Handover

**Created:** 2026-02-16 (laptop session)
**Purpose:** Continue KVM setup on desktop — fix Pi networking, test keyboard passthrough, verify HDMI capture, run end-to-end test.

## What Was Done (Laptop)

1. Created `src/hardware/pi_keyboard.py` — PiKeyboardClient HTTP wrapper for Pi Zero 2W keyboard API
2. Created `scripts/mouse_passthrough.py` — evdev mouse → ESP32 serial → BLE → Windows
3. Created `scripts/keyboard_passthrough.py` — evdev keyboard → Pi HTTP → BT → Windows
4. Created `scripts/video_viewer.sh` — ffplay wrapper for HDMI capture
5. Added `evdev`, `pyserial`, `requests` to dependencies
6. Updated `README.md` with correct architecture (desktop/laptop split)
7. All code compiles and imports verified

## What Needs To Be Done (Desktop)

### Step 1: Pull Latest Code

```bash
cd ~/Programs/UI-agent
git pull origin master
uv sync
```

### Step 2: Fix Pi Zero Networking

The Pi Zero 2W is connected via USB to the desktop (RNDIS gadget, interface `enp8s0f3u3u4`).

```bash
# Assign IP to the USB network interface
sudo ip addr add 10.55.0.1/24 dev enp8s0f3u3u4
sudo ip link set enp8s0f3u3u4 up

# Test connectivity
ping -c 3 10.55.0.2

# If no response: power cycle Pi (unplug/replug USB), wait 45s, retry
# If still no response: may need to re-flash SD card

# Verify keyboard API
curl http://10.55.0.2:8081/api/keyboard/health
```

**Success criteria:** `curl http://10.55.0.2:8081/api/keyboard/health` returns JSON with status "healthy" or "ok".

### Step 3: Verify HDMI Capture

```bash
# Check device exists and formats
v4l2-ctl --list-formats-ext -d /dev/video0

# Test live view
./scripts/video_viewer.sh
```

**Success criteria:** Windows desktop visible in ffplay window.

### Step 4: Test Keyboard Passthrough

```bash
# List keyboards
uv run python scripts/keyboard_passthrough.py --list

# Start passthrough (will need to select a keyboard)
uv run python scripts/keyboard_passthrough.py --device /dev/input/eventN
```

Press **Scroll Lock** → type → verify characters appear on Windows.

### Step 5: Verify Mouse Passthrough Works From Laptop

On **laptop**, start mouse passthrough:
```bash
cd ~/Programs/UI-agent
uv run python scripts/mouse_passthrough.py
```

Press **Scroll Lock** → move mouse → verify cursor moves on Windows.

### Step 6: End-to-End Test

Run all three simultaneously:
1. **Desktop:** `./scripts/video_viewer.sh` — see Windows screen
2. **Desktop:** `uv run python scripts/keyboard_passthrough.py` — Scroll Lock → type → Windows
3. **Laptop:** `uv run python scripts/mouse_passthrough.py` — Scroll Lock → move → Windows

## Key Files

| File | Machine | Purpose |
|------|---------|---------|
| `scripts/keyboard_passthrough.py` | Desktop | Keyboard evdev → Pi HTTP → BT → Windows |
| `scripts/video_viewer.sh` | Desktop | HDMI capture ffplay viewer |
| `scripts/mouse_passthrough.py` | Laptop | Mouse evdev → ESP32 serial → BLE → Windows |
| `src/hardware/pi_keyboard.py` | Both | Pi keyboard HTTP client library |
| `src/hardware/esp32_mouse.py` | Both | ESP32 mouse serial protocol reference |

## Troubleshooting

**Pi not responding at 10.55.0.2:**
- Check interface: `ip addr show enp8s0f3u3u4`
- Check USB: `lsusb | grep 0525:a4a2`
- Power cycle Pi (unplug/replug), wait 45s

**No HDMI signal:**
- Check device: `ls /dev/video0`
- Disable autosuspend: `echo "on" | sudo tee /sys/bus/usb/devices/1-3/power/control`
- Check if Windows machine HDMI is connected

**Permission denied on evdev:**
- `sudo usermod -aG input $USER` then re-login

**Scroll Lock not detected:**
- Some keyboards don't have a Scroll Lock key
- Check with `evtest /dev/input/eventN` — look for KEY_SCROLLLOCK (code 70)
