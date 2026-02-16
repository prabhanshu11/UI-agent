# UI-Agent

AI-driven UI automation system with distributed KVM capabilities for controlling a Windows machine from Linux.

## Architecture

```
                    DESKTOP                          LAPTOP                       WINDOWS
              ┌─────────────────┐            ┌─────────────────┐           ┌──────────┐
              │ AI Agent        │            │ BT Mouse        │           │          │
              │                 │            │ (physical,user)  │           │          │
              │ HDMI Capture    │◄── HDMI ───│                 │           │          │
              │ /dev/video0     │            │ mouse_passthru  │           │          │
              │                 │            │     ↓            │           │          │
              │ Pi Zero 2W USB  │            │ ESP32 serial    │── BLE ──→│  Mouse   │
              │ enp8s0f3u3u4    │            │ /dev/ttyUSB0    │           │          │
              │ 10.55.0.2:8081  │            │                 │           │          │
              │     ↓           │            │                 │           │          │
              │ keyboard_passthru│──→ Pi ──→ BT ──────────────→│  Keyboard│
              │                 │            │                 │           │          │
              └────────┬────────┘            └────────┬────────┘           └──────────┘
                       └──── Tailscale (local LAN) ───┘
```

### Component Locations

| Component | Machine | Device | Protocol |
|-----------|---------|--------|----------|
| HDMI Capture (MacroSilicon MS2109) | Desktop | `/dev/video0` | v4l2/MJPEG |
| Pi Zero 2W (BT Keyboard "K380") | Desktop USB | `10.55.0.2:8081` | HTTP → BT HID |
| ESP32 (BLE Mouse "M350 Pebble") | Laptop | `/dev/ttyUSB0` | Serial → BLE HID |
| User's BT Mouse | Laptop | evdev `/dev/input/eventN` | evdev capture |

### Non-Negotiable KVM Primitives

These are foundational capabilities that must always work:

1. **Mouse** — Physical BT mouse on laptop → evdev capture → ESP32 serial → BLE → Windows cursor
2. **Keyboard** — Desktop keyboard → evdev capture → Pi HTTP API → BT Classic → Windows keyboard
3. **Video** — Windows HDMI out → Desktop capture card `/dev/video0` → ffplay/recording
4. **Direct HID Streaming** — Real-time forwarding of mouse and keyboard controls from host devices to Windows, with Scroll Lock toggle

## Setup

Python environment managed by `uv`.

```bash
uv sync
```

### Prerequisites

Both machines need evdev access for input capture:
```bash
sudo usermod -aG input $USER
# Then re-login
```

## KVM Scripts

### Mouse Passthrough (runs on LAPTOP)

Captures a physical mouse and forwards movements/clicks to Windows via ESP32 BLE:

```bash
# List available input devices
uv run python scripts/mouse_passthrough.py --list

# Start passthrough (auto-detect mouse)
uv run python scripts/mouse_passthrough.py

# Specify device and enable TCP event streaming
uv run python scripts/mouse_passthrough.py --device /dev/input/event5 --tcp-port 9601
```

Press **Scroll Lock** to toggle between local and KVM mode.

### Keyboard Passthrough (runs on DESKTOP)

Captures keyboard events and forwards to Windows via Pi Zero BT:

```bash
# List available input devices
uv run python scripts/keyboard_passthrough.py --list

# Start passthrough (auto-detect keyboard)
uv run python scripts/keyboard_passthrough.py

# Specify device and Pi host
uv run python scripts/keyboard_passthrough.py --device /dev/input/event3 --pi-host 10.55.0.2
```

Press **Scroll Lock** to toggle between local and KVM mode.

### Video Viewer (runs on DESKTOP)

Displays the Windows screen via HDMI capture card:

```bash
# Default 1920x1080@30fps
./scripts/video_viewer.sh

# Custom resolution/fps
./scripts/video_viewer.sh 1280x720 60
```

## Hardware Modules

- `src/hardware/esp32_mouse.py` — ESP32 BLE mouse control (serial protocol)
- `src/hardware/pi_keyboard.py` — Pi Zero BT keyboard HTTP client
- `src/hardware/usb_capture.py` — HDMI capture card interface

## Commands

```bash
uv sync           # Install dependencies
uv run pytest     # Run tests
```
