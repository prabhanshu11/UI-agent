#!/usr/bin/env python3
"""Keyboard passthrough: evdev keyboard → Pi HTTP API → BT → Windows.

Captures a physical keyboard via evdev, forwards key events to the
Pi Zero 2W's HTTP API which sends them via Bluetooth HID to Windows.
Scroll Lock toggles between local mode and KVM passthrough mode.

Runs on: DESKTOP (where Pi Zero 2W is connected via USB)
"""

import argparse
import sys
import os

import evdev
import requests

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.hardware.pi_keyboard import PiKeyboardClient


EV_KEY = 0x01
KEY_SCROLLLOCK = 70

# evdev keycode → Pi API key name mapping
KEYCODE_MAP = {
    # Letters
    evdev.ecodes.KEY_A: 'a', evdev.ecodes.KEY_B: 'b', evdev.ecodes.KEY_C: 'c',
    evdev.ecodes.KEY_D: 'd', evdev.ecodes.KEY_E: 'e', evdev.ecodes.KEY_F: 'f',
    evdev.ecodes.KEY_G: 'g', evdev.ecodes.KEY_H: 'h', evdev.ecodes.KEY_I: 'i',
    evdev.ecodes.KEY_J: 'j', evdev.ecodes.KEY_K: 'k', evdev.ecodes.KEY_L: 'l',
    evdev.ecodes.KEY_M: 'm', evdev.ecodes.KEY_N: 'n', evdev.ecodes.KEY_O: 'o',
    evdev.ecodes.KEY_P: 'p', evdev.ecodes.KEY_Q: 'q', evdev.ecodes.KEY_R: 'r',
    evdev.ecodes.KEY_S: 's', evdev.ecodes.KEY_T: 't', evdev.ecodes.KEY_U: 'u',
    evdev.ecodes.KEY_V: 'v', evdev.ecodes.KEY_W: 'w', evdev.ecodes.KEY_X: 'x',
    evdev.ecodes.KEY_Y: 'y', evdev.ecodes.KEY_Z: 'z',

    # Numbers
    evdev.ecodes.KEY_1: '1', evdev.ecodes.KEY_2: '2', evdev.ecodes.KEY_3: '3',
    evdev.ecodes.KEY_4: '4', evdev.ecodes.KEY_5: '5', evdev.ecodes.KEY_6: '6',
    evdev.ecodes.KEY_7: '7', evdev.ecodes.KEY_8: '8', evdev.ecodes.KEY_9: '9',
    evdev.ecodes.KEY_0: '0',

    # Function keys
    evdev.ecodes.KEY_F1: 'f1', evdev.ecodes.KEY_F2: 'f2', evdev.ecodes.KEY_F3: 'f3',
    evdev.ecodes.KEY_F4: 'f4', evdev.ecodes.KEY_F5: 'f5', evdev.ecodes.KEY_F6: 'f6',
    evdev.ecodes.KEY_F7: 'f7', evdev.ecodes.KEY_F8: 'f8', evdev.ecodes.KEY_F9: 'f9',
    evdev.ecodes.KEY_F10: 'f10', evdev.ecodes.KEY_F11: 'f11', evdev.ecodes.KEY_F12: 'f12',

    # Special keys
    evdev.ecodes.KEY_ENTER: 'enter',
    evdev.ecodes.KEY_ESC: 'escape',
    evdev.ecodes.KEY_BACKSPACE: 'backspace',
    evdev.ecodes.KEY_TAB: 'tab',
    evdev.ecodes.KEY_SPACE: 'space',
    evdev.ecodes.KEY_CAPSLOCK: 'capslock',
    evdev.ecodes.KEY_INSERT: 'insert',
    evdev.ecodes.KEY_HOME: 'home',
    evdev.ecodes.KEY_PAGEUP: 'pageup',
    evdev.ecodes.KEY_DELETE: 'delete',
    evdev.ecodes.KEY_END: 'end',
    evdev.ecodes.KEY_PAGEDOWN: 'pagedown',

    # Arrow keys
    evdev.ecodes.KEY_UP: 'up',
    evdev.ecodes.KEY_DOWN: 'down',
    evdev.ecodes.KEY_LEFT: 'left',
    evdev.ecodes.KEY_RIGHT: 'right',

    # Punctuation / symbols
    evdev.ecodes.KEY_MINUS: '-',
    evdev.ecodes.KEY_EQUAL: '=',
    evdev.ecodes.KEY_LEFTBRACE: '[',
    evdev.ecodes.KEY_RIGHTBRACE: ']',
    evdev.ecodes.KEY_BACKSLASH: '\\',
    evdev.ecodes.KEY_SEMICOLON: ';',
    evdev.ecodes.KEY_APOSTROPHE: "'",
    evdev.ecodes.KEY_GRAVE: '`',
    evdev.ecodes.KEY_COMMA: ',',
    evdev.ecodes.KEY_DOT: '.',
    evdev.ecodes.KEY_SLASH: '/',

    # Numpad
    evdev.ecodes.KEY_KP0: 'kp_0', evdev.ecodes.KEY_KP1: 'kp_1',
    evdev.ecodes.KEY_KP2: 'kp_2', evdev.ecodes.KEY_KP3: 'kp_3',
    evdev.ecodes.KEY_KP4: 'kp_4', evdev.ecodes.KEY_KP5: 'kp_5',
    evdev.ecodes.KEY_KP6: 'kp_6', evdev.ecodes.KEY_KP7: 'kp_7',
    evdev.ecodes.KEY_KP8: 'kp_8', evdev.ecodes.KEY_KP9: 'kp_9',
    evdev.ecodes.KEY_KPENTER: 'kp_enter',
    evdev.ecodes.KEY_KPDOT: 'kp_period',
    evdev.ecodes.KEY_KPPLUS: 'kp_plus',
    evdev.ecodes.KEY_KPMINUS: 'kp_minus',
    evdev.ecodes.KEY_KPASTERISK: 'kp_multiply',
    evdev.ecodes.KEY_KPSLASH: 'kp_divide',

    # Print screen, pause
    evdev.ecodes.KEY_SYSRQ: 'printscreen',
    evdev.ecodes.KEY_PAUSE: 'pause',
    evdev.ecodes.KEY_NUMLOCK: 'numlock',
}

# Modifier keycodes
MODIFIER_KEYCODES = {
    evdev.ecodes.KEY_LEFTCTRL: 'ctrl',
    evdev.ecodes.KEY_RIGHTCTRL: 'ctrl',
    evdev.ecodes.KEY_LEFTSHIFT: 'shift',
    evdev.ecodes.KEY_RIGHTSHIFT: 'shift',
    evdev.ecodes.KEY_LEFTALT: 'alt',
    evdev.ecodes.KEY_RIGHTALT: 'alt',
    evdev.ecodes.KEY_LEFTMETA: 'gui',
    evdev.ecodes.KEY_RIGHTMETA: 'gui',
}


def list_input_devices():
    """List all input devices, highlighting keyboards."""
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    keyboards = []
    for dev in devices:
        caps = dev.capabilities(verbose=True)
        has_keys = any('EV_KEY' in str(k) for k in caps)
        # Check for letter keys to distinguish keyboards from mice
        key_caps = dev.capabilities().get(evdev.ecodes.EV_KEY, [])
        has_letters = evdev.ecodes.KEY_A in key_caps and evdev.ecodes.KEY_Z in key_caps
        if has_keys and has_letters:
            keyboards.append(dev)
            marker = " [KEYBOARD]"
        else:
            marker = ""
        print(f"  {dev.path}: {dev.name}{marker}")
    if keyboards:
        print(f"\nDetected keyboards: {', '.join(d.path for d in keyboards)}")
    else:
        print("\nNo keyboards detected. Are you in the 'input' group?")
    return keyboards


class KeyboardPassthrough:
    """Forward evdev keyboard events to Pi keyboard API."""

    def __init__(self, device_path: str, pi_host: str = '10.55.0.2', pi_port: int = 8081):
        self.device = evdev.InputDevice(device_path)
        self.pi = PiKeyboardClient(host=pi_host, port=pi_port)
        self.passthrough_active = False
        self.running = True

        # Track active modifiers
        self.active_modifiers: set[str] = set()

    def toggle_passthrough(self):
        """Toggle passthrough mode."""
        self.passthrough_active = not self.passthrough_active
        state = "ON" if self.passthrough_active else "OFF"
        print(f"\n[KVM] Keyboard passthrough {state}")
        if self.passthrough_active:
            self.device.grab()
            self.active_modifiers.clear()
            print("[KVM] Keyboard grabbed — exclusive control to Windows")
        else:
            self.device.ungrab()
            self.active_modifiers.clear()
            print("[KVM] Keyboard released — back to local control")

    def _handle_key_event(self, code: int, value: int):
        """Process a key event and forward to Pi API.

        Args:
            code: evdev keycode.
            value: 0=release, 1=press, 2=repeat.
        """
        # Track modifier state
        if code in MODIFIER_KEYCODES:
            mod = MODIFIER_KEYCODES[code]
            if value == 1:  # Press
                self.active_modifiers.add(mod)
            elif value == 0:  # Release
                self.active_modifiers.discard(mod)
            # Don't send modifier-only presses to the API — they're sent
            # along with the next regular key press
            return

        # Only forward key presses (not repeats or releases)
        if value != 1:
            return

        key_name = KEYCODE_MAP.get(code)
        if key_name is None:
            return

        modifiers = list(self.active_modifiers) if self.active_modifiers else None

        try:
            self.pi.key_press(key_name, modifiers)
            mod_str = f" +{'+'.join(modifiers)}" if modifiers else ""
            print(f"  → {key_name}{mod_str}", end='\r')
        except requests.RequestException as e:
            print(f"  [ERR] Pi API: {e}", end='\r')

    def run(self):
        """Main event loop."""
        print(f"Keyboard: {self.device.name} ({self.device.path})")
        print(f"Pi host: {self.pi.base_url}")

        # Check Pi connectivity
        try:
            health = self.pi.health()
            print(f"Pi health: {health.get('status', 'unknown')}")
        except requests.RequestException as e:
            print(f"WARNING: Pi not reachable ({e})")
            print("Continuing anyway — will retry on key press")

        print(f"Press Scroll Lock to toggle KVM passthrough")
        print(f"Ctrl+C to exit\n")

        try:
            for event in self.device.read_loop():
                if not self.running:
                    break

                if event.type != EV_KEY:
                    continue

                # Scroll Lock toggles passthrough
                if event.code == KEY_SCROLLLOCK and event.value == 1:
                    self.toggle_passthrough()
                    continue

                if not self.passthrough_active:
                    continue

                self._handle_key_event(event.code, event.value)

        except KeyboardInterrupt:
            print("\n[KVM] Shutting down...")
        finally:
            self.running = False
            if self.passthrough_active:
                try:
                    self.device.ungrab()
                except OSError:
                    pass
            print("[KVM] Keyboard passthrough stopped")


def main():
    parser = argparse.ArgumentParser(description='Keyboard KVM passthrough: evdev → Pi HTTP → BT → Windows')
    parser.add_argument('--list', action='store_true', help='List available input devices')
    parser.add_argument('--device', '-d', help='evdev device path (e.g. /dev/input/event3)')
    parser.add_argument('--pi-host', default='10.55.0.2', help='Pi Zero 2W IP (default: 10.55.0.2)')
    parser.add_argument('--pi-port', type=int, default=8081, help='Pi API port (default: 8081)')
    args = parser.parse_args()

    if args.list:
        list_input_devices()
        return

    if not args.device:
        print("Detecting keyboards...")
        keyboards = list_input_devices()
        if len(keyboards) == 1:
            args.device = keyboards[0].path
            print(f"\nAuto-selected: {keyboards[0].name}")
        elif len(keyboards) > 1:
            print("\nMultiple keyboards found. Use --device to select one.")
            return
        else:
            print("\nNo keyboards found. Use --list to see all devices.")
            return

    passthrough = KeyboardPassthrough(
        device_path=args.device,
        pi_host=args.pi_host,
        pi_port=args.pi_port,
    )
    passthrough.run()


if __name__ == '__main__':
    main()
