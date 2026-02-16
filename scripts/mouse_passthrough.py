#!/usr/bin/env python3
"""Mouse passthrough: evdev mouse → ESP32 serial → BLE → Windows.

Captures a physical Bluetooth mouse via evdev, forwards movement and
clicks directly to the ESP32 over serial. Scroll Lock toggles between
local mode (normal mouse) and passthrough mode (KVM control).

Performance-critical: writes directly to serial, bypassing ESP32Mouse
class which has sleep delays unsuitable for real-time passthrough.

Runs on: LAPTOP (where ESP32 is connected via USB serial)
"""

import argparse
import json
import os
import select
import socket
import struct
import sys
import threading
import time

import evdev
import serial


# evdev event codes
EV_REL = 0x02
EV_KEY = 0x01

REL_X = 0x00
REL_Y = 0x01
REL_WHEEL = 0x08
REL_WHEEL_HI_RES = 0x0B

BTN_LEFT = 0x110
BTN_RIGHT = 0x111
BTN_MIDDLE = 0x112

KEY_SCROLLLOCK = 70


def list_input_devices():
    """List all input devices with their capabilities."""
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    mice = []
    for dev in devices:
        caps = dev.capabilities(verbose=True)
        has_rel = any('EV_REL' in str(k) for k in caps)
        has_btn = any('BTN_LEFT' in str(v) for vals in caps.values() for v in vals)
        if has_rel and has_btn:
            mice.append(dev)
            marker = " [MOUSE]"
        else:
            marker = ""
        print(f"  {dev.path}: {dev.name}{marker}")
        if has_rel:
            print(f"    Capabilities: REL axes detected")
    if mice:
        print(f"\nDetected mice: {', '.join(d.path for d in mice)}")
    else:
        print("\nNo mice detected. Are you in the 'input' group?")
    return mice


class MousePassthrough:
    """Forward evdev mouse events to ESP32 serial port."""

    def __init__(self, device_path: str, serial_port: str = '/dev/ttyUSB0',
                 baud_rate: int = 115200, tcp_port: int = 0):
        self.device = evdev.InputDevice(device_path)
        self.ser = serial.Serial(serial_port, baud_rate, timeout=0.1)
        self.ser.reset_input_buffer()

        self.passthrough_active = False
        self.running = True

        # Optional TCP event streaming
        self.tcp_port = tcp_port
        self.tcp_clients: list[socket.socket] = []
        self.tcp_server: socket.socket | None = None

        # Accumulate relative movements between syncs
        self._dx = 0
        self._dy = 0

    def _send_mouse(self, dx: int, dy: int):
        """Send mouse movement directly to ESP32 serial. No sleeps."""
        if dx == 0 and dy == 0:
            return
        # Chunk if exceeding HID limits (-127 to 127)
        while dx != 0 or dy != 0:
            cdx = max(-127, min(127, dx))
            cdy = max(-127, min(127, dy))
            self.ser.write(f'MOUSE:{cdx},{cdy}\n'.encode())
            dx -= cdx
            dy -= cdy
        self.ser.flush()

    def _send_click(self, button: str):
        """Send mouse click to ESP32 serial."""
        self.ser.write(f'CLICK:{button}\n'.encode())
        self.ser.flush()

    def _send_scroll(self, amount: int):
        """Send scroll event to ESP32 serial."""
        self.ser.write(f'SCROLL:{amount}\n'.encode())
        self.ser.flush()

    def _stream_event(self, event_data: dict):
        """Stream event to TCP clients (non-blocking)."""
        if not self.tcp_clients:
            return
        line = json.dumps(event_data) + '\n'
        data = line.encode()
        dead = []
        for client in self.tcp_clients:
            try:
                client.sendall(data)
            except (BrokenPipeError, ConnectionResetError, OSError):
                dead.append(client)
        for client in dead:
            self.tcp_clients.remove(client)
            client.close()

    def _start_tcp_server(self):
        """Start TCP server for event streaming."""
        self.tcp_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp_server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.tcp_server.bind(('0.0.0.0', self.tcp_port))
        self.tcp_server.listen(5)
        self.tcp_server.setblocking(False)
        print(f"TCP event stream on port {self.tcp_port}")

        def accept_loop():
            while self.running:
                try:
                    readable, _, _ = select.select([self.tcp_server], [], [], 1.0)
                    if readable:
                        client, addr = self.tcp_server.accept()
                        print(f"TCP client connected: {addr}")
                        self.tcp_clients.append(client)
                except (OSError, ValueError):
                    break

        t = threading.Thread(target=accept_loop, daemon=True)
        t.start()

    def toggle_passthrough(self):
        """Toggle passthrough mode on/off."""
        self.passthrough_active = not self.passthrough_active
        state = "ON" if self.passthrough_active else "OFF"
        print(f"\n[KVM] Passthrough {state}")
        if self.passthrough_active:
            self.device.grab()
            print("[KVM] Mouse grabbed — exclusive control to Windows")
        else:
            self.device.ungrab()
            print("[KVM] Mouse released — back to local control")

    def run(self):
        """Main event loop."""
        print(f"Mouse: {self.device.name} ({self.device.path})")
        print(f"Serial: {self.ser.port} @ {self.ser.baudrate}")
        print(f"Press Scroll Lock to toggle KVM passthrough")
        print(f"Ctrl+C to exit\n")

        if self.tcp_port:
            self._start_tcp_server()

        try:
            for event in self.device.read_loop():
                if not self.running:
                    break

                # Scroll Lock toggles passthrough regardless of mode
                if event.type == EV_KEY and event.code == KEY_SCROLLLOCK and event.value == 1:
                    self.toggle_passthrough()
                    continue

                if not self.passthrough_active:
                    continue

                # Mouse movement
                if event.type == EV_REL:
                    if event.code == REL_X:
                        self._dx += event.value
                    elif event.code == REL_Y:
                        self._dy += event.value
                    elif event.code in (REL_WHEEL, REL_WHEEL_HI_RES):
                        self._send_scroll(event.value)
                        self._stream_event({'type': 'scroll', 'value': event.value})

                # SYN event — flush accumulated movement
                elif event.type == 0x00:  # EV_SYN
                    if self._dx != 0 or self._dy != 0:
                        self._send_mouse(self._dx, self._dy)
                        self._stream_event({'type': 'move', 'dx': self._dx, 'dy': self._dy})
                        self._dx = 0
                        self._dy = 0

                # Button press/release
                elif event.type == EV_KEY:
                    if event.value == 1:  # Press
                        btn_map = {BTN_LEFT: 'left', BTN_RIGHT: 'right', BTN_MIDDLE: 'middle'}
                        button = btn_map.get(event.code)
                        if button:
                            self._send_click(button)
                            self._stream_event({'type': 'click', 'button': button})

        except KeyboardInterrupt:
            print("\n[KVM] Shutting down...")
        finally:
            self.running = False
            if self.passthrough_active:
                try:
                    self.device.ungrab()
                except OSError:
                    pass
            self.ser.close()
            if self.tcp_server:
                self.tcp_server.close()
            for client in self.tcp_clients:
                client.close()
            print("[KVM] Mouse passthrough stopped")


def main():
    parser = argparse.ArgumentParser(description='Mouse KVM passthrough: evdev → ESP32 → BLE → Windows')
    parser.add_argument('--list', action='store_true', help='List available input devices')
    parser.add_argument('--device', '-d', help='evdev device path (e.g. /dev/input/event5)')
    parser.add_argument('--serial', '-s', default='/dev/ttyUSB0', help='ESP32 serial port (default: /dev/ttyUSB0)')
    parser.add_argument('--baud', type=int, default=115200, help='Serial baud rate (default: 115200)')
    parser.add_argument('--tcp-port', type=int, default=0, help='TCP port for event streaming (0=disabled)')
    args = parser.parse_args()

    if args.list:
        list_input_devices()
        return

    if not args.device:
        print("Detecting mice...")
        mice = list_input_devices()
        if len(mice) == 1:
            args.device = mice[0].path
            print(f"\nAuto-selected: {mice[0].name}")
        elif len(mice) > 1:
            print("\nMultiple mice found. Use --device to select one.")
            return
        else:
            print("\nNo mice found. Use --list to see all devices, or --device to specify manually.")
            return

    passthrough = MousePassthrough(
        device_path=args.device,
        serial_port=args.serial,
        baud_rate=args.baud,
        tcp_port=args.tcp_port,
    )
    passthrough.run()


if __name__ == '__main__':
    main()
