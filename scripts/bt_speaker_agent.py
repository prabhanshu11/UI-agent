#!/usr/bin/env python3
"""Bluetooth A2DP sink agent — spoofs Sony WF-1000XM5 identity.

Makes the desktop's Bluetooth adapter appear as "WF-1000XM5" headphones
to nearby devices. The target Windows laptop pairs with it and streams
audio, which PipeWire captures for recording/transcription.

Follows the KVM secrecy principle: the target only sees familiar
consumer devices. Same pattern as ESP32→"M350 Pebble" mouse and
Pi Zero→"Keyboard K380" keyboard.

Real Sony WF-1000XM5 fingerprint (from bluetoothctl info):
  Class: 0x00240404 (Audio/Video, Wearable Headset Device)
  Icon: audio-headset
  Modalias: usb:v054Cp0E63d0610 (Sony, product 0x0E63)
  UUIDs: A2DP Sink, AVRCP, Handsfree, Headset, HID

Usage:
    python3 scripts/bt_speaker_agent.py
    # Then on Windows: Settings > Bluetooth > Add device > "WF-1000XM5"
"""

import signal
import sys

import dbus
import dbus.mainloop.glib
import dbus.service
from gi.repository import GLib

BUS_NAME = "org.bluez"
AGENT_INTERFACE = "org.bluez.Agent1"
AGENT_PATH = "/ui_agent/bt_audio"

# Spoofed identity — matches real Sony WF-1000XM5
DEVICE_ALIAS = "WF-1000XM5"

# UUIDs the real XM5 advertises (from bluetoothctl info)
ALLOWED_UUIDS = {
    "00001108-0000-1000-8000-00805f9b34fb",  # Headset
    "0000110b-0000-1000-8000-00805f9b34fb",  # Audio Sink
    "0000110c-0000-1000-8000-00805f9b34fb",  # A/V Remote Control Target
    "0000110d-0000-1000-8000-00805f9b34fb",  # Advanced Audio Distribution
    "0000110e-0000-1000-8000-00805f9b34fb",  # A/V Remote Control
    "0000110f-0000-1000-8000-00805f9b34fb",  # A/V Remote Control Controller
    "0000111e-0000-1000-8000-00805f9b34fb",  # Handsfree
    "00001131-0000-1000-8000-00805f9b34fb",  # Headset HS
    "00001124-0000-1000-8000-00805f9b34fb",  # HID
    "00001200-0000-1000-8000-00805f9b34fb",  # PnP Information
}


class Rejected(dbus.DBusException):
    _dbus_error_name = "org.bluez.Error.Rejected"


class AudioSinkAgent(dbus.service.Object):
    @dbus.service.method(AGENT_INTERFACE, in_signature="", out_signature="")
    def Release(self):
        print("[xm5] Released")

    @dbus.service.method(AGENT_INTERFACE, in_signature="os", out_signature="")
    def AuthorizeService(self, device, uuid):
        if uuid.lower() in ALLOWED_UUIDS:
            print(f"[xm5] Authorized: {device} → {uuid}")
            return
        # Accept all services — real headphones don't reject
        print(f"[xm5] Authorized (permissive): {device} → {uuid}")
        return

    @dbus.service.method(AGENT_INTERFACE, in_signature="o", out_signature="s")
    def RequestPinCode(self, device):
        print(f"[xm5] PIN: {device}")
        return "0000"

    @dbus.service.method(AGENT_INTERFACE, in_signature="ou", out_signature="")
    def RequestConfirmation(self, device, passkey):
        print(f"[xm5] Confirmed: {device} passkey={passkey:06d}")
        return  # auto-confirm (real headphones auto-accept)

    @dbus.service.method(AGENT_INTERFACE, in_signature="o", out_signature="u")
    def RequestPasskey(self, device):
        print(f"[xm5] Passkey: {device}")
        return dbus.UInt32(0)

    @dbus.service.method(AGENT_INTERFACE, in_signature="ou", out_signature="")
    def DisplayPasskey(self, device, passkey):
        print(f"[xm5] Display: {device} → {passkey:06d}")

    @dbus.service.method(AGENT_INTERFACE, in_signature="", out_signature="")
    def Cancel(self):
        print("[xm5] Cancelled")


def main():
    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SystemBus()

    agent = AudioSinkAgent(bus, AGENT_PATH)
    mainloop = GLib.MainLoop()

    # Set adapter to impersonate Sony WF-1000XM5
    adapter = dbus.Interface(
        bus.get_object(BUS_NAME, "/org/bluez/hci0"),
        "org.freedesktop.DBus.Properties",
    )
    adapter.Set("org.bluez.Adapter1", "DiscoverableTimeout", dbus.UInt32(0))
    adapter.Set("org.bluez.Adapter1", "Discoverable", dbus.Boolean(True))
    adapter.Set("org.bluez.Adapter1", "Pairable", dbus.Boolean(True))
    adapter.Set("org.bluez.Adapter1", "Alias", dbus.String(DEVICE_ALIAS))

    # Try to force device class to match real XM5 (0x240404)
    # BlueZ may override this from registered services, but Class in
    # main.conf + this D-Bus call covers most cases
    try:
        import subprocess
        subprocess.run(
            ["bluetoothctl", "system-alias", DEVICE_ALIAS],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass

    # Read back what Windows will actually see
    actual_alias = adapter.Get("org.bluez.Adapter1", "Alias")
    actual_class = adapter.Get("org.bluez.Adapter1", "Class")
    print(f"[xm5] Adapter spoofed as '{actual_alias}'")
    print(f"[xm5] Class: 0x{int(actual_class):06x} (target: 0x240404 Audio/Headset)")
    print(f"[xm5] Discoverable + Pairable")
    if int(actual_class) != 0x240404:
        print(f"[xm5] WARNING: Class mismatch — Windows may show 'Computer' icon")
        print(f"[xm5]   but name '{DEVICE_ALIAS}' + A2DP Sink UUID should still work")

    # Register as default agent (NoInputNoOutput = headphones behavior)
    obj = bus.get_object(BUS_NAME, "/org/bluez")
    manager = dbus.Interface(obj, "org.bluez.AgentManager1")
    manager.RegisterAgent(AGENT_PATH, "NoInputNoOutput")
    manager.RequestDefaultAgent(AGENT_PATH)

    print(f"[xm5] Waiting for connections...")
    print(f"[xm5] On Windows: Settings > Bluetooth > Add device > '{DEVICE_ALIAS}'")

    def shutdown(signum, frame):
        print("\n[xm5] Shutting down...")
        try:
            adapter.Set("org.bluez.Adapter1", "Discoverable", dbus.Boolean(False))
            manager.UnregisterAgent(AGENT_PATH)
        except Exception:
            pass
        mainloop.quit()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    mainloop.run()


if __name__ == "__main__":
    main()
