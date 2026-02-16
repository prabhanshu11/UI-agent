"""Cursor locator using frame-diff sweep and verify-before-click protocol.

Implements the ASMP (Actuator-Sensor Mapping Protocol) pattern for detecting
cursor position on an HDMI-captured screen when using an ESP32 relative mouse.

The core problem: ESP32 HID mouse sends relative movements across the entire
Windows virtual desktop. On multi-monitor setups, the cursor may be on a
different monitor than the one captured by HDMI. This module sweeps the cursor
across the virtual desktop, using frame subtraction to detect when the cursor
enters the HDMI-captured screen.

Usage:
    from src.hardware.cursor_locator import CursorLocator
    from src.hardware.esp32_mouse import ESP32Mouse
    from src.hardware.hdmi_sensor import HDMISensor

    mouse = ESP32Mouse('/dev/ttyUSB0')
    sensor = HDMISensor('/dev/video0')
    locator = CursorLocator(mouse, sensor)

    pos = locator.locate()
    if pos:
        locator.verified_click(target_x, target_y)
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .esp32_mouse import ESP32Mouse
from .hdmi_sensor import HDMISensor
from .cursor_detector import detect_cursor_movement, detect_cursor_after_movement


@dataclass
class CursorPosition:
    """Detected cursor position on the HDMI-captured screen."""
    x: int  # X on HDMI screen (0 to screen_width)
    y: int  # Y on HDMI screen (0 to screen_height)
    virtual_x_offset: int  # X offset in virtual desktop to reach this screen
    confidence: float


class CursorLocator:
    """Locates and verifies cursor position using ESP32 mouse + HDMI capture.

    Implements three protocols:
    1. locate() — sweep cursor across virtual desktop, detect via frame-diff
    2. shake() — wiggle cursor to make it visually obvious
    3. verified_click() — move, verify position via screenshot, then click
    """

    def __init__(
        self,
        mouse: ESP32Mouse,
        sensor: HDMISensor,
        probe_dx: int = 50,
        probe_dy: int = 50,
        sweep_step: int = 200,
        max_virtual_width: int = 5000,
        settle_time: float = 0.3,
    ):
        """
        Args:
            mouse: ESP32Mouse instance (connected to target).
            sensor: HDMISensor instance (capturing target's screen).
            probe_dx: Small X movement for frame-diff detection.
            probe_dy: Small Y movement for frame-diff detection.
            sweep_step: Pixels to move right between sweep probes.
            max_virtual_width: Maximum virtual desktop width to sweep.
            settle_time: Seconds to wait after movement before capture.
        """
        self.mouse = mouse
        self.sensor = sensor
        self.probe_dx = probe_dx
        self.probe_dy = probe_dy
        self.sweep_step = sweep_step
        self.max_virtual_width = max_virtual_width
        self.settle_time = settle_time

    def locate(self, verbose: bool = True) -> Optional[CursorPosition]:
        """Locate cursor on the HDMI-captured screen via sweep + frame-diff.

        Algorithm:
        1. Park cursor at far-left of virtual desktop
        2. Move cursor to vertical center
        3. Sweep right in sweep_step increments
        4. At each step, take before/after frames around a small probe move
        5. Frame-diff detects cursor presence
        6. When found, set ESP32Mouse estimated position accordingly

        Args:
            verbose: Print progress to stdout.

        Returns:
            CursorPosition if found, None if cursor never appeared on screen.
        """
        if verbose:
            print("Parking cursor at far-left of virtual desktop...")

        # Park at top-left of virtual desktop
        self.mouse._send_chunked(-self.max_virtual_width, -3000)
        time.sleep(0.5)

        # Move to vertical center (better chance of being in usable screen area)
        self.mouse._send_chunked(0, self.sensor.height // 2)
        time.sleep(self.settle_time)

        steps = self.max_virtual_width // self.sweep_step
        if verbose:
            print(f"Sweeping right: {steps} steps of {self.sweep_step}px each")

        for step in range(steps + 1):
            x_offset = step * self.sweep_step

            # Capture frame BEFORE probe move
            frame_before = self.sensor.capture()
            time.sleep(0.1)

            # Small known movement
            self.mouse._send_chunked(self.probe_dx, self.probe_dy)
            time.sleep(self.settle_time)

            # Capture frame AFTER probe move
            frame_after = self.sensor.capture()

            # Check frame diff
            detection = detect_cursor_movement(frame_before, frame_after)

            if detection and detection.confidence > 0.1:
                # Cursor is on this screen!
                # Undo the probe movement to restore position
                self.mouse._send_chunked(-self.probe_dx, -self.probe_dy)
                time.sleep(0.1)

                # The detection centroid is approximately where the cursor
                # was between the two frames. Offset by half probe for
                # pre-probe position.
                cursor_x = detection.x - self.probe_dx // 2
                cursor_y = detection.y - self.probe_dy // 2

                # Update mouse's estimated position to match reality
                self.mouse.set_position(cursor_x, cursor_y)

                result = CursorPosition(
                    x=cursor_x,
                    y=cursor_y,
                    virtual_x_offset=x_offset,
                    confidence=detection.confidence,
                )

                if verbose:
                    print(f"Cursor found at ({cursor_x}, {cursor_y}) "
                          f"[virtual offset: {x_offset}px, "
                          f"confidence: {detection.confidence:.2f}, "
                          f"pixels: {detection.pixel_count}]")

                return result

            # Undo probe, then sweep right
            self.mouse._send_chunked(-self.probe_dx, -self.probe_dy)
            time.sleep(0.1)
            self.mouse._send_chunked(self.sweep_step, 0)
            time.sleep(0.2)

            if verbose and step % 5 == 0:
                print(f"  Step {step}/{steps} (offset ~{x_offset}px) — not found yet")

        if verbose:
            print("Cursor NOT found on HDMI screen after full sweep.")
        return None

    def shake(self, amplitude: int = 30, shakes: int = 3):
        """Quick left-right shake to make cursor visually obvious.

        Useful for visual confirmation after locate().

        Args:
            amplitude: Pixels to move left/right per shake.
            shakes: Number of shake cycles.
        """
        for _ in range(shakes):
            self.mouse._send_raw(amplitude, 0)
            time.sleep(0.05)
            self.mouse._send_raw(-amplitude * 2, 0)
            time.sleep(0.05)
            self.mouse._send_raw(amplitude, 0)
            time.sleep(0.1)

    def verify_position(
        self,
        expected_x: int,
        expected_y: int,
        tolerance: int = 50,
    ) -> Optional[Tuple[int, int]]:
        """Verify cursor is near expected position using frame-diff.

        Takes a before/after pair around a small probe movement and checks
        if the detected cursor is near the expected position.

        Args:
            expected_x: Expected X position on HDMI screen.
            expected_y: Expected Y position on HDMI screen.
            tolerance: Pixel distance tolerance for match.

        Returns:
            (actual_x, actual_y) if cursor found near expected position,
            None if cursor not detected or too far from expected.
        """
        frame_before = self.sensor.capture()
        time.sleep(0.1)

        self.mouse._send_chunked(self.probe_dx, self.probe_dy)
        time.sleep(self.settle_time)

        frame_after = self.sensor.capture()

        # Undo probe
        self.mouse._send_chunked(-self.probe_dx, -self.probe_dy)
        time.sleep(0.1)

        result = detect_cursor_after_movement(
            frame_after, frame_before,
            self.probe_dx, self.probe_dy,
        )

        if result is None:
            return None

        actual_x, actual_y = result
        dist = ((actual_x - expected_x) ** 2 + (actual_y - expected_y) ** 2) ** 0.5

        if dist <= tolerance:
            return (actual_x, actual_y)
        return None

    def verified_click(
        self,
        x: int,
        y: int,
        button: str = "left",
        verify: bool = True,
        tolerance: int = 80,
        save_verification: Optional[str] = None,
    ) -> bool:
        """Move to position, optionally verify via frame-diff, then click.

        Args:
            x: Target X on HDMI screen.
            y: Target Y on HDMI screen.
            button: Mouse button ('left', 'right', 'middle').
            verify: If True, verify cursor position before clicking.
            tolerance: Pixel distance tolerance for verification.
            save_verification: If set, save verification frame to this path.

        Returns:
            True if click executed (and verification passed if enabled).
        """
        self.mouse.move_to(x, y)
        time.sleep(self.settle_time)

        if verify:
            actual = self.verify_position(x, y, tolerance)
            if actual is None:
                print(f"WARNING: Cursor verification failed at ({x}, {y}). "
                      f"Cursor not detected near target. Click aborted.")
                return False

            # Update estimated position with verified actual
            self.mouse.set_position(actual[0], actual[1])

            if save_verification:
                frame = self.sensor.capture()
                crop = self.sensor.crop(frame, x - 100, y - 100, 200, 200)
                self.sensor.save_frame(crop, save_verification)

        self.mouse.click(button)
        return True

    def screenshot_region(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        output_path: str,
    ):
        """Capture and crop a region from the HDMI screen.

        Useful for reading UI text or verifying visual state.

        Args:
            x: Left edge.
            y: Top edge.
            w: Width.
            h: Height.
            output_path: Where to save the cropped image.
        """
        frame = self.sensor.capture()
        crop = self.sensor.crop(frame, x, y, w, h)
        self.sensor.save_frame(crop, output_path)


# CLI interface for testing
if __name__ == "__main__":
    import sys

    print("Cursor Locator — ASMP Cursor Detection")
    print("=" * 40)

    device = "/dev/video0"
    serial_port = "/dev/ttyUSB0"

    # Parse optional args
    args = sys.argv[1:]
    if "--device" in args:
        device = args[args.index("--device") + 1]
    if "--serial" in args:
        serial_port = args[args.index("--serial") + 1]

    print(f"HDMI device: {device}")
    print(f"Serial port: {serial_port}")

    mouse = ESP32Mouse(serial_port, screen_width=1920, screen_height=1080)
    sensor = HDMISensor(device)
    locator = CursorLocator(mouse, sensor)

    print("\nChecking ESP32 mouse connection...")
    status = mouse.status()
    print(f"Mouse status: {status}")

    if status != "CONNECTED":
        print("ERROR: Mouse not connected. Aborting.")
        sys.exit(1)

    print("\nStarting cursor sweep...")
    pos = locator.locate(verbose=True)

    if pos:
        print(f"\nCursor located at ({pos.x}, {pos.y})")
        print(f"Virtual desktop offset: {pos.virtual_x_offset}px")
        print(f"Confidence: {pos.confidence:.2f}")
        print("\nShaking cursor for visual confirmation...")
        locator.shake()
        print("Done. Check HDMI screen for cursor wiggle.")
    else:
        print("\nCursor not found on HDMI screen.")
        print("Try: ensure cursor is not hidden, check HDMI input is active.")

    mouse.close()
