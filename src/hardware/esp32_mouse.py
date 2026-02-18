"""ESP32 BT HID mouse control with boundary detection.

Provides mouse control via ESP32 Bluetooth HID emulation with:
- Screen boundary detection to prevent cursor leaving target screen
- Position tracking (estimated, since HID mouse is relative-only)
- Chunked movement for values > 127 (HID limit)
- Circle and pattern movements

MULTI-MONITOR WARNING:
    This class assumes screen_width/height maps to a single monitor. On Windows
    with multiple monitors, HID relative movements operate on the entire virtual
    desktop (e.g. 3840x1080 for two side-by-side 1920x1080 monitors).

    calibrate_to_corner() will park the cursor at the virtual desktop edge, NOT
    necessarily the HDMI-captured monitor. If the captured display is the right
    monitor, the cursor ends up on the wrong screen entirely.

    For multi-monitor setups, use cursor_locator.locate_cursor() which performs
    a frame-diff sweep across the virtual desktop to find the cursor on the
    HDMI-captured screen, then calls set_position() with the correct offset.

    See also: docs/MOUSE_CONTROL_LEARNINGS.md
"""

import serial
import threading
import time
import math
import random
from typing import Optional, Tuple


class ESP32Mouse:
    """ESP32 BT HID mouse control with boundary detection."""

    # HID mouse reports limited to -127 to 127 per axis
    HID_MAX = 127

    def __init__(
        self,
        serial_port: str = '/dev/ttyUSB0',
        baud_rate: int = 115200,
        screen_width: int = 1920,
        screen_height: int = 1080,
        buffer_percent: float = 0.5  # 0.5% buffer to prevent edge escape
    ):
        """Initialize ESP32 mouse controller.

        Args:
            serial_port: Serial port for ESP32 connection.
            baud_rate: Serial baud rate.
            screen_width: Target screen width in pixels.
            screen_height: Target screen height in pixels.
            buffer_percent: Percentage buffer from screen edge (0.5% = ~10px).
        """
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Calculate buffer in pixels (0.5% of each dimension)
        self.buffer_x = int(screen_width * buffer_percent / 100)
        self.buffer_y = int(screen_height * buffer_percent / 100)

        # Estimated cursor position (start at center)
        self.estimated_x = screen_width // 2
        self.estimated_y = screen_height // 2

        # Serial connection (lazy init)
        self._ser: Optional[serial.Serial] = None

    @property
    def ser(self) -> serial.Serial:
        """Get serial connection, opening if needed."""
        if self._ser is None or not self._ser.is_open:
            self._ser = serial.Serial(self.serial_port, self.baud_rate, timeout=2)
            self._ser.reset_input_buffer()
            time.sleep(0.3)
        return self._ser

    def close(self):
        """Close serial connection and stop heartbeat/anti-sleep."""
        self.stop_heartbeat()
        self.stop_anti_sleep()
        if self._ser and self._ser.is_open:
            self._ser.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def status(self) -> str:
        """Check ESP32 connection status.

        Returns:
            Status string (e.g., "CONNECTED" or "DISCONNECTED").
        """
        self.ser.write(b'STATUS\n')
        self.ser.flush()
        time.sleep(0.3)

        status = ""
        while self.ser.in_waiting:
            line = self.ser.readline().decode().strip()
            if "Mouse:" in line:
                status = line.split("Mouse:")[-1].strip()
        return status

    def is_connected(self) -> bool:
        """Check if mouse is connected to target device."""
        return self.status() == "CONNECTED"

    def reconnect(self, timeout: float = 5.0) -> bool:
        """Reset ESP32 via DTR toggle and wait for BLE reconnect.

        The ESP32 BLE HID connection goes stale within ~1s of no activity.
        When stale, STATUS still reports CONNECTED but cursor doesn't move.
        A DTR toggle forces a hardware reset → fresh BLE pairing.

        Args:
            timeout: Max seconds to wait for CONNECTED status after reset.

        Returns:
            True if reconnected successfully.
        """
        ser = self.ser
        ser.dtr = False
        time.sleep(0.1)
        ser.dtr = True
        ser.reset_input_buffer()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.5)
            try:
                if self.is_connected():
                    return True
            except serial.SerialException:
                pass
        return False

    def start_heartbeat(self, interval: float = 0.1):
        """Start daemon thread sending zero-movement reports to keep BLE alive.

        Sends MOUSE:0,0 every `interval` seconds. This prevents the BLE HID
        connection from going stale without actually moving the cursor.

        Args:
            interval: Seconds between heartbeat reports (0.1 = 100ms).
        """
        if hasattr(self, '_heartbeat_stop') and not self._heartbeat_stop.is_set():
            return  # Already running

        self._heartbeat_stop = threading.Event()

        def _beat():
            while not self._heartbeat_stop.is_set():
                try:
                    self.ser.write(b'MOUSE:0,0\n')
                    self.ser.flush()
                except (serial.SerialException, OSError):
                    break
                self._heartbeat_stop.wait(interval)

        self._heartbeat_thread = threading.Thread(target=_beat, daemon=True)
        self._heartbeat_thread.start()

    def stop_heartbeat(self):
        """Stop the heartbeat daemon thread."""
        if hasattr(self, '_heartbeat_stop'):
            self._heartbeat_stop.set()
            if hasattr(self, '_heartbeat_thread'):
                self._heartbeat_thread.join(timeout=2)

    def start_anti_sleep(self, interval: float = 30.0, pixels: int = 3):
        """Start daemon thread sending real mouse jitter to prevent Windows sleep.

        Unlike heartbeat (MOUSE:0,0 which keeps BLE alive but doesn't move cursor),
        this sends actual ±Npx movement every `interval` seconds so Windows resets
        its idle timer and doesn't enter sleep mode.

        The movement is: right N px, pause 100ms, left N px — net zero displacement.

        Args:
            interval: Seconds between jitter bursts (30s default).
            pixels: Jitter amplitude in pixels (3px default — invisible to user).
        """
        if hasattr(self, '_anti_sleep_stop') and not self._anti_sleep_stop.is_set():
            return  # Already running

        self._anti_sleep_stop = threading.Event()
        self._last_jitter_time = 0.0
        self._jitter_pixels = pixels

        def _jitter():
            while not self._anti_sleep_stop.is_set():
                try:
                    self.ser.write(f'MOUSE:{pixels},0\n'.encode())
                    self.ser.flush()
                    self._last_jitter_time = time.monotonic()
                    time.sleep(0.1)
                    self.ser.write(f'MOUSE:{-pixels},0\n'.encode())
                    self.ser.flush()
                except (serial.SerialException, OSError):
                    break
                self._anti_sleep_stop.wait(interval)

        self._anti_sleep_thread = threading.Thread(target=_jitter, daemon=True)
        self._anti_sleep_thread.start()

    def stop_anti_sleep(self):
        """Stop the anti-sleep jitter daemon thread."""
        if hasattr(self, '_anti_sleep_stop'):
            self._anti_sleep_stop.set()
            if hasattr(self, '_anti_sleep_thread'):
                self._anti_sleep_thread.join(timeout=2)

    def _send_raw(self, dx: int, dy: int):
        """Send raw mouse movement command (single HID report).

        Args:
            dx: X movement (-127 to 127).
            dy: Y movement (-127 to 127).
        """
        dx = max(-self.HID_MAX, min(self.HID_MAX, dx))
        dy = max(-self.HID_MAX, min(self.HID_MAX, dy))

        self.ser.write(f'MOUSE:{dx},{dy}\n'.encode())
        self.ser.flush()
        time.sleep(0.02)

    def _send_chunked(self, dx: int, dy: int):
        """Send movement in chunks if values exceed HID limits.

        Args:
            dx: Total X movement (any value).
            dy: Total Y movement (any value).
        """
        while dx != 0 or dy != 0:
            chunk_dx = max(-self.HID_MAX, min(self.HID_MAX, dx))
            chunk_dy = max(-self.HID_MAX, min(self.HID_MAX, dy))

            self._send_raw(chunk_dx, chunk_dy)

            dx -= chunk_dx
            dy -= chunk_dy

    def move(self, dx: int, dy: int, respect_bounds: bool = True) -> Tuple[int, int]:
        """Move mouse with optional boundary clamping.

        Args:
            dx: Desired X movement in pixels.
            dy: Desired Y movement in pixels.
            respect_bounds: If True, clamp movement to screen boundaries.

        Returns:
            Tuple of (actual_dx, actual_dy) after any clamping.
        """
        if respect_bounds:
            # Calculate new position
            new_x = self.estimated_x + dx
            new_y = self.estimated_y + dy

            # Clamp to screen bounds (with small buffer)
            new_x = max(self.buffer_x, min(self.screen_width - self.buffer_x, new_x))
            new_y = max(self.buffer_y, min(self.screen_height - self.buffer_y, new_y))

            # Calculate actual movement after clamping
            dx = new_x - self.estimated_x
            dy = new_y - self.estimated_y

        # Send the movement
        self._send_chunked(dx, dy)

        # Update estimated position
        self.estimated_x += dx
        self.estimated_y += dy

        return dx, dy

    def move_to(self, x: int, y: int, respect_bounds: bool = True) -> Tuple[int, int]:
        """Move cursor to estimated absolute position.

        Note: Since HID mouse is relative-only, this relies on estimated position.
        Use calibrate() first for accuracy.

        Args:
            x: Target X position.
            y: Target Y position.
            respect_bounds: If True, clamp to screen boundaries.

        Returns:
            Tuple of (actual_dx, actual_dy) movement made.
        """
        dx = x - self.estimated_x
        dy = y - self.estimated_y
        return self.move(dx, dy, respect_bounds)

    def click(self, button: str = 'left'):
        """Send mouse click.

        Args:
            button: Button to click ('left', 'right', 'middle').
        """
        self.ser.write(f'CLICK:{button}\n'.encode())
        self.ser.flush()
        time.sleep(0.05)

    def circle(
        self,
        radius: int = 200,
        steps: int = 24,
        rotations: int = 1,
        jitter: int = 0,
        delay: float = 0.05
    ):
        """Draw a circle pattern.

        WARNING: This uses estimated position. Use safe_circle() for guaranteed
        on-screen movement.

        Args:
            radius: Circle radius in pixels (200+ recommended for visibility).
            steps: Number of segments (24 = 15° per step, good visibility).
            rotations: Number of complete circles.
            jitter: Random jitter +/- pixels for natural movement.
            delay: Delay between movements in seconds.
        """
        for rotation in range(rotations):
            for i in range(steps):
                angle1 = (2 * math.pi * i) / steps
                angle2 = (2 * math.pi * (i + 1)) / steps

                dx = radius * (math.cos(angle2) - math.cos(angle1))
                dy = radius * (math.sin(angle2) - math.sin(angle1))

                # Add jitter if requested
                if jitter > 0:
                    dx += random.randint(-jitter, jitter)
                    dy += random.randint(-jitter, jitter)

                dx = int(dx)
                dy = int(dy)

                # Use move() to respect boundaries
                self.move(dx, dy, respect_bounds=True)

                # Variable delay for natural movement
                if jitter > 0:
                    time.sleep(delay + random.uniform(-0.01, 0.02))
                else:
                    time.sleep(delay)

    def safe_circle(
        self,
        radius: int = 200,
        steps: int = 24,
        rotations: int = 1,
        delay: float = 0.05
    ):
        """Draw a circle guaranteed to stay on screen.

        This function:
        1. First pushes cursor to top-left corner (known position)
        2. Moves to screen center
        3. Draws circle with radius clamped to fit within screen

        Args:
            radius: Desired circle radius (will be reduced if too large).
            steps: Number of segments.
            rotations: Number of complete circles.
            delay: Delay between movements.
        """
        # Step 1: Force cursor to top-left corner
        # Send massive negative movement to guarantee hitting corner
        print("Calibrating to top-left corner...")
        overshoot = 3000  # More than any reasonable screen
        self._send_chunked(-overshoot, -overshoot)
        time.sleep(0.3)

        # Now we KNOW cursor is at top-left (0,0 clamped by Windows)
        self.estimated_x = 0
        self.estimated_y = 0

        # Step 2: Move to screen center
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        print(f"Moving to center ({center_x}, {center_y})...")
        self._send_chunked(center_x, center_y)
        time.sleep(0.2)

        self.estimated_x = center_x
        self.estimated_y = center_y

        # Step 3: Clamp radius to fit within screen from center
        max_radius_x = min(center_x, self.screen_width - center_x) - 50  # 50px margin
        max_radius_y = min(center_y, self.screen_height - center_y) - 50
        max_radius = min(max_radius_x, max_radius_y)

        if radius > max_radius:
            print(f"Reducing radius from {radius} to {max_radius} to fit screen")
            radius = max_radius

        # Step 4: Draw circle (no boundary checking needed - we're centered)
        print(f"Drawing circle: radius={radius}, steps={steps}, rotations={rotations}")
        for rotation in range(rotations):
            for i in range(steps):
                angle1 = (2 * math.pi * i) / steps
                angle2 = (2 * math.pi * (i + 1)) / steps

                dx = int(radius * (math.cos(angle2) - math.cos(angle1)))
                dy = int(radius * (math.sin(angle2) - math.sin(angle1)))

                self._send_raw(dx, dy)
                time.sleep(delay)

        print("Circle complete!")

    def calibrate_to_corner(self, corner: str = 'top_left'):
        """Calibrate position by moving to a known corner.

        Sends large movements to ensure cursor hits screen edge,
        then sets estimated position to that corner.

        Args:
            corner: Which corner ('top_left', 'top_right', 'bottom_left', 'bottom_right').
        """
        # Move far beyond screen bounds to ensure we hit the edge
        overshoot = max(self.screen_width, self.screen_height) + 500

        if corner == 'top_left':
            self._send_chunked(-overshoot, -overshoot)
            self.estimated_x = self.buffer_x
            self.estimated_y = self.buffer_y
        elif corner == 'top_right':
            self._send_chunked(overshoot, -overshoot)
            self.estimated_x = self.screen_width - self.buffer_x
            self.estimated_y = self.buffer_y
        elif corner == 'bottom_left':
            self._send_chunked(-overshoot, overshoot)
            self.estimated_x = self.buffer_x
            self.estimated_y = self.screen_height - self.buffer_y
        elif corner == 'bottom_right':
            self._send_chunked(overshoot, overshoot)
            self.estimated_x = self.screen_width - self.buffer_x
            self.estimated_y = self.screen_height - self.buffer_y

    def get_position(self) -> Tuple[int, int]:
        """Get estimated cursor position.

        Returns:
            Tuple of (x, y) estimated position.
        """
        return self.estimated_x, self.estimated_y

    def set_position(self, x: int, y: int):
        """Manually set estimated position (e.g., after visual verification).

        Args:
            x: Actual X position from screenshot analysis.
            y: Actual Y position from screenshot analysis.
        """
        self.estimated_x = x
        self.estimated_y = y


# CLI interface for testing
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python esp32_mouse.py <command> [args...]")
        print("Commands:")
        print("  status              - Check connection status")
        print("  move <dx> <dy>      - Move relative")
        print("  click [button]      - Click (left/right/middle)")
        print("  circle [radius]     - Draw circle")
        print("  calibrate [corner]  - Calibrate to corner")
        sys.exit(1)

    cmd = sys.argv[1]

    with ESP32Mouse() as mouse:
        if cmd == 'status':
            print(f"Mouse: {mouse.status()}")
            print(f"Estimated position: {mouse.get_position()}")

        elif cmd == 'move':
            dx = int(sys.argv[2])
            dy = int(sys.argv[3])
            actual = mouse.move(dx, dy)
            print(f"Moved: {actual}, Position: {mouse.get_position()}")

        elif cmd == 'click':
            button = sys.argv[2] if len(sys.argv) > 2 else 'left'
            mouse.click(button)
            print(f"Clicked: {button}")

        elif cmd == 'circle':
            radius = int(sys.argv[2]) if len(sys.argv) > 2 else 200
            print(f"Drawing circle with radius {radius}...")
            mouse.circle(radius=radius, steps=24, rotations=2)
            print("Done")

        elif cmd == 'calibrate':
            corner = sys.argv[2] if len(sys.argv) > 2 else 'top_left'
            print(f"Calibrating to {corner}...")
            mouse.calibrate_to_corner(corner)
            print(f"Position set to: {mouse.get_position()}")

        else:
            print(f"Unknown command: {cmd}")
