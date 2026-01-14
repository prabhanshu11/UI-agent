"""ESP32 BT HID mouse control with boundary detection.

Provides mouse control via ESP32 Bluetooth HID emulation with:
- Screen boundary detection to prevent cursor leaving target screen
- Position tracking (estimated, since HID mouse is relative-only)
- Chunked movement for values > 127 (HID limit)
- Circle and pattern movements
"""

import serial
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
        """Close serial connection."""
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
