"""USB HDMI capture card interface for multi-screen capture.

Supports MacroSilicon MS2109 and compatible devices.
Handles USB power management and V4L2 video capture.
"""

import subprocess
import re
from pathlib import Path
from typing import Optional, Tuple


class USBCaptureCard:
    """Interface for USB HDMI capture devices (e.g., Pibox/MacroSilicon MS2109)."""

    # Known USB IDs for MacroSilicon capture cards
    VENDOR_ID = "534d"
    PRODUCT_ID = "2109"

    def __init__(self, device_path: Optional[str] = None):
        """Initialize USB capture card interface.

        Args:
            device_path: Path to V4L2 device (e.g., /dev/video4).
                        If None, will auto-detect.
        """
        self.device_path = device_path or self._detect_device()
        self.usb_bus_port = None

        if self.device_path:
            self.usb_bus_port = self._get_usb_bus_port()

    def _detect_device(self) -> Optional[str]:
        """Auto-detect MacroSilicon USB capture card device.

        Returns:
            Path to V4L2 device or None if not found.
        """
        try:
            # List all video devices with their names
            result = subprocess.run(
                ["v4l2-ctl", "--list-devices"],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse output to find USB3.0 capture device
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if "USB3. 0 capture" in line or "MacroSilicon" in line:
                    # Next lines contain device paths
                    if i + 1 < len(lines):
                        device_match = re.search(r'/dev/video\d+', lines[i + 1])
                        if device_match:
                            return device_match.group(0)

            return None

        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    def _get_usb_bus_port(self) -> Optional[str]:
        """Get USB bus-port identifier for this device.

        Returns:
            Bus-port string (e.g., "1-3") or None.
        """
        if not self.device_path:
            return None

        try:
            # Get device number from path (e.g., video4 -> 4)
            device_num = self.device_path.split('video')[-1]

            # Find USB device path via sysfs
            video_dev_path = Path(f"/sys/class/video4linux/video{device_num}/device")
            if not video_dev_path.exists():
                return None

            # Resolve to USB device
            usb_device = video_dev_path.resolve()

            # Extract bus-port (e.g., /devices/pci.../usb1/1-3 -> 1-3)
            match = re.search(r'/(\d+-\d+)(?:/|$)', str(usb_device))
            if match:
                return match.group(1)

            return None

        except Exception:
            return None

    def disable_autosuspend(self) -> bool:
        """Disable USB autosuspend for this device.

        Critical for MacroSilicon cards to prevent screen flickering.
        Requires sudo privileges.

        Returns:
            True if successful, False otherwise.
        """
        if not self.usb_bus_port:
            return False

        control_path = f"/sys/bus/usb/devices/{self.usb_bus_port}/power/control"

        try:
            subprocess.run(
                ["sudo", "tee", control_path],
                input=b"on",
                check=True,
                capture_output=True
            )
            return True

        except subprocess.CalledProcessError:
            return False

    def get_capabilities(self) -> dict:
        """Get device capabilities (formats, resolutions, framerates).

        Returns:
            Dict with device capabilities or empty dict on error.
        """
        if not self.device_path:
            return {}

        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", self.device_path, "--list-formats-ext"],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse output to extract capabilities
            capabilities = {
                "device": self.device_path,
                "formats": [],
                "raw_output": result.stdout
            }

            # Extract formats (MJPG, YUYV, etc.)
            format_matches = re.findall(r"\[\d+\]: '(\w+)'", result.stdout)
            capabilities["formats"] = list(set(format_matches))

            return capabilities

        except (subprocess.CalledProcessError, FileNotFoundError):
            return {}

    def capture_frame(
        self,
        output_path: str,
        width: int = 1920,
        height: int = 1080,
        input_format: str = "mjpeg"
    ) -> Tuple[bool, str]:
        """Capture a single frame from the device.

        Args:
            output_path: Where to save the captured frame.
            width: Frame width in pixels.
            height: Frame height in pixels.
            input_format: V4L2 input format (mjpeg or yuyv).

        Returns:
            Tuple of (success: bool, message: str).
        """
        if not self.device_path:
            return False, "No device detected"

        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-f", "v4l2",
                    "-input_format", input_format,
                    "-video_size", f"{width}x{height}",
                    "-i", self.device_path,
                    "-update", "1",
                    "-frames:v", "1",
                    "-y",
                    output_path
                ],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                return True, f"Frame captured to {output_path}"
            else:
                return False, f"FFmpeg error: {result.stderr[-200:]}"

        except subprocess.TimeoutExpired:
            return False, "Capture timed out"
        except FileNotFoundError:
            return False, "ffmpeg not found - install it first"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def __repr__(self) -> str:
        return f"USBCaptureCard(device={self.device_path}, bus_port={self.usb_bus_port})"
