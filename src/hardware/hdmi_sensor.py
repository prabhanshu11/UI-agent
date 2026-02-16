"""HDMI capture sensor for frame-based analysis.

Wraps ffmpeg V4L2 capture to provide numpy arrays for frame operations
like subtraction, diffing, and region cropping. Designed for use with
cursor_detector and cursor_locator in the ASMP pipeline.

Requires: ffmpeg, numpy, Pillow
Device: USB HDMI capture card at /dev/video0 (YUYV format)
"""

import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


class HDMISensor:
    """HDMI capture card sensor providing numpy frame arrays."""

    def __init__(
        self,
        device: str = "/dev/video0",
        width: int = 1920,
        height: int = 1080,
        input_format: str = "yuyv422",
    ):
        self.device = device
        self.width = width
        self.height = height
        self.input_format = input_format
        self._tmp_path = Path(tempfile.mkdtemp()) / "hdmi_frame.png"

    def capture(self, settle_frames: int = 3) -> np.ndarray:
        """Capture a frame and return as numpy array.

        Args:
            settle_frames: Number of frames to capture (uses last one).
                More frames = more stable but slower. 3 is good default.

        Returns:
            numpy array of shape (height, width, 3) in RGB.

        Raises:
            RuntimeError: If ffmpeg capture fails.
        """
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "v4l2",
                "-input_format", self.input_format,
                "-video_size", f"{self.width}x{self.height}",
                "-i", self.device,
                "-frames:v", str(settle_frames),
                "-update", "1",
                str(self._tmp_path),
            ],
            capture_output=True,
            timeout=10,
        )

        if not self._tmp_path.exists():
            raise RuntimeError(
                f"HDMI capture failed: {result.stderr[-200:].decode(errors='replace')}"
            )

        frame = np.array(Image.open(self._tmp_path).convert("RGB"))
        return frame

    def capture_to_file(self, output_path: str, settle_frames: int = 3) -> bool:
        """Capture a frame and save to file.

        Args:
            output_path: Where to save the PNG.
            settle_frames: Number of frames to capture.

        Returns:
            True if capture succeeded.
        """
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "v4l2",
                "-input_format", self.input_format,
                "-video_size", f"{self.width}x{self.height}",
                "-i", self.device,
                "-frames:v", str(settle_frames),
                "-update", "1",
                output_path,
            ],
            capture_output=True,
            timeout=10,
        )
        return Path(output_path).exists()

    def crop(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> np.ndarray:
        """Crop a region from a frame.

        Args:
            frame: Source frame array.
            x: Left edge of crop region.
            y: Top edge of crop region.
            w: Width of crop region.
            h: Height of crop region.

        Returns:
            Cropped numpy array.
        """
        # Clamp to frame bounds
        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        return frame[y:y + h, x:x + w]

    def save_frame(self, frame: np.ndarray, output_path: str):
        """Save a numpy frame to an image file.

        Args:
            frame: numpy array (H, W, 3) RGB.
            output_path: Path to save (PNG, JPG, etc.).
        """
        Image.fromarray(frame).save(output_path)
