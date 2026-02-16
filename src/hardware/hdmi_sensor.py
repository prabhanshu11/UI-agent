"""HDMI capture sensor for frame-based analysis.

Wraps ffmpeg V4L2 capture to provide numpy arrays for frame operations
like subtraction, diffing, and region cropping. Also supports continuous
video recording for motion-based cursor detection.

Requires: ffmpeg, numpy, Pillow
Device: USB HDMI capture card at /dev/video0 (YUYV format)
"""

import os
import signal
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

    def start_recording(
        self,
        output_path: str,
        fps: int = 30,
        input_format: str = "mjpeg",
    ) -> subprocess.Popen:
        """Start ffmpeg video recording as a background process.

        Uses MJPEG input for lower latency than YUYV. Writes to an MKV
        container which handles incomplete writes gracefully (important
        since we stop recording mid-stream).

        Args:
            output_path: Path for output video file (e.g. /tmp/cursor_detect.mkv).
            fps: Capture framerate.
            input_format: V4L2 input format (mjpeg recommended for video).

        Returns:
            subprocess.Popen handle. Caller must stop via stop_recording().
        """
        cmd = [
            "ffmpeg", "-y",
            "-f", "v4l2",
            "-input_format", input_format,
            "-framerate", str(fps),
            "-video_size", f"{self.width}x{self.height}",
            "-i", self.device,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", "18",
            output_path,
        ]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return proc

    def stop_recording(self, proc: subprocess.Popen):
        """Stop a running ffmpeg recording gracefully.

        Sends 'q' to ffmpeg's stdin (its standard quit signal) so it
        flushes the container trailer and produces a valid file.
        Falls back to SIGINT if stdin write fails.

        Args:
            proc: The Popen handle from start_recording().
        """
        if proc.poll() is not None:
            return  # Already exited

        try:
            proc.stdin.write(b"q")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            proc.send_signal(signal.SIGINT)

        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def load_video_frames(self, video_path: str) -> list[np.ndarray]:
        """Extract all frames from a video file as numpy RGB arrays.

        Uses ffmpeg to decode the video and output raw RGB frames to a pipe,
        then reshapes into numpy arrays. This avoids writing individual PNGs
        to disk which is much faster.

        Args:
            video_path: Path to video file (MKV, MP4, etc.).

        Returns:
            List of numpy arrays, each (height, width, 3) in RGB.

        Raises:
            RuntimeError: If ffmpeg fails to decode the video.
        """
        # Probe video dimensions (use our known dimensions as default)
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-v", "error",
            "pipe:1",
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=30)

        if result.returncode != 0 and len(result.stdout) == 0:
            raise RuntimeError(
                f"Failed to decode video: {result.stderr[-300:].decode(errors='replace')}"
            )

        raw = result.stdout
        frame_size = self.width * self.height * 3

        if len(raw) < frame_size:
            return []

        n_frames = len(raw) // frame_size
        frames = []
        for i in range(n_frames):
            frame_bytes = raw[i * frame_size : (i + 1) * frame_size]
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                (self.height, self.width, 3)
            )
            frames.append(frame)

        return frames

    def save_frame(self, frame: np.ndarray, output_path: str):
        """Save a numpy frame to an image file.

        Args:
            frame: numpy array (H, W, 3) RGB.
            output_path: Path to save (PNG, JPG, etc.).
        """
        Image.fromarray(frame).save(output_path)
