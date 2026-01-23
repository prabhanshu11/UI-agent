"""Screenshot Extractor Agent - Extracts keyframes from video recordings.

Monitors task.md for work, extracts frames based on strategies,
writes results to output.md.
"""

import os
import subprocess
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import yaml

from PIL import Image
import numpy as np


# Supported image formats for vision models
SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
PREFERRED_FORMAT = 'PNG'  # Best for screenshots - lossless


def validate_image(image_path: Path) -> Tuple[bool, str]:
    """Validate image is in a format supported by vision models.

    Args:
        image_path: Path to image file

    Returns:
        (is_valid, error_message) tuple
    """
    if not image_path.exists():
        return False, f"File does not exist: {image_path}"

    # Check extension
    if image_path.suffix.lower() not in SUPPORTED_FORMATS:
        return False, f"Unsupported format: {image_path.suffix}. Use: {SUPPORTED_FORMATS}"

    # Verify it's actually a valid image
    try:
        with Image.open(image_path) as img:
            img.verify()
        # Re-open after verify (verify closes the file)
        with Image.open(image_path) as img:
            # Check dimensions are reasonable
            if img.width < 10 or img.height < 10:
                return False, f"Image too small: {img.width}x{img.height}"
            if img.width > 10000 or img.height > 10000:
                return False, f"Image too large: {img.width}x{img.height}"
            # Check mode is supported
            if img.mode not in ('RGB', 'RGBA', 'L', 'P'):
                return False, f"Unsupported image mode: {img.mode}"
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

    return True, ""


def convert_to_png(image_path: Path, output_path: Optional[Path] = None) -> Path:
    """Convert image to PNG format for consistent handling.

    Args:
        image_path: Source image path
        output_path: Destination path (default: same name with .png)

    Returns:
        Path to converted image
    """
    if output_path is None:
        output_path = image_path.with_suffix('.png')

    with Image.open(image_path) as img:
        # Convert to RGB if necessary (handles RGBA, P mode)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        img.save(output_path, 'PNG')

    return output_path


class ScreenshotExtractor:
    """Extract screenshots from video files based on various strategies."""

    def __init__(self, agent_dir: Path):
        """Initialize the extractor.

        Args:
            agent_dir: Path to agent directory (contains task.md, output.md)
        """
        self.agent_dir = Path(agent_dir)
        self.task_file = self.agent_dir / "task.md"
        self.output_file = self.agent_dir / "output.md"
        self.last_frame_hash: Optional[str] = None

    def check_for_task(self) -> Optional[dict]:
        """Check if there's a task to process.

        Returns:
            Task dict if task.md exists and has pending work, None otherwise.
        """
        if not self.task_file.exists():
            return None

        content = self.task_file.read_text()

        # Parse YAML front matter
        if "```yaml" in content:
            yaml_block = content.split("```yaml")[1].split("```")[0]
            try:
                task = yaml.safe_load(yaml_block)
                if task.get("status") == "pending":
                    return task
            except yaml.YAMLError:
                pass

        return None

    def extract_interval_frames(
        self,
        video_path: Path,
        output_dir: Path,
        interval_seconds: float = 30.0,
        max_frames: int = 100
    ) -> list[Path]:
        """Extract frames at regular intervals.

        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            interval_seconds: Seconds between frames
            max_frames: Maximum frames to extract

        Returns:
            List of paths to extracted frames
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = []

        # Get video duration
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, text=True
        )

        try:
            duration = float(result.stdout.strip())
        except ValueError:
            return frames

        # Extract frames
        timestamp = 0.0
        frame_num = 0

        while timestamp < duration and frame_num < max_frames:
            output_path = output_dir / f"frame_{frame_num:04d}_{int(timestamp):05d}s.png"

            subprocess.run(
                ["ffmpeg", "-ss", str(timestamp), "-i", str(video_path),
                 "-frames:v", "1", "-y", str(output_path)],
                capture_output=True
            )

            if output_path.exists():
                frames.append(output_path)

            timestamp += interval_seconds
            frame_num += 1

        return frames

    def extract_change_frames(
        self,
        video_path: Path,
        output_dir: Path,
        threshold: float = 0.15,
        min_interval: float = 2.0,
        max_frames: int = 100
    ) -> list[tuple[Path, float, float]]:
        """Extract frames when significant change is detected.

        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            threshold: Change threshold (0-1, fraction of pixels different)
            min_interval: Minimum seconds between frames
            max_frames: Maximum frames to extract

        Returns:
            List of (path, timestamp, change_ratio) tuples
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        frames = []

        # Get video info
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate,duration",
             "-of", "default=noprint_wrappers=1", str(video_path)],
            capture_output=True, text=True
        )

        # Parse frame rate
        fps = 20  # default
        duration = 60  # default
        for line in result.stdout.split('\n'):
            if line.startswith("r_frame_rate="):
                rate = line.split('=')[1]
                if '/' in rate:
                    num, den = rate.split('/')
                    fps = float(num) / float(den)
            elif line.startswith("duration="):
                try:
                    duration = float(line.split('=')[1])
                except ValueError:
                    pass

        # Sample at lower rate for change detection
        sample_interval = min_interval / 2  # Check twice per min_interval
        timestamp = 0.0
        last_image: Optional[np.ndarray] = None
        last_capture_time = -min_interval  # Allow first frame
        frame_num = 0

        while timestamp < duration and frame_num < max_frames:
            # Extract frame to temp file
            temp_path = output_dir / "_temp_frame.png"
            subprocess.run(
                ["ffmpeg", "-ss", str(timestamp), "-i", str(video_path),
                 "-frames:v", "1", "-y", str(temp_path)],
                capture_output=True
            )

            if temp_path.exists():
                # Load and compare
                img = Image.open(temp_path)
                img_array = np.array(img.convert('L'))  # Grayscale

                if last_image is not None:
                    # Calculate change
                    diff = np.abs(img_array.astype(float) - last_image.astype(float))
                    change_ratio = np.mean(diff > 30) # Pixels that changed by >30

                    if change_ratio > threshold and (timestamp - last_capture_time) >= min_interval:
                        # Significant change - save frame
                        output_path = output_dir / f"keyframe_{frame_num:04d}_{int(timestamp):05d}s.png"
                        temp_path.rename(output_path)
                        frames.append((output_path, timestamp, change_ratio))
                        last_capture_time = timestamp
                        frame_num += 1
                else:
                    # First frame - always save
                    output_path = output_dir / f"keyframe_{frame_num:04d}_{int(timestamp):05d}s.png"
                    temp_path.rename(output_path)
                    frames.append((output_path, timestamp, 1.0))
                    last_capture_time = timestamp
                    frame_num += 1

                last_image = img_array

                # Clean up temp if not saved
                if temp_path.exists():
                    temp_path.unlink()

            timestamp += sample_interval

        return frames

    def capture_live_frame(
        self,
        device: str = "/dev/video4",
        output_path: Optional[Path] = None
    ) -> tuple[bool, Optional[Path], Optional[str]]:
        """Capture a single frame from live video device.

        Args:
            device: V4L2 device path
            output_path: Where to save (auto-generates if None)

        Returns:
            (success, path, frame_hash) tuple
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.agent_dir / "captures" / f"live_{timestamp}.png"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["ffmpeg", "-f", "v4l2", "-input_format", "mjpeg",
             "-video_size", "1920x1080", "-i", device,
             "-update", "1", "-frames:v", "1", "-y", str(output_path)],
            capture_output=True, timeout=10
        )

        if result.returncode == 0 and output_path.exists():
            # Calculate hash for change detection
            frame_hash = hashlib.md5(output_path.read_bytes()).hexdigest()[:8]
            return True, output_path, frame_hash

        return False, None, None

    def is_frame_different(self, new_hash: str) -> bool:
        """Check if frame is different from last captured.

        Args:
            new_hash: Hash of new frame

        Returns:
            True if different or first frame
        """
        if self.last_frame_hash is None:
            self.last_frame_hash = new_hash
            return True

        is_different = new_hash != self.last_frame_hash
        self.last_frame_hash = new_hash
        return is_different

    def write_output(self, results: dict):
        """Write results to output.md.

        Args:
            results: Dict with extraction results
        """
        timestamp = datetime.now().isoformat()

        output = f"""# Screenshot Extraction Results

**Generated**: {timestamp}
**Status**: completed

## Summary

- **Total frames extracted**: {results.get('frame_count', 0)}
- **Method**: {results.get('method', 'unknown')}
- **Source**: {results.get('source', 'unknown')}

## Frames

```yaml
frames:
"""

        for frame in results.get('frames', []):
            if isinstance(frame, tuple):
                path, ts, change = frame
                output += f"""  - path: "{path}"
    timestamp: {ts}
    change_ratio: {change:.3f}
"""
            else:
                output += f"""  - path: "{frame}"
"""

        output += "```\n"

        self.output_file.write_text(output)

    def run_once(self) -> bool:
        """Check for task and process if available.

        Returns:
            True if task was processed
        """
        task = self.check_for_task()
        if not task:
            return False

        # Process task
        video_path = Path(task.get('video_path', ''))
        output_dir = Path(task.get('output_dir', self.agent_dir / 'output'))
        method = task.get('method', 'interval')

        if method == 'interval':
            frames = self.extract_interval_frames(
                video_path, output_dir,
                interval_seconds=task.get('interval', 30.0)
            )
            results = {
                'frame_count': len(frames),
                'method': 'interval',
                'source': str(video_path),
                'frames': frames
            }
        elif method == 'change':
            frames = self.extract_change_frames(
                video_path, output_dir,
                threshold=task.get('threshold', 0.15),
                min_interval=task.get('min_interval', 2.0)
            )
            results = {
                'frame_count': len(frames),
                'method': 'change_detection',
                'source': str(video_path),
                'frames': frames
            }
        else:
            return False

        self.write_output(results)

        # Mark task complete
        task_content = self.task_file.read_text()
        task_content = task_content.replace('status: pending', 'status: completed')
        self.task_file.write_text(task_content)

        return True


if __name__ == "__main__":
    # Test the extractor
    import sys

    agent_dir = Path(__file__).parent.parent.parent / "agents" / "screenshot-extractor"
    extractor = ScreenshotExtractor(agent_dir)

    # Quick test - capture live frame
    success, path, hash = extractor.capture_live_frame()
    if success:
        print(f"Captured: {path} (hash: {hash})")
    else:
        print("Capture failed")
