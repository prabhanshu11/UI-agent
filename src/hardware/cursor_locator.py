"""Cursor locator via Lissajous curve + video motion analysis.

Locates the mouse cursor on an HDMI-captured screen when controlling a
Windows machine via an ESP32 relative HID mouse. The machine may have
any number of monitors in any arrangement — we don't know which one the
HDMI capture card sees or where the cursor currently is.

Approach:
1. RECORD: ffmpeg captures HDMI to video file continuously
2. MOVE: ESP32 sends HID reports tracing a Lissajous curve across the
   entire virtual desktop (space-filling, smooth, analytically known)
3. ANALYZE: periodically extract frames from the growing video and run
   curvature matching to detect the Lissajous signature in motion blobs

When the analyze step detects the curve, all threads stop. The cursor's
position is the centroid of the last matched motion blob.

Usage:
    from src.hardware.cursor_locator import CursorLocator
    from src.hardware.esp32_mouse import ESP32Mouse
    from src.hardware.hdmi_sensor import HDMISensor

    mouse = ESP32Mouse('/dev/ttyUSB0')
    sensor = HDMISensor('/dev/video0')
    locator = CursorLocator(mouse, sensor)

    pos = locator.locate()
    if pos:
        print(f"Cursor at ({pos.x}, {pos.y})")
        locator.shake()
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .cursor_detector import (
    CurveMatch,
    LissajousCurve,
    extract_motion_blobs,
    match_curve_segment,
)
from .esp32_mouse import ESP32Mouse
from .hdmi_sensor import HDMISensor


@dataclass
class CursorPosition:
    """Detected cursor position on the HDMI-captured screen."""

    x: int                      # Pixel X on HDMI screen
    y: int                      # Pixel Y on HDMI screen
    t_param: float              # Lissajous time parameter at detection
    correlation: float          # Match quality (0-1)
    pixels_per_hid: float = 1.0  # Set after calibrate()


class CursorLocator:
    """Locates cursor via Lissajous sweep + video curvature matching.

    Three concurrent activities:
    - RECORD: ffmpeg captures HDMI at 30fps to an MKV file
    - MOVE: ESP32 sends HID reports sampling a Lissajous curve
    - ANALYZE: periodically loads latest frames, runs motion matching

    When the analyzer detects the Lissajous curvature signature in the
    video, it signals all threads to stop. Early termination means we
    don't have to sweep the entire virtual desktop.
    """

    def __init__(
        self,
        mouse: ESP32Mouse,
        sensor: HDMISensor,
        curve: Optional[LissajousCurve] = None,
        max_duration: float = 10.0,
        video_fps: int = 30,
    ):
        """
        Args:
            mouse: ESP32Mouse instance.
            sensor: HDMISensor instance.
            curve: Lissajous curve parameters. Default covers a 6000x3000
                virtual desktop with irrational frequency ratio.
            max_duration: Maximum seconds to sweep before giving up.
            video_fps: HDMI capture framerate.
        """
        self.mouse = mouse
        self.sensor = sensor
        self.curve = curve or LissajousCurve()
        self.max_duration = max_duration
        self.video_fps = video_fps
        self._stop_event = threading.Event()
        self._move_start_time: float = 0.0

    def locate(self, verbose: bool = True) -> Optional[CursorPosition]:
        """Locate cursor on the HDMI-captured screen.

        Runs the 3-activity pipeline:
        1. Start ffmpeg recording
        2. Start Lissajous movement in a background thread
        3. Run analysis loop (in calling thread) checking for curve match
        4. On detection or timeout, stop everything and return

        Args:
            verbose: Print progress to stdout.

        Returns:
            CursorPosition if found, None if cursor never appeared.
        """
        self._stop_event.clear()
        video_path = "/tmp/cursor_detect.mkv"

        if verbose:
            print("Starting HDMI video recording...")

        # Start continuous recording
        ffmpeg_proc = self.sensor.start_recording(video_path, fps=self.video_fps)
        time.sleep(0.5)  # Let recording stabilize and capture initial frames

        if verbose:
            print(f"Starting Lissajous sweep (max {self.max_duration}s)...")

        # Start mouse movement in background thread
        self._move_start_time = time.monotonic()
        move_thread = threading.Thread(target=self._execute_curve, daemon=True)
        move_thread.start()

        # Run analysis loop in this thread
        result = self._analyze_loop(video_path, verbose=verbose)

        # Stop everything
        self._stop_event.set()
        move_thread.join(timeout=3)
        self.sensor.stop_recording(ffmpeg_proc)

        if result and verbose:
            print(f"Cursor detected at ({result.x}, {result.y}) "
                  f"[t={result.t_param:.2f}s, correlation={result.correlation:.2f}]")

        if result:
            self.mouse.set_position(result.x, result.y)

        return result

    def _execute_curve(self):
        """Send Lissajous HID reports until stop_event fires."""
        reports = self.curve.sample_hid_reports(0, self.max_duration)
        for dx, dy in reports:
            if self._stop_event.is_set():
                break
            self.mouse._send_raw(dx, dy)

    def _analyze_loop(
        self,
        video_path: str,
        check_interval: float = 1.0,
        verbose: bool = True,
    ) -> Optional[CursorPosition]:
        """Periodically load frames from growing video and check for curve match.

        Runs until a match is found, stop_event is set, or max_duration expires.

        Args:
            video_path: Path to the growing MKV file.
            check_interval: Seconds between analysis attempts.
            verbose: Print progress.

        Returns:
            CursorPosition if detected, None otherwise.
        """
        frames_analyzed = 0
        start_time = time.monotonic()

        while not self._stop_event.is_set():
            elapsed = time.monotonic() - start_time
            if elapsed > self.max_duration + 2:  # Extra 2s for processing
                if verbose:
                    print(f"Timeout after {elapsed:.1f}s — cursor not found.")
                break

            time.sleep(check_interval)

            # Load all frames decoded so far
            try:
                all_frames = self.sensor.load_video_frames(video_path)
            except (RuntimeError, Exception):
                continue  # File not ready or still being written

            if len(all_frames) < frames_analyzed + 10:
                continue  # Not enough new frames to analyze

            if verbose:
                print(f"  [{elapsed:.1f}s] Analyzing {len(all_frames)} frames "
                      f"({len(all_frames) - frames_analyzed} new)...")

            # Analyze new frames (keep 1 frame overlap for diff continuity)
            start_idx = max(0, frames_analyzed - 1)
            new_frames = all_frames[start_idx:]
            frames_analyzed = len(all_frames)

            # Extract motion blobs for consecutive frame pairs
            blobs_seq = []
            for i in range(len(new_frames) - 1):
                blobs = extract_motion_blobs(new_frames[i], new_frames[i + 1])
                blobs_seq.append(blobs)

            if not blobs_seq:
                continue

            # Check for Lissajous curve match
            match = match_curve_segment(
                blobs_seq,
                self.curve,
                fps=float(self.video_fps),
            )

            if match and match.matched:
                if verbose:
                    print(f"  Curve match found! correlation={match.correlation:.2f}")
                self._stop_event.set()
                return CursorPosition(
                    x=match.screen_pos[0],
                    y=match.screen_pos[1],
                    t_param=match.t_end,
                    correlation=match.correlation,
                )

        return None

    def calibrate(self, verbose: bool = True) -> float:
        """Calibrate pixel-to-HID-unit ratio after locate().

        Moves a known HID distance horizontally while recording,
        then measures the pixel displacement to compute the ratio.

        Returns:
            pixels_per_hid_unit (typically ~1.0 at 100% DPI, ~1.5 at 150%).
        """
        video_path = "/tmp/cursor_calibrate.mkv"
        hid_distance = 100  # Known horizontal movement in HID units

        if verbose:
            print(f"Calibrating: moving {hid_distance} HID units right...")

        # Record 1.5s video during horizontal movement
        ffmpeg_proc = self.sensor.start_recording(video_path, fps=self.video_fps)
        time.sleep(0.3)

        # Capture start frame index
        frame_before = self.sensor.capture()
        time.sleep(0.2)

        # Move known distance (pure horizontal)
        self.mouse._send_chunked(hid_distance, 0)
        time.sleep(0.3)

        # Capture end frame
        frame_after = self.sensor.capture()

        self.sensor.stop_recording(ffmpeg_proc)

        # Detect motion between start and end frames
        blobs = extract_motion_blobs(frame_before, frame_after, min_pixels=5)
        if not blobs:
            if verbose:
                print("Calibration failed: no motion detected.")
            return 1.0

        # Use the largest blob's centroid displacement
        # For calibration we need start+end, so use frames from video
        try:
            frames = self.sensor.load_video_frames(video_path)
        except RuntimeError:
            if verbose:
                print("Calibration failed: couldn't load video.")
            return 1.0

        if len(frames) < 10:
            if verbose:
                print("Calibration failed: too few frames.")
            return 1.0

        # Find motion blob in first and last few frames
        early_blobs = extract_motion_blobs(frames[0], frames[5], min_pixels=5)
        late_blobs = extract_motion_blobs(frames[-6], frames[-1], min_pixels=5)

        if not early_blobs or not late_blobs:
            if verbose:
                print("Calibration failed: couldn't track cursor in video.")
            return 1.0

        # Pixel displacement
        start_x = early_blobs[0].centroid[0]
        end_x = late_blobs[0].centroid[0]
        pixel_disp = abs(end_x - start_x)

        if pixel_disp < 5:
            if verbose:
                print("Calibration failed: negligible pixel displacement.")
            return 1.0

        ratio = pixel_disp / hid_distance
        if verbose:
            print(f"Calibration: {pixel_disp}px / {hid_distance} HID = "
                  f"{ratio:.2f} pixels/HID unit")

        return ratio

    def shake(self, amplitude: int = 30, shakes: int = 3):
        """Quick left-right wiggle for visual confirmation.

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

    def verified_click(
        self,
        x: int,
        y: int,
        button: str = "left",
        verify: bool = True,
    ) -> bool:
        """Move to (x,y), optionally verify via short recording, then click.

        For verification: moves a small known distance, records briefly,
        checks that cursor is near the expected position.

        Args:
            x: Target X on HDMI screen.
            y: Target Y on HDMI screen.
            button: Mouse button ('left', 'right', 'middle').
            verify: If True, verify cursor position before clicking.

        Returns:
            True if click executed (and verification passed if enabled).
        """
        self.mouse.move_to(x, y)
        time.sleep(0.3)

        if verify:
            # Quick verification: small probe movement + frame diff
            frame_before = self.sensor.capture()
            time.sleep(0.1)

            probe_dx, probe_dy = 20, 0
            self.mouse._send_chunked(probe_dx, probe_dy)
            time.sleep(0.2)

            frame_after = self.sensor.capture()

            # Undo probe
            self.mouse._send_chunked(-probe_dx, -probe_dy)
            time.sleep(0.1)

            blobs = extract_motion_blobs(frame_before, frame_after, min_pixels=5)
            if not blobs:
                print(f"WARNING: Cursor verification failed at ({x}, {y}). "
                      f"Click aborted.")
                return False

            # Check blob is near expected position
            actual_x, actual_y = blobs[0].centroid
            dist = ((actual_x - x) ** 2 + (actual_y - y) ** 2) ** 0.5
            if dist > 100:
                print(f"WARNING: Cursor at ({actual_x}, {actual_y}), "
                      f"expected ({x}, {y}), dist={dist:.0f}px. Click aborted.")
                return False

            self.mouse.set_position(actual_x, actual_y)

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

    print("Cursor Locator — Lissajous Curve Detection")
    print("=" * 45)

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

    print("\nStarting Lissajous cursor detection...")
    pos = locator.locate(verbose=True)

    if pos:
        print(f"\nCursor located at ({pos.x}, {pos.y})")
        print(f"Lissajous t={pos.t_param:.2f}s")
        print(f"Match correlation: {pos.correlation:.2f}")

        print("\nShaking cursor for visual confirmation...")
        locator.shake()
        print("Done. Check HDMI screen for cursor wiggle.")

        print("\nCalibrating pixel-to-HID ratio...")
        ratio = locator.calibrate(verbose=True)
        print(f"Calibrated: {ratio:.2f} pixels per HID unit")
    else:
        print("\nCursor not found on HDMI screen.")
        print("Possible causes:")
        print("  - Cursor is hidden (fullscreen app, screen saver)")
        print("  - HDMI capture input is not active")
        print("  - Virtual desktop is larger than expected (increase max_duration)")

    mouse.close()
