"""Background cursor position tracker with three-tier recovery.

Maintains last_known_position by periodically probing the cursor via
frame-diff on the HDMI capture. When the cursor is lost, escalates:

    Tier 1: Micro-shake (±3px) — cheap, invisible, confirms position
    Tier 2: Macro-shake (±15px) — visible, recovers small drift
    Tier 3: Lissajous re-detect — expensive, full sweep, last resort

All operations run at 960×540 (downsampled from 1920×1080) for speed.
The session log showed blob extraction at full-res takes ~160ms/pair;
at half-res this drops to ~40ms/pair.

Usage:
    tracker = CursorTracker(mouse, sensor, logger)
    tracker.start()         # Begins background tracking
    pos = tracker.position  # Current known position (or None)
    tracker.stop()          # Graceful shutdown
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .cursor_detector import extract_motion_blobs
from .cursor_locator import CursorLocator, CursorPosition
from .esp32_mouse import ESP32Mouse
from .experience_logger import ExperienceLogger
from .hdmi_sensor import HDMISensor


# Downsampled resolution for fast frame-diff
_LOW_W = 960
_LOW_H = 540


@dataclass
class TrackedPosition:
    """Current cursor position with confidence metadata."""

    x: int
    y: int
    timestamp: float        # time.monotonic() of last confirmation
    method: str             # "micro_shake", "macro_shake", "lissajous"
    confidence: float = 1.0  # 0-1, decays over time without confirmation


class CursorTracker:
    """Background service that maintains cursor position awareness.

    Runs a daemon thread that periodically confirms cursor position
    via small probe movements + frame-diff. If confirmation fails,
    escalates to larger probes and eventually a full Lissajous sweep.
    """

    def __init__(
        self,
        mouse: ESP32Mouse,
        sensor: HDMISensor,
        logger: Optional[ExperienceLogger] = None,
        check_interval: float = 2.0,
        stale_threshold: float = 10.0,
    ):
        """
        Args:
            mouse: ESP32Mouse for sending probe movements.
            sensor: HDMISensor for capturing frames.
            logger: ExperienceLogger for recording events.
            check_interval: Seconds between position checks.
            stale_threshold: Seconds without confirmation before escalating.
        """
        self.mouse = mouse
        self.sensor = sensor
        self.logger = logger
        self.check_interval = check_interval
        self.stale_threshold = stale_threshold

        self._position: Optional[TrackedPosition] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def position(self) -> Optional[TrackedPosition]:
        """Current tracked position (thread-safe read)."""
        with self._lock:
            return self._position

    def set_position(self, x: int, y: int, method: str = "manual"):
        """Manually set position (e.g., after Lissajous locate)."""
        with self._lock:
            self._position = TrackedPosition(
                x=x, y=y,
                timestamp=time.monotonic(),
                method=method,
            )
        self.mouse.set_position(x, y)
        if self.logger:
            self.logger.log("cursor_moved", position=(x, y),
                            details={"method": method})

    def start(self):
        """Start background tracking thread."""
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._track_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop background tracking."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _track_loop(self):
        """Main tracking loop — runs in daemon thread."""
        while not self._stop_event.is_set():
            self._stop_event.wait(self.check_interval)
            if self._stop_event.is_set():
                break

            pos = self.position
            if pos is None:
                # No known position — need full Lissajous detect
                self._escalate_lissajous()
                continue

            age = time.monotonic() - pos.timestamp
            if age < self.stale_threshold:
                continue  # Position is fresh, no action needed

            # Position stale — try to confirm
            if not self._probe_micro():
                if not self._probe_macro():
                    self._log_event("cursor_lost", (pos.x, pos.y),
                                    {"age_s": round(age, 1)})
                    self._escalate_lissajous()

    def _capture_low_res(self) -> Optional[np.ndarray]:
        """Capture a frame and downsample to 960×540."""
        try:
            frame = self.sensor.capture(settle_frames=2)
            return cv2.resize(frame, (_LOW_W, _LOW_H), interpolation=cv2.INTER_AREA)
        except (RuntimeError, Exception):
            return None

    def _probe_micro(self) -> bool:
        """Tier 1: ±3px micro-shake to confirm cursor is still at known position.

        Returns True if cursor detected at approximately the expected location.
        """
        pos = self.position
        if pos is None:
            return False

        frame_before = self._capture_low_res()
        if frame_before is None:
            return False

        # Micro-shake: move right 3px then back
        self.mouse._send_raw(3, 0)
        time.sleep(0.15)

        frame_after = self._capture_low_res()
        if frame_after is None:
            self.mouse._send_raw(-3, 0)
            return False

        # Undo the probe
        self.mouse._send_raw(-3, 0)

        # Check for motion near expected position (scaled to low-res)
        blobs = extract_motion_blobs(
            frame_before, frame_after,
            threshold=20, min_pixels=3, max_pixels=2000,
        )

        if not blobs:
            return False

        # Scale expected position to low-res coords
        expect_x = pos.x * _LOW_W // 1920
        expect_y = pos.y * _LOW_H // 1080

        # Check if any blob is near expected position
        for blob in blobs:
            bx, by = blob.centroid
            dist = ((bx - expect_x) ** 2 + (by - expect_y) ** 2) ** 0.5
            if dist < 60:  # ~120px tolerance at full-res
                # Confirmed — update timestamp
                with self._lock:
                    if self._position:
                        self._position.timestamp = time.monotonic()
                        self._position.method = "micro_shake"
                return True

        return False

    def _probe_macro(self) -> bool:
        """Tier 2: ±15px macro-shake — visible, catches small drift.

        If detected, updates position to actual blob centroid (corrects drift).
        """
        pos = self.position
        if pos is None:
            return False

        frame_before = self._capture_low_res()
        if frame_before is None:
            return False

        # Macro-shake: move right 15px then back
        self.mouse._send_raw(15, 0)
        time.sleep(0.2)

        frame_after = self._capture_low_res()
        if frame_after is None:
            self.mouse._send_raw(-15, 0)
            return False

        self.mouse._send_raw(-15, 0)

        blobs = extract_motion_blobs(
            frame_before, frame_after,
            threshold=20, min_pixels=5, max_pixels=3000,
        )

        if not blobs:
            return False

        # Use the largest blob as cursor position (scale back to full-res)
        bx, by = blobs[0].centroid
        full_x = bx * 1920 // _LOW_W
        full_y = by * 1080 // _LOW_H

        self.set_position(full_x, full_y, method="macro_shake")
        return True

    def _escalate_lissajous(self):
        """Tier 3: Full Lissajous sweep — expensive last resort."""
        self._log_event("lissajous_start", None, {})

        locator = CursorLocator(self.mouse, self.sensor)
        result = locator.locate(verbose=False)

        if result:
            self.set_position(result.x, result.y, method="lissajous")
            self._log_event("cursor_detected", (result.x, result.y), {
                "method": "lissajous",
                "correlation": round(result.correlation, 3),
                "t_param": round(result.t_param, 2),
            })
        else:
            self._log_event("lissajous_failed", None, {})

    def _log_event(
        self,
        event: str,
        position: Optional[tuple[int, int]],
        details: dict,
    ):
        """Log to experience logger if available."""
        if self.logger:
            self.logger.log(event, position=position, details=details)
