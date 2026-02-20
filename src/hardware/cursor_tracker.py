"""Background cursor position tracker with continuous verification.

Maintains cursor position awareness through a fusion of signals:

    1. CNN patch verification (<2ms, runs every check cycle)
    2. Silhouette template matching (when recognizer available)
    3. Probe movements (micro-shake, macro-shake) as fallback
    4. Lissajous full sweep as last resort

V2 improvements (from 102 loss event analysis):
    - CNN verification BEFORE probe movements (no mouse movement needed)
    - Multi-signal consensus before declaring loss
    - Ghost position (6,15) guard
    - CNN confidence EMA smoothing (prevents single-frame drops)
    - Faster check interval (0.5s default, was 2.0s)
    - Shorter stale threshold (3.0s default, was 10.0s)

Usage:
    tracker = CursorTracker(mouse, sensor, recognizer=recognizer, logger=logger)
    tracker.start()         # Begins background tracking
    pos = tracker.position  # Current known position (or None)
    tracker.stop()          # Graceful shutdown
"""

import threading
import time
from dataclasses import dataclass, field
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

# CNN confidence thresholds (derived from 102-event analysis)
_CNN_CONFIRM_THRESHOLD = 0.8   # Above this = position confirmed
_CNN_CONCERN_THRESHOLD = 0.3   # Below this = trigger immediate escalation
_CNN_EMA_ALPHA = 0.3           # Smoothing factor (0.3 = 30% new, 70% history)

# Ghost position guard: reject positions near (0,0) from manual init
_GHOST_MIN_X = 20
_GHOST_MIN_Y = 20


@dataclass
class TrackedPosition:
    """Current cursor position with confidence metadata."""

    x: int
    y: int
    timestamp: float        # time.monotonic() of last confirmation
    method: str             # "cnn_verify", "micro_shake", "macro_shake", "lissajous"
    confidence: float = 1.0  # 0-1, decays over time without confirmation
    cnn_ema: float = 0.0    # Smoothed CNN confidence (EMA)


class CursorTracker:
    """Background service that maintains cursor position awareness.

    V2: Uses continuous CNN verification as the primary confirmation
    method. Probe movements are now fallback, not primary. Multi-signal
    consensus prevents false loss declarations.
    """

    def __init__(
        self,
        mouse: ESP32Mouse,
        sensor: HDMISensor,
        recognizer=None,
        logger: Optional[ExperienceLogger] = None,
        check_interval: float = 0.5,
        stale_threshold: float = 3.0,
    ):
        """
        Args:
            mouse: ESP32Mouse for sending probe movements.
            sensor: HDMISensor for capturing frames.
            recognizer: CursorRecognizer for CNN verification (optional).
            logger: ExperienceLogger for recording events.
            check_interval: Seconds between position checks (0.5s default).
            stale_threshold: Seconds without confirmation before escalating (3.0s).
        """
        self.mouse = mouse
        self.sensor = sensor
        self.recognizer = recognizer
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
        """Manually set position (e.g., after Lissajous locate).

        Guards against ghost positions near (0,0) from uninitialized state.
        Analysis showed 11% of loss events traced to bogus (6,15) position.
        """
        # Ghost position guard
        if (x < _GHOST_MIN_X and y < _GHOST_MIN_Y
                and method == "manual"):
            if self.logger:
                self.logger.log("ghost_position_rejected",
                                position=(x, y),
                                details={"method": method})
            return

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

    # ------------------------------------------------------------------
    # CNN verification (Phase A quick win: no mouse movement needed)
    # ------------------------------------------------------------------

    def _cnn_verify(self, frame: np.ndarray) -> float:
        """Check CNN confidence at current position. <2ms, no mouse movement.

        Returns CNN confidence 0.0-1.0, or -1.0 if no recognizer/position.
        """
        if self.recognizer is None:
            return -1.0

        pos = self.position
        if pos is None:
            return -1.0

        # Extract 64x64 patch at current position from full-res frame
        patch = self._extract_patch(frame, pos.x, pos.y)
        if patch is None:
            return -1.0

        return self.recognizer.classify_patch(patch)

    def _extract_patch(self, frame: np.ndarray, x: int, y: int,
                       size: int = 64) -> Optional[np.ndarray]:
        """Extract a size×size grayscale patch centered at (x, y)."""
        if frame is None:
            return None

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        h, w = gray.shape[:2]
        half = size // 2
        x0 = max(0, x - half)
        y0 = max(0, y - half)
        x1 = min(w, x + half)
        y1 = min(h, y + half)

        patch = gray[y0:y1, x0:x1]
        if patch.shape[0] != size or patch.shape[1] != size:
            # Pad if near edge
            padded = np.zeros((size, size), dtype=np.uint8)
            ph, pw = patch.shape[:2]
            padded[:ph, :pw] = patch
            return padded

        return patch

    # ------------------------------------------------------------------
    # Main tracking loop (V2: CNN-first, consensus-based)
    # ------------------------------------------------------------------

    def _track_loop(self):
        """Main tracking loop — runs in daemon thread.

        V2 flow:
          1. Capture frame
          2. CNN verify at current position (no mouse movement, <2ms)
          3. If CNN confirms (>0.8) → position fresh, continue
          4. If CNN uncertain (0.3-0.8) → try micro-shake
          5. If CNN low (<0.3) OR no CNN → check multi-signal consensus
          6. If consensus says lost → escalate immediately (no 10s wait)
        """
        while not self._stop_event.is_set():
            self._stop_event.wait(self.check_interval)
            if self._stop_event.is_set():
                break

            pos = self.position
            if pos is None:
                self._escalate_lissajous()
                continue

            age = time.monotonic() - pos.timestamp

            # Step 1: Try CNN verification first (no mouse movement)
            try:
                frame = self.sensor.capture(settle_frames=2)
            except (RuntimeError, Exception):
                frame = None

            if frame is not None and self.recognizer is not None:
                cnn_conf = self._cnn_verify(frame)

                if cnn_conf >= 0:
                    # Update EMA
                    with self._lock:
                        if self._position:
                            old_ema = self._position.cnn_ema
                            self._position.cnn_ema = (
                                _CNN_EMA_ALPHA * cnn_conf +
                                (1 - _CNN_EMA_ALPHA) * old_ema
                            )

                    ema = self._position.cnn_ema if self._position else 0.0

                    if ema >= _CNN_CONFIRM_THRESHOLD:
                        # CNN confirms cursor is here — refresh timestamp
                        with self._lock:
                            if self._position:
                                self._position.timestamp = time.monotonic()
                                self._position.method = "cnn_verify"
                                self._position.confidence = ema
                        continue

                    if ema < _CNN_CONCERN_THRESHOLD and age > 1.0:
                        # CNN is very uncertain AND position is aging
                        # Skip the 10s wait — escalate now
                        self._log_event("cnn_concern", (pos.x, pos.y), {
                            "cnn_ema": round(ema, 3),
                            "age_s": round(age, 1),
                        })
                        if not self._probe_micro():
                            if not self._probe_macro():
                                self._log_event("cursor_lost", (pos.x, pos.y),
                                                {"age_s": round(age, 1),
                                                 "cnn_ema": round(ema, 3)})
                                self._escalate_lissajous()
                        continue

            # Step 2: No CNN or CNN uncertain — fall back to age-based check
            if age < self.stale_threshold:
                continue

            # Step 3: Position stale — try probes
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
