"""Commanded movement system — unified jitter + probe + Lissajous.

Key insight: Anti-sleep jitter (±3px) and Lissajous sweep are the SAME thing
at different scales. Every commanded movement is a position verification
opportunity using the silhouette tracker's ROI-constrained detection.

Movement profiles:
  jitter:    ±3px, instant, invisible — anti-sleep + free health check
  probe:     ±15px, 200ms, visible — small recovery
  big_probe: ±50px, 200ms, visible — medium recovery
  lissajous: full sweep, 5s, guaranteed — brute-force finder

Escalation chain (all use same code, just different amplitudes):
  cursor lost?
    → ±3px jitter verify (free, already happening)
    → ±15px probe verify (small, visible)
    → ±50px probe verify (medium, noticeable)
    → Full Lissajous sweep (5s, GUARANTEED)

Usage:
    cm = CommandedMovement(mouse, sil_tracker, sensor=sensor)
    pos = cm.jitter_verify(500, 300)   # Quick ±3px check
    pos = cm.probe_verify(500, 300)    # ±15px recovery
    pos = cm.escalate(500, 300)        # Try everything in order
"""

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .silhouette_tracker import SilhouetteTracker, TrackResult


@dataclass
class MovementResult:
    """Result of a commanded movement verification."""

    x: int
    y: int
    amplitude: int          # Movement amplitude that succeeded
    method: str             # "jitter", "probe_15", "probe_50", "lissajous"
    confidence: float       # Detection confidence
    latency_ms: float       # Total time for move + verify


class CommandedMovement:
    """Unified commanded movement executor with silhouette-based verification.

    Sends known movements via ESP32 and uses the silhouette tracker's
    ROI-constrained detection to verify the cursor responded. This is
    faster and more reliable than full-frame blob detection because:
    1. ROI is small (~200x200) → detection is <1ms
    2. We KNOW what movement to expect → can validate direction
    3. Same code path for all amplitudes → fewer bugs
    """

    def __init__(
        self,
        mouse,
        sil_tracker: SilhouetteTracker,
        sensor=None,
        daemon_client=None,
    ):
        """
        Args:
            mouse: ESP32Mouse for sending movements.
            sil_tracker: SilhouetteTracker for ROI-based verification.
            sensor: HDMISensor for frame capture (when no daemon).
            daemon_client: DaemonClient for frame capture (when daemon running).
        """
        self.mouse = mouse
        self.sil_tracker = sil_tracker
        self.sensor = sensor
        self.daemon_client = daemon_client

    def _capture(self) -> Optional[np.ndarray]:
        """Capture a frame from daemon or sensor."""
        if self.daemon_client and self.daemon_client.is_running():
            jpeg = self.daemon_client.read_jpeg()
            if jpeg:
                frame = cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.sensor:
            return self.sensor.capture(settle_frames=1)
        return None

    def _verify_movement(
        self,
        last_x: int,
        last_y: int,
        dx: int,
        dy: int,
        settle_ms: float = 100.0,
    ) -> Optional[tuple[int, int, float]]:
        """Core verification: capture before, move, capture after, check ROI.

        Args:
            last_x: Known cursor X before movement.
            last_y: Known cursor Y before movement.
            dx: Horizontal movement in pixels (HID units).
            dy: Vertical movement in pixels (HID units).
            settle_ms: Milliseconds to wait after movement for frame to settle.

        Returns:
            (new_x, new_y, confidence) if cursor motion detected, None otherwise.
        """
        frame_before = self._capture()
        if frame_before is None:
            return None

        # Convert to grayscale for ROI extraction
        if len(frame_before.shape) == 3:
            gray_before = cv2.cvtColor(frame_before, cv2.COLOR_RGB2GRAY)
        else:
            gray_before = frame_before

        # Execute movement
        self.mouse._send_raw(dx, dy)
        time.sleep(settle_ms / 1000.0)

        frame_after = self._capture()
        if frame_after is None:
            # Undo movement
            self.mouse._send_raw(-dx, -dy)
            return None

        if len(frame_after.shape) == 3:
            gray_after = cv2.cvtColor(frame_after, cv2.COLOR_RGB2GRAY)
        else:
            gray_after = frame_after

        # Extract ROIs at cursor position from both frames
        roi_size = max(self.sil_tracker.ROI_MIN, abs(dx) * 4 + 100, abs(dy) * 4 + 100)
        half = roi_size // 2
        h, w = gray_before.shape[:2]

        x0 = max(0, last_x - half)
        y0 = max(0, last_y - half)
        x1 = min(w, last_x + half)
        y1 = min(h, last_y + half)

        roi_before = gray_before[y0:y1, x0:x1]
        roi_after = gray_after[y0:y1, x0:x1]

        rh = min(roi_before.shape[0], roi_after.shape[0])
        rw = min(roi_before.shape[1], roi_after.shape[1])

        if rh < 20 or rw < 20:
            self.mouse._send_raw(-dx, -dy)
            return None

        # Frame difference in ROI
        diff = cv2.absdiff(roi_before[:rh, :rw], roi_after[:rh, :rw])
        _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            self.mouse._send_raw(-dx, -dy)
            return None

        # Find the motion blob closest to expected new position
        expected_roi_x = (last_x - x0) + dx
        expected_roi_y = (last_y - y0) + dy

        best_dist = 9999.0
        best_cx, best_cy = 0, 0
        best_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 3 or area > 3000:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            dist = ((cx - expected_roi_x) ** 2 + (cy - expected_roi_y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_cx = cx
                best_cy = cy
                best_area = area

        if best_area == 0:
            self.mouse._send_raw(-dx, -dy)
            return None

        # Convert ROI coords back to full-frame
        full_x = x0 + best_cx
        full_y = y0 + best_cy

        # Confidence based on how close the detected motion is to expected
        max_expected_dist = (dx ** 2 + dy ** 2) ** 0.5 + 20  # tolerance
        confidence = max(0.0, 1.0 - best_dist / max(max_expected_dist, 1))

        # Undo the movement
        self.mouse._send_raw(-dx, -dy)
        time.sleep(settle_ms / 1000.0)

        return (full_x, full_y, confidence)

    def jitter_verify(self, last_x: int, last_y: int) -> Optional[MovementResult]:
        """Fire ±3px jitter and verify cursor moved using ROI detection.

        This is the cheapest verification — same as anti-sleep jitter but
        with explicit frame capture before/after for reliable detection.

        Returns MovementResult if cursor detected, None if not.
        """
        t0 = time.monotonic()
        result = self._verify_movement(last_x, last_y, dx=3, dy=0, settle_ms=100)

        if result is None:
            return None

        x, y, confidence = result
        return MovementResult(
            x=x, y=y,
            amplitude=3,
            method="jitter",
            confidence=confidence,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    def probe_verify(
        self,
        last_x: int,
        last_y: int,
        amplitude: int = 15,
    ) -> Optional[MovementResult]:
        """Fire ±Npx probe in TWO directions and verify using ROI detection.

        Dual-direction probe (right then down) eliminates false positives:
        only the real cursor responds to both movements.

        Returns MovementResult if cursor detected in both directions, None otherwise.
        """
        t0 = time.monotonic()

        # First probe: horizontal
        result_h = self._verify_movement(last_x, last_y, dx=amplitude, dy=0, settle_ms=150)
        if result_h is None:
            return None

        x_h, y_h, conf_h = result_h

        # Second probe: vertical (from detected position)
        result_v = self._verify_movement(x_h, y_h, dx=0, dy=amplitude, settle_ms=150)
        if result_v is None:
            return None

        x_v, y_v, conf_v = result_v

        # Both directions confirmed — high confidence
        confidence = min(conf_h, conf_v)
        method = f"probe_{amplitude}"

        return MovementResult(
            x=x_v, y=y_v,
            amplitude=amplitude,
            method=method,
            confidence=confidence,
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    def escalate(self, last_x: int, last_y: int) -> Optional[MovementResult]:
        """Escalating recovery chain — try increasingly aggressive movements.

        Order:
        1. ±3px jitter (invisible, ~300ms)
        2. ±15px probe (small, ~600ms)
        3. ±50px probe (noticeable, ~600ms)
        4. Full Lissajous sweep (5s, GUARANTEED) — not implemented here,
           caller should use CursorLocator directly.

        Returns the first successful result, or None if all probes fail.
        """
        # Tier 1: Micro jitter
        result = self.jitter_verify(last_x, last_y)
        if result:
            return result

        # Tier 2: Small probe (dual-direction)
        result = self.probe_verify(last_x, last_y, amplitude=15)
        if result:
            return result

        # Tier 3: Big probe (dual-direction)
        result = self.probe_verify(last_x, last_y, amplitude=50)
        if result:
            return result

        # Tier 4: Lissajous not handled here — caller should use CursorLocator
        # This is intentional: Lissajous requires recording + offline analysis,
        # which is a different paradigm from the instant probe-based verification.
        return None
