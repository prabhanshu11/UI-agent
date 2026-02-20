"""Fast ROI-based cursor silhouette tracking.

Once the cursor position is known (from Phase 1 Claude Vision, Lissajous, or
motion detection), this module tracks the cursor by:

1. Template matching in a small ROI around last known position (stationary cursor)
2. Frame differencing in ROI + shape scoring (moving cursor)
3. Adaptive ROI expansion when cursor leaves the tracking window

Designed to run inside the main motion detection loop at ~20fps.
ROI tracking at 200x200 is ~100x faster than full-frame 1920x1080.

Usage:
    st = SilhouetteTracker(recognizer)
    result = st.track(frame, last_x, last_y)
    if result:
        print(f"Cursor at ({result.x}, {result.y}) via {result.method}")
"""

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .cursor_recognizer import CursorRecognizer, PATCH_SIZE


@dataclass
class TrackResult:
    """Result of one silhouette tracking step."""

    x: int                  # Full-frame cursor X
    y: int                  # Full-frame cursor Y
    confidence: float       # 0.0-1.0 match quality
    method: str             # "template" or "motion_roi"
    roi_size: int           # Current ROI window size
    latency_ms: float = 0.0  # Tracking latency for this frame


@dataclass
class CandidateBlob:
    """A motion candidate region for YOLO crop validation."""

    x: int                  # Full-frame center X
    y: int                  # Full-frame center Y
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) full-frame
    area: int               # Contour area in pixels
    score: float            # Shape-likeness score


class SilhouetteTracker:
    """Fast ROI-constrained cursor tracking.

    Two-strategy tracker that operates in a small window around the
    last known cursor position:

    Strategy 1 — Template matching:
      Uses the recognizer's learned cursor template (from confirmed positions)
      or synthetic arrow template. Works for stationary cursor. ~0.2ms.

    Strategy 2 — ROI frame differencing:
      Computes abs-diff between current and previous ROI, finds motion contours,
      scores by cursor shape likeness. Works for moving cursor. ~0.5ms.

    Adaptive ROI:
      Starts at 200x200 (tight). On consecutive misses, expands by 200px
      up to 800x800. Resets to 200 on successful track. If 800 also fails,
      returns None so the caller can fall back to full-frame detection or
      Phase 1 visual scan.
    """

    # ROI size parameters
    ROI_MIN = 200
    ROI_STEP = 200
    ROI_MAX = 800

    # Template matching threshold (TM_CCOEFF_NORMED range: -1 to 1)
    TEMPLATE_THRESHOLD = 0.45

    # Motion detection thresholds
    MOTION_THRESHOLD = 15       # Grayscale diff threshold for "changed" pixel
    MIN_BLOB_PIXELS = 5         # Minimum contour area
    MAX_BLOB_PIXELS = 3000      # Maximum contour area
    SHAPE_MIN_SCORE = 0.3       # Minimum cursor shape score

    # Expansion trigger
    MISS_EXPAND_AFTER = 2       # Expand ROI after N consecutive misses

    def __init__(self, recognizer: CursorRecognizer):
        self.recognizer = recognizer
        self._roi_size = self.ROI_MIN
        self._prev_gray: Optional[np.ndarray] = None
        self._consecutive_misses = 0
        self._last_track_time = 0.0
        self._tracking_hz = 0.0

    @property
    def roi_size(self) -> int:
        """Current ROI window size in pixels."""
        return self._roi_size

    @property
    def tracking_hz(self) -> float:
        """Measured tracking frequency."""
        return self._tracking_hz

    @property
    def consecutive_misses(self) -> int:
        """Number of consecutive frames where cursor was not found in ROI."""
        return self._consecutive_misses

    def reset(self):
        """Reset tracker state (call after Phase 1 re-detection)."""
        self._roi_size = self.ROI_MIN
        self._prev_gray = None
        self._consecutive_misses = 0

    def track(
        self,
        frame: np.ndarray,
        last_x: int,
        last_y: int,
    ) -> Optional[TrackResult]:
        """Try to find/confirm cursor near (last_x, last_y) using ROI tracking.

        Args:
            frame: Current full frame (H, W, 3) RGB or (H, W) grayscale.
            last_x: Last known cursor X in full-frame coordinates.
            last_y: Last known cursor Y in full-frame coordinates.

        Returns:
            TrackResult if cursor found in ROI, None if lost.
        """
        t0 = time.monotonic()

        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        # Extract ROI from current frame
        roi, x0, y0 = self._extract_roi(gray, last_x, last_y)

        result = None

        # Strategy 1: Template matching (stationary + slow-moving cursor)
        result = self._template_match(roi, x0, y0)

        # Strategy 2: Motion detection within ROI (moving cursor)
        if result is None and self._prev_gray is not None:
            prev_roi, _, _ = self._extract_roi(self._prev_gray, last_x, last_y)
            result = self._motion_track(roi, prev_roi, x0, y0)

        # Measure latency
        elapsed_ms = (time.monotonic() - t0) * 1000

        if result:
            result.latency_ms = elapsed_ms
            self._consecutive_misses = 0
            self._roi_size = self.ROI_MIN
        else:
            self._consecutive_misses += 1
            if self._consecutive_misses >= self.MISS_EXPAND_AFTER:
                self._roi_size = min(self.ROI_MAX, self._roi_size + self.ROI_STEP)

        # Save full gray for next frame's ROI extraction
        self._prev_gray = gray

        # Update Hz measurement
        now = time.monotonic()
        if self._last_track_time > 0:
            dt = now - self._last_track_time
            if dt > 0:
                # Exponential moving average for smooth Hz reading
                instant_hz = 1.0 / dt
                self._tracking_hz = 0.8 * self._tracking_hz + 0.2 * instant_hz
        self._last_track_time = now

        return result

    def roi_bounds(self, last_x: int, last_y: int, frame_shape: tuple) -> tuple[int, int, int, int]:
        """Get current ROI bounds for overlay drawing.

        Returns (x0, y0, x1, y1) in full-frame coordinates.
        """
        h, w = frame_shape[:2]
        half = self._roi_size // 2
        x0 = max(0, last_x - half)
        y0 = max(0, last_y - half)
        x1 = min(w, last_x + half)
        y1 = min(h, last_y + half)
        return (x0, y0, x1, y1)

    def get_candidates(
        self,
        frame: np.ndarray,
        last_x: int,
        last_y: int,
    ) -> list[CandidateBlob]:
        """Return all motion candidates in the ROI for YOLO crop validation.

        Unlike track() which returns only the best blob, this returns ALL blobs
        passing the size/shape filters. Each candidate includes a full-frame
        bounding box suitable for crop extraction.

        Args:
            frame: Current full frame (H, W, 3) RGB or (H, W) grayscale.
            last_x: Last known cursor X in full-frame coordinates.
            last_y: Last known cursor Y in full-frame coordinates.

        Returns:
            List of CandidateBlob, sorted by score (best first).
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame

        roi_curr, x0, y0 = self._extract_roi(gray, last_x, last_y)

        if self._prev_gray is None:
            return []

        roi_prev, _, _ = self._extract_roi(self._prev_gray, last_x, last_y)

        rh = min(roi_curr.shape[0], roi_prev.shape[0])
        rw = min(roi_curr.shape[1], roi_prev.shape[1])
        if rh < 20 or rw < 20:
            return []

        curr = roi_curr[:rh, :rw]
        prev = roi_prev[:rh, :rw]

        diff = cv2.absdiff(curr, prev)
        _, mask = cv2.threshold(diff, self.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        candidates = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_BLOB_PIXELS or area > self.MAX_BLOB_PIXELS:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw > 100 or bh > 100:
                continue
            if max(bw, bh) > 0:
                aspect = min(bw, bh) / max(bw, bh)
                if aspect < 0.15:
                    continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            score = 1.0
            max_dim = max(bw, bh)
            if max_dim < 8:
                score *= 0.3
            elif max_dim > 80:
                score *= 0.5

            # Shape scoring via recognizer
            if self.recognizer.cursor_template is not None:
                half = PATCH_SIZE // 2
                if (cx >= half and cy >= half and
                        cx + half <= rw and cy + half <= rh):
                    patch = curr[cy - half:cy + half, cx - half:cx + half]
                    if patch.shape == (PATCH_SIZE, PATCH_SIZE):
                        shape_score = self.recognizer.score_blob_shape(patch)
                        score *= (0.5 + shape_score)

            # Full-frame bounding box with padding for YOLO crop
            pad = 32
            fx1 = max(0, x0 + bx - pad)
            fy1 = max(0, y0 + by - pad)
            fx2 = x0 + bx + bw + pad
            fy2 = y0 + by + bh + pad

            candidates.append(CandidateBlob(
                x=x0 + cx,
                y=y0 + cy,
                bbox=(fx1, fy1, fx2, fy2),
                area=area,
                score=score,
            ))

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _extract_roi(
        self, gray: np.ndarray, cx: int, cy: int,
    ) -> tuple[np.ndarray, int, int]:
        """Extract square ROI centered at (cx, cy), clamped to frame bounds.

        Returns (roi, x0, y0) where x0/y0 is the top-left corner in frame coords.
        """
        h, w = gray.shape[:2]
        half = self._roi_size // 2

        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(w, cx + half)
        y1 = min(h, cy + half)

        return gray[y0:y1, x0:x1], x0, y0

    def _template_match(
        self, roi: np.ndarray, roi_x0: int, roi_y0: int,
    ) -> Optional[TrackResult]:
        """Match cursor template within ROI using cv2.matchTemplate.

        Tries all available templates (learned, arrow, I-beam) and picks
        the best match. This handles cursor shape changes (arrow → I-beam → hand).
        """
        templates = self._get_all_templates()
        if not templates:
            return None

        rh, rw = roi.shape[:2]
        best_val = 0.0
        best_loc = (0, 0)
        best_tw, best_th = 0, 0

        for tmpl in templates:
            th, tw = tmpl.shape[:2]
            if rh < th + 2 or rw < tw + 2:
                continue
            result = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_tw, best_th = tw, th

        if best_val < self.TEMPLATE_THRESHOLD:
            return None

        match_x = roi_x0 + best_loc[0] + best_tw // 2
        match_y = roi_y0 + best_loc[1] + best_th // 2

        return TrackResult(
            x=match_x,
            y=match_y,
            confidence=float(best_val),
            method="template",
            roi_size=self._roi_size,
        )

    def _motion_track(
        self,
        roi_curr: np.ndarray,
        roi_prev: np.ndarray,
        x0: int,
        y0: int,
    ) -> Optional[TrackResult]:
        """Detect cursor motion within ROI via frame differencing.

        Both ROIs are extracted at the same (last_x, last_y) position,
        ensuring pixel-aligned comparison.
        """
        # Handle size mismatch (cursor near frame edge)
        rh = min(roi_curr.shape[0], roi_prev.shape[0])
        rw = min(roi_curr.shape[1], roi_prev.shape[1])

        if rh < 20 or rw < 20:
            return None

        curr = roi_curr[:rh, :rw]
        prev = roi_prev[:rh, :rw]

        # Frame difference
        diff = cv2.absdiff(curr, prev)
        _, mask = cv2.threshold(diff, self.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Score each contour by cursor-likeness
        best_score = 0.0
        best_cx, best_cy = 0, 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.MIN_BLOB_PIXELS or area > self.MAX_BLOB_PIXELS:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)

            # Filter out large UI redraws
            if bw > 100 or bh > 100:
                continue

            # Filter out very thin blobs (text cursor blinks)
            if max(bw, bh) > 0:
                aspect = min(bw, bh) / max(bw, bh)
                if aspect < 0.15:
                    continue

            # Centroid via moments
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Base score from size (prefer cursor-sized: 15-50px)
            score = 1.0
            max_dim = max(bw, bh)
            if max_dim < 8:
                score *= 0.3
            elif max_dim > 80:
                score *= 0.5

            # Shape-informed scoring via recognizer template
            if self.recognizer.cursor_template is not None:
                half = PATCH_SIZE // 2
                if (cx >= half and cy >= half and
                        cx + half <= rw and cy + half <= rh):
                    patch = curr[cy - half:cy + half, cx - half:cx + half]
                    if patch.shape == (PATCH_SIZE, PATCH_SIZE):
                        shape_score = self.recognizer.score_blob_shape(patch)
                        score *= (0.5 + shape_score)

            if score > best_score:
                best_score = score
                best_cx = cx
                best_cy = cy

        if best_score <= 0:
            return None

        # Convert ROI coords to full-frame coords
        full_x = x0 + best_cx
        full_y = y0 + best_cy

        return TrackResult(
            x=full_x,
            y=full_y,
            confidence=min(1.0, best_score),
            method="motion_roi",
            roi_size=self._roi_size,
        )

    def _get_all_templates(self) -> list[np.ndarray]:
        """Get all available templates for multi-template matching.

        Returns list of 32x32 grayscale templates to try:
        - Learned cursor template (from probe-confirmed positions) — highest priority
        - Synthetic arrow template
        - Synthetic I-beam template
        - Synthetic hand/pointer template
        """
        templates = []
        if self.recognizer.cursor_template is not None:
            templates.append(self.recognizer.cursor_template[16:48, 16:48])
        templates.append(self.recognizer._arrow_template)
        templates.append(self.recognizer._ibeam_template)
        if hasattr(self.recognizer, '_hand_template'):
            templates.append(self.recognizer._hand_template)
        return templates
