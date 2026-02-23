"""Position Fusion Engine — continuous cursor position awareness.

Replaces the reactive stale-threshold approach with continuous
multi-signal fusion:

    1. Dead reckoning from sent HID commands (50 Hz, in ESP32Mouse)
    2. CNN patch verification at predicted position (<2ms, every frame)
    3. Silhouette tracker ROI matching (20 Hz)
    4. YOLO full-frame ground truth anchor (1-2 Hz)

Derived from analysis of 102 cursor loss events which showed:
    - 68.6% of events never recovered (recovery is broken)
    - Mean time to lose: 0.32s (loss is near-instant)
    - Mean time to recover: 8.6s (recovery is painfully slow)
    - Retro CNN saw cursor during "lost" phase in 63% of events
    - Live CNN was nearly always zero (not running during tracking)

Usage:
    fusion = PositionFusionEngine(recognizer, yolo_detector)
    fusion.update(frame, esp32_position)  # Call every frame
    pos, conf = fusion.position, fusion.confidence
"""

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# Confidence thresholds
CONFIRM_THRESHOLD = 0.8      # Position confirmed
CONCERN_THRESHOLD = 0.3      # Trigger increased scanning
YOLO_TRUST_THRESHOLD = 0.7   # Trust YOLO detection even if far from predicted

# YOLO scan intervals (seconds)
YOLO_NORMAL_INTERVAL = 1.0   # Normal scanning rate
YOLO_URGENT_INTERVAL = 0.3   # When confidence is medium
YOLO_CRITICAL_INTERVAL = 0.1 # When confidence is low

# Drift correction: max distance to accept YOLO correction without reset
MAX_DRIFT_CORRECTION_PX = 200


@dataclass
class FusedPosition:
    """Position estimate from the fusion engine."""
    x: int
    y: int
    confidence: float       # 0.0-1.0 fused confidence
    source: str             # Primary source: "cnn", "yolo", "silhouette", "dead_reckoning"
    cnn_conf: float = 0.0   # Raw CNN confidence
    yolo_conf: float = 0.0  # Last YOLO confidence (0 if not scanned recently)
    sil_conf: float = 0.0   # Silhouette tracker confidence
    timestamp: float = 0.0  # time.monotonic() of this estimate


class CursorVerifier:
    """Unified CPU-fast cursor verification (CNN + silhouette).

    Runs both TinyCursorNet and template matching in a single call.
    Returns fused confidence score. Total latency: <3ms on CPU.

    Original TinyCursorNet and SilhouetteTracker code is preserved
    in their respective files — this wraps them for convenience.
    """

    def __init__(self, recognizer):
        """
        Args:
            recognizer: CursorRecognizer instance (has CNN + templates).
        """
        self.recognizer = recognizer

    def verify(self, frame: np.ndarray, x: int, y: int,
               prev_frame: Optional[np.ndarray] = None) -> tuple[float, str]:
        """Combined verification at (x, y). Returns (confidence, method).

        Args:
            frame: Current frame (H, W, 3) or (H, W) grayscale.
            x: Cursor X position to verify.
            y: Cursor Y position to verify.
            prev_frame: Previous frame for motion check (optional).

        Returns:
            (confidence, method) where method is "cnn" or "template".
        """
        patch = self._extract_patch(frame, x, y)
        if patch is None:
            return 0.0, "none"

        # CNN check
        cnn_conf = self.recognizer.classify_patch(patch)

        # Template check
        template_conf = self.recognizer.score_blob_shape(patch)

        # Fuse: take the higher of CNN or template
        if cnn_conf >= template_conf:
            return cnn_conf, "cnn"
        return template_conf, "template"

    def _extract_patch(self, frame: np.ndarray, x: int, y: int,
                       size: int = 64) -> Optional[np.ndarray]:
        """Extract size×size grayscale patch centered at (x, y)."""
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
            padded = np.zeros((size, size), dtype=np.uint8)
            ph, pw = patch.shape[:2]
            padded[:ph, :pw] = patch
            return padded

        return patch


class YOLOAnchor:
    """Periodic YOLO full-frame scan for absolute position correction.

    Runs YOLO at 1-2 Hz normally, faster when confidence drops.
    Provides ground truth that corrects dead reckoning drift.
    """

    def __init__(self, yolo_detector, interval: float = YOLO_NORMAL_INTERVAL):
        """
        Args:
            yolo_detector: YOLOCursorDetector instance.
            interval: Seconds between scans.
        """
        self.yolo = yolo_detector
        self.interval = interval
        self._last_scan_time = 0.0
        self._last_detection = None

    @property
    def last_detection(self):
        """Most recent YOLO detection result."""
        return self._last_detection

    def check(self, frame: np.ndarray,
              predicted_x: int, predicted_y: int) -> Optional[tuple]:
        """Run YOLO if interval elapsed. Returns (x, y, confidence) or None.

        Args:
            frame: Full 1920x1080 frame.
            predicted_x: Current predicted cursor X.
            predicted_y: Current predicted cursor Y.

        Returns:
            (x, y, confidence) if cursor detected, None otherwise.
        """
        now = time.monotonic()
        if now - self._last_scan_time < self.interval:
            return None

        self._last_scan_time = now

        detections = self.yolo.detect(frame)
        if not detections:
            self._last_detection = None
            return None

        # Find detection closest to predicted position
        best = min(detections, key=lambda d:
            ((d.cx - predicted_x) ** 2 + (d.cy - predicted_y) ** 2) ** 0.5)

        dist = ((best.cx - predicted_x) ** 2 +
                (best.cy - predicted_y) ** 2) ** 0.5

        self._last_detection = best

        # Accept if close to predicted position (drift correction)
        if dist < MAX_DRIFT_CORRECTION_PX:
            return (best.cx, best.cy, best.confidence)

        # Accept if far but YOLO is very confident (cursor moved by user)
        if best.confidence > YOLO_TRUST_THRESHOLD:
            return (best.cx, best.cy, best.confidence)

        return None

    def set_urgency(self, confidence: float):
        """Adjust scan rate based on current confidence level."""
        if confidence >= CONFIRM_THRESHOLD:
            self.interval = YOLO_NORMAL_INTERVAL
        elif confidence >= CONCERN_THRESHOLD:
            self.interval = YOLO_URGENT_INTERVAL
        else:
            self.interval = YOLO_CRITICAL_INTERVAL


class PositionFusionEngine:
    """Continuous cursor position fusion from multiple signal sources.

    This is the main orchestrator. Call update() with each new frame
    to get a continuously refined position estimate.

    Track D enhancements:
      D3. Trail-informed YOLO scoring (1.5x boost near trail, 0.6x far)
      D4. Confidence gate (high/medium/low replaces 3.0s stale threshold)
      D5. Dead reckoning feeds motion trail for velocity even without vision
    """

    # D3: Trail-informed scoring parameters
    TRAIL_BOOST = 1.5          # Multiplier for detections near trail prediction
    TRAIL_REDUCE = 0.6         # Multiplier for detections far from prediction during stillness
    TRAIL_NEAR_PX = 100        # "Near" = within 100px of trail prediction
    TRAIL_FAR_PX = 300         # "Far" = beyond 300px of trail prediction

    # D4: Confidence gate levels
    CONF_HIGH = 0.7            # CNN + YOLO agree
    CONF_MEDIUM = 0.3          # One source only
    CONF_LOW = 0.1             # No recent confirmation

    def __init__(self, recognizer=None, yolo_detector=None, motion_trail=None):
        """
        Args:
            recognizer: CursorRecognizer for CNN verification.
            yolo_detector: YOLOCursorDetector for ground truth anchoring.
            motion_trail: MotionTrail instance for trail-informed scoring (D3/D5).
        """
        self._verifier = CursorVerifier(recognizer) if recognizer else None
        self._yolo_anchor = YOLOAnchor(yolo_detector) if yolo_detector else None
        self._trail = motion_trail
        self._position: Optional[FusedPosition] = None
        self._prev_frame: Optional[np.ndarray] = None
        self._cnn_ema = 0.0
        self._confidence_level = "low"  # D4: high, medium, low

    @property
    def position(self) -> Optional[FusedPosition]:
        """Current fused position estimate."""
        return self._position

    @property
    def confidence(self) -> float:
        """Current fused confidence (0.0-1.0)."""
        return self._position.confidence if self._position else 0.0

    @property
    def confidence_level(self) -> str:
        """D4: Current confidence gate level (high, medium, low)."""
        return self._confidence_level

    def set_position(self, x: int, y: int, source: str = "external"):
        """Set position from external source (e.g., Lissajous locate)."""
        self._position = FusedPosition(
            x=x, y=y, confidence=1.0, source=source,
            timestamp=time.monotonic(),
        )
        self._cnn_ema = 0.5  # Reset EMA to neutral
        self._confidence_level = "medium"

    def update(self, frame: np.ndarray,
               dr_x: Optional[int] = None,
               dr_y: Optional[int] = None) -> Optional[FusedPosition]:
        """Process one frame through the fusion pipeline.

        Args:
            frame: Current 1920x1080 frame.
            dr_x: Dead reckoning X from ESP32Mouse (optional).
            dr_y: Dead reckoning Y from ESP32Mouse (optional).

        Returns:
            Updated FusedPosition, or None if no position known.
        """
        if self._position is None:
            # No position yet — rely on YOLO full-frame scan
            if self._yolo_anchor and frame is not None:
                # Scan center of screen as starting guess
                result = self._yolo_anchor.check(frame, 960, 540)
                if result:
                    self.set_position(result[0], result[1], source="yolo_init")
            return self._position

        pos = self._position
        now = time.monotonic()

        # D5: Feed dead reckoning to motion trail for velocity estimation
        if dr_x is not None and dr_y is not None:
            pos.x = dr_x
            pos.y = dr_y
            if self._trail:
                self._trail.append(
                    dr_x, dr_y, source="dead_reckoning")

        # Step 2: CNN verification at predicted position
        cnn_conf = 0.0
        cnn_method = "none"
        if self._verifier and frame is not None:
            cnn_conf, cnn_method = self._verifier.verify(
                frame, pos.x, pos.y, self._prev_frame)
            # Update EMA
            self._cnn_ema = 0.3 * cnn_conf + 0.7 * self._cnn_ema

        # Step 3: YOLO ground truth check (periodic)
        yolo_conf = 0.0
        if self._yolo_anchor and frame is not None:
            self._yolo_anchor.set_urgency(self._cnn_ema)
            yolo_result = self._yolo_anchor.check(frame, pos.x, pos.y)
            if yolo_result:
                yolo_x, yolo_y, yolo_conf = yolo_result

                # D3: Trail-informed scoring
                yolo_conf = self._trail_score_yolo(
                    yolo_x, yolo_y, yolo_conf)

                # Correct position to YOLO ground truth
                pos.x = yolo_x
                pos.y = yolo_y

        # Step 4: Fuse confidence
        fused_conf = self._fuse_confidence(self._cnn_ema, yolo_conf)

        # D4: Update confidence gate
        self._confidence_level = self._compute_confidence_level(
            self._cnn_ema, yolo_conf, now - pos.timestamp)

        # Determine primary source
        if yolo_conf > 0.7:
            source = "yolo"
        elif self._cnn_ema >= CONFIRM_THRESHOLD:
            source = "cnn"
        elif dr_x is not None:
            source = "dead_reckoning"
        else:
            source = pos.source

        # Update position
        self._position = FusedPosition(
            x=pos.x, y=pos.y,
            confidence=fused_conf,
            source=source,
            cnn_conf=self._cnn_ema,
            yolo_conf=yolo_conf,
            timestamp=now,
        )

        self._prev_frame = frame
        return self._position

    def _trail_score_yolo(self, yolo_x: int, yolo_y: int,
                          raw_conf: float) -> float:
        """D3: Adjust YOLO confidence based on motion trail prediction.

        Boost if detection is near the predicted position (consistent trajectory).
        Reduce if far from prediction during stillness (likely false positive).
        """
        if not self._trail:
            return raw_conf

        pred_x, pred_y = self._trail.extrapolate(dt=0.0)
        dist = ((yolo_x - pred_x) ** 2 + (yolo_y - pred_y) ** 2) ** 0.5

        is_still = not self._trail.is_moving(threshold_px_s=10.0)

        if dist < self.TRAIL_NEAR_PX:
            # Near trail prediction — boost confidence
            return min(1.0, raw_conf * self.TRAIL_BOOST)
        elif dist > self.TRAIL_FAR_PX and is_still:
            # Far from prediction during stillness — likely false positive
            return raw_conf * self.TRAIL_REDUCE

        return raw_conf

    def _compute_confidence_level(self, cnn_ema: float, yolo_conf: float,
                                  age_s: float) -> str:
        """D4: Compute confidence gate level from fused signals.

        Replaces the old age-based 3.0s stale threshold with signal-based levels.
        """
        if cnn_ema >= self.CONF_HIGH and yolo_conf >= 0.3:
            return "high"
        if cnn_ema >= self.CONF_MEDIUM or yolo_conf >= 0.3:
            return "medium"
        return "low"

    def _fuse_confidence(self, cnn_ema: float, yolo_conf: float) -> float:
        """Combine CNN and YOLO confidences into a single score."""
        if yolo_conf > 0:
            # YOLO provides strong signal — weight it heavily
            return 0.4 * cnn_ema + 0.6 * yolo_conf

        # CNN only
        return cnn_ema

    def to_dict(self) -> dict:
        """Status dict for /api/state integration."""
        pos = self._position
        return {
            "x": pos.x if pos else 0,
            "y": pos.y if pos else 0,
            "confidence": round(self.confidence, 3),
            "confidence_level": self._confidence_level,
            "source": pos.source if pos else "none",
            "cnn_ema": round(self._cnn_ema, 3),
            "age_s": round(time.monotonic() - pos.timestamp, 1) if pos else -1,
        }
