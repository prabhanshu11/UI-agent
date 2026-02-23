"""Temporal YOLO — dual-frame inference for cursor velocity estimation.

Track C from TEMPORAL_FUSION_PLAN.md: detect cursor in current frame
AND a lagged frame (~1s ago), match detections by proximity, compute
velocity vectors. This turns single-frame YOLO into a motion-aware
detector that provides speed, direction, and trajectory prediction.

Architecture:
  FrameBuffer: ring buffer of 16 timestamped frames (~3.2s at 5Hz)
  TemporalYOLODetector: dual-frame inference → velocity computation
  TemporalDetection: dataclass with position + velocity

Performance budget: ~30-60ms for dual inference on RTX 2060 SUPER.
Fits within the 200ms per-cycle budget of the detection pipeline.

Usage:
    from src.hardware.temporal_yolo import TemporalYOLODetector

    temporal = TemporalYOLODetector(yolo_detector, lag_s=1.0)
    temporal.push_frame(frame)  # Call every capture cycle

    result = temporal.detect()
    if result:
        print(f"Cursor at ({result.cx}, {result.cy}), "
              f"speed={result.speed:.0f}px/s, dir={result.direction:.0f}deg")
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.hardware.yolo_detector import YOLOCursorDetector, YOLODetection


# Matching threshold: max pixels between current and lagged detection
# to consider them the same cursor
MATCH_DISTANCE_PX = 400

# Minimum time delta between frames for velocity calculation
MIN_DT_S = 0.05  # 50ms — avoids noisy velocity from near-identical frames


@dataclass
class TimestampedFrame:
    """A frame with its capture timestamp."""
    frame: np.ndarray
    timestamp: float  # time.monotonic()


@dataclass
class TemporalDetection:
    """YOLO detection enriched with velocity from dual-frame matching."""
    # Current position (from latest frame)
    cx: int
    cy: int
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    inference_ms: float

    # Velocity (computed from lagged frame match)
    vx: float = 0.0       # px/s, positive = rightward
    vy: float = 0.0       # px/s, positive = downward
    speed: float = 0.0    # |velocity| in px/s
    direction: float = 0.0  # degrees, 0=right, 90=down

    # Lag info
    lag_s: float = 0.0        # Actual time delta between matched frames
    lag_matched: bool = False  # Whether a lagged detection was found
    lag_cx: int = 0            # Lagged detection position
    lag_cy: int = 0

    # Prediction
    predicted_x: int = 0  # Position extrapolated 0.1s into the future
    predicted_y: int = 0


class FrameBuffer:
    """Ring buffer of timestamped frames for temporal analysis.

    Stores up to `maxlen` frames with their capture timestamps.
    At 5Hz capture rate, 16 frames = 3.2s of history.
    """

    def __init__(self, maxlen: int = 16):
        self._buffer: deque[TimestampedFrame] = deque(maxlen=maxlen)

    def push(self, frame: np.ndarray) -> None:
        """Add a frame to the buffer."""
        self._buffer.append(TimestampedFrame(
            frame=frame,
            timestamp=time.monotonic(),
        ))

    def get_lagged(self, lag_s: float = 1.0) -> Optional[TimestampedFrame]:
        """Get the frame closest to `lag_s` seconds ago.

        Returns None if buffer is empty or all frames are too recent.
        """
        if not self._buffer:
            return None

        target_time = time.monotonic() - lag_s
        best = None
        best_delta = float('inf')

        for entry in self._buffer:
            delta = abs(entry.timestamp - target_time)
            if delta < best_delta:
                best_delta = delta
                best = entry

        # Reject if the best frame is too close to "now" (< half the lag)
        if best and best_delta > lag_s * 0.5:
            # Only reject if we don't have ANY old frames
            # (buffer just started filling)
            if best.timestamp > target_time:
                return None

        return best

    @property
    def count(self) -> int:
        return len(self._buffer)

    @property
    def span_s(self) -> float:
        """Time span covered by the buffer in seconds."""
        if len(self._buffer) < 2:
            return 0.0
        return self._buffer[-1].timestamp - self._buffer[0].timestamp


class TemporalYOLODetector:
    """Dual-frame YOLO inference for velocity-enriched cursor detection.

    On each detect() call:
      1. Run YOLO on the current (latest) frame
      2. Run YOLO on a lagged frame (~1s ago)
      3. Match detections by spatial proximity
      4. Compute velocity vector from the matched pair
    """

    def __init__(
        self,
        yolo: YOLOCursorDetector,
        lag_s: float = 1.0,
        buffer_size: int = 16,
    ):
        """
        Args:
            yolo: YOLOCursorDetector instance.
            lag_s: Target lag in seconds for the comparison frame.
            buffer_size: Number of frames to keep in the ring buffer.
        """
        self._yolo = yolo
        self._lag_s = lag_s
        self._buffer = FrameBuffer(maxlen=buffer_size)

        # Stats
        self._detect_count = 0
        self._match_count = 0
        self._total_ms = 0.0
        self._last_result: Optional[TemporalDetection] = None

    def push_frame(self, frame: np.ndarray) -> None:
        """Add a frame to the temporal buffer.

        Call this every capture cycle (e.g., at 5Hz from motion_detection_loop).
        """
        self._buffer.push(frame)

    def detect(self, frame: Optional[np.ndarray] = None) -> Optional[TemporalDetection]:
        """Run dual-frame YOLO inference and compute velocity.

        Args:
            frame: Current frame. If None, uses the latest buffered frame.

        Returns:
            TemporalDetection with position + velocity, or None if no cursor found.
        """
        t0 = time.monotonic()

        # Use provided frame or latest from buffer
        if frame is None:
            if self._buffer.count == 0:
                return None
            # The latest frame is in the buffer, but we need to detect on it
            # This shouldn't happen in normal flow — caller should provide frame
            return None

        # 1. Detect on current frame
        current_dets = self._yolo.detect(frame)
        if not current_dets:
            elapsed_ms = (time.monotonic() - t0) * 1000
            self._total_ms += elapsed_ms
            self._detect_count += 1
            self._last_result = None
            return None

        # Best current detection (highest confidence)
        best_current = max(current_dets, key=lambda d: d.confidence)

        # 2. Get lagged frame and detect
        lagged_entry = self._buffer.get_lagged(self._lag_s)
        lag_matched = False
        vx, vy = 0.0, 0.0
        lag_cx, lag_cy = 0, 0
        actual_lag_s = 0.0

        if lagged_entry is not None:
            lagged_dets = self._yolo.detect(lagged_entry.frame)
            actual_lag_s = time.monotonic() - lagged_entry.timestamp

            if lagged_dets and actual_lag_s > MIN_DT_S:
                # 3. Match by proximity to current detection
                best_lagged = min(
                    lagged_dets,
                    key=lambda d: (
                        (d.cx - best_current.cx) ** 2 +
                        (d.cy - best_current.cy) ** 2
                    ),
                )

                dist = math.sqrt(
                    (best_current.cx - best_lagged.cx) ** 2 +
                    (best_current.cy - best_lagged.cy) ** 2
                )

                if dist < MATCH_DISTANCE_PX:
                    # 4. Compute velocity
                    dt = actual_lag_s
                    vx = (best_current.cx - best_lagged.cx) / dt
                    vy = (best_current.cy - best_lagged.cy) / dt
                    lag_matched = True
                    lag_cx = int(best_lagged.cx)
                    lag_cy = int(best_lagged.cy)
                    self._match_count += 1

        speed = math.sqrt(vx * vx + vy * vy)
        direction = math.degrees(math.atan2(vy, vx)) if speed > 1.0 else 0.0

        # Predict 100ms ahead
        pred_dt = 0.1
        pred_x = max(0, min(1919, int(best_current.cx + vx * pred_dt)))
        pred_y = max(0, min(1079, int(best_current.cy + vy * pred_dt)))

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._total_ms += elapsed_ms
        self._detect_count += 1

        result = TemporalDetection(
            cx=int(best_current.cx),
            cy=int(best_current.cy),
            confidence=float(best_current.confidence),
            bbox=(int(best_current.x1), int(best_current.y1),
                  int(best_current.x2), int(best_current.y2)),
            inference_ms=round(elapsed_ms, 2),
            vx=round(vx, 1),
            vy=round(vy, 1),
            speed=round(speed, 1),
            direction=round(direction, 1),
            lag_s=round(actual_lag_s, 3),
            lag_matched=lag_matched,
            lag_cx=lag_cx,
            lag_cy=lag_cy,
            predicted_x=pred_x,
            predicted_y=pred_y,
        )

        self._last_result = result
        return result

    @property
    def last_result(self) -> Optional[TemporalDetection]:
        return self._last_result

    @property
    def match_rate(self) -> float:
        """Fraction of detections that found a lagged match."""
        if self._detect_count == 0:
            return 0.0
        return self._match_count / self._detect_count

    @property
    def avg_ms(self) -> float:
        """Average dual-inference time in ms."""
        if self._detect_count == 0:
            return 0.0
        return self._total_ms / self._detect_count

    def status(self) -> dict:
        """Status dict for /api/state integration."""
        result = {
            "lag_target_s": self._lag_s,
            "buffer_count": self._buffer.count,
            "buffer_span_s": round(self._buffer.span_s, 2),
            "detect_count": self._detect_count,
            "match_count": self._match_count,
            "match_rate": round(self.match_rate, 3),
            "avg_ms": round(self.avg_ms, 2),
        }
        if self._last_result:
            r = self._last_result
            result["last"] = {
                "cx": r.cx, "cy": r.cy,
                "confidence": round(r.confidence, 3),
                "speed": r.speed,
                "direction": r.direction,
                "lag_matched": r.lag_matched,
                "lag_s": r.lag_s,
                "predicted_x": r.predicted_x,
                "predicted_y": r.predicted_y,
            }
        return result
