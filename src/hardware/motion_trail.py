"""Motion Trail — thread-safe ring buffer of cursor position history.

Central abstraction consumed by CNN, YOLO, tracker, and dashboard.
Replaces inline velocity_history with a unified trail that:
  - Computes velocity from consecutive points
  - Extrapolates predicted position from recent motion
  - Provides speed estimates for silhouette tracker
  - Exposes trail data for dashboard visualization

Sources append points as they detect/estimate the cursor:
  - "daemon_blob"     — primary motion blob from C daemon
  - "yolo"            — YOLO full-frame detection
  - "cnn_verify"      — CNN patch verification
  - "dead_reckoning"  — ESP32 HID command accumulator
  - "passthrough"     — passthrough mode click/move events

Thread Safety:
  All public methods use a threading.Lock for safe concurrent access
  from the capture loop, YOLO thread, and dashboard API.
"""

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MotionTrailPoint:
    """Single point in the motion trail."""
    x: int
    y: int
    timestamp: float           # time.monotonic()
    velocity_x: float = 0.0   # px/s, computed from previous point
    velocity_y: float = 0.0   # px/s
    pixel_count: int = 0       # blob pixel count (0 if non-blob source)
    confidence: float = 0.0    # from CNN/YOLO, 0.0 if unknown
    source: str = "unknown"    # origin of this point


class MotionTrail:
    """Thread-safe ring buffer of recent cursor positions (~120 points).

    At 30fps with motion, this covers ~4 seconds of history.
    Idle frames (no motion) don't generate points, so coverage
    extends longer during still periods.
    """

    def __init__(self, max_points: int = 120):
        self._points: list[MotionTrailPoint] = []
        self._max_points = max_points
        self._lock = threading.Lock()

    def append(self, x: int, y: int, *,
               pixel_count: int = 0,
               confidence: float = 0.0,
               source: str = "unknown") -> None:
        """Add a new point to the trail, auto-computing velocity.

        Args:
            x, y: Pixel coordinates on the 1920x1080 frame.
            pixel_count: Blob pixel count (for daemon_blob source).
            confidence: Model confidence (for yolo/cnn sources).
            source: Origin identifier string.
        """
        now = time.monotonic()

        with self._lock:
            vx, vy = 0.0, 0.0
            if self._points:
                prev = self._points[-1]
                dt = now - prev.timestamp
                if dt > 0.001:  # Avoid division by near-zero
                    vx = (x - prev.x) / dt
                    vy = (y - prev.y) / dt

            point = MotionTrailPoint(
                x=x, y=y,
                timestamp=now,
                velocity_x=vx, velocity_y=vy,
                pixel_count=pixel_count,
                confidence=confidence,
                source=source,
            )
            self._points.append(point)

            # Trim to max size
            if len(self._points) > self._max_points:
                self._points = self._points[-self._max_points:]

    def get_recent(self, max_age_s: float = 2.0) -> list[MotionTrailPoint]:
        """Return points within the last max_age_s seconds.

        Returns a copy — safe to iterate outside the lock.
        """
        cutoff = time.monotonic() - max_age_s
        with self._lock:
            return [p for p in self._points if p.timestamp >= cutoff]

    def get_last(self) -> Optional[MotionTrailPoint]:
        """Return the most recent point, or None if trail is empty."""
        with self._lock:
            return self._points[-1] if self._points else None

    def velocity_estimate(self, window_s: float = 0.3) -> tuple[float, float]:
        """Average velocity (vx, vy) in px/s over the recent window.

        Returns (0.0, 0.0) if fewer than 2 points in window.
        """
        recent = self.get_recent(max_age_s=window_s)
        if len(recent) < 2:
            return (0.0, 0.0)

        # Average the velocity vectors of recent points (skip first — no velocity)
        vx_sum = sum(p.velocity_x for p in recent[1:])
        vy_sum = sum(p.velocity_y for p in recent[1:])
        n = len(recent) - 1
        return (vx_sum / n, vy_sum / n)

    def speed_estimate(self, window_s: float = 0.3) -> float:
        """Scalar speed in px/s over the recent window."""
        vx, vy = self.velocity_estimate(window_s)
        return math.sqrt(vx * vx + vy * vy)

    def extrapolate(self, dt: float = 0.1) -> tuple[int, int]:
        """Predict cursor position dt seconds from the last known point.

        Uses velocity_estimate to project forward. Falls back to
        last known position if no velocity data.

        Args:
            dt: Seconds to extrapolate forward (0.0 = current predicted position).

        Returns:
            (x, y) predicted position, clamped to frame bounds.
        """
        last = self.get_last()
        if last is None:
            return (960, 540)  # Center of 1920x1080 as fallback

        vx, vy = self.velocity_estimate(window_s=0.3)

        # Time since last point + requested extrapolation
        elapsed = time.monotonic() - last.timestamp + dt
        px = int(last.x + vx * elapsed)
        py = int(last.y + vy * elapsed)

        # Clamp to frame bounds
        px = max(0, min(1919, px))
        py = max(0, min(1079, py))
        return (px, py)

    def is_moving(self, threshold_px_s: float = 10.0,
                  window_s: float = 0.5) -> bool:
        """Whether cursor appears to be in motion."""
        return self.speed_estimate(window_s) > threshold_px_s

    def direction_degrees(self, window_s: float = 0.3) -> float:
        """Movement direction in degrees (0=right, 90=down, etc.)."""
        vx, vy = self.velocity_estimate(window_s)
        if abs(vx) < 0.1 and abs(vy) < 0.1:
            return 0.0
        return math.degrees(math.atan2(vy, vx))

    def length(self) -> int:
        """Current number of points in the trail."""
        with self._lock:
            return len(self._points)

    def clear(self) -> None:
        """Clear all trail points."""
        with self._lock:
            self._points.clear()

    def to_api_dict(self, max_age_s: float = 2.0) -> dict:
        """Serialize for /api/state response.

        Returns a compact dict with recent points and velocity.
        """
        recent = self.get_recent(max_age_s)
        vx, vy = self.velocity_estimate()
        speed = self.speed_estimate()
        last = self.get_last()

        return {
            "points": [
                {"x": p.x, "y": p.y, "src": p.source,
                 "conf": round(p.confidence, 2),
                 "age_s": round(time.monotonic() - p.timestamp, 2)}
                for p in recent[-20:]  # Last 20 points for API (keep response small)
            ],
            "velocity": {"vx": round(vx, 1), "vy": round(vy, 1)},
            "speed": round(speed, 1),
            "direction_deg": round(self.direction_degrees(), 1),
            "moving": self.is_moving(),
            "total_points": self.length(),
            "last_x": last.x if last else None,
            "last_y": last.y if last else None,
        }
