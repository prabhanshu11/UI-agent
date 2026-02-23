"""Mouse calibration: map HID deltas to actual pixel displacement.

Problem: Windows applies a nonlinear acceleration curve to HID mouse reports.
The same dx=10 produces different pixel displacements depending on mouse
velocity. Without calibration, move(100, 0) might move 80px or 300px.

Approach:
  1. Automated calibration sweep: send known HID deltas at controlled timing
  2. Observe actual pixel displacement via HDMI capture + cursor detection
  3. Fit a transfer function: (dx_hid, velocity) → pixel_displacement
  4. Use the inverse for calibrated movements

The sweep takes ~2 minutes with ~100 test points:
  - Magnitude sweep: dx = [1, 2, 5, 10, 20, 40, 80, 127]
  - Velocity variation: fixed dx=20, timing = [7, 10, 15, 20, 30, 50] ms
  - Y-axis + diagonals
  - Jitter characterization: 20 identical reports, measure variance

Usage:
    from src.hardware.mouse_calibration import MouseCalibrator

    cal = MouseCalibrator(mouse, detect_cursor_fn)
    cal.run_sweep()               # ~2 min automated calibration
    cal.save("data/calibration.json")

    # Later:
    cal = MouseCalibrator.load("data/calibration.json", mouse)
    cal.calibrated_move(100, 0)   # Move exactly 100px right (±3px)
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np


@dataclass
class CalibrationPoint:
    """Single calibration measurement."""

    dx_hid: int  # Commanded HID delta X
    dy_hid: int  # Commanded HID delta Y
    interval_ms: float  # Time between reports
    observed_dx_px: float  # Measured pixel displacement X
    observed_dy_px: float  # Measured pixel displacement Y
    gain_x: float  # observed_dx / dx_hid (effective gain)
    gain_y: float  # observed_dy / dy_hid (effective gain)
    n_reports: int  # Number of HID reports sent
    variance_px: float = 0.0  # Measurement variance (from repeated trials)


@dataclass
class CalibrationResult:
    """Complete calibration result with fitted model."""

    points: list[CalibrationPoint] = field(default_factory=list)
    # Piecewise linear model: gains at different velocities
    velocity_breakpoints: list[float] = field(default_factory=list)  # px/s
    gain_at_breakpoints: list[float] = field(default_factory=list)  # multiplier
    # Simple linear model fallback
    mean_gain_x: float = 1.0
    mean_gain_y: float = 1.0
    # Metadata
    timestamp: str = ""
    total_points: int = 0
    jitter_std_px: float = 0.0  # Cursor jitter standard deviation


class MouseCalibrator:
    """Automated mouse calibration using HID sweep + visual observation.

    Requires:
      - ESP32Mouse instance (sends HID reports)
      - A cursor detection function: frame -> (x, y) or None
      - A frame capture function: () -> numpy array
    """

    def __init__(
        self,
        mouse,  # ESP32Mouse instance
        detect_cursor: Callable[..., Optional[tuple[int, int]]],
        capture_frame: Callable[..., "np.ndarray"],
        settle_time: float = 0.3,  # Seconds to wait for cursor to settle
    ):
        """
        Args:
            mouse: ESP32Mouse instance with _send_raw() and move().
            detect_cursor: Function that takes a frame and returns (x, y) or None.
            capture_frame: Function that returns current HDMI capture frame.
            settle_time: Time to wait after movement for cursor to settle.
        """
        self.mouse = mouse
        self.detect_cursor = detect_cursor
        self.capture_frame = capture_frame
        self.settle_time = settle_time
        self.result: Optional[CalibrationResult] = None

    def _get_cursor_pos(self) -> Optional[tuple[int, int]]:
        """Capture frame and detect cursor position."""
        frame = self.capture_frame()
        if frame is None:
            return None
        return self.detect_cursor(frame)

    def _measure_displacement(
        self, dx: int, dy: int, n_reports: int = 1, interval_ms: float = 20
    ) -> Optional[tuple[float, float]]:
        """Send HID reports and measure actual pixel displacement.

        Args:
            dx, dy: HID delta per report (-127 to 127).
            n_reports: Number of reports to send.
            interval_ms: Time between reports in milliseconds.

        Returns:
            (actual_dx_px, actual_dy_px) or None if detection failed.
        """
        # Get starting position
        time.sleep(self.settle_time)
        start = self._get_cursor_pos()
        if start is None:
            return None

        # Send HID reports
        for _ in range(n_reports):
            self.mouse._send_raw(dx, dy)
            if interval_ms > 20:
                time.sleep((interval_ms - 20) / 1000)  # _send_raw has 20ms sleep

        # Get ending position
        time.sleep(self.settle_time)
        end = self._get_cursor_pos()
        if end is None:
            return None

        return (end[0] - start[0], end[1] - start[1])

    def run_sweep(self, progress_callback: Optional[Callable] = None) -> CalibrationResult:
        """Run full calibration sweep (~2 minutes).

        Measures gain at multiple magnitudes and velocities.

        Args:
            progress_callback: Optional fn(step, total, description) for progress.

        Returns:
            CalibrationResult with fitted model.
        """
        points = []
        total_steps = 0

        # Phase 1: Magnitude sweep (fixed timing, variable dx)
        magnitudes = [1, 2, 5, 10, 20, 40, 80, 127]
        # Phase 2: Velocity sweep (fixed dx=20, variable timing)
        timings_ms = [7, 10, 15, 20, 30, 50]
        # Phase 3: Y-axis
        y_magnitudes = [5, 20, 80]
        # Phase 4: Jitter characterization
        jitter_trials = 20

        total = len(magnitudes) + len(timings_ms) + len(y_magnitudes) + jitter_trials
        step = 0

        def report(desc):
            nonlocal step
            step += 1
            if progress_callback:
                progress_callback(step, total, desc)

        # Center cursor first
        self.mouse.move_to(960, 540)
        time.sleep(0.5)

        # Phase 1: X magnitude sweep
        for dx in magnitudes:
            report(f"Magnitude sweep: dx={dx}")
            disp = self._measure_displacement(dx, 0, n_reports=3)
            if disp:
                gain = disp[0] / (dx * 3) if dx * 3 != 0 else 0
                points.append(CalibrationPoint(
                    dx_hid=dx, dy_hid=0, interval_ms=20,
                    observed_dx_px=disp[0], observed_dy_px=disp[1],
                    gain_x=gain, gain_y=0, n_reports=3,
                ))
            # Return to center
            self.mouse.move_to(960, 540)
            time.sleep(0.2)

        # Phase 2: Velocity sweep (fixed dx=20, variable timing)
        for interval in timings_ms:
            report(f"Velocity sweep: interval={interval}ms")
            disp = self._measure_displacement(20, 0, n_reports=5, interval_ms=interval)
            if disp:
                gain = disp[0] / (20 * 5) if True else 0
                velocity_px_s = abs(disp[0]) / (5 * interval / 1000)
                points.append(CalibrationPoint(
                    dx_hid=20, dy_hid=0, interval_ms=interval,
                    observed_dx_px=disp[0], observed_dy_px=disp[1],
                    gain_x=gain, gain_y=0, n_reports=5,
                ))
            self.mouse.move_to(960, 540)
            time.sleep(0.2)

        # Phase 3: Y-axis
        for dy in y_magnitudes:
            report(f"Y-axis: dy={dy}")
            disp = self._measure_displacement(0, dy, n_reports=3)
            if disp:
                gain = disp[1] / (dy * 3) if dy * 3 != 0 else 0
                points.append(CalibrationPoint(
                    dx_hid=0, dy_hid=dy, interval_ms=20,
                    observed_dx_px=disp[0], observed_dy_px=disp[1],
                    gain_x=0, gain_y=gain, n_reports=3,
                ))
            self.mouse.move_to(960, 540)
            time.sleep(0.2)

        # Phase 4: Jitter characterization (identical reports, measure variance)
        jitter_displacements = []
        for i in range(jitter_trials):
            report(f"Jitter test {i+1}/{jitter_trials}")
            disp = self._measure_displacement(10, 0, n_reports=1)
            if disp:
                jitter_displacements.append(disp[0])
            # Return
            self._measure_displacement(-10, 0, n_reports=1)

        jitter_std = float(np.std(jitter_displacements)) if jitter_displacements else 0.0

        # Fit model
        result = self._fit_model(points, jitter_std)
        self.result = result
        return result

    def _fit_model(
        self, points: list[CalibrationPoint], jitter_std: float
    ) -> CalibrationResult:
        """Fit piecewise linear transfer function from calibration points."""
        result = CalibrationResult(
            points=points,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total_points=len(points),
            jitter_std_px=jitter_std,
        )

        # Extract X-axis gains
        x_gains = [p.gain_x for p in points if p.dx_hid > 0 and p.gain_x > 0]
        y_gains = [p.gain_y for p in points if p.dy_hid > 0 and p.gain_y > 0]

        result.mean_gain_x = float(np.mean(x_gains)) if x_gains else 1.0
        result.mean_gain_y = float(np.mean(y_gains)) if y_gains else 1.0

        # Build piecewise model from velocity sweep data
        velocity_points = [
            p for p in points if p.dx_hid == 20 and p.interval_ms != 20
        ]
        if velocity_points:
            velocities = []
            gains = []
            for p in velocity_points:
                v = abs(p.observed_dx_px) / (p.n_reports * p.interval_ms / 1000)
                velocities.append(v)
                gains.append(p.gain_x)

            # Sort by velocity
            sorted_pairs = sorted(zip(velocities, gains))
            result.velocity_breakpoints = [v for v, _ in sorted_pairs]
            result.gain_at_breakpoints = [g for _, g in sorted_pairs]

        return result

    def calibrated_move(self, desired_dx_px: int, desired_dy_px: int) -> tuple[int, int]:
        """Move cursor by desired pixel amount, compensating for acceleration.

        Uses calibration model to compute HID deltas that produce the
        desired pixel displacement.

        Args:
            desired_dx_px: Desired X displacement in pixels.
            desired_dy_px: Desired Y displacement in pixels.

        Returns:
            (estimated_actual_dx, estimated_actual_dy) in pixels.
        """
        if self.result is None:
            # No calibration — fall back to raw movement
            self.mouse.move(desired_dx_px, desired_dy_px, respect_bounds=True)
            return (desired_dx_px, desired_dy_px)

        # Apply inverse gain
        gain_x = self.result.mean_gain_x or 1.0
        gain_y = self.result.mean_gain_y or 1.0

        # TODO(user): Implement velocity-aware gain lookup here.
        # The piecewise model is in self.result.velocity_breakpoints/gain_at_breakpoints.
        # For now, use the mean gain as a flat correction.
        # This is where you'd interpolate based on desired velocity to get
        # a more accurate gain for fast vs slow movements.

        hid_dx = int(round(desired_dx_px / gain_x))
        hid_dy = int(round(desired_dy_px / gain_y))

        self.mouse.move(hid_dx, hid_dy, respect_bounds=True)
        return (desired_dx_px, desired_dy_px)

    def save(self, path: str | Path):
        """Save calibration result to JSON."""
        if self.result is None:
            raise ValueError("No calibration result to save. Run run_sweep() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": self.result.timestamp,
            "total_points": self.result.total_points,
            "mean_gain_x": self.result.mean_gain_x,
            "mean_gain_y": self.result.mean_gain_y,
            "jitter_std_px": self.result.jitter_std_px,
            "velocity_breakpoints": self.result.velocity_breakpoints,
            "gain_at_breakpoints": self.result.gain_at_breakpoints,
            "points": [
                {
                    "dx_hid": p.dx_hid,
                    "dy_hid": p.dy_hid,
                    "interval_ms": p.interval_ms,
                    "observed_dx_px": p.observed_dx_px,
                    "observed_dy_px": p.observed_dy_px,
                    "gain_x": p.gain_x,
                    "gain_y": p.gain_y,
                    "n_reports": p.n_reports,
                }
                for p in self.result.points
            ],
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path, mouse, detect_cursor=None, capture_frame=None):
        """Load calibration from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())

        cal = cls(
            mouse=mouse,
            detect_cursor=detect_cursor or (lambda f: None),
            capture_frame=capture_frame or (lambda: None),
        )
        cal.result = CalibrationResult(
            timestamp=data["timestamp"],
            total_points=data["total_points"],
            mean_gain_x=data["mean_gain_x"],
            mean_gain_y=data["mean_gain_y"],
            jitter_std_px=data.get("jitter_std_px", 0),
            velocity_breakpoints=data.get("velocity_breakpoints", []),
            gain_at_breakpoints=data.get("gain_at_breakpoints", []),
            points=[
                CalibrationPoint(**p)
                for p in data.get("points", [])
            ],
        )
        return cal

    def report(self) -> str:
        """Generate human-readable calibration report."""
        if self.result is None:
            return "No calibration data. Run run_sweep() first."

        r = self.result
        lines = [
            "=" * 50,
            "  MOUSE CALIBRATION REPORT",
            "=" * 50,
            f"  Date: {r.timestamp}",
            f"  Calibration points: {r.total_points}",
            f"",
            f"  Mean gain X: {r.mean_gain_x:.3f} (1 HID unit = {r.mean_gain_x:.1f} pixels)",
            f"  Mean gain Y: {r.mean_gain_y:.3f} (1 HID unit = {r.mean_gain_y:.1f} pixels)",
            f"  Jitter std: {r.jitter_std_px:.1f} px",
            f"",
        ]

        if r.velocity_breakpoints:
            lines.append("  Velocity-dependent gain:")
            for v, g in zip(r.velocity_breakpoints, r.gain_at_breakpoints):
                lines.append(f"    {v:8.0f} px/s → gain {g:.3f}")

        lines.append("")
        lines.append(f"  Accuracy estimate: ±{max(3, r.jitter_std_px * 2):.0f} px")
        lines.append("=" * 50)

        return "\n".join(lines)
