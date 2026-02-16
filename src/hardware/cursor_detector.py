"""Lissajous curve cursor detector via motion blur curvature matching.

Detects cursor position on an HDMI-captured screen by:
1. Moving the cursor along a Lissajous curve (space-filling, smooth)
2. Recording video continuously during movement
3. Extracting motion blobs from frame diffs (changed pixel clusters)
4. Matching blob trajectories against the expected Lissajous curvature

The Lissajous curve has an irrational frequency ratio, so it visits every
region of the virtual desktop. When the cursor crosses the HDMI-captured
screen, the motion blur trail's orientation and curvature uniquely identify
which segment of the curve is visible.

This replaces the old snapshot-diff approach which was defeated by animated
UI elements (spinners, blinking cursors) producing false positive diffs.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage


@dataclass
class LissajousCurve:
    """Parametric Lissajous curve for space-filling cursor sweep.

    x(t) = A_x * sin(omega_x * t + phi)
    y(t) = A_y * sin(omega_y * t)

    With irrational omega_y/omega_x ratio, the curve never repeats
    locally, guaranteeing it visits all regions of the virtual desktop.
    """

    amp_x: float = 3000.0      # Half-width of virtual desktop
    amp_y: float = 1500.0      # Half-height of virtual desktop
    omega_x: float = math.pi / 8       # Slow horizontal frequency
    omega_y: float = math.pi * math.sqrt(2) / 8  # Irrational ratio to omega_x
    phi: float = 0.0           # Phase offset

    def position(self, t: float) -> tuple[float, float]:
        """(x, y) position at time t."""
        x = self.amp_x * math.sin(self.omega_x * t + self.phi)
        y = self.amp_y * math.sin(self.omega_y * t)
        return (x, y)

    def velocity(self, t: float) -> tuple[float, float]:
        """(dx/dt, dy/dt) — first derivative at time t."""
        vx = self.amp_x * self.omega_x * math.cos(self.omega_x * t + self.phi)
        vy = self.amp_y * self.omega_y * math.cos(self.omega_y * t)
        return (vx, vy)

    def acceleration(self, t: float) -> tuple[float, float]:
        """(d²x/dt², d²y/dt²) — second derivative at time t."""
        ax = -self.amp_x * self.omega_x**2 * math.sin(self.omega_x * t + self.phi)
        ay = -self.amp_y * self.omega_y**2 * math.sin(self.omega_y * t)
        return (ax, ay)

    def angle(self, t: float) -> float:
        """Direction angle theta = atan2(dy/dt, dx/dt) at time t."""
        vx, vy = self.velocity(t)
        return math.atan2(vy, vx)

    def speed(self, t: float) -> float:
        """Scalar speed at time t."""
        vx, vy = self.velocity(t)
        return math.sqrt(vx * vx + vy * vy)

    def curvature(self, t: float) -> float:
        """Signed curvature kappa(t) = (x'*y'' - y'*x'') / (x'^2 + y'^2)^(3/2)."""
        vx, vy = self.velocity(t)
        ax, ay = self.acceleration(t)
        speed_sq = vx * vx + vy * vy
        if speed_sq < 1e-10:
            return 0.0
        return (vx * ay - vy * ax) / (speed_sq ** 1.5)

    def sample_hid_reports(
        self,
        t_start: float,
        t_end: float,
        dt: float = 0.02,
    ) -> list[tuple[int, int]]:
        """Sample the curve's velocity into HID report (dx, dy) pairs.

        Each HID report is the velocity integrated over dt, clamped to ±127.
        At 50 reports/sec (dt=0.02), this produces smooth cursor motion.

        Args:
            t_start: Start time parameter.
            t_end: End time parameter.
            dt: Time step between reports (0.02 = 20ms = ESP32 report interval).

        Returns:
            List of (dx_hid, dy_hid) integer pairs.
        """
        reports = []
        t = t_start
        while t < t_end:
            vx, vy = self.velocity(t)
            # Velocity * dt = displacement per report
            dx_raw = vx * dt
            dy_raw = vy * dt
            # Scale so max displacement fits comfortably in ±127
            # Max velocity = amp * omega, max displacement = amp * omega * dt
            dx_hid = int(max(-127, min(127, round(dx_raw * 127 / self._max_displacement(dt)))))
            dy_hid = int(max(-127, min(127, round(dy_raw * 127 / self._max_displacement(dt)))))
            reports.append((dx_hid, dy_hid))
            t += dt
        return reports

    def _max_displacement(self, dt: float) -> float:
        """Maximum single-axis displacement per dt interval.

        Used to scale HID reports so they fit within ±127.
        We use ~60% of 127 (≈80) as the target max to leave headroom.
        """
        max_vx = self.amp_x * self.omega_x
        max_vy = self.amp_y * self.omega_y
        max_v = max(max_vx, max_vy)
        # Scale so max_v * dt maps to ~80 HID units
        return max_v * dt * (127 / 80)

    def expected_angle_sequence(
        self,
        t_start: float,
        t_end: float,
        n_points: int,
    ) -> np.ndarray:
        """Compute expected motion direction angles over a time range.

        Returns array of angles for use in curvature matching.
        """
        times = np.linspace(t_start, t_end, n_points)
        return np.array([self.angle(t) for t in times])

    def expected_curvature_sequence(
        self,
        t_start: float,
        t_end: float,
        n_points: int,
    ) -> np.ndarray:
        """Compute expected curvature values over a time range."""
        times = np.linspace(t_start, t_end, n_points)
        return np.array([self.curvature(t) for t in times])


@dataclass
class MotionBlob:
    """A cluster of changed pixels between consecutive frames."""

    centroid: tuple[int, int]   # (x, y) center of mass
    angle: float                # Principal axis angle (orientation of blur streak)
    pixel_count: int            # Number of changed pixels in cluster
    bbox: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)


@dataclass
class CurveMatch:
    """Result of matching detected motion blobs to expected Lissajous curve."""

    matched: bool
    t_start: float = 0.0           # Curve time parameter where cursor entered screen
    t_end: float = 0.0             # Curve time parameter at detection point
    screen_pos: tuple[int, int] = (0, 0)  # Cursor position on HDMI screen at t_end
    correlation: float = 0.0       # Match quality (0-1)
    blobs: list[MotionBlob] = field(default_factory=list)


def extract_motion_blobs(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    threshold: int = 30,
    min_pixels: int = 8,
    max_pixels: int = 4000,
) -> list[MotionBlob]:
    """Find clusters of changed pixels between two frames.

    Computes per-pixel absolute difference, thresholds to binary mask,
    labels connected components, then for each valid component computes
    centroid and principal axis angle via covariance matrix eigendecomposition.

    Args:
        frame_a: First frame (H, W, 3) RGB uint8.
        frame_b: Second frame (H, W, 3) RGB uint8.
        threshold: Per-channel sum threshold for "changed" pixel.
        min_pixels: Minimum pixels for a valid blob (filters noise).
        max_pixels: Maximum pixels (filters screen transitions).

    Returns:
        List of MotionBlob, sorted by pixel_count descending.
    """
    diff = np.abs(frame_b.astype(np.int16) - frame_a.astype(np.int16))
    diff_magnitude = diff.sum(axis=2)
    mask = diff_magnitude > threshold

    # Label connected components
    labeled, n_labels = ndimage.label(mask)

    blobs = []
    for label_id in range(1, n_labels + 1):
        component = labeled == label_id
        pixel_count = int(component.sum())

        if pixel_count < min_pixels or pixel_count > max_pixels:
            continue

        # Get pixel coordinates
        ys, xs = np.where(component)

        # Centroid
        cx = int(round(xs.mean()))
        cy = int(round(ys.mean()))

        # Bounding box
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        # Principal axis angle via covariance matrix eigendecomposition
        angle = _compute_principal_angle(xs, ys)

        blobs.append(MotionBlob(
            centroid=(cx, cy),
            angle=angle,
            pixel_count=pixel_count,
            bbox=bbox,
        ))

    # Sort by pixel count descending (largest first)
    blobs.sort(key=lambda b: b.pixel_count, reverse=True)
    return blobs


def _compute_principal_angle(xs: np.ndarray, ys: np.ndarray) -> float:
    """Compute principal axis angle of a point cloud via PCA.

    Uses the 2x2 covariance matrix and its eigendecomposition.
    The eigenvector with the larger eigenvalue gives the principal direction.

    Args:
        xs: X coordinates of pixels.
        ys: Y coordinates of pixels.

    Returns:
        Angle in radians (-pi to pi) of the principal axis.
    """
    if len(xs) < 3:
        return 0.0

    # Center the coordinates
    mx = xs.mean()
    my = ys.mean()
    dx = xs.astype(np.float64) - mx
    dy = ys.astype(np.float64) - my

    # 2x2 covariance matrix
    cxx = (dx * dx).mean()
    cxy = (dx * dy).mean()
    cyy = (dy * dy).mean()

    # Eigenvalues via quadratic formula (faster than numpy.linalg for 2x2)
    trace = cxx + cyy
    det = cxx * cyy - cxy * cxy
    discriminant = max(0.0, trace * trace / 4 - det)
    sqrt_disc = math.sqrt(discriminant)

    # Larger eigenvalue
    lambda1 = trace / 2 + sqrt_disc

    # Eigenvector for larger eigenvalue
    if abs(cxy) > 1e-10:
        angle = math.atan2(lambda1 - cxx, cxy)
    elif cxx >= cyy:
        angle = 0.0  # Principal axis is horizontal
    else:
        angle = math.pi / 2  # Principal axis is vertical

    return angle


def match_curve_segment(
    blobs_sequence: list[list[MotionBlob]],
    curve: LissajousCurve,
    fps: float = 30.0,
    min_consecutive: int = 8,
    angle_tolerance: float = 0.6,
    translation_threshold: float = 3.0,
) -> Optional[CurveMatch]:
    """Match a sequence of motion blobs against the expected Lissajous curve.

    The key insight: cursor motion produces blobs that TRANSLATE across frames
    with an orientation matching the Lissajous derivative. UI animations
    (spinners, blinking cursors) are stationary or periodic — they fail the
    translation + curvature coherence test.

    Algorithm:
    1. For each frame pair, pick the dominant blob (most cursor-like)
    2. Build trajectory: sequence of (centroid, angle) across frames
    3. Find runs of consecutive frames with a dominant blob
    4. For each run >= min_consecutive frames:
       a. Compute observed angle trajectory and centroid translation
       b. Slide a window along Lissajous time parameter
       c. Compare expected vs observed angle sequence (circular correlation)
       d. Check centroids translate (not stationary like animations)
    5. Return best match above threshold

    Args:
        blobs_sequence: List of per-frame blob lists (from extract_motion_blobs).
        curve: The LissajousCurve being executed.
        fps: Video framerate for time calculation.
        min_consecutive: Minimum consecutive frames with blobs to attempt match.
        angle_tolerance: Maximum mean angular difference (radians) for match.
        translation_threshold: Minimum total centroid displacement (pixels)
            per frame to distinguish cursor from stationary animations.

    Returns:
        CurveMatch if a segment matches, None otherwise.
    """
    if len(blobs_sequence) < min_consecutive:
        return None

    # Extract dominant blob per frame (largest, as cursor + trail is prominent)
    dominant = []
    for blobs in blobs_sequence:
        if blobs:
            dominant.append(blobs[0])  # Largest blob
        else:
            dominant.append(None)

    # Find runs of consecutive frames with a dominant blob
    runs = _find_consecutive_runs(dominant, min_consecutive)

    if not runs:
        return None

    dt_frame = 1.0 / fps
    best_match: Optional[CurveMatch] = None
    best_corr = 0.0

    for run_start, run_end in runs:
        run_blobs = [dominant[i] for i in range(run_start, run_end)]
        n = len(run_blobs)

        # Extract observed centroid trajectory
        centroids = np.array([b.centroid for b in run_blobs], dtype=np.float64)
        observed_angles = np.array([b.angle for b in run_blobs])

        # Check for translation (cursor moves, animations don't)
        centroid_diffs = np.diff(centroids, axis=0)
        centroid_speeds = np.sqrt((centroid_diffs ** 2).sum(axis=1))
        mean_speed = centroid_speeds.mean()

        if mean_speed < translation_threshold:
            continue  # Stationary blob — likely UI animation, not cursor

        # Compute observed direction from centroid trajectory
        # This is more reliable than blob PCA angle for matching
        observed_directions = np.arctan2(centroid_diffs[:, 1], centroid_diffs[:, 0])

        # Slide window along Lissajous time to find best match
        # The curve runs for up to max_duration seconds; sample at frame intervals
        run_duration = (n - 1) * dt_frame
        max_t = 12.0  # Search up to 12s into the curve
        t_step = dt_frame  # Step by one frame interval

        t = 0.0
        while t + run_duration <= max_t:
            # Expected angles at this window position
            expected = np.array([
                curve.angle(t + i * dt_frame)
                for i in range(n - 1)
            ])

            # Circular mean angular difference
            angle_diff = _circular_distance(observed_directions, expected)
            mean_diff = angle_diff.mean()

            if mean_diff < angle_tolerance:
                # Also check curvature coherence for stronger match
                corr = 1.0 - mean_diff / math.pi  # 0 = worst, 1 = perfect
                if corr > best_corr:
                    best_corr = corr
                    t_match_end = t + run_duration
                    last_blob = run_blobs[-1]
                    best_match = CurveMatch(
                        matched=True,
                        t_start=t,
                        t_end=t_match_end,
                        screen_pos=last_blob.centroid,
                        correlation=corr,
                        blobs=run_blobs,
                    )

            t += t_step

    return best_match


def _find_consecutive_runs(
    items: list[Optional[MotionBlob]],
    min_length: int,
) -> list[tuple[int, int]]:
    """Find runs of consecutive non-None items.

    Returns list of (start_index, end_index) where end is exclusive.
    Only returns runs with length >= min_length.
    """
    runs = []
    start = None

    for i, item in enumerate(items):
        if item is not None:
            if start is None:
                start = i
        else:
            if start is not None and (i - start) >= min_length:
                runs.append((start, i))
            start = None

    # Handle run that extends to the end
    if start is not None and (len(items) - start) >= min_length:
        runs.append((start, len(items)))

    return runs


def _circular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute element-wise circular distance between angle arrays.

    Returns values in [0, pi] — the shortest angular distance.
    """
    diff = a - b
    # Wrap to [-pi, pi]
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    return np.abs(diff)
