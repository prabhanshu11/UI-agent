"""KVM Control Dashboard — Live motion detection + cursor tracking + jitter.

Shows real-time:
- HDMI capture with motion blob overlays (highlighted rectangles + centroids)
- Cursor position crosshair
- Three-tier tracker status
- Anti-sleep jitter activity
- ESP32 mouse + Pi keyboard connection status

Usage:
    uv run python -m src.web.kvm_dashboard
"""

import base64
import io
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.hardware.esp32_mouse import ESP32Mouse
from src.hardware.hdmi_sensor import HDMISensor
from src.hardware.cursor_detector import extract_motion_blobs, MotionBlob
from src.hardware.cursor_tracker import CursorTracker
from src.hardware.cursor_sample_collector import CursorSampleCollector, extract_gray_patch
from src.hardware.cursor_recognizer import CursorRecognizer
from src.hardware.daemon_client import DaemonClient
from src.hardware.cursor_vision_detector import ClaudeVisionCursorDetector
from src.hardware.silhouette_tracker import SilhouetteTracker
from src.hardware.commanded_movement import CommandedMovement
from src.hardware.yolo_detector import YOLOCursorDetector

def utc_now_ms() -> str:
    """UTC timestamp with millisecond precision, e.g. '2026-02-20T14:30:05.123Z'."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


app = FastAPI(title="KVM Control Dashboard")


# ── Shared state ────────────────────────────────────────────────────

@dataclass
class MotionState:
    """Thread-safe shared state between motion service and API."""
    frame_raw: Optional[np.ndarray] = None
    frame_overlay: Optional[bytes] = None
    blobs: list[MotionBlob] = field(default_factory=list)
    blob_count: int = 0
    cursor_x: int = 0
    cursor_y: int = 0
    cursor_method: str = "unknown"
    cursor_age_s: float = 999.0
    tracker_tier: str = "none"
    mouse_connected: bool = False
    heartbeat_active: bool = False
    anti_sleep_active: bool = False
    last_jitter_time: float = 0.0
    jitter_count: int = 0
    fps: float = 0.0
    frame_count: int = 0
    pi_keyboard_connected: Optional[bool] = None
    cnn_confidence: float = 0.0
    cnn_model_version: str = "none"
    cnn_inference_ms: float = 0.0
    cursor_validated: bool = False
    sample_count_pos: int = 0
    sample_count_neg: int = 0
    # Silhouette tracking (Phase 2)
    silhouette_active: bool = False
    silhouette_method: str = "none"
    silhouette_hz: float = 0.0
    silhouette_roi_size: int = 200
    silhouette_confidence: float = 0.0
    silhouette_latency_ms: float = 0.0
    silhouette_misses: int = 0
    error: Optional[str] = None
    # YOLO full-screen detection
    yolo_active: bool = False
    yolo_x: int = 0
    yolo_y: int = 0
    yolo_confidence: float = 0.0
    yolo_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2
    yolo_inference_ms: float = 0.0
    yolo_timestamp: float = 0.0  # monotonic time of last detection
    yolo_detection_count: int = 0
    # Claude Vision (ground truth)
    vision_x: int = 0
    vision_y: int = 0
    vision_confidence: str = "none"  # "high", "medium", "low", "none"
    vision_cursor_type: str = "unknown"
    vision_latency_ms: float = 0.0
    vision_timestamp: float = 0.0  # monotonic time of last detection
    vision_count: int = 0
    # Pipeline profiling
    prof_jpeg_decode_ms: float = 0.0
    prof_overlay_draw_ms: float = 0.0
    prof_jpeg_encode_ms: float = 0.0
    prof_frame_total_ms: float = 0.0
    prof_daemon_timestamp_ns: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

state = MotionState()


@dataclass
class DashboardConfig:
    """Runtime-tunable parameters exposed via /api/config."""
    frame_fps: float = 5.0          # Overlayed video render rate (Hz)
    cnn_interval_s: float = 30.0    # CNN validation cycle (seconds)
    jitter_interval_s: float = 30.0 # Anti-sleep jitter interval
    jitter_pixels: int = 3          # Jitter amplitude (px)
    verify_every_n: int = 3         # Run verification probe every Nth jitter
    verify_drift_px: int = 100      # Drift threshold: probe vs tracked position
    lock: threading.Lock = field(default_factory=threading.Lock)

config = DashboardConfig()

# ── Pipeline profiler state ────────────────────────────────────────
profiler_samples: list[dict] = []
profiler_running: bool = False
profiler_start_time: float = 0.0
PROFILER_DURATION_S = 180  # 3 minutes

# ── Velocity test state ───────────────────────────────────────────
_vtest_running: bool = False
_vtest_results: list[dict] = []
_vtest_progress: dict = {"profile": "", "step": 0, "total": 0, "phase": "idle"}
_vtest_summary: dict = {}
_VTEST_DIR = Path(__file__).parent.parent.parent / "data" / "velocity_test"

VELOCITY_PROFILES = {
    "V0": {"desc": "Stationary", "dx": 0, "dy": 0, "steps": 50, "delay_ms": 200},
    "V1": {"desc": "Slow diagonal", "dx": 2, "dy": 2, "steps": 25, "delay_ms": 200},
    "V2": {"desc": "Medium horizontal", "dx": 10, "dy": 0, "steps": 30, "delay_ms": 100},
    "V3": {"desc": "Fast sweep", "dx": 50, "dy": 0, "steps": 10, "delay_ms": 80},
    "V4": {"desc": "Snap to corner", "dx": 400, "dy": 300, "steps": 1, "delay_ms": 2000},
    "A1": {"desc": "Accelerating", "accel": "up", "start_speed": 2, "end_speed": 30, "steps": 20, "delay_ms": 150},
    "A2": {"desc": "Decelerate+stop", "accel": "down", "start_speed": 30, "end_speed": 0, "steps": 20, "delay_ms": 150},
    "A3": {"desc": "Tiny jitter", "dx": 2, "dy": 0, "steps": 50, "delay_ms": 200, "alternate": True},
}


def _auto_label(error_px: float) -> str:
    if error_px < 20:
        return "accurate"
    if error_px < 80:
        return "drift"
    if error_px < 200:
        return "wrong"
    return "lost"


# ── Loss event sampling (ring buffer + trigger) ─────────────────
from collections import deque

_LOSS_EVENT_DIR = Path(__file__).parent.parent.parent / "data" / "loss_events"
_RING_BUFFER_SIZE = 300   # ~60s at 5Hz
_RING_BUFFER_HZ = 5.0
_LOSS_PRE_WINDOW_S = 10.0
_LOSS_POST_WINDOW_S = 15.0
_LOSS_COOLDOWN_S = 30.0

_ring_buffer: deque = deque(maxlen=_RING_BUFFER_SIZE)
_ring_last_t: float = 0.0
_ring_lock = threading.Lock()

_loss_event_active: bool = False
_loss_event_cooldown: float = 0.0
_loss_prev = {"validated": False, "cnn": 0.0, "sil": 0.0, "x": 0, "y": 0}
_loss_events_log: list[dict] = []  # in-memory index of captured events


mouse: Optional[ESP32Mouse] = None
sensor: Optional[HDMISensor] = None
tracker: Optional[CursorTracker] = None
recognizer: Optional[CursorRecognizer] = None
collector: Optional[CursorSampleCollector] = None
daemon_client: Optional[DaemonClient] = None
claude_detector: Optional[ClaudeVisionCursorDetector] = None
sil_tracker: Optional[SilhouetteTracker] = None
cmd_movement: Optional[CommandedMovement] = None
yolo_detector: Optional[YOLOCursorDetector] = None

# Sustained low-confidence tracker for jitter-based re-acquisition
_low_confidence_since: float = 0.0  # monotonic time when CNN confidence first went <0.7
_vision_lockout_until: float = 0.0  # monotonic time until which motion blobs can't override position
_VISION_LOCKOUT_S: float = 30.0     # how long Vision position is protected from motion override
_YOLO_LOCKOUT_S: float = 5.0       # how long a high-confidence YOLO position is protected
_yolo_lockout_until: float = 0.0   # monotonic time until which YOLO position is protected

# Noise grid: shared between daemon loop (writer) and probe (reader)
_NOISE_GRID_W, _NOISE_GRID_H = 48, 27
_NOISE_CELL_PX = 40
_NOISE_THRESHOLD = 8
_noise_grid: list[list[int]] = [[_NOISE_THRESHOLD] * _NOISE_GRID_W for _ in range(_NOISE_GRID_H)]


def noise_grid_features() -> dict:
    """Extract compact features from the noise grid for logging/CNN input.

    Converts the 48x27 = 1296-cell grid into ~20 meaningful features:
    - Per-quadrant activity counts (4 features)
    - Total noisy cells, active cells, clear cells (3 features)
    - Cluster count and sizes (2 features)
    - Spatial centroid of activity (2 features)
    - Activity density per 8x8 macro-block (reduces to ~6x3=18 → top-3 hottest blocks)
    - Motion complexity score (entropy-like)
    """
    grid = _noise_grid
    w, h = _NOISE_GRID_W, _NOISE_GRID_H
    thresh = _NOISE_THRESHOLD
    mid_x, mid_y = w // 2, h // 2

    # Per-quadrant counts of noisy cells (cells at or above threshold)
    q_counts = [0, 0, 0, 0]  # Q1=TL, Q2=TR, Q3=BL, Q4=BR
    total_noisy = 0
    total_active = 0  # cells with any activity (>0)
    sum_x, sum_y, sum_w = 0.0, 0.0, 0.0

    for gy in range(h):
        for gx in range(w):
            v = grid[gy][gx]
            if v > 0:
                total_active += 1
            if v >= thresh:
                total_noisy += 1
                qi = (0 if gx < mid_x else 1) + (0 if gy < mid_y else 2)
                q_counts[qi] += 1
                # Weighted centroid
                sum_x += gx * v
                sum_y += gy * v
                sum_w += v

    centroid_x = round(sum_x / sum_w, 1) if sum_w > 0 else -1
    centroid_y = round(sum_y / sum_w, 1) if sum_w > 0 else -1

    # Macro-blocks: 8x8 cells → 6x3 = 18 blocks, report top 3 hottest
    macro_w, macro_h = 8, 9  # 48/8=6, 27/9=3
    macros = []
    for my in range(0, h, macro_h):
        for mx in range(0, w, macro_w):
            count = 0
            for gy in range(my, min(my + macro_h, h)):
                for gx in range(mx, min(mx + macro_w, w)):
                    if grid[gy][gx] >= thresh:
                        count += 1
            if count > 0:
                macros.append({"x": mx, "y": my, "count": count})
    macros.sort(key=lambda m: -m["count"])

    # Complexity: ratio of noisy cells to total cells (0 = still, 1 = everything moving)
    complexity = round(total_noisy / (w * h), 4) if w * h > 0 else 0

    return {
        "total_noisy": total_noisy,
        "total_active": total_active,
        "total_clear": w * h - total_active,
        "quadrant_noisy": q_counts,
        "centroid": [centroid_x, centroid_y],
        "complexity": complexity,
        "hotspots": macros[:3],
    }


# ── Motion detection service (background thread) ───────────────────

def draw_blob_overlay(frame: np.ndarray, blobs: list[MotionBlob],
                      cursor_pos: tuple[int, int],
                      pointer_blob_idx: int = -1,
                      roi_bounds: Optional[tuple[int, int, int, int]] = None,
                      yolo_bbox: Optional[tuple[int, int, int, int]] = None,
                      yolo_conf: float = 0.0) -> np.ndarray:
    """Draw motion blob rectangles and cursor crosshair on frame.

    Args:
        frame: RGB frame to draw on.
        blobs: Motion blobs to highlight.
        cursor_pos: (x, y) of known cursor position.
        pointer_blob_idx: Index of the blob identified as cursor (-1 if none).
            This blob gets green coloring; all others get blue.
        roi_bounds: Optional (x0, y0, x1, y1) silhouette tracking ROI to draw.
    """
    overlay = frame.copy()

    # Draw silhouette tracking ROI (yellow dashed rectangle)
    if roi_bounds:
        rx0, ry0, rx1, ry1 = roi_bounds
        # Draw dashed rectangle by drawing short segments
        color_roi = (0, 255, 255)  # Yellow
        dash_len = 10
        # Top edge
        for x in range(rx0, rx1, dash_len * 2):
            cv2.line(overlay, (x, ry0), (min(x + dash_len, rx1), ry0), color_roi, 1)
        # Bottom edge
        for x in range(rx0, rx1, dash_len * 2):
            cv2.line(overlay, (x, ry1), (min(x + dash_len, rx1), ry1), color_roi, 1)
        # Left edge
        for y in range(ry0, ry1, dash_len * 2):
            cv2.line(overlay, (rx0, y), (rx0, min(y + dash_len, ry1)), color_roi, 1)
        # Right edge
        for y in range(ry0, ry1, dash_len * 2):
            cv2.line(overlay, (rx1, y), (rx1, min(y + dash_len, ry1)), color_roi, 1)

    for i, blob in enumerate(blobs):
        x_min, y_min, x_max, y_max = blob.bbox
        cx, cy = blob.centroid
        if i == pointer_blob_idx:
            # Pointer blob: green
            color = (0, 255, 0)
            thickness = 2
        else:
            # General motion: blue
            color = (80, 160, 255)
            thickness = 1
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, thickness)
        cv2.circle(overlay, (cx, cy), 4, color, -1)
        label = str(blob.pixel_count) + "px"
        cv2.putText(overlay, label, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # YOLO bounding box (cyan) — shows full-screen detection result
    if yolo_bbox and yolo_conf > 0.25:
        yx1, yy1, yx2, yy2 = yolo_bbox
        color_yolo = (0, 255, 255)  # Cyan
        cv2.rectangle(overlay, (yx1, yy1), (yx2, yy2), color_yolo, 2)
        yolo_label = f"YOLO {yolo_conf:.0%}"
        cv2.putText(overlay, yolo_label, (yx1, yy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_yolo, 1)

    return overlay


def frame_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _draw_tracking_hud(overlay: np.ndarray, cursor_age: float,
                       cursor_pos: tuple[int, int]):
    """Draw tracking quality indicator directly on the video frame.

    Shows: green TRACKING / yellow STALE / red LOST with age and coords.
    This gives immediate visual feedback without reading the sidebar.
    """
    h, w = overlay.shape[:2]
    if cursor_age < 5:
        color = (0, 200, 0)
        label = f"TRACKING ({cursor_pos[0]},{cursor_pos[1]})"
    elif cursor_age < 30:
        color = (255, 200, 0)
        label = f"STALE {cursor_age:.0f}s ({cursor_pos[0]},{cursor_pos[1]})"
    else:
        color = (255, 50, 50)
        label = f"LOST {cursor_age:.0f}s"

    # Bottom-left corner
    cv2.putText(overlay, label, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)  # shadow
    cv2.putText(overlay, label, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw crosshair at cursor position if tracking
    if cursor_age < 30 and cursor_pos[0] > 0:
        cx, cy = cursor_pos
        cv2.drawMarker(overlay, (cx, cy), color,
                       cv2.MARKER_CROSS, 20, 1, cv2.LINE_AA)


def _daemon_motion_loop():
    """Read blobs + frames from C daemon shm. Cursor ID stays in Python.

    Three tracking strategies run in the same loop:
    1. Motion blobs from daemon (~20Hz) — tracks moving cursor via frame-diff
    2. Template matching via silhouette tracker (at frame_fps) — tracks stationary cursor
    3. Staleness auto-detection — marks position lost when age exceeds threshold
    4. Daemon health check — detects frozen daemon via stale frame_count
    """
    last_frame_time = 0.0
    last_daemon_seq = 0
    last_daemon_seq_time = time.monotonic()

    # Velocity estimation for predictive cursor tracking
    velocity_history: list[tuple[float, float, float]] = []  # (dx, dy, timestamp)
    MAX_VELOCITY_SAMPLES = 5

    # Persistent blob filter: detects UI animations (clocks, tickers, etc.)
    # Grid of 48x27 cells (40px each at 1920x1080). Each cell tracks
    # consecutive frames where blobs appear in that region.
    # Starts FULL (all regions presumed noisy). Regions that DON'T have
    # blobs decay to 0 and become "clear" after ~8 frames (~0.4s).
    # UI animation regions stay above threshold because they get blobs
    # every frame. Cursor regions clear quickly because cursor motion
    # is transient (only during movement).
    global _noise_grid, _yolo_lockout_until
    _noise_grid = [[_NOISE_THRESHOLD] * _NOISE_GRID_W for _ in range(_NOISE_GRID_H)]

    while True:
        try:
            ds = daemon_client.read_state()
            if ds is None:
                time.sleep(0.05)
                continue

            now = time.monotonic()

            # ── Daemon health check ──
            if ds.frame_count != last_daemon_seq:
                last_daemon_seq = ds.frame_count
                last_daemon_seq_time = now
            elif now - last_daemon_seq_time > 5.0:
                state.error = f"Daemon frozen (seq={ds.frame_count} for {now - last_daemon_seq_time:.0f}s)"

            # ── Fast path: cursor tracking from daemon blobs (~20Hz) ──
            # Tighter blob filter: real cursor is 5-200px at 1080p.
            # Also reject very elongated blobs (UI text redraws, scrollbars).
            pos = tracker.position if tracker else None
            cursor_blobs = []
            for b in ds.blobs:
                if b.pixel_count < 5 or b.pixel_count > 200:
                    continue
                bx0, by0, bx1, by1 = b.bbox
                bw = max(1, bx1 - bx0)
                bh = max(1, by1 - by0)
                # Reject very elongated blobs (aspect < 0.2 = text/scrollbar)
                if min(bw, bh) / max(bw, bh) < 0.2:
                    continue
                # Reject very large bounding boxes (UI element redraws)
                if bw > 80 or bh > 80:
                    continue
                cursor_blobs.append(b)

            # ── Persistent blob filter ──
            # Update noise grid: mark cells that have blobs this frame
            active_cells = set()
            for b in cursor_blobs:
                gx = min(_NOISE_GRID_W - 1, b.centroid[0] // _NOISE_CELL_PX)
                gy = min(_NOISE_GRID_H - 1, b.centroid[1] // _NOISE_CELL_PX)
                active_cells.add((gx, gy))

            for gy in range(_NOISE_GRID_H):
                for gx in range(_NOISE_GRID_W):
                    if (gx, gy) in active_cells:
                        _noise_grid[gy][gx] = min(_noise_grid[gy][gx] + 1, _NOISE_THRESHOLD + 5)
                    else:
                        _noise_grid[gy][gx] = max(0, _noise_grid[gy][gx] - 1)

            # Remove blobs in persistent-motion cells (UI animations)
            filtered_blobs = []
            for b in cursor_blobs:
                gx = min(_NOISE_GRID_W - 1, b.centroid[0] // _NOISE_CELL_PX)
                gy = min(_NOISE_GRID_H - 1, b.centroid[1] // _NOISE_CELL_PX)
                if _noise_grid[gy][gx] >= _NOISE_THRESHOLD:
                    continue  # Skip: persistent motion = UI animation
                filtered_blobs.append(b)
            cursor_blobs = filtered_blobs

            primary_blob = None
            if cursor_blobs:
                if pos and (now - pos.timestamp < 10):
                    # Predict next position from velocity history
                    pred_x, pred_y = pos.x, pos.y
                    if len(velocity_history) >= 2:
                        recent = velocity_history[-MAX_VELOCITY_SAMPLES:]
                        avg_dx = sum(v[0] for v in recent) / len(recent)
                        avg_dy = sum(v[1] for v in recent) / len(recent)
                        dt = now - pos.timestamp
                        pred_x = pos.x + int(avg_dx * min(dt * 20, 3))
                        pred_y = pos.y + int(avg_dy * min(dt * 20, 3))

                    def _blob_score(b):
                        # Distance from predicted AND last-known, use smaller
                        dist_pred = ((b.centroid[0] - pred_x) ** 2 +
                                     (b.centroid[1] - pred_y) ** 2) ** 0.5
                        dist_last = ((b.centroid[0] - pos.x) ** 2 +
                                     (b.centroid[1] - pos.y) ** 2) ** 0.5
                        dist = min(dist_pred, dist_last)
                        # Size sweet spot: 15-40px, penalize above AND below
                        size = b.pixel_count
                        if size < 15:
                            size_penalty = (15 - size) * 1.0
                        elif size > 40:
                            size_penalty = (size - 40) * 0.8
                        else:
                            size_penalty = 0
                        return dist + size_penalty

                    best = min(cursor_blobs, key=_blob_score)
                    dist = ((best.centroid[0] - pos.x) ** 2 +
                            (best.centroid[1] - pos.y) ** 2) ** 0.5
                    if dist < 300:
                        # Silhouette confidence gate: if tracker is confident,
                        # reject blobs too far from silhouette position
                        accept = True
                        if (sil_tracker and state.silhouette_confidence > 0.6
                                and state.silhouette_latency_ms < 50):
                            sil_dist = ((best.centroid[0] - state.cursor_x) ** 2 +
                                        (best.centroid[1] - state.cursor_y) ** 2) ** 0.5
                            if sil_dist > 100:
                                accept = False

                        # Vision lockout: don't let motion blobs override
                        # a Vision-confirmed position for _VISION_LOCKOUT_S
                        if accept and now < _vision_lockout_until:
                            blob_dist_from_vision = (
                                (best.centroid[0] - state.vision_x) ** 2 +
                                (best.centroid[1] - state.vision_y) ** 2) ** 0.5
                            if blob_dist_from_vision > 80:
                                accept = False  # Blob too far from Vision truth

                        # YOLO lockout: don't let blobs override YOLO-confirmed pos
                        if accept and now < _yolo_lockout_until and state.yolo_active:
                            blob_dist_from_yolo = (
                                (best.centroid[0] - state.yolo_x) ** 2 +
                                (best.centroid[1] - state.yolo_y) ** 2) ** 0.5
                            if blob_dist_from_yolo > 80:
                                accept = False  # Blob too far from YOLO detection

                        if accept:
                            primary_blob = best
                            bx, by = best.centroid
                            # Update velocity history
                            dx = bx - pos.x
                            dy = by - pos.y
                            velocity_history.append((dx, dy, now))
                            if len(velocity_history) > MAX_VELOCITY_SAMPLES * 2:
                                velocity_history = velocity_history[-MAX_VELOCITY_SAMPLES:]
                            if tracker:
                                tracker.set_position(bx, by, method="motion_track")
                            pos = tracker.position
                            if sil_tracker:
                                sil_tracker.reset()
                else:
                    # No known position — use Vision as anchor if recent,
                    # otherwise pick blob closest to typical cursor size
                    if now < _vision_lockout_until:
                        pass  # Don't pick random blobs during Vision lockout
                    elif (state.vision_x > 0 and state.vision_timestamp > 0
                          and now - state.vision_timestamp < 120):
                        # Vision position is <2min old — use as anchor
                        # Only accept blobs within 150px of Vision position
                        def _vision_dist(b):
                            return ((b.centroid[0] - state.vision_x) ** 2 +
                                    (b.centroid[1] - state.vision_y) ** 2) ** 0.5
                        near_blobs = [b for b in cursor_blobs if _vision_dist(b) < 150]
                        if near_blobs:
                            best = min(near_blobs, key=lambda b: _vision_dist(b))
                            primary_blob = best
                            bx, by = best.centroid
                            velocity_history.clear()
                            if tracker:
                                tracker.set_position(bx, by, method="motion_track")
                            pos = tracker.position
                            if sil_tracker:
                                sil_tracker.reset()
                    else:
                        def _cursor_likelihood(b):
                            ideal = 30
                            return abs(b.pixel_count - ideal)
                        best = min(cursor_blobs, key=_cursor_likelihood)
                        primary_blob = best
                        bx, by = best.centroid
                        velocity_history.clear()
                        if tracker:
                            tracker.set_position(bx, by, method="motion_track")
                        pos = tracker.position
                        if sil_tracker:
                            sil_tracker.reset()

            # ── Slow path: frame decode + template match + overlay (5Hz) ──
            frame_interval = 1.0 / max(0.1, config.frame_fps)
            roi_bounds = None

            if now - last_frame_time >= frame_interval:
                jpeg = daemon_client.read_jpeg()
                if jpeg:
                    t_decode = time.monotonic()
                    frame = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8),
                        cv2.IMREAD_COLOR)
                    if frame is not None:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        state.prof_jpeg_decode_ms = (time.monotonic() - t_decode) * 1000

                        # ── YOLO full-screen detection (~10fps) ──
                        if yolo_detector and yolo_detector.loaded:
                            yolo_dets = yolo_detector.detect(frame_rgb)
                            if yolo_dets:
                                best_yolo = yolo_dets[0]
                                state.yolo_active = True
                                state.yolo_x = best_yolo.cx
                                state.yolo_y = best_yolo.cy
                                state.yolo_confidence = best_yolo.confidence
                                state.yolo_bbox = (best_yolo.x1, best_yolo.y1,
                                                   best_yolo.x2, best_yolo.y2)
                                state.yolo_inference_ms = best_yolo.inference_ms
                                state.yolo_timestamp = now
                                state.yolo_detection_count += 1

                                pos = tracker.position if tracker else None

                                # Priority 1: No position / stale / lost → YOLO seeds
                                if (not pos or
                                        (now - pos.timestamp > 5) or
                                        state.silhouette_method == "lost"):
                                    if tracker and best_yolo.confidence > 0.5:
                                        tracker.set_position(
                                            best_yolo.cx, best_yolo.cy,
                                            method="yolo")
                                        pos = tracker.position
                                        if sil_tracker:
                                            sil_tracker.reset()
                                        _yolo_lockout_until = now + _YOLO_LOCKOUT_S

                                # Priority 2: YOLO disagrees with silhouette → anchor correction
                                # If YOLO high-conf and silhouette is tracking a different position
                                elif pos and best_yolo.confidence > 0.7:
                                    drift = ((best_yolo.cx - pos.x) ** 2 +
                                             (best_yolo.cy - pos.y) ** 2) ** 0.5
                                    if drift > 80:
                                        # YOLO sees cursor at a different position than
                                        # silhouette is tracking → YOLO wins (full-screen)
                                        if tracker:
                                            tracker.set_position(
                                                best_yolo.cx, best_yolo.cy,
                                                method="yolo_anchor")
                                            pos = tracker.position
                                        if sil_tracker:
                                            sil_tracker.reset()
                                        _yolo_lockout_until = now + _YOLO_LOCKOUT_S
                                        state.cursor_validated = False
                            else:
                                state.yolo_active = False

                        # ── Silhouette tracking on decoded frame ──
                        pos = tracker.position if tracker else None
                        if sil_tracker and pos:
                            track_result = sil_tracker.track(frame_rgb, pos.x, pos.y)
                            roi_bounds = sil_tracker.roi_bounds(
                                pos.x, pos.y, frame_rgb.shape)

                            state.silhouette_active = True
                            state.silhouette_hz = round(sil_tracker.tracking_hz, 1)
                            state.silhouette_roi_size = sil_tracker.roi_size
                            state.silhouette_misses = sil_tracker.consecutive_misses

                            if track_result:
                                # CNN confidence gate: template matching alone
                                # can't be trusted — cursors look like many UI
                                # elements (icons, text, dark spots). Only trust
                                # silhouette results when:
                                #  a) CNN has validated the current position, OR
                                #  b) Method is motion_roi (actual frame-diff movement)
                                #  c) Result is close to Vision position during lockout
                                # Template-only results are NEVER trusted without
                                # CNN backing, no matter how high the confidence.
                                vision_close = False
                                if now < _vision_lockout_until and state.vision_x > 0:
                                    vd = ((track_result.x - state.vision_x) ** 2 +
                                          (track_result.y - state.vision_y) ** 2) ** 0.5
                                    vision_close = vd < 60
                                sil_trusted = (
                                    state.cursor_validated or
                                    track_result.method == "motion_roi" or
                                    vision_close
                                )
                                if sil_trusted and tracker:
                                    tracker.set_position(
                                        track_result.x, track_result.y,
                                        method="sil_" + track_result.method)
                                    pos = tracker.position
                                elif tracker:
                                    # Gated but found — refresh timestamp to prevent
                                    # staleness (don't change x,y, just keep alive)
                                    tracker.set_position(
                                        pos.x, pos.y, method=pos.method)
                                    pos = tracker.position
                                state.silhouette_method = track_result.method
                                state.silhouette_confidence = round(
                                    track_result.confidence, 3)
                                state.silhouette_latency_ms = round(
                                    track_result.latency_ms, 2)
                            else:
                                state.silhouette_method = "searching"
                                if sil_tracker.roi_size >= SilhouetteTracker.ROI_MAX:
                                    state.silhouette_method = "lost"
                                    # Auto-recovery: probe to re-establish position
                                    recovery = _jitter_reacquire(True)
                                    if recovery:
                                        rx, ry = recovery
                                        sil_tracker.reset()
                                        state.silhouette_method = "recovered"
                        else:
                            state.silhouette_active = False

                        # ── Draw overlay with tracking info ──
                        t_overlay = time.monotonic()
                        cursor_pos = (pos.x, pos.y) if pos else (0, 0)
                        overlay = draw_blob_overlay(
                            frame_rgb, ds.blobs, cursor_pos,
                            roi_bounds=roi_bounds,
                            yolo_bbox=state.yolo_bbox if state.yolo_active else None,
                            yolo_conf=state.yolo_confidence)

                        # ── HUD: tracking quality on the frame ──
                        cursor_age = (now - pos.timestamp) if pos else 999.0
                        _draw_tracking_hud(overlay, cursor_age, cursor_pos)
                        state.prof_overlay_draw_ms = (time.monotonic() - t_overlay) * 1000

                        t_encode = time.monotonic()
                        overlay_jpeg = frame_to_jpeg(overlay)
                        state.prof_jpeg_encode_ms = (time.monotonic() - t_encode) * 1000

                        state.prof_frame_total_ms = (time.monotonic() - t_decode) * 1000
                        state.prof_daemon_timestamp_ns = ds.timestamp_ns

                        # Collect profiler sample if test is running
                        if profiler_running:
                            profiler_samples.append({
                                "utc": utc_now_ms(),
                                "jpeg_decode_ms": round(state.prof_jpeg_decode_ms, 2),
                                "silhouette_ms": round(state.silhouette_latency_ms, 2),
                                "overlay_draw_ms": round(state.prof_overlay_draw_ms, 2),
                                "jpeg_encode_ms": round(state.prof_jpeg_encode_ms, 2),
                                "frame_total_ms": round(state.prof_frame_total_ms, 2),
                            })

                        with state.lock:
                            state.frame_overlay = overlay_jpeg
                            state.frame_raw = frame_rgb
                        last_frame_time = now

            # ── Staleness check: auto-expire validated status ──
            pos = tracker.position if tracker else None
            cursor_age = (now - pos.timestamp) if pos else 999.0
            if cursor_age > 15:
                state.cursor_validated = False

            # ── Update shared state (~20Hz) ──
            cursor_pos = (0, 0)
            with state.lock:
                state.blobs = ds.blobs
                state.blob_count = ds.blob_count
                state.fps = ds.fps
                state.frame_count = ds.frame_count
                if pos:
                    state.cursor_x = pos.x
                    state.cursor_y = pos.y
                    state.cursor_age_s = round(cursor_age, 1)
                    state.cursor_method = pos.method
                    cursor_pos = (pos.x, pos.y)
                else:
                    state.cursor_age_s = 999.0

            # Feed loss-event ring buffer (rate-limited internally to ~5Hz)
            _ring_capture(getattr(state, 'frame_overlay', None))

            time.sleep(0.05)  # 20Hz state polling

        except Exception as e:
            state.error = str(e)
            time.sleep(1)


def motion_detection_loop():
    """Background thread: uses C daemon if available, else Python capture."""
    if daemon_client and daemon_client.is_running():
        print("  Daemon detected — using shared memory for motion detection")
        _daemon_motion_loop()
    else:
        print("  Daemon not running — using Python motion detection")
        _python_motion_loop()


def _python_motion_loop():
    """Python motion detection loop with Phase 2 silhouette tracking fast path.

    When cursor position is known and fresh:
      → Silhouette tracker runs first (ROI-based, <1ms)
      → If it finds cursor, skip expensive full-frame blob detection
      → Yellow dashed ROI rectangle shown on overlay

    When position is stale or silhouette tracker fails:
      → Full-frame blob detection (existing code, ~5-10ms)
      → Shape-informed or proximity-based blob selection
    """
    prev_frame = None
    frame_times = []

    while True:
        try:
            t0 = time.monotonic()
            frame = sensor.capture(settle_frames=1)
            state.frame_count += 1

            pos = tracker.position if tracker else None
            cursor_pos = (0, 0)
            roi_bounds = None

            if pos:
                cursor_pos = (pos.x, pos.y)
                state.cursor_x = pos.x
                state.cursor_y = pos.y
                state.cursor_method = pos.method
                state.cursor_age_s = time.monotonic() - pos.timestamp

            # ── YOLO full-screen detection ──────────────────────
            if yolo_detector and yolo_detector.loaded:
                yolo_dets = yolo_detector.detect(frame)
                if yolo_dets:
                    best_yolo = yolo_dets[0]
                    state.yolo_active = True
                    state.yolo_x = best_yolo.cx
                    state.yolo_y = best_yolo.cy
                    state.yolo_confidence = best_yolo.confidence
                    state.yolo_bbox = (best_yolo.x1, best_yolo.y1,
                                       best_yolo.x2, best_yolo.y2)
                    state.yolo_inference_ms = best_yolo.inference_ms
                    state.yolo_timestamp = t0
                    state.yolo_detection_count += 1

                    # Seed position from YOLO if stale or lost
                    if (not pos or
                            (t0 - pos.timestamp > 5) or
                            not state.silhouette_active):
                        if tracker and best_yolo.confidence > 0.5:
                            tracker.set_position(
                                best_yolo.cx, best_yolo.cy, method="yolo")
                            pos = tracker.position
                            cursor_pos = (pos.x, pos.y)
                            if sil_tracker:
                                sil_tracker.reset()
                else:
                    state.yolo_active = False

            # ── FAST PATH: Silhouette tracking in ROI ──────────────
            silhouette_handled = False
            position_fresh = pos and (time.monotonic() - pos.timestamp < 10)

            if sil_tracker and position_fresh:
                track_result = sil_tracker.track(frame, pos.x, pos.y)

                state.silhouette_active = True
                state.silhouette_hz = round(sil_tracker.tracking_hz, 1)
                state.silhouette_roi_size = sil_tracker.roi_size
                state.silhouette_misses = sil_tracker.consecutive_misses
                roi_bounds = sil_tracker.roi_bounds(pos.x, pos.y, frame.shape)

                if track_result:
                    bx, by = track_result.x, track_result.y
                    method = f"sil_{track_result.method}"
                    if tracker:
                        tracker.set_position(bx, by, method=method)
                    cursor_pos = (bx, by)
                    state.cursor_x = bx
                    state.cursor_y = by
                    state.cursor_method = method
                    state.cursor_age_s = 0.0
                    state.silhouette_method = track_result.method
                    state.silhouette_confidence = round(track_result.confidence, 3)
                    state.silhouette_latency_ms = round(track_result.latency_ms, 2)
                    silhouette_handled = True

                    # Update cursor template on high-confidence template matches
                    if (recognizer and track_result.method == "template"
                            and track_result.confidence > 0.6):
                        patch = extract_gray_patch(frame, bx, by)
                        if patch is not None:
                            recognizer.update_cursor_template(patch)
                else:
                    state.silhouette_method = "searching"
                    # If ROI maxed out, let full-frame take over
                    if sil_tracker.roi_size >= SilhouetteTracker.ROI_MAX:
                        state.silhouette_active = False
                        state.silhouette_method = "lost"
            else:
                state.silhouette_active = False

            # ── SLOW PATH: Full-frame blob detection ───────────────
            blobs = []
            pointer_blob_idx = -1

            if not silhouette_handled:
                if prev_frame is not None:
                    blobs = extract_motion_blobs(
                        prev_frame, frame,
                        threshold=15, min_pixels=5, max_pixels=3000,
                    )

                if blobs:
                    cursor_blobs = [b for b in blobs if 5 <= b.pixel_count <= 500]
                    position_stale = not pos or (time.monotonic() - pos.timestamp > 10)
                    if cursor_blobs:
                        best = None
                        bx, by = 0, 0

                        # Shape-informed scoring
                        if recognizer and recognizer.cursor_template is not None:
                            scored = []
                            for b in cursor_blobs:
                                bx_, by_ = b.centroid
                                blob_patch = extract_gray_patch(frame, bx_, by_)
                                if blob_patch is not None:
                                    score = recognizer.score_blob_shape(blob_patch)
                                    scored.append((b, score))
                            scored.sort(key=lambda x: x[1], reverse=True)

                            if position_stale:
                                if scored and scored[0][1] > 0.5:
                                    best = scored[0][0]
                                    bx, by = best.centroid
                                    state.cursor_method = "shape_track"
                            else:
                                good = [(b, s) for b, s in scored if s > 0.4]
                                if good:
                                    best_dist = 9999
                                    for b, s in good:
                                        bx_, by_ = b.centroid
                                        dist = ((bx_ - pos.x) ** 2 + (by_ - pos.y) ** 2) ** 0.5
                                        if dist < best_dist:
                                            best_dist = dist
                                            best = b
                                    if best and best_dist > 600:
                                        best = None
                                    if best:
                                        bx, by = best.centroid
                                        state.cursor_method = "shape_track"

                        # Fallback: size-based heuristic
                        if best is None:
                            if position_stale:
                                cursor_blobs.sort(key=lambda b: b.pixel_count)
                                best = cursor_blobs[0]
                                bx, by = best.centroid
                            else:
                                best_dist = 9999
                                for b in cursor_blobs:
                                    bx_, by_ = b.centroid
                                    dist = ((bx_ - pos.x) ** 2 + (by_ - pos.y) ** 2) ** 0.5
                                    if dist < best_dist:
                                        best_dist = dist
                                        best = b
                                if best and best_dist > 600:
                                    best = None
                                if best:
                                    bx, by = best.centroid

                        if best:
                            try:
                                pointer_blob_idx = blobs.index(best)
                            except ValueError:
                                pointer_blob_idx = -1
                            if tracker:
                                tracker.set_position(bx, by, method=state.cursor_method if state.cursor_method in ("shape_track",) else "motion_track")
                            cursor_pos = (bx, by)
                            state.cursor_x = bx
                            state.cursor_y = by
                            if state.cursor_method != "shape_track":
                                state.cursor_method = "motion_track"
                            state.cursor_age_s = 0.0

                            # Reset silhouette tracker on fresh position from full-frame
                            if sil_tracker:
                                sil_tracker.reset()

            overlay = draw_blob_overlay(
                frame, blobs, cursor_pos, pointer_blob_idx, roi_bounds,
                yolo_bbox=state.yolo_bbox if state.yolo_active else None,
                yolo_conf=state.yolo_confidence)
            jpeg = frame_to_jpeg(overlay)

            with state.lock:
                state.frame_raw = frame
                state.frame_overlay = jpeg
                state.blobs = blobs
                state.blob_count = len(blobs)

            frame_times.append(time.monotonic())
            frame_times = [t for t in frame_times if t > time.monotonic() - 2.0]
            state.fps = len(frame_times) / 2.0 if len(frame_times) > 1 else 0

            prev_frame = frame

            # Feed loss-event ring buffer
            _ring_capture(getattr(state, 'frame_overlay', None))

            elapsed = time.monotonic() - t0
            time.sleep(max(0, 0.05 - elapsed))

        except Exception as e:
            state.error = str(e)
            time.sleep(1)


# ── Loss Event Sampling ──────────────────────────────────────────

def _ring_capture(jpeg_bytes: bytes | None):
    """Feed ring buffer from motion loop. Rate-limited to ~5Hz."""
    global _ring_last_t
    now = time.monotonic()
    if now - _ring_last_t < 1.0 / _RING_BUFFER_HZ:
        return
    _ring_last_t = now

    # Snapshot blob centroids — these are all the candidate cursor positions
    blobs_snap = []
    try:
        for b in (state.blobs or [])[:10]:
            blobs_snap.append({
                "centroid": b.centroid if hasattr(b, 'centroid') else (0, 0),
                "bbox": b.bbox if hasattr(b, 'bbox') else (0, 0, 0, 0),
                "px": b.pixel_count if hasattr(b, 'pixel_count') else 0,
            })
    except Exception:
        pass

    entry = {
        "t": now,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "jpeg": jpeg_bytes,
        "cx": state.cursor_x, "cy": state.cursor_y,
        "method": state.cursor_method,
        "validated": state.cursor_validated,
        "cnn_conf": state.cnn_confidence,
        "cnn_model": state.cnn_model_version,
        "cnn_ms": state.cnn_inference_ms,
        "sil_conf": state.silhouette_confidence,
        "sil_method": state.silhouette_method,
        "sil_roi": state.silhouette_roi_size,
        "sil_misses": state.silhouette_misses,
        "blob_count": state.blob_count,
        "blobs": blobs_snap,
        "fps": state.fps,
        "vision_x": state.vision_x, "vision_y": state.vision_y,
        "yolo_x": state.yolo_x, "yolo_y": state.yolo_y,
        "yolo_conf": state.yolo_confidence,
        "yolo_active": state.yolo_active,
    }
    with _ring_lock:
        _ring_buffer.append(entry)

    # Check for loss event trigger
    _check_loss_trigger(now)


def _check_loss_trigger(now: float):
    """Detect confidence transitions that indicate cursor loss."""
    global _loss_event_active, _loss_event_cooldown, _loss_prev

    if _loss_event_active or now < _loss_event_cooldown:
        return

    trigger = None
    prev = _loss_prev

    # Validated True → False
    if prev["validated"] and not state.cursor_validated:
        trigger = "validation_lost"

    # CNN drops from high (>0.7) to low (<0.5)
    if prev["cnn"] > 0.7 and state.cnn_confidence < 0.5:
        trigger = "cnn_drop"

    # Silhouette drops from good (>0.7) to bad (<0.3)
    if prev["sil"] > 0.7 and state.silhouette_confidence < 0.3:
        trigger = "silhouette_drop"

    # Position jump > 200px
    if prev["x"] > 0 or prev["y"] > 0:
        dx = state.cursor_x - prev["x"]
        dy = state.cursor_y - prev["y"]
        jump = (dx * dx + dy * dy) ** 0.5
        if jump > 200:
            trigger = f"position_jump_{int(jump)}px"

    # Update previous state
    _loss_prev = {
        "validated": state.cursor_validated,
        "cnn": state.cnn_confidence,
        "sil": state.silhouette_confidence,
        "x": state.cursor_x, "y": state.cursor_y,
    }

    if trigger:
        _start_loss_event(trigger, now)


def _start_loss_event(trigger: str, trigger_time: float):
    """Snapshot ring buffer (pre-event) and start post-event capture thread."""
    global _loss_event_active
    _loss_event_active = True

    with _ring_lock:
        cutoff = trigger_time - _LOSS_PRE_WINDOW_S
        pre = [e for e in _ring_buffer if e["t"] >= cutoff]

    event_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    print(f"[loss_event] TRIGGERED: {trigger} — capturing {len(pre)} pre-frames → {event_id}")

    threading.Thread(
        target=_capture_loss_post,
        args=(event_id, trigger, trigger_time, list(pre)),
        daemon=True,
    ).start()


def _capture_loss_post(event_id: str, trigger: str, trigger_time: float, pre: list[dict]):
    """Background: capture post-event frames until recovery or timeout, then save."""
    global _loss_event_active, _loss_event_cooldown

    post: list[dict] = []
    deadline = trigger_time + _LOSS_POST_WINDOW_S
    recovered = False
    recovery_time = 0.0

    try:
        while time.monotonic() < deadline:
            time.sleep(1.0 / _RING_BUFFER_HZ)

            # Grab frame
            jpeg = None
            try:
                if daemon_client and daemon_client.is_running():
                    jpeg = daemon_client.read_jpeg()
                elif sensor:
                    frame = sensor.capture(settle_frames=1)
                    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    jpeg = buf.tobytes()
            except Exception:
                continue

            entry = {
                "t": time.monotonic(),
                "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                "jpeg": jpeg,
                "cx": state.cursor_x, "cy": state.cursor_y,
                "method": state.cursor_method,
                "validated": state.cursor_validated,
                "cnn_conf": state.cnn_confidence,
                "cnn_model": state.cnn_model_version,
                "sil_conf": state.silhouette_confidence,
                "sil_method": state.silhouette_method,
                "sil_roi": state.silhouette_roi_size,
                "sil_misses": state.silhouette_misses,
                "blob_count": state.blob_count,
                "fps": state.fps,
            }
            post.append(entry)

            # Check recovery: validated + CNN confident
            if state.cursor_validated and state.cnn_confidence > 0.7:
                recovered = True
                recovery_time = time.monotonic()
                # Capture a few more frames after recovery
                for _ in range(int(_RING_BUFFER_HZ * 2)):
                    time.sleep(1.0 / _RING_BUFFER_HZ)
                    try:
                        jpeg2 = None
                        if daemon_client and daemon_client.is_running():
                            jpeg2 = daemon_client.read_jpeg()
                        elif sensor:
                            f2 = sensor.capture(settle_frames=1)
                            _, b2 = cv2.imencode('.jpg', f2, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            jpeg2 = b2.tobytes()
                    except Exception:
                        jpeg2 = None
                    post.append({
                        "t": time.monotonic(),
                        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                        "jpeg": jpeg2,
                        "cx": state.cursor_x, "cy": state.cursor_y,
                        "method": state.cursor_method,
                        "validated": state.cursor_validated,
                        "cnn_conf": state.cnn_confidence,
                        "cnn_model": state.cnn_model_version,
                        "sil_conf": state.silhouette_confidence,
                        "sil_method": state.silhouette_method,
                        "sil_roi": state.silhouette_roi_size,
                        "sil_misses": state.silhouette_misses,
                        "blob_count": state.blob_count,
                        "fps": state.fps,
                    })
                break

        # Save everything to disk
        _save_loss_event(event_id, trigger, trigger_time, pre, post, recovered, recovery_time)

    finally:
        _loss_event_cooldown = time.monotonic() + _LOSS_COOLDOWN_S
        _loss_event_active = False


def _save_loss_event(event_id, trigger, trigger_time, pre, post, recovered, recovery_time):
    """Write loss event to disk with full analysis:
    - Frames + patches at tracked position (CNN re-inference)
    - Patches at EACH blob centroid (candidate cursor positions → CNN scoring)
    - Velocity/acceleration profile across the timeline
    - Edge case tags (partial cursor, near-edge, noisy background)
    - Peak velocity/acceleration markers
    """
    event_dir = _LOSS_EVENT_DIR / event_id
    frames_dir = event_dir / "frames"
    patches_dir = event_dir / "patches"
    blob_patches_dir = event_dir / "blob_patches"
    frames_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)
    blob_patches_dir.mkdir(parents=True, exist_ok=True)

    all_entries = pre + post
    timeline = []

    # ── Pass 1: Compute velocity/acceleration across timeline ──
    velocities = []   # px/s at each frame
    accels = []       # px/s² at each frame
    for i in range(len(all_entries)):
        if i == 0:
            velocities.append(0.0)
            accels.append(0.0)
            continue
        e, prev_e = all_entries[i], all_entries[i - 1]
        dt = e["t"] - prev_e["t"]
        if dt <= 0:
            velocities.append(velocities[-1] if velocities else 0.0)
            accels.append(0.0)
            continue
        dx = e["cx"] - prev_e["cx"]
        dy = e["cy"] - prev_e["cy"]
        speed = ((dx * dx + dy * dy) ** 0.5) / dt  # px/s
        velocities.append(round(speed, 1))
        if len(velocities) >= 2:
            dv = velocities[-1] - velocities[-2]
            accels.append(round(dv / dt, 1))
        else:
            accels.append(0.0)

    # Find velocity/acceleration peaks
    peak_vel_idx = max(range(len(velocities)), key=lambda i: velocities[i]) if velocities else 0
    peak_accel_idx = max(range(len(accels)), key=lambda i: abs(accels[i])) if accels else 0

    # Frame dimensions for edge detection (assume 1920x1080)
    FRAME_W, FRAME_H = 1920, 1080
    EDGE_MARGIN = 32  # cursor is 64x64, so within 32px of edge = partial

    # ── Pass 2: Process each frame ──
    for i, e in enumerate(all_entries):
        t_offset = e["t"] - trigger_time

        # Phase
        if e["t"] < trigger_time:
            phase = "had_it"
        elif recovered and e["t"] > recovery_time:
            phase = "recovered"
        elif e["validated"]:
            phase = "recovering"
        else:
            phase = "lost"

        # Edge case tags
        edge_cases = []
        cx, cy = e["cx"], e["cy"]
        if cx < EDGE_MARGIN or cy < EDGE_MARGIN or cx > FRAME_W - EDGE_MARGIN or cy > FRAME_H - EDGE_MARGIN:
            edge_cases.append("partial_cursor")
        if e["blob_count"] > 5:
            edge_cases.append("noisy_background")
        if velocities[i] > 500:
            edge_cases.append("high_speed")
        if abs(accels[i]) > 2000:
            edge_cases.append("high_accel")
        if i == peak_vel_idx and velocities[i] > 50:
            edge_cases.append("peak_velocity")
        if i == peak_accel_idx and abs(accels[i]) > 100:
            edge_cases.append("peak_acceleration")

        # Save frame JPEG
        frame_name = f"f{i:04d}_{t_offset:+06.1f}s.jpg"
        if e["jpeg"]:
            (frames_dir / frame_name).write_bytes(e["jpeg"])

        frame_np = None
        if e["jpeg"]:
            try:
                frame_np = cv2.imdecode(np.frombuffer(e["jpeg"], np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                pass

        # ── Patch at tracked position → CNN re-inference ──
        patch_name = None
        retro_cnn = 0.0
        retro_ms = 0.0
        if frame_np is not None and cx > 0 and cy > 0:
            try:
                patch = extract_gray_patch(frame_np, cx, cy)
                if patch is not None and recognizer:
                    t0 = time.monotonic()
                    retro_cnn = recognizer.classify_patch(patch)
                    retro_ms = (time.monotonic() - t0) * 1000
                    patch_name = f"p{i:04d}_{t_offset:+06.1f}s.png"
                    cv2.imwrite(str(patches_dir / patch_name), patch)
            except Exception:
                pass

        # ── Patches at EACH blob centroid → CNN scoring ──
        # These are all the places the system thought could be the cursor
        blob_results = []
        if frame_np is not None and recognizer and e.get("blobs"):
            for bi, blob in enumerate(e["blobs"][:6]):
                bx, by = blob["centroid"]
                if bx <= 0 or by <= 0:
                    continue
                try:
                    bp = extract_gray_patch(frame_np, bx, by)
                    if bp is not None:
                        bt0 = time.monotonic()
                        b_cnn = recognizer.classify_patch(bp)
                        b_ms = (time.monotonic() - bt0) * 1000
                        bp_name = f"b{i:04d}_{bi}_{t_offset:+06.1f}s.png"
                        cv2.imwrite(str(blob_patches_dir / bp_name), bp)
                        blob_results.append({
                            "centroid": [bx, by],
                            "bbox": blob["bbox"],
                            "cnn_score": round(b_cnn, 4),
                            "cnn_ms": round(b_ms, 1),
                            "patch": f"blob_patches/{bp_name}",
                        })
                except Exception:
                    pass

        timeline.append({
            "index": i,
            "t_offset_s": round(t_offset, 2),
            "utc": e["utc"],
            "phase": phase,
            "cursor": {"x": cx, "y": cy, "method": e["method"], "validated": e["validated"]},
            "live_cnn": {"confidence": round(e["cnn_conf"], 4), "model": e["cnn_model"],
                         "inference_ms": e.get("cnn_ms", 0)},
            "retrospective_cnn": {"confidence": round(retro_cnn, 4), "inference_ms": round(retro_ms, 1)},
            "silhouette": {
                "confidence": round(e["sil_conf"], 3),
                "method": e["sil_method"],
                "roi_size": e["sil_roi"],
                "misses": e["sil_misses"],
            },
            "velocity_px_s": velocities[i],
            "acceleration_px_s2": accels[i],
            "blob_count": e["blob_count"],
            "blob_candidates": blob_results,
            "edge_cases": edge_cases,
            "vision": {"x": e.get("vision_x", 0), "y": e.get("vision_y", 0)},
            "frame": f"frames/{frame_name}" if e["jpeg"] else None,
            "patch": f"patches/{patch_name}" if patch_name else None,
        })

    # ── Velocity/acceleration summary ──
    vel_summary = {}
    if velocities:
        vel_summary = {
            "max_velocity_px_s": round(max(velocities), 1),
            "max_velocity_at_offset_s": round(all_entries[peak_vel_idx]["t"] - trigger_time, 2),
            "max_accel_px_s2": round(max(abs(a) for a in accels), 1),
            "max_accel_at_offset_s": round(all_entries[peak_accel_idx]["t"] - trigger_time, 2),
            "avg_velocity_pre": round(
                sum(velocities[:len(pre)]) / max(1, len(pre)), 1),
            "avg_velocity_post": round(
                sum(velocities[len(pre):]) / max(1, len(post)), 1),
        }

    # Count total blob patches generated
    total_blob_patches = sum(len(t.get("blob_candidates", [])) for t in timeline)

    # Manifest
    manifest = {
        "event_id": event_id,
        "trigger": trigger,
        "trigger_utc": datetime.fromtimestamp(
            time.time() - (time.monotonic() - trigger_time), tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
        "recovered": recovered,
        "duration_s": round(all_entries[-1]["t"] - all_entries[0]["t"], 1) if all_entries else 0,
        "total_frames": len(all_entries),
        "pre_frames": len(pre),
        "post_frames": len(post),
        "total_tracked_patches": sum(1 for t in timeline if t["patch"]),
        "total_blob_patches": total_blob_patches,
        "phases": {
            "had_it": sum(1 for t in timeline if t["phase"] == "had_it"),
            "lost": sum(1 for t in timeline if t["phase"] == "lost"),
            "recovering": sum(1 for t in timeline if t["phase"] == "recovering"),
            "recovered": sum(1 for t in timeline if t["phase"] == "recovered"),
        },
        "velocity": vel_summary,
        "edge_cases_found": list(set(
            tag for t in timeline for tag in t.get("edge_cases", [])
        )),
    }

    with open(event_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(event_dir / "timeline.jsonl", "w") as f:
        for t_entry in timeline:
            f.write(json.dumps(t_entry) + "\n")

    _loss_events_log.append(manifest)

    print(f"[loss_event] Saved {event_id}: {trigger} | {len(all_entries)} frames "
          f"({len(pre)} pre + {len(post)} post) | {total_blob_patches} blob patches | "
          f"peak_vel={vel_summary.get('max_velocity_px_s', 0)}px/s | recovered={recovered}")


def jitter_loop():
    """Anti-sleep jitter with opportunistic cursor re-acquisition.

    Replaces ESP32Mouse's built-in anti-sleep thread so we control the
    timing. Every jitter_interval_s (30s):

    Normal mode: ±Npx jitter (prevents Windows sleep, invisible to user).

    Re-acquisition mode: when CNN confidence has been <0.7 for >15s,
    the jitter fires a correlation probe instead of simple ±3px.
    Uses _jitter_reacquire() which does 15px dual-direction probe —
    large enough for reliable blob correlation but still serves as
    anti-sleep (any mouse movement resets Windows idle timer).
    """
    global _low_confidence_since, _vision_lockout_until

    while True:
        time.sleep(config.jitter_interval_s)

        # Pause jitter during velocity tests — mouse is under test control
        if _vtest_running:
            continue

        # Update hardware monitoring state
        # Check _ser (private) to avoid triggering the lazy-init property.
        # If heartbeat thread is alive, the serial IS working.
        try:
            hb_alive = (
                mouse is not None and
                hasattr(mouse, '_heartbeat_stop') and
                not mouse._heartbeat_stop.is_set()
            )
            ser_open = (
                mouse is not None and
                mouse._ser is not None and
                mouse._ser.is_open
            )
            state.mouse_connected = hb_alive or ser_open
            state.heartbeat_active = hb_alive
            state.anti_sleep_active = True  # We ARE the anti-sleep now
        except Exception as e:
            import traceback
            traceback.print_exc()

        if not mouse:
            continue

        use_daemon = daemon_client and daemon_client.is_running()
        now = time.monotonic()

        # Check if re-acquisition needed. Two independent triggers:
        # 1. CNN: confidence < 0.7 for entire jitter cycle
        # 2. Silhouette: tracker struggling (ROI expanded OR high misses)
        # Silhouette trigger bypasses CNN false-positive problem.
        cnn_trigger = (
            _low_confidence_since > 0 and
            (now - _low_confidence_since) > config.jitter_interval_s
        )
        sil_trigger = (
            sil_tracker and (
                sil_tracker.roi_size >= sil_tracker.ROI_MAX or
                sil_tracker.consecutive_misses >= 10
            )
        )
        need_reacquire = (cnn_trigger or sil_trigger) and (use_daemon or sensor)

        if need_reacquire:
            # Try YOLO first: if YOLO recently detected cursor, use that position
            # instead of the slow jitter probe (saves ~2s)
            yolo_age = now - state.yolo_timestamp if state.yolo_timestamp > 0 else 999
            if (yolo_detector and state.yolo_active
                    and state.yolo_confidence > 0.5 and yolo_age < 5.0):
                result = (state.yolo_x, state.yolo_y)
            else:
                # Correlation probe: the jitter IS the probe signal
                result = _jitter_reacquire(use_daemon)
            if result:
                rx, ry = result
                state.cursor_x = rx
                state.cursor_y = ry
                state.cursor_method = "jitter_reacquire"
                state.cursor_age_s = 0.0
                state.cursor_validated = False
                state.cnn_confidence = 0.0
                _low_confidence_since = 0.0  # Found it — reset
                if sil_tracker:
                    sil_tracker.reset()
            else:
                # Probe failed — still do normal jitter for anti-sleep
                try:
                    px = config.jitter_pixels
                    mouse._send_raw(px, 0)
                    time.sleep(0.1)
                    mouse._send_raw(-px, 0)
                except Exception:
                    pass
        elif (state.jitter_count % config.verify_every_n == 0
              and (use_daemon or sensor)):
            # Verification probe: dual-direction check to catch CNN false positives.
            # The tracker might be confidently tracking the WRONG thing.
            result = _jitter_reacquire(use_daemon)
            if result:
                rx, ry = result
                drift = ((rx - state.cursor_x) ** 2 +
                         (ry - state.cursor_y) ** 2) ** 0.5
                if drift > config.verify_drift_px:
                    # Probe found cursor far from tracked position — tracker is wrong
                    print(f"[verify] drift={drift:.0f}px: probe=({rx},{ry}) "
                          f"vs tracked=({state.cursor_x},{state.cursor_y}) → re-lock")
                    state.cursor_x = rx
                    state.cursor_y = ry
                    state.cursor_method = "verify_probe"
                    state.cursor_age_s = 0.0
                    state.cursor_validated = False
                    state.cnn_confidence = 0.0
                    _low_confidence_since = 0.0
                    # Protect position from motion override (reuse Vision lockout)
                    _vision_lockout_until = time.monotonic() + _VISION_LOCKOUT_S
                    state.vision_x = rx
                    state.vision_y = ry
                    state.vision_timestamp = time.monotonic()
                    if sil_tracker:
                        sil_tracker.reset()
                    if recognizer and (use_daemon or sensor):
                        # Capture fresh template at probe-confirmed position
                        try:
                            if use_daemon:
                                jpeg = daemon_client.read_jpeg()
                                if jpeg:
                                    frame = cv2.imdecode(
                                        np.frombuffer(jpeg, dtype=np.uint8),
                                        cv2.IMREAD_COLOR)
                                    patch = extract_gray_patch(frame, rx, ry)
                                    if patch is not None:
                                        recognizer.update_cursor_template(patch)
                        except Exception:
                            pass
                # else: drift is small — tracker is on the right thing, no action
        else:
            # Normal anti-sleep jitter: ±Npx, net zero displacement
            try:
                px = config.jitter_pixels
                mouse._send_raw(px, 0)
                time.sleep(0.1)
                mouse._send_raw(-px, 0)
            except Exception:
                pass

        state.last_jitter_time = time.monotonic()
        state.jitter_count += 1


def pi_keyboard_monitor_loop():
    import urllib.request
    while True:
        try:
            req = urllib.request.urlopen("http://10.55.0.2:8081/api/keyboard/health", timeout=2)
            data = json.loads(req.read())
            state.pi_keyboard_connected = data.get("bluetooth_connected", False)
        except Exception:
            state.pi_keyboard_connected = None
        time.sleep(5)


def _jitter_reacquire(use_daemon):
    """Dual-direction probe to reliably find cursor position.

    Moves cursor RIGHT then DOWN, captures 3 frames, matches blobs
    that follow both movements. Only the cursor responds to both —
    UI animations and video noise don't follow our mouse commands.

    Same proven pattern as /api/locate but with smaller (15px) movements.

    Returns (x, y) of detected cursor, or None.
    """
    if not mouse:
        return None

    amp = 15  # Small enough to not disrupt user, large enough to detect

    def _capture():
        if use_daemon:
            jpeg = daemon_client.read_jpeg()
            if jpeg:
                return cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            return None
        if sensor:
            return sensor.capture(settle_frames=1)
        return None

    frame_a = _capture()
    if frame_a is None:
        return None

    # Move RIGHT — wait 250ms for daemon JPEG buffer to refresh
    mouse._send_raw(amp, 0)
    time.sleep(0.25)
    frame_b = _capture()

    # Move DOWN
    mouse._send_raw(0, amp)
    time.sleep(0.25)
    frame_c = _capture()

    # Undo both movements
    mouse._send_raw(-amp, -amp)

    if frame_b is None or frame_c is None:
        return None

    # Extract motion blobs from each direction
    blobs_ab = extract_motion_blobs(
        frame_a, frame_b, threshold=15, min_pixels=5, max_pixels=2000)
    blobs_bc = extract_motion_blobs(
        frame_b, frame_c, threshold=15, min_pixels=5, max_pixels=2000)

    if not blobs_ab or not blobs_bc:
        # Fallback: if only one direction got blobs, pick smallest from it
        fallback_blobs = blobs_ab or blobs_bc
        if fallback_blobs:
            small = [b for b in fallback_blobs if b.pixel_count <= 500]
            if small:
                best = min(small, key=lambda b: b.pixel_count)
                bx, by = best.centroid
                if tracker:
                    tracker.set_position(bx, by, method="dual_probe")
                return (bx, by)
        return None

    # Match blob pairs across both directions.
    # After moving right by amp, cursor blob in A→B is near (original_x + amp/2, original_y).
    # After moving down by amp, cursor blob in B→C is offset from A→B blob by ~(0, amp/2).
    half = amp / 2.0

    # Pre-filter: reject blobs in known-noisy cells (clock, toolbar animations)
    # AND reject large blobs (>500px, UI redraws not cursor movement).
    def _probe_filter(blobs):
        out = []
        for b in blobs:
            if b.pixel_count > 500:
                continue
            gx = min(_NOISE_GRID_W - 1, b.centroid[0] // _NOISE_CELL_PX)
            gy = min(_NOISE_GRID_H - 1, b.centroid[1] // _NOISE_CELL_PX)
            if _noise_grid[gy][gx] >= _NOISE_THRESHOLD:
                continue  # Known-noisy region (toolbar, clock)
            out.append(b)
        return out

    blobs_ab_filt = _probe_filter(blobs_ab)
    blobs_bc_filt = _probe_filter(blobs_bc)

    # Log blob counts for diagnostics
    print(f"[probe] blobs A→B: {len(blobs_ab)} total, {len(blobs_ab_filt)} clean | "
          f"B→C: {len(blobs_bc)} total, {len(blobs_bc_filt)} clean")

    # Fall back to unfiltered if filtering removed everything
    ab_search = blobs_ab_filt or blobs_ab
    bc_search = blobs_bc_filt or blobs_bc

    best_match = None
    best_dist = 9999
    for ba in ab_search:
        for bc in bc_search:
            # B→C blob should be shifted ~(0, +half) from A→B blob
            dx = bc.centroid[0] - ba.centroid[0]
            dy = bc.centroid[1] - ba.centroid[1]
            dist = ((dx - 0) ** 2 + (dy - half) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_match = (ba, bc)

    if best_match and best_dist < amp * 3:
        ba, bc = best_match
        # Cursor was at: A→B blob centroid minus half the rightward movement
        cursor_x = ba.centroid[0] - int(half)
        cursor_y = ba.centroid[1]
        print(f"[probe] match dist={best_dist:.1f} → cursor=({cursor_x},{cursor_y}) "
              f"ba_px={ba.pixel_count} bc_px={bc.pixel_count} "
              f"ba_pos={ba.centroid} bc_pos={bc.centroid}")

        if tracker:
            tracker.set_position(cursor_x, cursor_y, method="dual_probe")

        # Save training data
        new_patch = extract_gray_patch(frame_b, ba.centroid[0], ba.centroid[1])
        if new_patch is not None:
            if recognizer:
                recognizer.update_cursor_template(new_patch)
            if collector:
                collector.collect_from_tracking(
                    frame_b, cursor_x, cursor_y, confidence=0.9)
                state.sample_count_pos = collector.positive_count
                state.sample_count_neg = collector.negative_count

        return (cursor_x, cursor_y)

    # No dual match — fallback to smallest blob from A→B diff
    small_ab = [b for b in blobs_ab if b.pixel_count <= 500]
    if small_ab:
        best = min(small_ab, key=lambda b: b.pixel_count)
        bx, by = best.centroid
        if tracker:
            tracker.set_position(bx, by, method="dual_probe")
        return (bx, by)

    return None


def cursor_validation_loop():
    """Background thread: validate cursor position + re-acquire if lost.

    Every cnn_interval_s (default 30s, configurable via /api/config):

    1. If no position or very stale (>60s): dual-probe re-acquire immediately
    2. If position fresh: CNN classify_patch at (x, y)
       - confidence > 0.7: confirmed, reset low-confidence clock
       - confidence < 0.7: start low-confidence clock, try re-acquire
       - If re-acquire fails, jitter_loop() will retry on next jitter cycle

    The dual-probe re-acquire moves cursor right then down, captures 3
    frames, and matches blobs across both directions. Only the real cursor
    follows both movements — UI noise doesn't.
    """
    global _low_confidence_since
    while True:
        time.sleep(config.cnn_interval_s)
        if not recognizer or not tracker:
            continue

        use_daemon = daemon_client and daemon_client.is_running()
        if not use_daemon and not sensor:
            continue

        try:
            pos = tracker.position
            now = time.monotonic()
            position_stale = (not pos) or (now - pos.timestamp > 60)

            if position_stale:
                # Position unknown or very stale — try YOLO first (no mouse needed),
                # then commanded movement, then legacy probe
                reacquired = False
                yolo_age = now - state.yolo_timestamp if state.yolo_timestamp > 0 else 999
                if (yolo_detector and state.yolo_active
                        and state.yolo_confidence > 0.5 and yolo_age < 5.0):
                    tracker.set_position(state.yolo_x, state.yolo_y, method="yolo")
                    state.cursor_x = state.yolo_x
                    state.cursor_y = state.yolo_y
                    state.cursor_method = "yolo"
                    state.cursor_age_s = 0.0
                    reacquired = True
                if not reacquired and cmd_movement and pos:
                    mr = cmd_movement.escalate(pos.x, pos.y)
                    if mr:
                        tracker.set_position(mr.x, mr.y, method=mr.method)
                        state.cursor_x = mr.x
                        state.cursor_y = mr.y
                        state.cursor_method = mr.method
                        state.cursor_age_s = 0.0
                        reacquired = True
                if not reacquired:
                    result = _jitter_reacquire(use_daemon)
                    if result:
                        reacquired = True
                if reacquired:
                    _low_confidence_since = 0.0  # Reset on success
                    state.cursor_validated = False
                    state.cnn_confidence = 0.0
                    if sil_tracker:
                        sil_tracker.reset()
                else:
                    # Failed to reacquire — signal jitter_loop to try on next cycle
                    if _low_confidence_since == 0.0:
                        _low_confidence_since = time.monotonic()
                continue

            # Position is fresh — validate with CNN
            cx, cy = pos.x, pos.y
            t0 = time.monotonic()

            if use_daemon:
                jpeg = daemon_client.read_jpeg()
                if not jpeg:
                    continue
                frame = cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
            else:
                frame = sensor.capture(settle_frames=1)

            patch = extract_gray_patch(frame, cx, cy)
            if patch is None:
                continue

            confidence = recognizer.classify_patch(patch)
            inference_ms = (time.monotonic() - t0) * 1000

            state.cnn_confidence = confidence
            state.cnn_inference_ms = round(inference_ms, 1)
            state.cnn_model_version = recognizer.model_version

            if confidence > 0.7:
                # Cursor confirmed — update template + reset silhouette to tight ROI
                _low_confidence_since = 0.0  # Reset sustained-low tracker
                recognizer.update_cursor_template(patch)
                state.cursor_validated = True
                if sil_tracker:
                    sil_tracker.reset()
                if collector:
                    collector.collect_from_tracking(
                        frame, cx, cy, confidence=confidence)
                    state.sample_count_pos = collector.positive_count
                    state.sample_count_neg = collector.negative_count
            else:
                # CNN says cursor NOT here — start/keep low-confidence clock.
                # Do NOT reset clock on re-acquisition "success" — the position
                # might still be wrong. Only CNN > 0.7 resets the clock.
                # This lets jitter_loop() see sustained low confidence and fire
                # its correlation probe as a backup.
                if _low_confidence_since == 0.0:
                    _low_confidence_since = time.monotonic()
                state.cursor_validated = False
                if sil_tracker:
                    sil_tracker.reset()
                # Immediate re-acquisition attempt — updates position but
                # does NOT reset the low-confidence clock
                if cmd_movement:
                    mr = cmd_movement.escalate(cx, cy)
                    if mr:
                        tracker.set_position(mr.x, mr.y, method=mr.method)
                        state.cursor_x = mr.x
                        state.cursor_y = mr.y
                        state.cursor_method = mr.method
                else:
                    _jitter_reacquire(use_daemon)

        except Exception:
            pass  # Non-critical — will retry next cycle


# ── API endpoints ───────────────────────────────────────────────────

@app.get("/")
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/profiler")
async def profiler_page():
    return HTMLResponse(PROFILER_HTML)


@app.get("/concepts")
async def concepts_page():
    return HTMLResponse(CONCEPTS_HTML)


@app.get("/api/frame")
async def get_frame():
    with state.lock:
        jpeg = state.frame_overlay
    if jpeg:
        return Response(content=jpeg, media_type="image/jpeg",
                        headers={"X-Frame-UTC": utc_now_ms()})
    return {"error": "No frame yet"}


@app.get("/api/state")
async def get_state():
    return {
        "blob_count": state.blob_count,
        "blobs": [
            {"centroid": b.centroid, "bbox": b.bbox,
             "pixel_count": b.pixel_count, "angle": round(b.angle, 3)}
            for b in state.blobs[:10]
        ],
        "cursor": {
            "x": state.cursor_x, "y": state.cursor_y,
            "method": state.cursor_method,
            "age_s": round(state.cursor_age_s, 1),
            "quality": "good" if state.cursor_age_s < 5
                       else "stale" if state.cursor_age_s < 30
                       else "lost",
        },
        "hardware": {
            "mouse_connected": state.mouse_connected,
            "heartbeat_active": state.heartbeat_active,
            "anti_sleep_active": state.anti_sleep_active,
        },
        "jitter": {
            "last_time": state.last_jitter_time,
            "count": state.jitter_count,
            "reacquire_pending": (
                (_low_confidence_since > 0 and
                 (time.monotonic() - _low_confidence_since) > config.jitter_interval_s) or
                (sil_tracker and (
                    sil_tracker.roi_size >= sil_tracker.ROI_MAX or
                    sil_tracker.consecutive_misses >= 10))
            ),
            "trigger": "cnn" if (_low_confidence_since > 0 and
                (time.monotonic() - _low_confidence_since) > config.jitter_interval_s)
                else "silhouette" if (sil_tracker and (
                    sil_tracker.roi_size >= sil_tracker.ROI_MAX or
                    sil_tracker.consecutive_misses >= 10))
                else "none",
            "low_confidence_s": round(time.monotonic() - _low_confidence_since, 1)
                if _low_confidence_since > 0 else 0.0,
            "verify_in": config.verify_every_n - (state.jitter_count % config.verify_every_n),
        },
        "pi_keyboard": state.pi_keyboard_connected,
        "cnn": {
            "confidence": round(state.cnn_confidence, 3),
            "model_version": state.cnn_model_version,
            "inference_ms": state.cnn_inference_ms,
            "cursor_validated": state.cursor_validated,
            "samples": {
                "pos": state.sample_count_pos,
                "neg": state.sample_count_neg,
            },
        },
        "yolo": {
            "active": state.yolo_active,
            "x": state.yolo_x, "y": state.yolo_y,
            "confidence": round(state.yolo_confidence, 3),
            "bbox": state.yolo_bbox,
            "inference_ms": round(state.yolo_inference_ms, 1),
            "age_s": round(time.monotonic() - state.yolo_timestamp, 1)
                if state.yolo_timestamp > 0 else -1,
            "detection_count": state.yolo_detection_count,
        },
        "silhouette": {
            "active": state.silhouette_active,
            "method": state.silhouette_method,
            "hz": state.silhouette_hz,
            "roi_size": state.silhouette_roi_size,
            "confidence": state.silhouette_confidence,
            "latency_ms": state.silhouette_latency_ms,
            "misses": state.silhouette_misses,
        },
        "vision": {
            "x": state.vision_x,
            "y": state.vision_y,
            "confidence": state.vision_confidence,
            "cursor_type": state.vision_cursor_type,
            "latency_ms": state.vision_latency_ms,
            "age_s": round(time.monotonic() - state.vision_timestamp, 1)
                if state.vision_timestamp > 0 else -1,
            "count": state.vision_count,
        },
        "fps": round(state.fps, 1),
        "frame_count": state.frame_count,
        "claude_detector": claude_detector.to_dict() if claude_detector else None,
        "error": state.error,
        "timestamp": utc_now_ms(),
    }


@app.get("/api/mouse/move")
async def mouse_move(dx: int = 0, dy: int = 0):
    if mouse:
        mouse.move(dx, dy)
        return {"moved": (dx, dy), "position": mouse.get_position()}
    return {"error": "Mouse not initialized"}


@app.get("/api/mouse/click")
async def mouse_click(button: str = "left"):
    if mouse:
        mouse.click(button)
        return {"clicked": button}
    return {"error": "Mouse not initialized"}


@app.get("/api/probe")
async def probe_cursor(amplitude: int = 30):
    """Detect actual cursor position via probe movement + frame diff.

    Pauses motion loop, moves mouse by amplitude, captures diff,
    returns detected position. Updates tracker state.
    """
    if not mouse:
        return {"error": "Mouse not initialized"}

    # In daemon mode, use daemon JPEG frames instead of sensor
    use_daemon = daemon_client and daemon_client.is_running()
    if not use_daemon and not sensor:
        return {"error": "Hardware not initialized"}

    import cv2 as _cv2
    import numpy as _np
    import hashlib as _hl
    try:
        # Capture frame before probe
        if use_daemon:
            jpeg = daemon_client.read_jpeg()
            if not jpeg:
                return {"error": "No frame from daemon"}
            frame_before = _cv2.imdecode(
                _np.frombuffer(jpeg, dtype=_np.uint8), _cv2.IMREAD_COLOR)
        else:
            frame_before = sensor.capture(settle_frames=1)

        h_before = _hl.md5(jpeg).hexdigest()[:8] if use_daemon else "sensor"

        mouse._send_raw(amplitude, 0)
        time.sleep(0.30)  # Wait for daemon JPEG buffer to refresh

        if use_daemon:
            jpeg2 = daemon_client.read_jpeg()
            frame_after = _cv2.imdecode(
                _np.frombuffer(jpeg2, dtype=_np.uint8), _cv2.IMREAD_COLOR) if jpeg2 else frame_before
        else:
            frame_after = sensor.capture(settle_frames=1)
        mouse._send_raw(-amplitude, 0)

        h_after = _hl.md5(jpeg2).hexdigest()[:8] if (use_daemon and jpeg2) else "sensor"
        frames_same = (h_before == h_after)

        # Use cursor shape filter to distinguish cursor from UI changes
        blobs = extract_motion_blobs(
            frame_before, frame_after,
            threshold=15, min_pixels=5, max_pixels=500,
            cursor_shape_filter=True,
        )

        # Also get unfiltered for comparison
        all_blobs = extract_motion_blobs(
            frame_before, frame_after,
            threshold=15, min_pixels=5, max_pixels=3000,
        )

        if not blobs:
            small_blobs = [b for b in all_blobs if b.pixel_count < 200]
            small_blobs.sort(key=lambda b: b.pixel_count)
            if small_blobs:
                blobs = small_blobs

        if not blobs:
            return {
                "error": "No cursor detected",
                "total_blobs": len(all_blobs),
                "debug": {
                    "frames_same": frames_same,
                    "hash_before": h_before,
                    "hash_after": h_after,
                    "all_blob_sizes": [b.pixel_count for b in all_blobs[:10]],
                },
            }

        bx, by = blobs[0].centroid
        if tracker:
            tracker.set_position(bx, by, method="api_probe")
        mouse.set_position(bx, by)
        return {
            "cursor": {"x": bx, "y": by},
            "blob_count": len(blobs),
            "blob_pixels": blobs[0].pixel_count,
            "total_blobs": len(all_blobs),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/locate")
async def locate_cursor():
    """Dual-direction probe to reliably find cursor position.

    Moves cursor RIGHT then DOWN, captures 3 frames, matches blobs
    that follow both movements. The cursor is the only thing that
    moves consistently in both directions.
    """
    if not mouse:
        return {"error": "Mouse not initialized"}

    use_daemon = daemon_client and daemon_client.is_running()
    if not use_daemon and not sensor:
        return {"error": "Hardware not initialized"}

    def _capture():
        if use_daemon:
            import cv2, numpy as np
            jpeg = daemon_client.read_jpeg()
            if jpeg:
                return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            return None
        return sensor.capture(settle_frames=1)

    try:
        frame_a = _capture()

        mouse._send_chunked(120, 0)
        time.sleep(0.2)
        frame_b = _capture()

        mouse._send_chunked(0, 120)
        time.sleep(0.2)
        frame_c = _capture()

        # Return cursor to original position
        mouse._send_chunked(-120, -120)

        blobs_ab = extract_motion_blobs(
            frame_a, frame_b, threshold=12, min_pixels=5, max_pixels=2000)
        blobs_bc = extract_motion_blobs(
            frame_b, frame_c, threshold=12, min_pixels=5, max_pixels=2000)

        if not blobs_ab or not blobs_bc:
            return {
                "error": "No motion detected",
                "blobs_right": len(blobs_ab),
                "blobs_down": len(blobs_bc),
            }

        # Match: B->C blob should be offset ~(60, 60) from A->B blob
        best_match = None
        best_dist = 9999
        for ba in blobs_ab:
            for bc in blobs_bc:
                dx = bc.centroid[0] - ba.centroid[0]
                dy = bc.centroid[1] - ba.centroid[1]
                dist = ((dx - 60) ** 2 + (dy - 60) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_match = (ba, bc)

        if best_match and best_dist < 150:
            ba, bc = best_match
            cursor_x = ba.centroid[0] - 60
            cursor_y = ba.centroid[1]
            if tracker:
                tracker.set_position(cursor_x, cursor_y, method="dual_probe")
            mouse.set_position(cursor_x, cursor_y)
            return {
                "cursor": {"x": cursor_x, "y": cursor_y},
                "match_dist": round(best_dist, 1),
                "blobs_right": len(blobs_ab),
                "blobs_down": len(blobs_bc),
            }
        else:
            return {
                "error": "No consistent cursor match",
                "best_dist": round(best_dist, 1),
                "blobs_right": len(blobs_ab),
                "blobs_down": len(blobs_bc),
            }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/escalate")
async def escalate_cursor():
    """Escalating commanded movement recovery chain.

    Tries increasingly aggressive movements to re-find the cursor:
    1. ±3px jitter (invisible)
    2. ±15px dual-direction probe
    3. ±50px dual-direction probe
    Falls back to legacy _jitter_reacquire() if CommandedMovement not available.
    """
    if not mouse:
        return {"error": "Mouse not initialized"}
    if not tracker:
        return {"error": "Tracker not initialized"}

    pos = tracker.position
    last_x = pos.x if pos else 960
    last_y = pos.y if pos else 540

    if cmd_movement:
        mr = cmd_movement.escalate(last_x, last_y)
        if mr:
            tracker.set_position(mr.x, mr.y, method=mr.method)
            if sil_tracker:
                sil_tracker.reset()
            return {
                "cursor": {"x": mr.x, "y": mr.y},
                "method": mr.method,
                "amplitude": mr.amplitude,
                "confidence": round(mr.confidence, 3),
                "latency_ms": round(mr.latency_ms, 1),
            }

    # Fallback to legacy
    use_daemon = daemon_client and daemon_client.is_running()
    result = _jitter_reacquire(use_daemon)
    if result:
        return {"cursor": {"x": result[0], "y": result[1]}, "method": "legacy_dual_probe"}

    return {"error": "All recovery methods failed"}


@app.get("/api/screenshot")
async def screenshot(label: str = "capture"):
    """Capture and save an overlayed frame (what the user sees).

    In daemon mode, returns the latest overlayed frame (with blob rects
    and cursor crosshair). This is the same image visible in the dashboard,
    suitable for the UI agent to see the current screen state.
    """
    try:
        use_daemon = daemon_client and daemon_client.is_running()

        if use_daemon:
            # Return the latest overlayed frame from the motion loop
            with state.lock:
                jpeg = state.frame_overlay
            if not jpeg:
                return {"error": "No frame available yet"}
            path = "/tmp/nav_" + label + ".jpg"
            with open(path, "wb") as f:
                f.write(jpeg)
            return {"saved": path, "source": "daemon_overlay"}
        elif sensor:
            frame = sensor.capture(settle_frames=1)
            path = "/tmp/nav_" + label + ".jpg"
            import cv2 as _cv2
            bgr = _cv2.cvtColor(frame, _cv2.COLOR_RGB2BGR)
            _cv2.imwrite(path, bgr, [_cv2.IMWRITE_JPEG_QUALITY, 90])
            return {"saved": path, "shape": list(frame.shape), "source": "sensor"}
        else:
            return {"error": "No frame source available"}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/config")
async def get_config():
    """Return current dashboard configuration."""
    return {
        "frame_fps": config.frame_fps,
        "cnn_interval_s": config.cnn_interval_s,
        "jitter_interval_s": config.jitter_interval_s,
        "jitter_pixels": config.jitter_pixels,
        "verify_every_n": config.verify_every_n,
        "verify_drift_px": config.verify_drift_px,
    }


@app.post("/api/config")
async def set_config(request: Request):
    """Update dashboard configuration at runtime.

    Accepts JSON body with any subset of config keys. The UI agent
    can tune these parameters based on the current task needs.

    Example:
        POST /api/config {"frame_fps": 5, "cnn_interval_s": 10}
    """
    data = await request.json()

    if "frame_fps" in data:
        config.frame_fps = max(0.1, min(30.0, float(data["frame_fps"])))
    if "cnn_interval_s" in data:
        config.cnn_interval_s = max(1.0, min(300.0, float(data["cnn_interval_s"])))
    if "jitter_interval_s" in data:
        config.jitter_interval_s = max(5.0, min(300.0, float(data["jitter_interval_s"])))
        # jitter_loop() reads config each cycle — no restart needed
    if "jitter_pixels" in data:
        config.jitter_pixels = max(1, min(30, int(data["jitter_pixels"])))
    if "verify_every_n" in data:
        config.verify_every_n = max(1, min(20, int(data["verify_every_n"])))
    if "verify_drift_px" in data:
        config.verify_drift_px = max(20, min(500, int(data["verify_drift_px"])))

    return await get_config()


# ── Claude Vision Detection ────────────────────────────────────────

@app.get("/api/claude_detect")
async def claude_detect():
    """Trigger Claude Vision cursor detection (~5-15s).

    Captures current frame, chops into 4 quadrants, sends to Claude Agent SDK.
    Returns structured per-quadrant analysis with cursor position.

    If high confidence, updates tracker position and saves CNN training samples.
    """
    if not claude_detector:
        return {"error": "Claude detector not initialized"}

    # Use the EXACT same frame the rest of the system sees (state.frame_raw).
    # This ensures Vision's ground truth matches the frame used for blob detection,
    # silhouette tracking, and CNN validation — no timing skew from separate capture.
    frame_capture_time = time.monotonic()
    with state.lock:
        frame = state.frame_raw.copy() if state.frame_raw is not None else None

    if frame is None:
        return {"error": "No frame available"}

    # Get current estimated position
    pos = tracker.position if tracker else None
    est_x = pos.x if pos else None
    est_y = pos.y if pos else None

    # Run detection
    t0 = time.monotonic()
    result = await claude_detector.detect(frame, est_x, est_y)
    detect_ms = (time.monotonic() - t0) * 1000

    # Always update vision state for dashboard display
    state.vision_timestamp = time.monotonic()
    state.vision_latency_ms = round(detect_ms, 0)
    state.vision_count += 1
    if result.cursor_found and result.cursor_x is not None:
        state.vision_x = result.cursor_x
        state.vision_y = result.cursor_y
        state.vision_confidence = result.confidence or "unknown"
        state.vision_cursor_type = result.cursor_type or "unknown"
    else:
        state.vision_confidence = "not_found"

    # If high confidence and not confusing, update tracker and save samples.
    # When Vision position differs from tracker by >50px and CNN had validated,
    # the old position is a CNN false positive → save as negative training data.
    saved = 0
    old_x, old_y = state.cursor_x, state.cursor_y
    old_validated = state.cursor_validated
    if result.cursor_found and result.confidence in ("high", "medium"):
        if result.cursor_x is not None and result.cursor_y is not None:
            delta = ((old_x - result.cursor_x) ** 2 +
                     (old_y - result.cursor_y) ** 2) ** 0.5
            if delta > 50 and old_validated and collector:
                # Save old position as NEGATIVE sample (CNN was wrong here)
                # confidence=0.0 marks it as definite non-cursor
                collector.collect_from_tracking(
                    frame, old_x, old_y, confidence=0.0)
                state.sample_count_neg = collector.negative_count

            # Also reset low-confidence clock — Claude Vision is ground truth
            global _low_confidence_since, _vision_lockout_until
            _low_confidence_since = 0.0
            _vision_lockout_until = time.monotonic() + _VISION_LOCKOUT_S
            if tracker:
                tracker.set_position(
                    result.cursor_x, result.cursor_y, method="claude_vision")
                state.cursor_x = result.cursor_x
                state.cursor_y = result.cursor_y
                state.cursor_method = "claude_vision"
                state.cursor_age_s = 0.0
                state.cursor_validated = True
                state.cnn_confidence = 1.0  # Claude Vision overrides CNN
            if sil_tracker:
                sil_tracker.reset()  # Re-lock silhouette to Claude-found position
            # Capture fresh template from Vision-confirmed position so
            # silhouette tracks the ACTUAL cursor, not old UI element match
            if recognizer and frame is not None:
                patch = extract_gray_patch(frame, result.cursor_x, result.cursor_y)
                if patch is not None:
                    recognizer.update_cursor_template(patch)
            if collector:
                saved = claude_detector.save_training_samples(
                    frame, result, collector)
                state.sample_count_pos = collector.positive_count
                state.sample_count_neg = collector.negative_count

    # ── Vision Dataset Logging ──
    # Every Vision call is logged with full context for analysis
    _log_vision_dataset(result, detect_ms, est_x, est_y, old_x, old_y,
                        old_validated, frame)

    return {
        **claude_detector.to_dict(),
        "samples_saved": saved,
    }


# ── Vision Dataset ─────────────────────────────────────────────────

_VISION_DATASET_DIR = Path(__file__).parent.parent.parent / "data" / "vision_dataset"
_VISION_DATASET_DIR.mkdir(parents=True, exist_ok=True)
_VISION_DATASET_LOG = _VISION_DATASET_DIR / "log.jsonl"


def _log_vision_dataset(result, detect_ms, est_x, est_y,
                        tracker_x, tracker_y, tracker_validated, frame):
    """Log every Claude Vision inference for accuracy tracking.

    Dataset captures:
    - Vision output: coordinates, confidence, reasoning, per-quadrant analysis
    - Input context: tracker position, CNN state, estimated position
    - Coordinate sanity: whether reported position matches claimed quadrant
    - Frame snapshot for replay
    """
    try:
        entry_id = int(time.time() * 1000)

        # Sanity check: does cursor_x,cursor_y match the claimed quadrant?
        quadrant_valid = None
        cursor_quadrant = None
        if result.cursor_found and result.cursor_x is not None:
            cx, cy = result.cursor_x, result.cursor_y
            # Which quadrant do the coordinates fall in?
            if cx < 960:
                cursor_quadrant = 1 if cy < 540 else 3
            else:
                cursor_quadrant = 2 if cy < 540 else 4
            # Which quadrant did Vision say has the cursor?
            claimed_q = None
            for q in result.quadrants:
                if q.is_cursor_present:
                    claimed_q = q.quadrant
                    break
            quadrant_valid = (cursor_quadrant == claimed_q) if claimed_q else None

        # Save frame snapshot
        frame_path = None
        if frame is not None:
            fname = f"frame_{entry_id}.jpg"
            fpath = _VISION_DATASET_DIR / fname
            cv2.imwrite(str(fpath),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_path = fname

        entry = {
            "id": entry_id,
            "utc": utc_now_ms(),
            "latency_ms": round(detect_ms, 1),
            "input": {
                "est_x": est_x, "est_y": est_y,
                "tracker_x": tracker_x, "tracker_y": tracker_y,
                "tracker_validated": tracker_validated,
                "cnn_confidence": round(state.cnn_confidence, 3),
            },
            "output": {
                "cursor_found": result.cursor_found,
                "cursor_x": result.cursor_x,
                "cursor_y": result.cursor_y,
                "cursor_type": result.cursor_type,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "has_confusing": result.has_confusing_quadrants,
            },
            "quadrants": [
                {
                    "q": q.quadrant,
                    "cursor": q.is_cursor_present,
                    "dist": q.distance_from_estimated_point,
                    "confusing": q.is_the_image_confusing,
                }
                for q in result.quadrants
            ],
            "sanity": {
                "coord_quadrant": cursor_quadrant,
                "quadrant_valid": quadrant_valid,
            },
            "delta_from_tracker": round(
                ((tracker_x - result.cursor_x) ** 2 +
                 (tracker_y - result.cursor_y) ** 2) ** 0.5, 1
            ) if result.cursor_x is not None else None,
            "frame": frame_path,
        }

        with open(_VISION_DATASET_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

    except Exception as e:
        print(f"Vision dataset log error: {e}")


# ── Experience Logging ─────────────────────────────────────────────

_EXPERIENCE_DIR = Path(__file__).parent.parent.parent / "data" / "experience"
_EXPERIENCE_DIR.mkdir(parents=True, exist_ok=True)
_EXPERIENCE_LOG = _EXPERIENCE_DIR / "log.jsonl"


@app.post("/api/experience/log")
async def log_experience_snapshot(request: Request):
    """Log full sensor stack snapshot with optional user note.

    POST body: {"note": "cursor is at approximately (500, 400)", "test_id": "V1"}

    Every Claude Vision detection auto-logs. This endpoint also allows
    manual snapshots from Claude Code or the dashboard.
    """
    body = await request.json() if request.headers.get("content-type") == "application/json" else {}
    now_mono = time.monotonic()

    snapshot = {
        "utc": utc_now_ms(),
        "note": body.get("note", ""),
        "test_id": body.get("test_id", "manual"),
        "tracker": {
            "x": state.cursor_x, "y": state.cursor_y,
            "method": state.cursor_method,
            "age_s": round(state.cursor_age_s, 1),
            "validated": state.cursor_validated,
        },
        "cnn": {
            "confidence": round(state.cnn_confidence, 3),
            "validated": state.cursor_validated,
            "model": state.cnn_model_version,
        },
        "silhouette": {
            "active": state.silhouette_active,
            "method": state.silhouette_method,
            "confidence": state.silhouette_confidence,
            "roi_size": state.silhouette_roi_size,
            "misses": state.silhouette_misses,
            "hz": state.silhouette_hz,
        },
        "yolo": {
            "active": state.yolo_active,
            "x": state.yolo_x, "y": state.yolo_y,
            "confidence": round(state.yolo_confidence, 3),
            "bbox": state.yolo_bbox,
            "inference_ms": round(state.yolo_inference_ms, 1),
            "age_s": round(now_mono - state.yolo_timestamp, 1)
                if state.yolo_timestamp > 0 else -1,
            "detection_count": state.yolo_detection_count,
        },
        "vision": {
            "x": state.vision_x, "y": state.vision_y,
            "confidence": state.vision_confidence,
            "age_s": round(now_mono - state.vision_timestamp, 1)
                if state.vision_timestamp > 0 else -1,
        },
        "jitter": {
            "count": state.jitter_count,
            "low_confidence_s": round(now_mono - _low_confidence_since, 1)
                if _low_confidence_since > 0 else 0,
        },
        "motion_blobs": state.blob_count,
        "fps": round(state.fps, 1),
        "frame_count": state.frame_count,
    }

    # Append to JSONL log
    with open(_EXPERIENCE_LOG, "a") as f:
        f.write(json.dumps(snapshot) + "\n")

    return snapshot


# ── Validation GUI ─────────────────────────────────────────────────

_VALIDATION_SAMPLES_DIR = Path(__file__).parent.parent.parent / "data" / "cursor_samples"
_VALIDATION_REVIEWS_PATH = _VALIDATION_SAMPLES_DIR / "reviews.jsonl"


@app.get("/validation")
async def validation_page():
    """Validation GUI for reviewing CNN training samples."""
    return HTMLResponse(VALIDATION_HTML)


@app.get("/api/validation/samples")
async def get_validation_samples(page: int = 0, page_size: int = 20,
                                  sort: str = "priority",
                                  filter: str = "all",
                                  label: str = "all",
                                  source_filter: str = "all"):
    """Get paginated sample list for validation with server-side filtering.

    Sort: priority, recent, confidence_asc, confidence_desc, source
    Filter: all, unreviewed, reviewed, confusing, claude, disagreement
    Label: all, pos, neg
    Source: all, motion_track, claude_vision, ground_truth
    """
    index_path = _VALIDATION_SAMPLES_DIR / "index.jsonl"
    if not index_path.exists():
        return {"samples": [], "total": 0, "page": page, "filter_counts": {}}

    entries = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # Load reviews
    reviewed = {}
    file_tags = {}  # filename → {col: value, ...} aggregated from all tag actions
    if _VALIDATION_REVIEWS_PATH.exists():
        with open(_VALIDATION_REVIEWS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        review = json.loads(line)
                        fn = review["filename"]
                        if review.get("action") == "tag" and "tags" in review:
                            # Multi-dimensional tags — merge into file_tags
                            if fn not in file_tags:
                                file_tags[fn] = {}
                            file_tags[fn].update(review["tags"])
                        else:
                            reviewed[fn] = review
                    except (json.JSONDecodeError, KeyError):
                        continue

    # Score all entries (CNN inference for priority)
    model = recognizer._cnn_model if recognizer else None
    import torch

    scored_entries = []
    for entry in entries:
        filename = entry.get("filename", "")
        review = reviewed.get(filename)
        is_reviewed = review is not None

        cnn_conf = None
        if model is not None:
            patch_path = _VALIDATION_SAMPLES_DIR / filename
            if patch_path.exists():
                patch = cv2.imread(str(patch_path), cv2.IMREAD_GRAYSCALE)
                if patch is not None and patch.shape == (64, 64):
                    tensor = torch.from_numpy(
                        patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        cnn_conf = model(tensor).item()

        priority = 5.0
        entry_label = entry.get("label", "")
        entry_source = entry.get("source", "")
        is_confusing = entry.get("is_confusing", 0)

        # Normalize label (some entries use 1/0 instead of pos/neg)
        if entry_label == 1 or entry_label == "1":
            entry_label = "pos"
        elif entry_label == 0 or entry_label == "0":
            entry_label = "neg"

        cnn_disagrees = False
        if is_reviewed:
            priority = 99.0
        elif is_confusing:
            priority = 1.0
        elif cnn_conf is not None:
            predicted_pos = cnn_conf > 0.5
            labeled_pos = entry_label == "pos"
            if predicted_pos != labeled_pos:
                priority = 0.0 + abs(cnn_conf - 0.5)
                cnn_disagrees = True
            elif abs(cnn_conf - 0.5) < 0.2:
                priority = 2.0
            elif "claude" in entry_source:
                priority = 3.0
            else:
                priority = 4.0

        scored_entries.append({
            **entry,
            "label": entry_label,  # normalized
            "cnn_confidence": round(cnn_conf, 3) if cnn_conf is not None else None,
            "priority": round(priority, 3),
            "review": review,
            "cnn_disagrees": cnn_disagrees,
            "tags": file_tags.get(filename, {}),
        })

    # Compute filter counts BEFORE filtering (so tabs show totals)
    filter_counts = {
        "all": len(scored_entries),
        "unreviewed": sum(1 for e in scored_entries if not e["review"]),
        "reviewed": sum(1 for e in scored_entries if e["review"]),
        "confusing": sum(1 for e in scored_entries if e.get("is_confusing") or
                        (e.get("review") and e["review"].get("is_confusing_to_human"))),
        "claude": sum(1 for e in scored_entries if "claude" in e.get("source", "")),
        "disagreement": sum(1 for e in scored_entries if e.get("cnn_disagrees")),
        "pos": sum(1 for e in scored_entries if e["label"] == "pos"),
        "neg": sum(1 for e in scored_entries if e["label"] == "neg"),
    }
    # Source breakdown
    source_counts = {}
    for e in scored_entries:
        src = e.get("source", "unknown")
        # Group by prefix (motion_track, claude_vision, ground_truth)
        prefix = src.split("_neg")[0] if "_neg" in src else src
        source_counts[prefix] = source_counts.get(prefix, 0) + 1
    filter_counts["sources"] = source_counts

    # Apply server-side filters
    filtered = scored_entries
    if filter == "unreviewed":
        filtered = [e for e in filtered if not e["review"]]
    elif filter == "reviewed":
        filtered = [e for e in filtered if e["review"]]
    elif filter == "confusing":
        filtered = [e for e in filtered if e.get("is_confusing") or
                    (e.get("review") and e["review"].get("is_confusing_to_human"))]
    elif filter == "claude":
        filtered = [e for e in filtered if "claude" in e.get("source", "")]
    elif filter == "disagreement":
        filtered = [e for e in filtered if e.get("cnn_disagrees")]

    if label != "all":
        filtered = [e for e in filtered if e["label"] == label]

    if source_filter != "all":
        filtered = [e for e in filtered if source_filter in e.get("source", "")]

    # Sort
    if sort == "priority":
        filtered.sort(key=lambda e: e["priority"])
    elif sort == "recent":
        filtered.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
    elif sort == "confidence_asc":
        filtered.sort(key=lambda e: e.get("cnn_confidence") or 0)
    elif sort == "confidence_desc":
        filtered.sort(key=lambda e: e.get("cnn_confidence") or 0, reverse=True)
    elif sort == "source":
        filtered.sort(key=lambda e: e.get("source", ""))

    # Paginate
    total = len(filtered)
    start = page * page_size
    end = start + page_size
    page_entries = filtered[start:end]

    return {
        "samples": page_entries,
        "total": total,
        "page": page,
        "page_size": page_size,
        "filter_counts": filter_counts,
    }


@app.get("/api/validation/patch/{filename}")
async def get_validation_patch(filename: str):
    """Serve a 64x64 PNG patch image for the validation UI."""
    # Sanitize filename to prevent path traversal
    if "/" in filename or ".." in filename:
        return {"error": "Invalid filename"}
    patch_path = _VALIDATION_SAMPLES_DIR / filename
    if not patch_path.exists():
        return {"error": "File not found"}
    return Response(
        content=patch_path.read_bytes(),
        media_type="image/png",
    )


@app.post("/api/validation/review")
async def submit_review(request: Request):
    """Submit a review for a sample.

    Body: {"filename": "pos_000123.png", "action": "correct"|"wrong"|"delete"|"confusing"|"annotate_tip",
           "cursor_type": "arrow"|null, "tip_x": int|null, "tip_y": int|null}
    """
    data = await request.json()
    filename = data.get("filename")
    action = data.get("action")

    valid_actions = ("correct", "wrong", "confusing", "delete", "annotate_tip")
    if not filename or action not in valid_actions:
        return {"error": f"Invalid review: need filename and action ({'/'.join(valid_actions)})"}

    review = {
        "filename": filename,
        "action": action,
        "timestamp": time.time(),
    }
    if data.get("cursor_type"):
        review["cursor_type"] = data["cursor_type"]
    if action == "confusing":
        review["is_confusing_to_human"] = True
    if action == "wrong":
        current_label = "pos" if filename.startswith("pos_") else "neg"
        review["corrected_label"] = "neg" if current_label == "pos" else "pos"
    if data.get("tip_x") is not None and data.get("tip_y") is not None:
        review["tip_x"] = int(data["tip_x"])
        review["tip_y"] = int(data["tip_y"])

    with open(_VALIDATION_REVIEWS_PATH, "a") as f:
        f.write(json.dumps(review) + "\n")

    return {"ok": True, "review": review}


@app.post("/api/validation/tag")
async def submit_tags(request: Request):
    """Multi-dimensional tagging: set multiple independent features on a sample.

    Body: {"filename": "pos_000123.png", "tags": {"quality": "ok", "confusing": "yes",
           "cursor_type": "arrow", "visibility": "partial", "background": "noisy"}}

    Tags are key-value pairs. Each tag is independent — setting one doesn't affect others.
    Previous tags for the same key on the same file are superseded by the latest.
    """
    data = await request.json()
    filename = data.get("filename")
    tags = data.get("tags", {})

    if not filename or not tags:
        return {"error": "Need filename and tags dict"}

    record = {
        "filename": filename,
        "action": "tag",
        "tags": tags,
        "timestamp": time.time(),
    }

    with open(_VALIDATION_REVIEWS_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    return {"ok": True, "record": record}


@app.get("/api/validation/tag_schema")
async def get_tag_schema():
    """Return the current tag columns and their possible values.

    Columns are discovered from existing reviews + a default schema.
    """
    default_schema = {
        "quality": {"values": ["ok", "wrong", "unsure"], "color": "#3498db"},
        "confusing": {"values": ["yes", "no"], "color": "#e67e22"},
        "cursor_type": {"values": ["arrow", "hand", "ibeam", "busy", "crosshair", "move", "unknown"], "color": "#9b59b6"},
        "visibility": {"values": ["full", "partial", "hidden", "offscreen"], "color": "#2ecc71"},
        "background": {"values": ["clean", "noisy", "animated", "textured"], "color": "#e74c3c"},
    }

    # Discover extra columns from existing reviews
    if _VALIDATION_REVIEWS_PATH.exists():
        for line in _VALIDATION_REVIEWS_PATH.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get("action") == "tag" and "tags" in r:
                    for k, v in r["tags"].items():
                        if k not in default_schema:
                            default_schema[k] = {"values": [], "color": "#888"}
                        if v not in default_schema[k]["values"]:
                            default_schema[k]["values"].append(v)
            except Exception:
                pass

    return {"schema": default_schema}


@app.post("/api/validation/retrain")
async def trigger_retrain():
    """Trigger model retraining as a background subprocess."""
    import subprocess
    project_root = Path(__file__).parent.parent.parent
    script = project_root / "scripts" / "train_cursor_model.py"
    venv_python = project_root / ".venv" / "bin" / "python"

    proc = subprocess.Popen(
        [str(venv_python), str(script), "--epochs", "30", "--patience", "5"],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    return {
        "status": "training_started",
        "pid": proc.pid,
        "command": f"python scripts/train_cursor_model.py --epochs 30 --patience 5",
    }


@app.get("/api/validation/stats")
async def get_validation_stats():
    """Get validation stats: sample counts, model version, review progress."""
    # Count reviews
    review_count = 0
    if _VALIDATION_REVIEWS_PATH.exists():
        with open(_VALIDATION_REVIEWS_PATH) as f:
            review_count = sum(1 for line in f if line.strip())

    # Count samples by source
    source_counts = {}
    index_path = _VALIDATION_SAMPLES_DIR / "index.jsonl"
    if index_path.exists():
        with open(index_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        src = entry.get("source", "unknown")
                        source_counts[src] = source_counts.get(src, 0) + 1
                    except json.JSONDecodeError:
                        continue

    return {
        "pos_count": collector.positive_count if collector else 0,
        "neg_count": collector.negative_count if collector else 0,
        "total_samples": collector.total_samples if collector else 0,
        "review_count": review_count,
        "model_version": recognizer.model_version if recognizer else "none",
        "source_distribution": source_counts,
    }


# ── Pipeline Profiler API ────────────────────────────────────────────


@app.get("/api/profiler")
async def get_profiler():
    """Live pipeline timing + all algorithm states with confidence scores.

    Provides a complete snapshot of every subsystem for timing analysis.
    """
    now_mono = time.monotonic()
    return {
        "utc": utc_now_ms(),
        "monotonic": now_mono,
        "pipeline": {
            "jpeg_decode_ms": round(state.prof_jpeg_decode_ms, 2),
            "silhouette_ms": round(state.silhouette_latency_ms, 2),
            "overlay_draw_ms": round(state.prof_overlay_draw_ms, 2),
            "jpeg_encode_ms": round(state.prof_jpeg_encode_ms, 2),
            "frame_total_ms": round(state.prof_frame_total_ms, 2),
        },
        "cursor": {
            "x": state.cursor_x,
            "y": state.cursor_y,
            "method": state.cursor_method,
            "age_s": round(state.cursor_age_s, 1),
            "validated": state.cursor_validated,
        },
        "cnn": {
            "confidence": round(state.cnn_confidence, 4),
            "model": state.cnn_model_version,
            "inference_ms": state.cnn_inference_ms,
            "validated": state.cursor_validated,
            "uncertainty": round(1.0 - abs(state.cnn_confidence - 0.5) * 2, 3),
        },
        "silhouette": {
            "method": state.silhouette_method,
            "hz": state.silhouette_hz,
            "confidence": round(state.silhouette_confidence, 4),
            "misses": state.silhouette_misses,
            "roi_size": state.silhouette_roi_size,
            "latency_ms": round(state.silhouette_latency_ms, 2),
        },
        "vision": {
            "x": state.vision_x,
            "y": state.vision_y,
            "confidence": state.vision_confidence,
            "cursor_type": state.vision_cursor_type,
            "latency_ms": state.vision_latency_ms,
            "age_s": round(now_mono - state.vision_timestamp, 1) if state.vision_timestamp > 0 else -1,
            "count": state.vision_count,
        },
        "jitter": {
            "count": state.jitter_count,
            "trigger": "cnn" if (_low_confidence_since > 0 and
                (now_mono - _low_confidence_since) > config.jitter_interval_s)
                else "silhouette" if (sil_tracker and (
                    sil_tracker.roi_size >= sil_tracker.ROI_MAX or
                    sil_tracker.consecutive_misses >= 10))
                else "none",
            "low_confidence_s": round(now_mono - _low_confidence_since, 1)
                if _low_confidence_since > 0 else 0.0,
            "verify_in": config.verify_every_n - (state.jitter_count % config.verify_every_n),
            "lockout_remaining_s": round(max(0, _vision_lockout_until - now_mono), 1),
        },
        "daemon": {
            "fps": state.fps,
            "frame_count": state.frame_count,
            "blob_count": state.blob_count,
            "timestamp_ns": state.prof_daemon_timestamp_ns,
        },
        "hardware": {
            "mouse_connected": state.mouse_connected,
            "heartbeat_active": state.heartbeat_active,
        },
        "noise_grid": noise_grid_features(),
    }


@app.post("/api/profiler/start")
async def start_profiler():
    """Start a 3-minute profiling test. Collects per-frame timing samples."""
    global profiler_samples, profiler_running, profiler_start_time
    profiler_samples = []
    profiler_running = True
    profiler_start_time = time.monotonic()
    return {"status": "started", "duration_s": PROFILER_DURATION_S, "utc": utc_now_ms()}


@app.post("/api/profiler/stop")
async def stop_profiler():
    """Stop the profiling test early."""
    global profiler_running
    profiler_running = False
    return {"status": "stopped", "samples": len(profiler_samples), "utc": utc_now_ms()}


@app.get("/api/profiler/results")
async def get_profiler_results():
    """Get profiler results: per-stage min/avg/max/p95 over the test period."""
    global profiler_running
    # Auto-stop after duration
    if profiler_running and (time.monotonic() - profiler_start_time) >= PROFILER_DURATION_S:
        profiler_running = False

    if not profiler_samples:
        return {"status": "no_data", "running": profiler_running}

    def stats(values):
        if not values:
            return {"min": 0, "avg": 0, "max": 0, "p95": 0}
        s = sorted(values)
        return {
            "min": round(s[0], 2),
            "avg": round(sum(s) / len(s), 2),
            "max": round(s[-1], 2),
            "p95": round(s[int(len(s) * 0.95)], 2) if len(s) >= 2 else round(s[-1], 2),
        }

    keys = ["jpeg_decode_ms", "silhouette_ms", "overlay_draw_ms",
            "jpeg_encode_ms", "frame_total_ms"]
    result = {}
    for key in keys:
        result[key] = stats([s[key] for s in profiler_samples])

    elapsed = time.monotonic() - profiler_start_time if profiler_running else (
        PROFILER_DURATION_S if profiler_samples else 0)

    return {
        "status": "running" if profiler_running else "complete",
        "samples": len(profiler_samples),
        "elapsed_s": round(elapsed, 1),
        "duration_s": PROFILER_DURATION_S,
        "stages": result,
        "utc": utc_now_ms(),
    }


@app.post("/api/profiler/measure_latency")
async def measure_latency():
    """Capture frame, OCR the displayed UTC time, compare with local UTC.

    Requires a UTC millisecond clock website to be open on the target machine.
    Uses Claude Agent SDK to read the time from the captured HDMI frame.
    Returns timing breakdown: capture, inference, and the UTC delta.
    """
    import tempfile

    # Step 1: Capture frame + record local UTC
    t_capture_start = time.monotonic()
    capture_utc = utc_now_ms()
    with state.lock:
        jpeg = state.frame_overlay
        raw_frame = state.frame_raw
    t_capture_ms = (time.monotonic() - t_capture_start) * 1000

    if raw_frame is None:
        return {"error": "No frame available"}

    # Step 2: Save frame as temp JPEG for the agent
    t_infer_start = time.monotonic()
    try:
        import os
        os.environ.pop("CLAUDECODE", None)

        from claude_agent_sdk import (
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, prefix="utc_ocr_") as f:
            tmppath = f.name
            bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            f.write(buf.tobytes())

        options = ClaudeAgentOptions(
            system_prompt=(
                "You read UTC timestamps from screenshots of clock websites. "
                "You will be given a screenshot path. Read it using the Read tool, "
                "find the displayed UTC time (look for HH:MM:SS.mmm or similar formats), "
                "and output ONLY a JSON object: "
                '{\"utc_time\": \"HH:MM:SS.mmm\", \"confidence\": \"high/low\", '
                '\"notes\": \"brief description of what you see\"}'
            ),
            allowed_tools=["Read"],
            permission_mode="bypassPermissions",
            model="haiku",
            max_turns=3,
        )

        prompt = f"Read this screenshot and find the UTC time displayed: {tmppath}"

        ocr_text = ""
        async for msg in query(prompt=prompt, options=options):
            if isinstance(msg, ResultMessage):
                for block in msg.content:
                    if isinstance(block, TextBlock):
                        ocr_text += block.text

        # Clean up temp file
        try:
            os.unlink(tmppath)
        except OSError:
            pass

        t_infer_ms = (time.monotonic() - t_infer_start) * 1000

        # Step 3: Parse the OCR result
        import re
        json_match = re.search(r'\{[^}]+\}', ocr_text)
        ocr_result = {}
        displayed_utc = None
        if json_match:
            try:
                ocr_result = json.loads(json_match.group())
                displayed_utc = ocr_result.get("utc_time")
            except json.JSONDecodeError:
                pass

        # Step 4: Calculate delta if we got a time
        delta_ms = None
        if displayed_utc:
            try:
                # Parse capture_utc (YYYY-MM-DDTHH:MM:SS.mmmZ)
                cap_time_str = capture_utc[11:23]  # HH:MM:SS.mmm
                cap_parts = cap_time_str.split(':')
                cap_ms = (int(cap_parts[0]) * 3600 + int(cap_parts[1]) * 60 +
                          float(cap_parts[2])) * 1000

                # Parse displayed time (HH:MM:SS.mmm or HH:MM:SS)
                disp_parts = displayed_utc.split(':')
                disp_sec = disp_parts[2] if len(disp_parts) > 2 else "0"
                disp_ms = (int(disp_parts[0]) * 3600 + int(disp_parts[1]) * 60 +
                           float(disp_sec)) * 1000

                delta_ms = round(cap_ms - disp_ms)
            except (ValueError, IndexError):
                pass

        return {
            "capture_utc": capture_utc,
            "capture_ms": round(t_capture_ms, 2),
            "inference_ms": round(t_infer_ms, 2),
            "displayed_utc": displayed_utc,
            "delta_ms": delta_ms,
            "ocr_result": ocr_result,
            "ocr_raw": ocr_text[:500],
        }

    except ImportError:
        return {"error": "Claude Agent SDK not installed"}
    except Exception as e:
        t_infer_ms = (time.monotonic() - t_infer_start) * 1000
        return {"error": str(e), "inference_ms": round(t_infer_ms, 2)}


# ── Ground truth snapshot log ──────────────────────────────────────

_GROUND_TRUTH_DIR = Path(__file__).parent.parent.parent / "data" / "ground_truth"
_GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)
_GROUND_TRUTH_LOG = _GROUND_TRUTH_DIR / "log.jsonl"


@app.post("/api/profiler/snapshot")
async def profiler_snapshot():
    """Capture comprehensive ground truth: all algorithms + Vision + frame.

    Runs Claude Vision on the SAME frame the system sees, records every
    subsystem's output with timestamps and confidence/uncertainty scores.
    Saves frame JPEG + JSONL record for training data.

    This is the foundation for reliable Phase 2+ training loops.
    """
    now_mono = time.monotonic()
    now_utc = utc_now_ms()

    # 1. Grab the exact frame the system is using
    with state.lock:
        frame = state.frame_raw.copy() if state.frame_raw is not None else None
    if frame is None:
        return {"error": "No frame available"}

    # 2. Record all algorithm states AT THIS MOMENT
    snapshot = {
        "utc": now_utc,
        "monotonic": now_mono,
        "cursor": {
            "x": state.cursor_x, "y": state.cursor_y,
            "method": state.cursor_method,
            "age_s": round(state.cursor_age_s, 2),
            "validated": state.cursor_validated,
        },
        "cnn": {
            "confidence": round(state.cnn_confidence, 4),
            "model": state.cnn_model_version,
            "inference_ms": state.cnn_inference_ms,
            "uncertainty": round(1.0 - abs(state.cnn_confidence - 0.5) * 2, 3),
        },
        "silhouette": {
            "method": state.silhouette_method,
            "hz": state.silhouette_hz,
            "confidence": round(state.silhouette_confidence, 4),
            "misses": state.silhouette_misses,
            "roi_size": state.silhouette_roi_size,
        },
        "noise_grid": noise_grid_features(),
        "daemon": {
            "fps": state.fps,
            "blob_count": state.blob_count,
        },
        "pipeline": {
            "frame_total_ms": round(state.prof_frame_total_ms, 2),
            "silhouette_ms": round(state.silhouette_latency_ms, 2),
        },
    }

    # 3. Run Vision on the SAME frame (this takes 5-15s)
    vision_result = None
    if claude_detector:
        pos = tracker.position if tracker else None
        est_x = pos.x if pos else None
        est_y = pos.y if pos else None
        t0 = time.monotonic()
        try:
            frame_rgb = frame  # frame_raw is already RGB
            result = await claude_detector.detect(frame_rgb, est_x, est_y)
            detect_ms = (time.monotonic() - t0) * 1000
            vision_result = {
                "cursor_found": result.cursor_found,
                "x": result.cursor_x,
                "y": result.cursor_y,
                "cursor_type": result.cursor_type,
                "confidence": result.confidence,
                "latency_ms": round(detect_ms, 1),
                "reasoning": result.reasoning,
            }
            # Sanity check: does coord match claimed quadrant?
            if result.cursor_found and result.cursor_x is not None:
                cx, cy = result.cursor_x, result.cursor_y
                coord_q = (1 if cx < 960 else 2) if cy < 540 else (3 if cx < 960 else 4)
                claimed_q = None
                for q in result.quadrants:
                    if q.is_cursor_present:
                        claimed_q = q.quadrant
                        break
                vision_result["coord_quadrant"] = coord_q
                vision_result["claimed_quadrant"] = claimed_q
                vision_result["quadrant_match"] = coord_q == claimed_q if claimed_q else None
        except Exception as e:
            vision_result = {"error": str(e)}

    snapshot["vision"] = vision_result

    # 4. Save frame and log record
    ts_id = int(now_mono * 1000)
    frame_path = _GROUND_TRUTH_DIR / f"frame_{ts_id}.jpg"
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(frame_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    snapshot["frame"] = frame_path.name

    with open(_GROUND_TRUTH_LOG, "a") as f:
        f.write(json.dumps(snapshot) + "\n")

    return snapshot


# ── Manual tagging endpoint ─────────────────────────────────────────

_TAG_DIR = Path(__file__).parent.parent.parent / "data" / "tagged"
_TAG_DIR.mkdir(parents=True, exist_ok=True)
_TAG_LOG = _TAG_DIR / "tags.jsonl"


@app.post("/api/tag")
async def tag_frame(request: Request):
    """Save a manually tagged frame with multi-dimensional labels.

    The user sees all algorithm outputs on the dashboard and clicks
    tag buttons. Each tag saves the current frame + all state + labels.

    Expected JSON body:
    {
        "cursor_correct": "yes" | "no" | "unsure",
        "cursor_type": "arrow" | "hand" | "caret" | "resize" | "other",
        "movement": "stationary" | "slow" | "medium" | "fast",
        "track_quality": "accurate" | "drifting" | "wrong_target" | "lost",
        "context": "desktop" | "browser" | "teams" | "app",
        "notes": "optional freeform text"
    }
    """
    data = await request.json()
    now_mono = time.monotonic()
    now_utc = utc_now_ms()

    # Grab frame
    with state.lock:
        frame = state.frame_raw.copy() if state.frame_raw is not None else None

    if frame is None:
        return {"error": "No frame available"}

    # Save frame
    ts_id = int(now_mono * 1000)
    frame_path = _TAG_DIR / f"frame_{ts_id}.jpg"
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(frame_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # Record: tags + all algorithm states
    record = {
        "id": ts_id,
        "utc": now_utc,
        "tags": {
            "cursor_correct": data.get("cursor_correct", "unsure"),
            "cursor_type": data.get("cursor_type", "unknown"),
            "movement": data.get("movement", "unknown"),
            "track_quality": data.get("track_quality", "unknown"),
            "context": data.get("context", "unknown"),
            "notes": data.get("notes", ""),
        },
        "state": {
            "cursor_x": state.cursor_x,
            "cursor_y": state.cursor_y,
            "cursor_method": state.cursor_method,
            "cursor_validated": state.cursor_validated,
            "cnn_confidence": round(state.cnn_confidence, 4),
            "cnn_model": state.cnn_model_version,
            "silhouette_confidence": round(state.silhouette_confidence, 4),
            "silhouette_method": state.silhouette_method,
            "silhouette_roi": state.silhouette_roi_size,
            "silhouette_misses": state.silhouette_misses,
            "blob_count": state.blob_count,
            "vision_x": state.vision_x,
            "vision_y": state.vision_y,
            "vision_confidence": state.vision_confidence,
            "vision_age_s": round(now_mono - state.vision_timestamp, 1) if state.vision_timestamp > 0 else -1,
        },
        "noise_grid": noise_grid_features(),
        "frame": frame_path.name,
    }

    with open(_TAG_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    return {"status": "tagged", "id": ts_id, "frame": frame_path.name}


@app.get("/api/tags")
async def get_tags():
    """Return all tagged samples for review."""
    if not _TAG_LOG.exists():
        return {"tags": [], "count": 0}
    with open(_TAG_LOG) as f:
        tags = [json.loads(line) for line in f if line.strip()]
    return {"tags": tags[-50:], "count": len(tags)}


# ── Velocity / Acceleration Testing ────────────────────────────────

def _capture_snapshot(profile_id: str, step: int, direction: str,
                      commanded_dx: int, commanded_dy: int,
                      commanded_total_x: float, commanded_total_y: float,
                      run_dir: Path) -> dict:
    """Capture full sensor state for one velocity test step.

    Returns a dict suitable for JSONL logging.
    """
    now_mono = time.monotonic()

    # Save frame for CNN training
    frame_name = f"frame_{profile_id}_{direction}_{step:03d}.jpg"
    frame_path = run_dir / "frames" / frame_name
    if daemon_client and daemon_client.is_running():
        jpeg = daemon_client.read_jpeg()
        if jpeg:
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            frame_path.write_bytes(jpeg)

    tracked_x = state.cursor_x
    tracked_y = state.cursor_y
    error_px = ((tracked_x - commanded_total_x) ** 2 +
                (tracked_y - commanded_total_y) ** 2) ** 0.5

    return {
        "utc": utc_now_ms(),
        "monotonic": now_mono,
        "profile": profile_id,
        "step": step,
        "direction": direction,
        "commanded": {
            "dx": commanded_dx,
            "dy": commanded_dy,
            "total_x": round(commanded_total_x, 1),
            "total_y": round(commanded_total_y, 1),
        },
        "tracked": {
            "x": tracked_x,
            "y": tracked_y,
            "method": state.cursor_method,
            "age_s": round(state.cursor_age_s, 1),
            "validated": state.cursor_validated,
        },
        "silhouette": {
            "confidence": round(state.silhouette_confidence, 3),
            "method": state.silhouette_method,
            "roi_size": state.silhouette_roi_size,
            "misses": state.silhouette_misses,
            "hz": state.silhouette_hz,
        },
        "cnn": {
            "confidence": round(state.cnn_confidence, 3),
            "validated": state.cursor_validated,
            "model": state.cnn_model_version,
        },
        "daemon": {
            "fps": round(state.fps, 1),
            "blob_count": state.blob_count,
        },
        "noise_grid": noise_grid_features(),
        "error_px": round(error_px, 1),
        "label": _auto_label(error_px),
        "frame": frame_name,
    }


def _run_velocity_test(profiles: list[str]):
    """Background thread: run velocity test profiles sequentially.

    For each profile:
      1. Establish starting position (current tracker + ESP32 estimated)
      2. Move RIGHT for N steps, recording at each step
      3. Return to start
      4. Move DOWN for N steps, recording at each step
      5. Return to start
    """
    global _vtest_running, _vtest_results, _vtest_progress, _vtest_summary

    _vtest_running = True
    _vtest_results = []

    # Create run directory
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = _VTEST_DIR / f"run_{run_ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "frames").mkdir(exist_ok=True)

    log_path = run_dir / "results.jsonl"

    try:
        for pid in profiles:
            if not _vtest_running:
                break  # Stopped early
            if pid not in VELOCITY_PROFILES:
                continue

            prof = VELOCITY_PROFILES[pid]
            steps = prof["steps"]
            delay_s = prof["delay_ms"] / 1000.0
            is_accel = "accel" in prof
            is_alternate = prof.get("alternate", False)

            # Run test in two directions: RIGHT then DOWN
            for direction, axis in [("right", 0), ("down", 1)]:
                if not _vtest_running:
                    break

                total_steps = steps
                _vtest_progress = {
                    "profile": pid,
                    "direction": direction,
                    "step": 0,
                    "total": total_steps,
                    "phase": "testing",
                }

                # Record starting position
                start_esp_x = mouse.estimated_x if mouse else 0
                start_esp_y = mouse.estimated_y if mouse else 0
                cum_dx, cum_dy = 0.0, 0.0

                for step_i in range(total_steps):
                    if not _vtest_running:
                        break

                    # Calculate step displacement.
                    # Speed magnitude comes from profile; axis selects X or Y.
                    if is_accel:
                        t = step_i / max(1, total_steps - 1)
                        speed = round(prof["start_speed"] + t * (prof["end_speed"] - prof["start_speed"]))
                    elif is_alternate:
                        sign = 1 if step_i % 2 == 0 else -1
                        speed = sign * (prof["dx"] or prof.get("dy", 2))
                    else:
                        # Use dx for RIGHT, dy for DOWN (fallback to dx if dy==0)
                        speed = prof["dx"] if axis == 0 else (prof["dy"] if prof["dy"] else prof["dx"])

                    dx = speed if axis == 0 else 0
                    dy = speed if axis == 1 else 0

                    # Move mouse
                    if mouse and (dx != 0 or dy != 0):
                        actual_dx, actual_dy = mouse.move(dx, dy)
                        cum_dx += actual_dx
                        cum_dy += actual_dy
                    else:
                        # V0 stationary — no movement, just observe
                        pass

                    # Wait for tracker to process the movement
                    time.sleep(delay_s)

                    # Capture state
                    commanded_x = start_esp_x + cum_dx
                    commanded_y = start_esp_y + cum_dy
                    snapshot = _capture_snapshot(
                        pid, step_i, direction, dx, dy,
                        commanded_x, commanded_y, run_dir,
                    )
                    _vtest_results.append(snapshot)

                    # Write to JSONL
                    with open(log_path, "a") as f:
                        f.write(json.dumps(snapshot) + "\n")

                    _vtest_progress["step"] = step_i + 1

                # Return to start position after direction run
                if mouse and (cum_dx != 0 or cum_dy != 0):
                    mouse.move(int(-cum_dx), int(-cum_dy))
                    time.sleep(0.5)  # Settle after return

        # Compute summary stats per profile
        _vtest_summary = _compute_vtest_summary(_vtest_results)

        # Save summary
        with open(run_dir / "summary.json", "w") as f:
            json.dump(_vtest_summary, f, indent=2)

    except Exception as e:
        import traceback
        traceback.print_exc()
        _vtest_progress["phase"] = f"error: {e}"
    finally:
        _vtest_running = False
        _vtest_progress["phase"] = "done"


def _compute_vtest_summary(results: list[dict]) -> dict:
    """Aggregate velocity test results per profile + direction."""
    from collections import defaultdict
    import statistics

    groups = defaultdict(list)
    for r in results:
        key = f"{r['profile']}_{r['direction']}"
        groups[key].append(r)

    summary = {}
    for key, records in groups.items():
        errors = [r["error_px"] for r in records]
        labels = [r["label"] for r in records]
        n = len(records)
        label_counts = {
            "accurate": labels.count("accurate"),
            "drift": labels.count("drift"),
            "wrong": labels.count("wrong"),
            "lost": labels.count("lost"),
        }
        summary[key] = {
            "count": n,
            "error_mean": round(statistics.mean(errors), 1) if errors else 0,
            "error_max": round(max(errors), 1) if errors else 0,
            "error_min": round(min(errors), 1) if errors else 0,
            "error_p95": round(sorted(errors)[int(n * 0.95)] if n >= 2 else errors[0], 1) if errors else 0,
            "labels": label_counts,
            "accuracy_pct": round(100 * label_counts["accurate"] / n, 1) if n > 0 else 0,
            "methods": list(set(r["tracked"]["method"] for r in records)),
        }

    return summary


@app.post("/api/velocity_test")
async def start_velocity_test(request: Request):
    """Start velocity/acceleration test in background thread.

    POST body (optional): {"profiles": ["V0","V1","V2","V3","V4","A1","A2","A3"]}
    Omit profiles to run all.
    """
    global _vtest_running
    if _vtest_running:
        return {"error": "Test already running", "progress": _vtest_progress}
    if not mouse:
        return {"error": "Mouse not initialized"}

    body = {}
    if request.headers.get("content-type") == "application/json":
        body = await request.json()
    profiles = body.get("profiles", list(VELOCITY_PROFILES.keys()))

    threading.Thread(
        target=_run_velocity_test, args=(profiles,), daemon=True
    ).start()

    return {"status": "started", "profiles": profiles, "utc": utc_now_ms()}


@app.post("/api/velocity_test/stop")
async def stop_velocity_test():
    """Stop a running velocity test early."""
    global _vtest_running
    if not _vtest_running:
        return {"error": "No test running"}
    _vtest_running = False
    return {"status": "stopping", "results_so_far": len(_vtest_results)}


@app.get("/api/velocity_test/status")
async def velocity_test_status():
    """Get current velocity test progress."""
    return {
        "running": _vtest_running,
        "progress": _vtest_progress,
        "results_count": len(_vtest_results),
    }


@app.get("/api/velocity_test/results")
async def velocity_test_results():
    """Get velocity test results and summary."""
    return {
        "running": _vtest_running,
        "summary": _vtest_summary,
        "total_steps": len(_vtest_results),
        "last_10": _vtest_results[-10:] if _vtest_results else [],
    }


# ── Loss Event Sampling API ─────────────────────────────────────

@app.get("/api/loss_events")
async def list_loss_events():
    """List all captured loss events (newest first)."""
    # Rebuild index from disk if in-memory list is empty
    if not _loss_events_log and _LOSS_EVENT_DIR.exists():
        for d in sorted(_LOSS_EVENT_DIR.iterdir(), reverse=True):
            mf = d / "manifest.json"
            if mf.exists():
                try:
                    _loss_events_log.append(json.loads(mf.read_text()))
                except Exception:
                    pass

    return {
        "events": list(reversed(_loss_events_log)),
        "total": len(_loss_events_log),
        "ring_buffer_size": len(_ring_buffer),
        "active": _loss_event_active,
    }


@app.get("/api/loss_events/{event_id}/timeline")
async def get_loss_timeline(event_id: str):
    """Get full timeline for a loss event."""
    tl_path = _LOSS_EVENT_DIR / event_id / "timeline.jsonl"
    if not tl_path.exists():
        return {"error": "Event not found"}
    entries = []
    for line in tl_path.read_text().splitlines():
        if line.strip():
            entries.append(json.loads(line))
    manifest = {}
    mf = _LOSS_EVENT_DIR / event_id / "manifest.json"
    if mf.exists():
        manifest = json.loads(mf.read_text())
    return {"manifest": manifest, "timeline": entries}


@app.get("/api/loss_events/{event_id}/frame/{frame_name:path}")
async def get_loss_frame(event_id: str, frame_name: str):
    """Serve a frame or patch image from a loss event."""
    file_path = _LOSS_EVENT_DIR / event_id / frame_name
    if not file_path.exists():
        return Response(status_code=404, content="Not found")
    suffix = file_path.suffix.lower()
    media = "image/jpeg" if suffix == ".jpg" else "image/png"
    return Response(content=file_path.read_bytes(), media_type=media)


@app.get("/loss_events", response_class=HTMLResponse)
async def loss_events_page():
    """Timeline viewer for cursor loss events."""
    return LOSS_EVENTS_HTML


# ── Loss Events HTML ────────────────────────────────────────────────

LOSS_EVENTS_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cursor Loss Events</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'JetBrains Mono', monospace; background: #0a0a0f; color: #c8c8d0; }
.header { background: #101018; padding: 0.6rem 1.5rem; border-bottom: 2px solid #e74c3c;
    display: flex; justify-content: space-between; align-items: center; }
.header h1 { color: #e74c3c; font-size: 1.1rem; letter-spacing: 1px; }
.nav a { color: #888; text-decoration: none; font-size: 0.75rem; margin-left: 1rem; }
.nav a:hover { color: #fff; }
.events-list { padding: 1rem; }
.event-card { background: #14141f; border: 1px solid #222; border-radius: 6px; padding: 1rem; margin-bottom: 0.8rem; cursor: pointer; }
.event-card:hover { border-color: #e74c3c; }
.event-card .meta { display: flex; gap: 1.5rem; font-size: 0.75rem; color: #888; margin-top: 0.4rem; }
.event-card .trigger { color: #e74c3c; font-weight: bold; }
.event-card .recovered { color: #2ecc71; }
.event-card .not-recovered { color: #e74c3c; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.65rem; }
.badge-had { background: #1a3a1a; color: #2ecc71; }
.badge-lost { background: #3a1a1a; color: #e74c3c; }
.badge-recovered { background: #1a2a3a; color: #3498db; }

/* Timeline view */
.timeline-view { display: none; padding: 1rem; }
.timeline-view.active { display: block; }
.back-btn { background: #222; color: #aaa; border: 1px solid #333; padding: 4px 12px; border-radius: 4px; cursor: pointer; margin-bottom: 1rem; font-size: 0.75rem; }
.back-btn:hover { color: #fff; border-color: #e74c3c; }
.tl-header { margin-bottom: 1rem; }
.tl-header h2 { color: #e74c3c; font-size: 1rem; }
.tl-header .tl-meta { font-size: 0.75rem; color: #888; margin-top: 0.3rem; }

.tl-strip { display: flex; overflow-x: auto; gap: 4px; padding: 0.5rem 0; }
.tl-frame { flex-shrink: 0; width: 120px; text-align: center; cursor: pointer; border: 2px solid transparent; border-radius: 4px; padding: 2px; }
.tl-frame:hover { border-color: #555; }
.tl-frame.selected { border-color: #e74c3c; }
.tl-frame.phase-had_it { border-bottom: 3px solid #2ecc71; }
.tl-frame.phase-lost { border-bottom: 3px solid #e74c3c; }
.tl-frame.phase-recovering { border-bottom: 3px solid #f39c12; }
.tl-frame.phase-recovered { border-bottom: 3px solid #3498db; }
.tl-frame img { width: 116px; height: 65px; object-fit: cover; border-radius: 2px; background: #000; }
.tl-frame .tl-label { font-size: 0.55rem; color: #888; margin-top: 2px; }
.tl-frame .tl-conf { font-size: 0.55rem; }

.detail-panel { display: flex; gap: 1rem; margin-top: 1rem; }
.detail-frame { flex: 1; }
.detail-frame img { width: 100%; image-rendering: auto; border-radius: 4px; border: 1px solid #333; }
.detail-patch { width: 200px; }
.detail-patch img { width: 192px; height: 192px; image-rendering: pixelated; border: 1px solid #333; border-radius: 4px; }
.detail-info { flex: 1; font-size: 0.7rem; }
.detail-info table { width: 100%; border-collapse: collapse; }
.detail-info td { padding: 3px 6px; border-bottom: 1px solid #1a1a2a; }
.detail-info td:first-child { color: #888; width: 120px; }
.conf-bar { display: inline-block; height: 8px; border-radius: 2px; }
.conf-high { background: #2ecc71; }
.conf-mid { background: #f39c12; }
.conf-low { background: #e74c3c; }
.empty-state { text-align: center; padding: 4rem 2rem; color: #555; }
.empty-state p { margin-top: 0.5rem; font-size: 0.8rem; }
.ring-status { font-size: 0.7rem; color: #555; margin-left: 1rem; }
</style>
</head>
<body>
<div class="header">
  <div style="display:flex;align-items:center;gap:1rem;">
    <h1>CURSOR LOSS EVENTS</h1>
    <span class="ring-status" id="ring-status"></span>
  </div>
  <div class="nav">
    <a href="/">Dashboard</a>
    <a href="/validation">Validation</a>
    <a href="/profiler">Profiler</a>
    <a href="/loss_events">Loss Events</a>
  </div>
</div>

<div class="events-list" id="events-list"></div>

<div class="timeline-view" id="timeline-view">
  <button class="back-btn" onclick="showList()">&larr; Back to events</button>
  <div class="tl-header" id="tl-header"></div>
  <div class="tl-strip" id="tl-strip"></div>
  <div class="detail-panel" id="detail-panel"></div>
</div>

<script>
var events = [];
var currentTimeline = [];
var currentEventId = null;
var selectedIdx = 0;

async function loadEvents() {
  var res = await fetch('/api/loss_events');
  var data = await res.json();
  events = data.events;
  document.getElementById('ring-status').textContent =
    'Ring buffer: ' + data.ring_buffer_size + ' frames | ' +
    (data.active ? 'CAPTURING...' : 'watching');
  renderEventsList();
}

function renderEventsList() {
  var el = document.getElementById('events-list');
  if (events.length === 0) {
    el.innerHTML = '<div class="empty-state"><h2>No loss events captured yet</h2>' +
      '<p>Events are recorded automatically when cursor confidence drops.<br>' +
      'The ring buffer holds the last 60 seconds — when a loss happens, it saves the timeline.</p></div>';
    return;
  }
  var html = '';
  for (var i = 0; i < events.length; i++) {
    var ev = events[i];
    var recClass = ev.recovered ? 'recovered' : 'not-recovered';
    var recText = ev.recovered ? 'Recovered' : 'Not recovered';
    html += '<div class="event-card" onclick="openEvent(\'' + ev.event_id + '\')">' +
      '<span class="trigger">' + ev.trigger + '</span>' +
      '<div class="meta">' +
      '<span>' + ev.trigger_utc + '</span>' +
      '<span>' + ev.total_frames + ' frames</span>' +
      '<span>' + ev.duration_s + 's duration</span>' +
      '<span class="' + recClass + '">' + recText + '</span>' +
      (ev.phases ? ' <span class="badge badge-had">' + ev.phases.had_it + ' had</span>' +
        '<span class="badge badge-lost">' + ev.phases.lost + ' lost</span>' +
        '<span class="badge badge-recovered">' + (ev.phases.recovering + ev.phases.recovered) + ' recovery</span>' : '') +
      '</div></div>';
  }
  el.innerHTML = html;
}

async function openEvent(eventId) {
  currentEventId = eventId;
  var res = await fetch('/api/loss_events/' + eventId + '/timeline');
  var data = await res.json();
  currentTimeline = data.timeline;

  document.getElementById('events-list').style.display = 'none';
  document.getElementById('timeline-view').classList.add('active');

  var m = data.manifest;
  document.getElementById('tl-header').innerHTML =
    '<h2>' + m.trigger + '</h2>' +
    '<div class="tl-meta">' + m.trigger_utc + ' | ' + m.total_frames + ' frames | ' +
    m.duration_s + 's | ' + (m.recovered ? 'Recovered' : 'Not recovered') + '</div>';

  renderStrip();
  selectFrame(0);
}

function renderStrip() {
  var html = '';
  for (var i = 0; i < currentTimeline.length; i++) {
    var t = currentTimeline[i];
    var confClass = t.live_cnn.confidence > 0.7 ? 'conf-high' : (t.live_cnn.confidence > 0.4 ? 'conf-mid' : 'conf-low');
    var src = t.frame ? '/api/loss_events/' + currentEventId + '/frame/' + t.frame : '';
    html += '<div class="tl-frame phase-' + t.phase + (i === selectedIdx ? ' selected' : '') + '" onclick="selectFrame(' + i + ')">' +
      (src ? '<img src="' + src + '" loading="lazy">' : '<div style="width:116px;height:65px;background:#111"></div>') +
      '<div class="tl-label">' + t.t_offset_s + 's · ' + t.phase + '</div>' +
      '<div class="tl-conf"><span class="conf-bar ' + confClass + '" style="width:' + Math.round(t.live_cnn.confidence * 60) + 'px"></span> ' +
      (t.live_cnn.confidence * 100).toFixed(0) + '%</div>' +
      '</div>';
  }
  document.getElementById('tl-strip').innerHTML = html;
}

function selectFrame(idx) {
  selectedIdx = idx;
  var t = currentTimeline[idx];

  // Update strip selection
  var frames = document.querySelectorAll('.tl-frame');
  frames.forEach(function(f, i) { f.classList.toggle('selected', i === idx); });
  if (frames[idx]) frames[idx].scrollIntoView({behavior: 'smooth', block: 'nearest', inline: 'center'});

  // Build detail panel
  var frameSrc = t.frame ? '/api/loss_events/' + currentEventId + '/frame/' + t.frame : '';
  var patchSrc = t.patch ? '/api/loss_events/' + currentEventId + '/frame/' + t.patch : '';

  var liveConf = t.live_cnn.confidence;
  var retroConf = t.retrospective_cnn.confidence;
  function confColor(c) { return c > 0.7 ? '#2ecc71' : (c > 0.4 ? '#f39c12' : '#e74c3c'); }

  var html = '<div class="detail-frame">' +
    (frameSrc ? '<img src="' + frameSrc + '">' : '<div style="height:300px;background:#111;border-radius:4px;"></div>') +
    '</div>' +
    '<div class="detail-patch">' +
    '<div style="font-size:0.65rem;color:#888;margin-bottom:4px">64x64 patch at (' + t.cursor.x + ',' + t.cursor.y + ')</div>' +
    (patchSrc ? '<img src="' + patchSrc + '">' : '<div style="width:192px;height:192px;background:#111;border-radius:4px;"></div>') +
    '</div>' +
    '<div class="detail-info"><table>' +
    '<tr><td>Time offset</td><td>' + t.t_offset_s + 's</td></tr>' +
    '<tr><td>Phase</td><td style="color:' + (t.phase === 'had_it' ? '#2ecc71' : t.phase === 'lost' ? '#e74c3c' : '#f39c12') + '">' + t.phase + '</td></tr>' +
    '<tr><td>Position</td><td>(' + t.cursor.x + ', ' + t.cursor.y + ')</td></tr>' +
    '<tr><td>Method</td><td>' + t.cursor.method + '</td></tr>' +
    '<tr><td>Validated</td><td>' + (t.cursor.validated ? '<span style="color:#2ecc71">YES</span>' : '<span style="color:#e74c3c">NO</span>') + '</td></tr>' +
    '<tr><td>Live CNN</td><td><span style="color:' + confColor(liveConf) + '">' + (liveConf * 100).toFixed(1) + '%</span> (' + t.live_cnn.model + ')</td></tr>' +
    '<tr><td>Retro CNN</td><td><span style="color:' + confColor(retroConf) + '">' + (retroConf * 100).toFixed(1) + '%</span> (' + t.retrospective_cnn.inference_ms + 'ms)</td></tr>' +
    '<tr><td>Silhouette</td><td>' + (t.silhouette.confidence * 100).toFixed(0) + '% · ' + t.silhouette.method + ' · ROI ' + t.silhouette.roi_size + ' · misses ' + t.silhouette.misses + '</td></tr>' +
    '<tr><td>Blobs</td><td>' + t.blob_count + '</td></tr>' +
    '<tr><td>Velocity</td><td>' + (t.velocity_px_s || 0) + ' px/s</td></tr>' +
    '<tr><td>Acceleration</td><td>' + (t.acceleration_px_s2 || 0) + ' px/s²</td></tr>' +
    (t.edge_cases && t.edge_cases.length > 0 ?
      '<tr><td>Edge cases</td><td>' + t.edge_cases.map(function(c) {
        return '<span style="background:#3a1a2a;color:#e74c3c;padding:1px 6px;border-radius:8px;font-size:0.6rem;margin-right:4px">' + c + '</span>';
      }).join('') + '</td></tr>' : '') +
    '</table>';

  // Blob candidates with CNN scores
  if (t.blob_candidates && t.blob_candidates.length > 0) {
    html += '<div style="margin-top:8px;font-size:0.65rem;color:#888">Blob candidates (CNN scored):</div>';
    html += '<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:4px">';
    for (var bi = 0; bi < t.blob_candidates.length; bi++) {
      var bc = t.blob_candidates[bi];
      var bcColor = bc.cnn_score > 0.7 ? '#2ecc71' : (bc.cnn_score > 0.4 ? '#f39c12' : '#e74c3c');
      var bcSrc = '/api/loss_events/' + currentEventId + '/frame/' + bc.patch;
      html += '<div style="text-align:center">' +
        '<img src="' + bcSrc + '" style="width:64px;height:64px;image-rendering:pixelated;border:1px solid #333;border-radius:2px">' +
        '<div style="font-size:0.55rem;color:' + bcColor + '">' + (bc.cnn_score * 100).toFixed(0) + '% (' + bc.centroid[0] + ',' + bc.centroid[1] + ')</div>' +
        '</div>';
    }
    html += '</div>';
  }

  html += '</div>';

  document.getElementById('detail-panel').innerHTML = html;
}

function showList() {
  document.getElementById('events-list').style.display = '';
  document.getElementById('timeline-view').classList.remove('active');
}

// Keyboard nav: left/right to scrub timeline
document.addEventListener('keydown', function(e) {
  if (!document.getElementById('timeline-view').classList.contains('active')) return;
  if (e.key === 'ArrowLeft' || e.key === 'j') { e.preventDefault(); if (selectedIdx > 0) selectFrame(selectedIdx - 1); }
  if (e.key === 'ArrowRight' || e.key === 'k') { e.preventDefault(); if (selectedIdx < currentTimeline.length - 1) selectFrame(selectedIdx + 1); }
});

loadEvents();
setInterval(loadEvents, 10000);
</script>
</body>
</html>"""


# ── Dashboard HTML ──────────────────────────────────────────────────

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>KVM Motion Detector</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            background: #0a0a0f;
            color: #c8c8d0;
        }
        .header {
            background: #101018;
            padding: 0.6rem 1.5rem;
            border-bottom: 2px solid #ff4444;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { color: #ff4444; font-size: 1.1rem; letter-spacing: 1px; }
        .header .status { font-size: 0.8rem; display: flex; gap: 1.5rem; }
        .dot {
            display: inline-block; width: 8px; height: 8px;
            border-radius: 50%; margin-right: 4px; vertical-align: middle;
        }
        .dot.green { background: #2ecc71; box-shadow: 0 0 6px #2ecc71; }
        .dot.red { background: #e74c3c; box-shadow: 0 0 6px #e74c3c; }
        .dot.yellow { background: #f1c40f; box-shadow: 0 0 6px #f1c40f; }
        .dot.gray { background: #555; }
        .main {
            display: grid;
            grid-template-columns: 1fr 320px;
            height: calc(100vh - 50px);
        }
        .feed {
            background: #000; overflow: hidden;
            display: flex; justify-content: center; align-items: center;
        }
        .video-wrapper {
            position: relative; display: flex; justify-content: center;
            align-items: center; width: 100%; height: 100%;
        }
        #video-area { position: relative; max-width: 100%; max-height: 100%; overflow: hidden; }
        #video-area img { display: block; max-width: 100%; max-height: calc(100vh - 50px); }
        .cursor-dot {
            position: absolute; width: 24px; height: 24px;
            border-radius: 50%; pointer-events: none;
            background: radial-gradient(circle, #fff 20%, #00ff00 50%, transparent 70%);
            box-shadow: 0 0 12px #00ff00, 0 0 24px #00ff0088;
            transform: translate(-50%, -50%);
            animation: cursor-pulse 1.5s ease-in-out infinite;
            z-index: 10; display: none;
        }
        .vision-dot {
            position: absolute; width: 18px; height: 18px;
            border-radius: 50%; pointer-events: none;
            background: radial-gradient(circle, #fff 20%, #a855f7 50%, transparent 70%);
            box-shadow: 0 0 10px #a855f7, 0 0 20px #a855f788;
            transform: translate(-50%, -50%);
            z-index: 9; display: none;
        }
        @keyframes cursor-pulse {
            0%, 100% { opacity: 1; box-shadow: 0 0 12px #00ff00, 0 0 24px #00ff0088; }
            50% { opacity: 0.7; box-shadow: 0 0 20px #00ff00, 0 0 40px #00ff0066; }
        }
        .sidebar {
            background: #101018; border-left: 1px solid #222;
            overflow-y: auto;
        }
        .panel { border-bottom: 1px solid #1a1a25; padding: 0.8rem; }
        .tag-btn {
            background: #333; color: #ccc; border: 1px solid #555; padding: 2px 6px;
            cursor: pointer; border-radius: 3px; font-size: 0.65rem; margin: 1px;
            transition: all 0.15s;
        }
        .tag-btn:hover { border-color: #f39c12; color: #fff; }
        .tag-btn.selected { border-color: #f39c12; color: #fff; box-shadow: 0 0 4px #f39c12; }
        .tag-group { display: flex; flex-wrap: wrap; gap: 2px; }
        .panel-title {
            font-size: 0.7rem; text-transform: uppercase;
            letter-spacing: 2px; color: #ff4444; margin-bottom: 0.6rem;
        }
        .blob-table { width: 100%; font-size: 0.75rem; border-collapse: collapse; }
        .blob-table th {
            text-align: left; color: #666; font-weight: normal;
            padding: 2px 6px; border-bottom: 1px solid #1a1a25;
        }
        .blob-table td { padding: 2px 6px; font-variant-numeric: tabular-nums; }
        .hw-row {
            display: flex; justify-content: space-between;
            padding: 3px 0; font-size: 0.8rem;
        }
        .hw-label { color: #666; }
        .hw-value { font-weight: 600; }
        .hw-value.ok { color: #2ecc71; }
        .hw-value.warn { color: #f1c40f; }
        .hw-value.err { color: #e74c3c; }
        .cursor-display {
            background: #0a0a12; border: 1px solid #222;
            border-radius: 4px; padding: 0.6rem; text-align: center;
        }
        .cursor-coords {
            font-size: 1.4rem; font-weight: bold;
            color: #ff4444; margin-bottom: 0.3rem;
        }
        .cursor-meta { font-size: 0.7rem; color: #666; }
        .jitter-bar {
            height: 20px; background: #0a0a12; border: 1px solid #222;
            border-radius: 3px; overflow: hidden; position: relative; margin-top: 4px;
        }
        .jitter-pulse {
            position: absolute; height: 100%;
            background: linear-gradient(90deg, transparent, #2ecc71, transparent);
            width: 30px; display: none;
            animation: jitter-sweep 30s linear infinite;
        }
        .jitter-bar.active .jitter-pulse { display: block; }
        @keyframes jitter-sweep {
            0% { left: -30px; }
            0.5% { left: 50%; }
            1% { left: -30px; }
            100% { left: -30px; }
        }
        .tier-indicator { display: flex; gap: 4px; margin-top: 4px; }
        .tier {
            flex: 1; text-align: center; padding: 3px 0; border-radius: 3px;
            font-size: 0.65rem; background: #1a1a25; color: #444;
            transition: all 0.3s;
        }
        .tier.active { color: #fff; font-weight: bold; }
        .tier.t1.active { background: #2ecc71; color: #000; }
        .tier.t2.active { background: #f1c40f; color: #000; }
        .tier.t3.active { background: #e74c3c; }
    </style>
</head>
<body>
    <div class="header">
        <div style="display:flex; align-items:center; gap:1rem;">
            <h1>KVM</h1>
            <a href="/profiler" style="color:#888; text-decoration:none; font-size:0.75rem;">Profiler</a>
            <a href="/validation" style="color:#888; text-decoration:none; font-size:0.75rem;">Validation</a>
            <a href="/loss_events" style="color:#888; text-decoration:none; font-size:0.75rem;">Loss Events</a>
            <a href="/concepts" style="color:#888; text-decoration:none; font-size:0.75rem;">Concepts</a>
        </div>
        <div class="status">
            <span><span id="dot-mouse" class="dot gray"></span> Mouse</span>
            <span><span id="dot-heartbeat" class="dot gray"></span> Heartbeat</span>
            <span><span id="dot-jitter" class="dot gray"></span> Jitter</span>
            <span><span id="dot-keyboard" class="dot gray"></span> Pi KB</span>
            <span id="fps-display" style="color: #666;">0 FPS</span>
        </div>
    </div>
    <div class="main">
        <div class="feed" id="feed-container">
            <div class="video-wrapper">
                <div id="video-area">
                    <img id="live-frame" src="" alt="Loading...">
                    <div class="cursor-dot" id="cursor-dot"></div>
                    <div class="vision-dot" id="vision-dot"></div>
                    <div id="pause-overlay" style="display:none; position:absolute; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.3); z-index:10; justify-content:center; align-items:center; font-size:1.5rem; color:#f39c12; font-weight:bold; letter-spacing:4px; text-shadow:0 0 10px #000;">PAUSED — TAG &amp; RESUME</div>
                </div>
            </div>
        </div>
        <div class="sidebar">
            <div class="panel">
                <div class="panel-title">Cursor Position</div>
                <div class="cursor-display">
                    <div class="cursor-coords" id="cursor-coords">(?, ?)</div>
                    <div class="cursor-meta">
                        <span id="cursor-method">—</span> |
                        age: <span id="cursor-age">—</span>s
                    </div>
                </div>
                <div class="tier-indicator">
                    <div class="tier t1" id="tier-1" title="Motion tracking or silhouette following — fast, per-frame">T1 Micro</div>
                    <div class="tier t2" id="tier-2" title="CNN re-acquisition or macro shake — cursor was lost, searching nearby">T2 Macro</div>
                    <div class="tier t3" id="tier-3" title="Lissajous sweep — brute-force finder, sweeps entire screen">T3 Lissajous</div>
                </div>
                <div class="panel-help" id="cursor-help">
                    Green = fresh position, yellow = stale (&gt;5s), red = lost (&gt;15s).<br>
                    <b>Method</b>: how the position was last determined — motion_track (frame diff), sil_template (silhouette match), dual_probe (jitter re-acquisition), vision_anchor (Claude Vision).
                </div>
            </div>
            <div class="panel">
                <div class="panel-title">Motion Blobs (<span id="blob-count">0</span>)</div>
                <table class="blob-table">
                    <thead><tr><th>Centroid</th><th>Size</th><th>Angle</th></tr></thead>
                    <tbody id="blob-list"></tbody>
                </table>
            </div>
            <div class="panel">
                <div class="panel-title" style="color:#0ff">YOLO Full-Screen</div>
                <div class="hw-row">
                    <span class="hw-label">Status</span>
                    <span class="hw-value" id="yolo-status">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Position</span>
                    <span class="hw-value" id="yolo-position" style="color:#888">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Confidence</span>
                    <span class="hw-value" id="yolo-confidence">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Inference</span>
                    <span class="hw-value" id="yolo-inference" style="color:#888">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Detections</span>
                    <span class="hw-value" id="yolo-count" style="color:#888">—</span>
                </div>
            </div>
            <div class="panel">
                <div class="panel-title">Silhouette Tracking</div>
                <div class="hw-row">
                    <span class="hw-label">Status</span>
                    <span class="hw-value" id="sil-status">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Method</span>
                    <span class="hw-value" id="sil-method" style="color:#888">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Hz</span>
                    <span class="hw-value" id="sil-hz" style="color:#888">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">ROI</span>
                    <span class="hw-value" id="sil-roi" style="color:#888">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Confidence</span>
                    <span class="hw-value" id="sil-confidence">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Latency</span>
                    <span class="hw-value" id="sil-latency" style="color:#888">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Trusted</span>
                    <span class="hw-value" id="sil-trusted" style="color:#888">—</span>
                </div>
            </div>
            <div class="panel">
                <div class="panel-title">CNN Cursor Recognition</div>
                <div class="hw-row">
                    <span class="hw-label">Confidence</span>
                    <span class="hw-value" id="cnn-confidence">—</span>
                </div>
                <div style="height:8px; background:#1a1a25; border-radius:4px; overflow:hidden; margin:4px 0;">
                    <div id="cnn-bar" style="height:100%; width:0%; transition:width 0.3s; border-radius:4px;"></div>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Model</span>
                    <span class="hw-value" id="cnn-model" style="color:#888">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Validated</span>
                    <span class="hw-value" id="cnn-validated">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Inference</span>
                    <span class="hw-value" id="cnn-inference" style="color:#888">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Samples</span>
                    <span class="hw-value" id="cnn-samples" style="color:#888">0 pos / 0 neg</span>
                </div>
            </div>
            <div class="panel">
                <div class="panel-title" style="color:#9b59b6">Claude Vision (Ground Truth)</div>
                <div class="hw-row">
                    <span class="hw-label">Position</span>
                    <span class="hw-value" id="vision-pos" style="color:#888">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Confidence</span>
                    <span class="hw-value" id="vision-confidence">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Cursor Type</span>
                    <span class="hw-value" id="vision-type" style="color:#888">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Latency</span>
                    <span class="hw-value" id="vision-latency" style="color:#888">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Last Run</span>
                    <span class="hw-value" id="vision-age" style="color:#888">Never</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Tracker Delta</span>
                    <span class="hw-value" id="vision-delta" style="color:#888">—</span>
                </div>
                <div style="margin-top:6px;">
                    <button onclick="runVisionDetect()" id="vision-btn" style="background:#9b59b6; color:#fff; border:none; padding:4px 12px; border-radius:4px; cursor:pointer; font-size:0.7rem;">Run Detection</button>
                </div>
            </div>
            <div class="panel">
                <div class="panel-title">Anti-Sleep Jitter</div>
                <div class="hw-row">
                    <span class="hw-label">Status</span>
                    <span class="hw-value" id="jitter-status">—</span>
                </div>
                <div class="jitter-bar" id="jitter-bar">
                    <div class="jitter-pulse"></div>
                </div>
                <div style="font-size:0.65rem; color:#444; margin-top:3px;" id="jitter-desc">
                    ±3px every 30s — prevents Windows sleep
                </div>
            </div>
            <div class="panel">
                <div class="panel-title">Hardware</div>
                <div class="hw-row">
                    <span class="hw-label">ESP32 Mouse</span>
                    <span class="hw-value" id="hw-mouse">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">BLE Heartbeat</span>
                    <span class="hw-value" id="hw-heartbeat">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Pi Keyboard</span>
                    <span class="hw-value" id="hw-keyboard">—</span>
                </div>
                <div class="hw-row">
                    <span class="hw-label">Frame Count</span>
                    <span class="hw-value" id="hw-frames" style="color:#888">0</span>
                </div>
            </div>
            <div class="panel" style="border-color:#f39c12">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;">
                    <div class="panel-title" style="color:#f39c12; margin-bottom:0">Tag Frame</div>
                    <button id="pause-btn" onclick="togglePause()" style="background:#f39c12; color:#000; border:none; padding:3px 10px; cursor:pointer; border-radius:3px; font-weight:bold; font-size:0.65rem;">PAUSE (TAG)</button>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:4px; font-size:0.7rem;">
                    <div>
                        <div style="color:#888; margin-bottom:2px;">Cursor Correct?</div>
                        <div class="tag-group" id="tag-cursor-correct">
                            <button class="tag-btn" data-dim="cursor_correct" data-val="yes" style="background:#27ae60">Yes</button>
                            <button class="tag-btn" data-dim="cursor_correct" data-val="no" style="background:#e74c3c">No</button>
                            <button class="tag-btn" data-dim="cursor_correct" data-val="unsure" style="background:#7f8c8d">?</button>
                        </div>
                    </div>
                    <div>
                        <div style="color:#888; margin-bottom:2px;">Cursor Type</div>
                        <div class="tag-group" id="tag-cursor-type">
                            <button class="tag-btn" data-dim="cursor_type" data-val="arrow">Arrow</button>
                            <button class="tag-btn" data-dim="cursor_type" data-val="hand">Hand</button>
                            <button class="tag-btn" data-dim="cursor_type" data-val="caret">Caret</button>
                            <button class="tag-btn" data-dim="cursor_type" data-val="other">Other</button>
                        </div>
                    </div>
                    <div>
                        <div style="color:#888; margin-bottom:2px;">Movement</div>
                        <div class="tag-group" id="tag-movement">
                            <button class="tag-btn" data-dim="movement" data-val="stationary">Still</button>
                            <button class="tag-btn" data-dim="movement" data-val="slow">Slow</button>
                            <button class="tag-btn" data-dim="movement" data-val="medium">Med</button>
                            <button class="tag-btn" data-dim="movement" data-val="fast">Fast</button>
                        </div>
                    </div>
                    <div>
                        <div style="color:#888; margin-bottom:2px;">Track Quality</div>
                        <div class="tag-group" id="tag-track-quality">
                            <button class="tag-btn" data-dim="track_quality" data-val="accurate" style="background:#27ae60">Good</button>
                            <button class="tag-btn" data-dim="track_quality" data-val="drifting" style="background:#f39c12">Drift</button>
                            <button class="tag-btn" data-dim="track_quality" data-val="wrong_target" style="background:#e74c3c">Wrong</button>
                            <button class="tag-btn" data-dim="track_quality" data-val="lost" style="background:#c0392b">Lost</button>
                        </div>
                    </div>
                    <div style="grid-column:span 2">
                        <div style="color:#888; margin-bottom:2px;">Context</div>
                        <div class="tag-group" id="tag-context">
                            <button class="tag-btn" data-dim="context" data-val="desktop">Desktop</button>
                            <button class="tag-btn" data-dim="context" data-val="browser">Browser</button>
                            <button class="tag-btn" data-dim="context" data-val="teams">Teams</button>
                            <button class="tag-btn" data-dim="context" data-val="app">App</button>
                        </div>
                    </div>
                </div>
                <div style="margin-top:6px; display:flex; gap:4px; align-items:center;">
                    <input id="tag-notes" type="text" placeholder="Notes..." style="flex:1; background:#222; border:1px solid #444; color:#fff; padding:3px 6px; font-size:0.7rem; border-radius:3px;">
                    <button id="tag-submit" onclick="submitTag()" style="background:#f39c12; color:#000; border:none; padding:4px 12px; cursor:pointer; border-radius:3px; font-weight:bold; font-size:0.7rem;">TAG</button>
                </div>
                <div id="tag-status" style="font-size:0.65rem; color:#888; margin-top:3px;"></div>
            </div>
            <div class="panel" id="error-panel" style="display:none">
                <div class="panel-title" style="color:#e74c3c">Error</div>
                <div id="error-text" style="font-size:0.75rem; color:#e74c3c;"></div>
            </div>
        </div>
    </div>
    <script>
        const frameImg = document.getElementById('live-frame');
        let seq = 0;

        function updateFrame() {
            frameImg.src = '/api/frame?' + seq++;
        }

        async function updateState() {
            try {
                const res = await fetch('/api/state');
                const s = await res.json();
                lastState = s;

                document.getElementById('fps-display').textContent = s.fps + ' FPS';

                setDot('dot-mouse', s.hardware.mouse_connected ? 'green' : 'red');
                setDot('dot-heartbeat', s.hardware.heartbeat_active ? 'green' : 'gray');
                setDot('dot-jitter', s.hardware.anti_sleep_active ? 'green' : 'gray');
                setDot('dot-keyboard',
                    s.pi_keyboard === true ? 'green' :
                    s.pi_keyboard === false ? 'yellow' : 'gray');

                var coordsEl = document.getElementById('cursor-coords');
                coordsEl.textContent =
                    '(' + s.cursor.x + ', ' + s.cursor.y + ')';
                coordsEl.style.color = s.cursor.quality === 'good' ? '#2ecc71' :
                    s.cursor.quality === 'stale' ? '#f1c40f' : '#e74c3c';
                document.getElementById('cursor-method').textContent =
                    s.cursor.method + ' [' + s.cursor.quality + ']';
                document.getElementById('cursor-age').textContent =
                    s.cursor.age_s + 's';
                updateCursorDot(s.cursor.x, s.cursor.y);

                var m = s.cursor.method;
                document.getElementById('tier-1').classList.toggle('active',
                    m === 'micro_shake' || m === 'motion_track' || m === 'shape_track');
                document.getElementById('tier-2').classList.toggle('active',
                    m === 'macro_shake' || m === 'cnn_reacquire');
                document.getElementById('tier-3').classList.toggle('active', m === 'lissajous');

                document.getElementById('blob-count').textContent = s.blob_count;
                var tbody = document.getElementById('blob-list');
                while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
                if (s.blobs.length > 0) {
                    s.blobs.slice(0, 8).forEach(function(b, i) {
                        var tr = document.createElement('tr');
                        if (i === 0) tr.style.color = '#2ecc71';
                        var td1 = document.createElement('td');
                        td1.textContent = '(' + b.centroid[0] + ', ' + b.centroid[1] + ')';
                        var td2 = document.createElement('td');
                        td2.textContent = b.pixel_count + 'px';
                        var td3 = document.createElement('td');
                        td3.textContent = Math.round(b.angle * 180 / Math.PI) + '\u00B0';
                        tr.appendChild(td1);
                        tr.appendChild(td2);
                        tr.appendChild(td3);
                        tbody.appendChild(tr);
                    });
                } else {
                    var tr = document.createElement('tr');
                    var td = document.createElement('td');
                    td.colSpan = 3;
                    td.style.color = '#444';
                    td.textContent = 'No motion';
                    tr.appendChild(td);
                    tbody.appendChild(tr);
                }

                var jitterStatus = document.getElementById('jitter-status');
                var jitterBar = document.getElementById('jitter-bar');
                var jitterDesc = document.getElementById('jitter-desc');
                if (s.jitter && s.jitter.reacquire_pending) {
                    jitterStatus.textContent = 'Re-acquiring';
                    jitterStatus.className = 'hw-value warn';
                    jitterBar.classList.add('active');
                    jitterBar.style.background = '#e67e22';
                    jitterDesc.textContent = 'Low confidence ' +
                        s.jitter.low_confidence_s.toFixed(0) +
                        's \u2014 next jitter will probe';
                    jitterDesc.style.color = '#e67e22';
                } else if (s.hardware.anti_sleep_active) {
                    jitterStatus.textContent = 'Active';
                    jitterStatus.className = 'hw-value ok';
                    jitterBar.classList.add('active');
                    jitterBar.style.background = '';
                    jitterDesc.textContent = '\u00B13px every 30s \u2014 prevents Windows sleep';
                    jitterDesc.style.color = '#444';
                } else {
                    jitterStatus.textContent = 'Inactive';
                    jitterStatus.className = 'hw-value warn';
                    jitterBar.classList.remove('active');
                    jitterBar.style.background = '';
                    jitterDesc.style.color = '#444';
                }

                setHW('hw-mouse', s.hardware.mouse_connected, 'Connected', 'Disconnected');
                setHW('hw-heartbeat', s.hardware.heartbeat_active, 'Active (100ms)', 'Inactive');
                setHW('hw-keyboard',
                    s.pi_keyboard === true, 'BT Connected',
                    s.pi_keyboard === false ? 'Discoverable' : 'Unreachable');
                document.getElementById('hw-frames').textContent = s.frame_count;

                // YOLO full-screen panel
                if (s.yolo) {
                    var yoloStatus = document.getElementById('yolo-status');
                    yoloStatus.textContent = s.yolo.active ? 'Detecting' : 'No detection';
                    yoloStatus.className = 'hw-value ' + (s.yolo.active ? 'ok' : 'warn');
                    document.getElementById('yolo-position').textContent =
                        s.yolo.active ? '(' + s.yolo.x + ', ' + s.yolo.y + ')' : '—';
                    var yoloConf = document.getElementById('yolo-confidence');
                    yoloConf.textContent = s.yolo.active ? (s.yolo.confidence * 100).toFixed(1) + '%' : '—';
                    yoloConf.className = 'hw-value ' + (s.yolo.confidence > 0.7 ? 'ok' :
                        s.yolo.confidence > 0.4 ? 'warn' : 'err');
                    document.getElementById('yolo-inference').textContent =
                        s.yolo.inference_ms.toFixed(1) + 'ms';
                    document.getElementById('yolo-count').textContent = s.yolo.detection_count;
                }

                // Silhouette tracking panel
                if (s.silhouette) {
                    var silStatus = document.getElementById('sil-status');
                    silStatus.textContent = s.silhouette.active ? 'Active' : 'Inactive';
                    silStatus.className = 'hw-value ' + (s.silhouette.active ? 'ok' : 'warn');
                    var silMethod = document.getElementById('sil-method');
                    silMethod.textContent = s.silhouette.method;
                    silMethod.style.color = s.silhouette.method === 'template' ? '#2ecc71' :
                        s.silhouette.method === 'motion_roi' ? '#3498db' :
                        s.silhouette.method === 'lost' ? '#e74c3c' : '#888';
                    document.getElementById('sil-hz').textContent = s.silhouette.hz + ' Hz';
                    var roiEl = document.getElementById('sil-roi');
                    roiEl.textContent = s.silhouette.roi_size + 'px';
                    roiEl.style.color = s.silhouette.roi_size <= 200 ? '#2ecc71' :
                        s.silhouette.roi_size <= 400 ? '#f1c40f' : '#e74c3c';
                    var silConf = document.getElementById('sil-confidence');
                    silConf.textContent = (s.silhouette.confidence * 100).toFixed(1) + '%';
                    silConf.className = 'hw-value ' + (s.silhouette.confidence > 0.5 ? 'ok' :
                        s.silhouette.confidence > 0.3 ? 'warn' : 'err');
                    document.getElementById('sil-latency').textContent =
                        s.silhouette.latency_ms.toFixed(2) + 'ms';
                    // Trust status: silhouette is trusted if CNN validated OR high confidence OR motion
                    var trusted = s.cnn.cursor_validated ||
                        s.silhouette.confidence > 0.7 ||
                        s.silhouette.method === 'motion_roi';
                    var trustEl = document.getElementById('sil-trusted');
                    trustEl.textContent = trusted ? 'Yes' : 'Gated (CNN < 70%)';
                    trustEl.className = 'hw-value ' + (trusted ? 'ok' : 'warn');
                }

                // CNN panel
                if (s.cnn) {
                    var conf = s.cnn.confidence;
                    var confEl = document.getElementById('cnn-confidence');
                    confEl.textContent = (conf * 100).toFixed(1) + '%';
                    confEl.className = 'hw-value ' + (conf > 0.7 ? 'ok' : conf > 0.3 ? 'warn' : 'err');
                    var bar = document.getElementById('cnn-bar');
                    bar.style.width = (conf * 100) + '%';
                    bar.style.background = conf > 0.7 ? '#2ecc71' : conf > 0.3 ? '#f1c40f' : '#e74c3c';
                    document.getElementById('cnn-model').textContent = s.cnn.model_version;
                    var valEl = document.getElementById('cnn-validated');
                    valEl.textContent = s.cnn.cursor_validated ? 'Yes' : 'No';
                    valEl.className = 'hw-value ' + (s.cnn.cursor_validated ? 'ok' : 'warn');
                    document.getElementById('cnn-inference').textContent = s.cnn.inference_ms + 'ms';
                    document.getElementById('cnn-samples').textContent =
                        s.cnn.samples.pos + ' pos / ' + s.cnn.samples.neg + ' neg';
                }

                // Claude Vision panel
                if (s.vision) {
                    var v = s.vision;
                    if (v.age_s >= 0) {
                        document.getElementById('vision-pos').textContent =
                            '(' + v.x + ', ' + v.y + ')';
                        var vconf = document.getElementById('vision-confidence');
                        vconf.textContent = v.confidence;
                        vconf.className = 'hw-value ' + (
                            v.confidence === 'high' ? 'ok' :
                            v.confidence === 'medium' ? 'warn' : 'err');
                        document.getElementById('vision-type').textContent = v.cursor_type;
                        document.getElementById('vision-latency').textContent =
                            (v.latency_ms / 1000).toFixed(1) + 's';
                        var ageStr = v.age_s < 60 ? v.age_s.toFixed(0) + 's ago' :
                            (v.age_s / 60).toFixed(1) + 'm ago';
                        document.getElementById('vision-age').textContent = ageStr;
                        // Delta between tracker and vision
                        var dx = Math.abs(s.cursor.x - v.x);
                        var dy = Math.abs(s.cursor.y - v.y);
                        var delta = Math.round(Math.sqrt(dx*dx + dy*dy));
                        var deltaEl = document.getElementById('vision-delta');
                        deltaEl.textContent = delta + 'px';
                        deltaEl.className = 'hw-value ' + (
                            delta < 30 ? 'ok' : delta < 100 ? 'warn' : 'err');
                    }
                    updateVisionDot(v.x, v.y, v.age_s);
                }

                var errPanel = document.getElementById('error-panel');
                if (s.error) {
                    errPanel.style.display = 'block';
                    document.getElementById('error-text').textContent = s.error;
                } else {
                    errPanel.style.display = 'none';
                }
            } catch (e) {
                console.error('State update failed:', e);
            }
        }

        function setDot(id, color) {
            document.getElementById(id).className = 'dot ' + color;
        }
        function setHW(id, ok, okText, failText) {
            var el = document.getElementById(id);
            el.textContent = ok ? okText : failText;
            el.className = 'hw-value ' + (ok ? 'ok' : 'err');
        }

        function updateCursorDot(cursorX, cursorY) {
            var img = document.getElementById('live-frame');
            var dot = document.getElementById('cursor-dot');
            if (!img.naturalWidth || (cursorX <= 0 && cursorY <= 0)) {
                dot.style.display = 'none';
                return;
            }
            var pctX = Math.max(0, Math.min(100, cursorX / img.naturalWidth * 100));
            var pctY = Math.max(0, Math.min(100, cursorY / img.naturalHeight * 100));
            dot.style.left = pctX + '%';
            dot.style.top = pctY + '%';
            dot.style.display = 'block';
        }

        function updateVisionDot(visionX, visionY, ageS) {
            var img = document.getElementById('live-frame');
            var dot = document.getElementById('vision-dot');
            if (!img.naturalWidth || visionX <= 0 || visionY <= 0 || ageS < 0) {
                dot.style.display = 'none';
                return;
            }
            var pctX = Math.max(0, Math.min(100, visionX / img.naturalWidth * 100));
            var pctY = Math.max(0, Math.min(100, visionY / img.naturalHeight * 100));
            dot.style.left = pctX + '%';
            dot.style.top = pctY + '%';
            // Fade out as vision ages: fully visible <30s, faded 30-120s, hidden >120s
            var opacity = ageS < 30 ? 1.0 : ageS < 120 ? 1.0 - (ageS - 30) / 90 : 0;
            dot.style.opacity = opacity;
            dot.style.display = opacity > 0.05 ? 'block' : 'none';
        }

        function runVisionDetect() {
            var btn = document.getElementById('vision-btn');
            btn.textContent = 'Detecting...';
            btn.disabled = true;
            fetch('/api/claude_detect')
                .then(function(r) { return r.json(); })
                .then(function(d) {
                    btn.textContent = 'Run Detection';
                    btn.disabled = false;
                })
                .catch(function() {
                    btn.textContent = 'Run Detection';
                    btn.disabled = false;
                });
        }

        // ── Pause / freeze for tagging ──
        var paused = false;
        var frozenState = null;
        var tagCount = 0;

        function togglePause() {
            paused = !paused;
            var btn = document.getElementById('pause-btn');
            var overlay = document.getElementById('pause-overlay');
            if (paused) {
                btn.textContent = 'RESUME (LIVE)';
                btn.style.background = '#e74c3c';
                overlay.style.display = 'flex';
                // Freeze current state for tagging
                frozenState = lastState;
            } else {
                btn.textContent = 'PAUSE (TAG)';
                btn.style.background = '#f39c12';
                overlay.style.display = 'none';
                frozenState = null;
                // Clear tag selections
                document.querySelectorAll('.tag-btn').forEach(function(b) {
                    b.classList.remove('selected');
                });
                tagState = {};
                document.getElementById('tag-notes').value = '';
                document.getElementById('tag-status').textContent = '';
            }
        }

        var lastState = null;

        // ── Frame updates at 5Hz (skipped when paused) ──
        function scheduleFrame() {
            if (!paused) updateFrame();
            setTimeout(scheduleFrame, 200);
        }
        var stateInterval = setInterval(function() {
            if (!paused) updateState();
        }, 200);
        scheduleFrame();
        updateState();

        // Store last state for freeze
        var origUpdateState = updateState;

        // ── Tagging system ──
        var tagState = {};
        document.querySelectorAll('.tag-btn').forEach(function(btn) {
            btn.addEventListener('click', function() {
                // Auto-pause on first tag button click
                if (!paused) togglePause();
                var dim = this.dataset.dim;
                var val = this.dataset.val;
                // Toggle selection in same group
                this.parentElement.querySelectorAll('.tag-btn').forEach(function(b) {
                    b.classList.remove('selected');
                });
                this.classList.add('selected');
                tagState[dim] = val;
            });
        });

        async function submitTag() {
            var notes = document.getElementById('tag-notes').value;
            tagState.notes = notes;
            var statusEl = document.getElementById('tag-status');
            statusEl.textContent = 'Saving...';
            try {
                var res = await fetch('/api/tag', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(tagState)
                });
                var data = await res.json();
                if (data.status === 'tagged') {
                    tagCount++;
                    statusEl.textContent = 'Tagged #' + tagCount + ' (' + data.frame + ')';
                    statusEl.style.color = '#27ae60';
                    // Auto-resume after successful tag
                    setTimeout(function() {
                        togglePause();
                    }, 800);
                } else {
                    statusEl.textContent = 'Error: ' + (data.error || 'unknown');
                    statusEl.style.color = '#e74c3c';
                }
            } catch(e) {
                statusEl.textContent = 'Error: ' + e.message;
                statusEl.style.color = '#e74c3c';
            }
        }
    </script>
</body>
</html>
"""


# ── Validation HTML ────────────────────────────────────────────────

VALIDATION_HTML = r"""<!DOCTYPE html>
<html>
<head>
<title>Cursor Sample Validation</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0a0f; color: #e0e0e0; font-family: 'JetBrains Mono', monospace; }
.header { background: #12121a; padding: 12px 20px; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #222; }
.header h1 { font-size: 16px; color: #ff4444; }
.header .nav { display: flex; gap: 6px; }
.nav-link { padding: 5px 12px; border: 1px solid #333; background: #1a1a24; color: #888; text-decoration: none; font-size: 11px; border-radius: 4px; }
.nav-link:hover { background: #252530; color: #ccc; }
.nav-link.active { border-color: #ff4444; color: #ff4444; }
.header .actions { display: flex; gap: 8px; }
.btn { padding: 6px 14px; border: 1px solid #333; background: #1a1a24; color: #e0e0e0; cursor: pointer; font-size: 12px; font-family: inherit; border-radius: 4px; }
.btn:hover { background: #252530; }
.btn-retrain { border-color: #ff4444; color: #ff4444; }
.btn-retrain:hover { background: #ff444420; }
.controls { padding: 10px 20px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; border-bottom: 1px solid #1a1a24; }
.filter-btn { padding: 4px 10px; border: 1px solid #333; background: transparent; color: #888; cursor: pointer; font-size: 11px; font-family: inherit; border-radius: 3px; }
.filter-btn.active { border-color: #ff4444; color: #ff4444; background: #ff444410; }
.filter-count { font-size: 9px; color: #555; margin-left: 2px; }
.filter-btn.active .filter-count { color: #ff444488; }
.sort-select, .source-select, .label-select { padding: 4px 8px; border: 1px solid #333; background: #12121a; color: #ccc; font-size: 11px; font-family: inherit; border-radius: 3px; }
.ctrl-label { font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-right: 2px; }
.ctrl-sep { width: 1px; height: 20px; background: #222; margin: 0 4px; }
.pagination { display: flex; gap: 8px; align-items: center; margin-left: auto; font-size: 11px; color: #666; }
.pagination .btn { padding: 3px 8px; font-size: 11px; }
.stats-bar { padding: 8px 20px; background: #0d0d14; font-size: 11px; color: #666; display: flex; gap: 16px; border-bottom: 1px solid #1a1a24; flex-wrap: wrap; }
.stats-bar span { color: #888; }
table { width: 100%; border-collapse: collapse; }
th { text-align: left; padding: 8px 12px; font-size: 11px; color: #666; background: #0d0d14; border-bottom: 1px solid #222; cursor: pointer; }
th:hover { color: #aaa; }
th.sorted { color: #ff4444; }
td { padding: 8px 12px; border-bottom: 1px solid #1a1a24; vertical-align: middle; font-size: 12px; }
tr:hover { background: #12121a; }
.patch-img { width: 96px; height: 96px; image-rendering: pixelated; border: 1px solid #333; border-radius: 2px; cursor: pointer; }
.patch-img:hover { border-color: #ff4444; }
.label-pos { color: #4CAF50; font-weight: bold; }
.label-neg { color: #ff5252; font-weight: bold; }
.source { color: #888; font-size: 11px; }
.source.claude { color: #7c4dff; }
.source.ground-truth { color: #2ecc71; }
.confusing-badge { color: #FF9800; font-size: 10px; margin-left: 4px; }
.disagree-badge { color: #ff5252; font-size: 10px; margin-left: 4px; }
.conf-bar { width: 60px; height: 6px; background: #1a1a24; border-radius: 3px; display: inline-block; vertical-align: middle; margin-right: 6px; }
.conf-fill { height: 100%; border-radius: 3px; }
.conf-high { background: #4CAF50; }
.conf-mid { background: #FF9800; }
.conf-low { background: #ff5252; }
.actions-cell { display: flex; gap: 4px; }
.btn-ok { border-color: #4CAF50; color: #4CAF50; }
.btn-ok:hover { background: #4CAF5020; }
.btn-wrong { border-color: #FF9800; color: #FF9800; }
.btn-wrong:hover { background: #FF980020; }
.btn-confuse { border-color: #9c27b0; color: #9c27b0; }
.btn-confuse:hover { background: #9c27b020; }
.btn-del { border-color: #ff5252; color: #ff5252; }
.btn-del:hover { background: #ff525220; }
.tag-cell { white-space: nowrap; }
.tag-chip { display: inline-block; padding: 1px 5px; margin: 1px; border: 1px solid #444; border-radius: 8px; font-size: 0.6rem; cursor: pointer; color: #888; transition: all 0.15s; }
.tag-chip:hover { color: #fff; background: #ffffff10; }
.tag-chip.active { color: #fff; font-weight: bold; }
.reviewed { opacity: 0.4; }
.toast { position: fixed; bottom: 20px; right: 20px; padding: 10px 16px; background: #1a1a24; border: 1px solid #4CAF50; color: #4CAF50; border-radius: 4px; font-size: 12px; display: none; z-index: 100; }
.preview-overlay { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.85); z-index: 200; justify-content: center; align-items: center; flex-direction: column; gap: 12px; }
.preview-overlay.active { display: flex; }
.preview-container { position: relative; }
.preview-overlay img { width: 512px; height: 512px; image-rendering: pixelated; border: 2px solid #444; cursor: crosshair; }
.tip-marker { position: absolute; width: 12px; height: 12px; border: 2px solid #ff0; border-radius: 50%; transform: translate(-50%, -50%); pointer-events: none; box-shadow: 0 0 8px #ff0; }
.preview-info { color: #888; font-size: 12px; text-align: center; }
.preview-info em { color: #ff0; font-style: normal; }
.cursor-type { font-size: 11px; padding: 2px 6px; border-radius: 3px; }
.cursor-type.arrow { color: #4CAF50; border: 1px solid #4CAF5040; }
.cursor-type.hand { color: #2196F3; border: 1px solid #2196F340; }
.cursor-type.ibeam { color: #FF9800; border: 1px solid #FF980040; }
.cursor-type.unknown { color: #666; border: 1px solid #33333340; }
.btn-confuse { border-color: #9C27B0; color: #9C27B0; }
.btn-confuse:hover { background: #9C27B020; }
.keyboard-hint { position: fixed; bottom: 20px; left: 20px; font-size: 10px; color: #444; }
.info-bar { background: #0d0d16; border-bottom: 1px solid #1a1a24; overflow: hidden; transition: max-height 0.3s ease; }
.info-bar.collapsed { max-height: 0; border: none; }
.info-bar-inner { padding: 12px 20px; display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; font-size: 11px; line-height: 1.6; color: #777; }
.info-bar-inner h3 { font-size: 11px; color: #ff4444; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; font-weight: 600; }
.info-bar-inner code { background: #1a1a24; padding: 1px 4px; border-radius: 2px; color: #aaa; font-size: 10px; }
.info-bar-inner b { color: #ccc; font-weight: 500; }
.info-toggle { font-size: 10px; color: #555; cursor: pointer; border: 1px solid #333; background: transparent; padding: 2px 8px; border-radius: 3px; font-family: inherit; }
.info-toggle:hover { color: #aaa; border-color: #555; }
.info-bar-inner .kv { display: flex; gap: 4px; }
.info-bar-inner .kv dt { color: #555; min-width: 80px; }
.info-bar-inner .kv dd { color: #aaa; }
.info-bar-inner ul { padding-left: 14px; margin: 2px 0; }
.info-bar-inner li { margin: 1px 0; }
</style>
</head>
<body>
<div class="header">
  <div style="display:flex;align-items:center;gap:16px">
    <h1>CURSOR SAMPLE VALIDATION</h1>
    <div class="nav">
      <a href="/" class="nav-link">Dashboard</a>
      <a href="/profiler" class="nav-link">Profiler</a>
      <a href="/validation" class="nav-link active">Validation</a>
      <a href="/loss_events" class="nav-link">Loss Events</a>
    </div>
  </div>
  <div class="actions">
    <button class="info-toggle" onclick="toggleInfo()">? Info</button>
    <button class="btn btn-retrain" onclick="retrain()">Retrain CNN</button>
  </div>
</div>

<div class="info-bar" id="info-bar">
<div class="info-bar-inner">
  <div>
    <h3>What is this page</h3>
    These are 64x64 grayscale patches extracted around where the system thinks the cursor is.
    Each patch is fed through <b>TinyCursorNet</b> (14K param CNN) to classify: <b>cursor</b> or <b>not cursor</b>.<br><br>
    <b>Your job</b>: review samples the CNN gets wrong, tag features, and retrain.
    The CNN improves only from data you label here.
    <ul>
      <li><b>OK</b> — CNN's label is correct for this patch</li>
      <li><b>X</b> (Wrong) — CNN got it backwards (pos is actually neg, or vice versa)</li>
      <li><b>Del</b> — bad sample, remove from training set</li>
    </ul>
  </div>
  <div>
    <h3>Columns &amp; Filters</h3>
    <ul>
      <li><b>Label</b>: <code>pos</code> = cursor was here, <code>neg</code> = no cursor at this position</li>
      <li><b>Type</b>: cursor shape when captured (arrow, hand, I-beam) — only from Claude Vision samples</li>
      <li><b>Source</b>: how the patch was collected:<br>
        <code>motion_track</code> — from live tracking loop<br>
        <code>claude_vision</code> — Vision-confirmed position<br>
        <code>ground_truth</code> — manually verified</li>
      <li><b>CNN Conf</b>: model's confidence this patch contains a cursor (0-1). Green &gt;0.7, yellow 0.3-0.7, red &lt;0.3</li>
      <li><b>Priority</b>: review order — CNN Disagrees (0) &gt; confusing (1) &gt; uncertain (2) &gt; Vision (3) &gt; easy (4) &gt; reviewed (99)</li>
    </ul>
  </div>
  <div>
    <h3>Feature Tags (right columns)</h3>
    Tags are <b>independent</b> — a sample can be "ok" AND "confusing" AND "partial" simultaneously.
    Click a chip to set it, click again to clear. Saves instantly.
    <ul>
      <li><b>quality</b> — ok / wrong / unsure (your judgment of the label)</li>
      <li><b>confusing</b> — yes / no (would a human struggle to identify the cursor here?)</li>
      <li><b>cursor_type</b> — arrow / hand / ibeam / busy / crosshair / move</li>
      <li><b>visibility</b> — full / partial / hidden / offscreen</li>
      <li><b>background</b> — clean / noisy / animated / textured</li>
    </ul>
    New columns appear automatically when you tag with custom keys via the API.
    <br><br>
    <h3>Keyboard Shortcuts</h3>
    <code>J/K</code> or arrows: navigate &nbsp; <code>1</code>: OK &nbsp; <code>2</code>: Wrong &nbsp; <code>3</code>: Confusing &nbsp; <code>4</code>: Del &nbsp; <code>Space</code>: open patch (click to annotate cursor tip)
  </div>
</div>
</div>

<div class="controls">
  <span class="ctrl-label">Filter</span>
  <button class="filter-btn active" data-filter="all" onclick="setFilter('all')" title="All samples in the training set">All <span class="filter-count" id="cnt-all"></span></button>
  <button class="filter-btn" data-filter="unreviewed" onclick="setFilter('unreviewed')" title="Samples you haven't reviewed yet — start here">Unreviewed <span class="filter-count" id="cnt-unreviewed"></span></button>
  <button class="filter-btn" data-filter="reviewed" onclick="setFilter('reviewed')" title="Samples you've already tagged (dimmed in table)">Reviewed <span class="filter-count" id="cnt-reviewed"></span></button>
  <button class="filter-btn" data-filter="claude" onclick="setFilter('claude')" title="Patches from Claude Vision detections — higher quality ground truth positions">Claude Vision <span class="filter-count" id="cnt-claude"></span></button>
  <button class="filter-btn" data-filter="disagreement" onclick="setFilter('disagreement')" title="CNN prediction differs from the assigned label — most valuable to review first">CNN Disagrees <span class="filter-count" id="cnt-disagreement"></span></button>
  <div class="ctrl-sep"></div>
  <span class="ctrl-label">Label</span>
  <select class="label-select" onchange="setLabel(this.value)">
    <option value="all">All</option>
    <option value="pos">Positive</option>
    <option value="neg">Negative</option>
  </select>
  <div class="ctrl-sep"></div>
  <span class="ctrl-label">Source</span>
  <select class="source-select" id="source-select" onchange="setSource(this.value)">
    <option value="all">All Sources</option>
  </select>
  <div class="ctrl-sep"></div>
  <span class="ctrl-label">Sort</span>
  <select class="sort-select" onchange="setSort(this.value)">
    <option value="priority">Priority (review first)</option>
    <option value="confidence_asc">CNN Conf (low first)</option>
    <option value="confidence_desc">CNN Conf (high first)</option>
    <option value="source">Source</option>
    <option value="recent">Recent</option>
  </select>
  <div class="pagination">
    <button class="btn" onclick="prevPage()">&laquo;</button>
    <span id="page-info">Page 1</span>
    <button class="btn" onclick="nextPage()">&raquo;</button>
  </div>
</div>
<div class="stats-bar" id="stats-bar">
  <span>Loading stats...</span>
</div>
<table id="samples-table">
  <thead>
    <tr>
      <th>Patch</th>
      <th onclick="setSort('source')">Label</th>
      <th>Type</th>
      <th onclick="setSort('source')">Source</th>
      <th onclick="setSort(currentSort==='confidence_asc'?'confidence_desc':'confidence_asc')">CNN Conf</th>
      <th onclick="setSort('priority')">Priority</th>
      <th>Actions</th>
    </tr>
  </thead>
  <tbody id="samples-body"></tbody>
</table>
<div class="toast" id="toast"></div>
<div class="preview-overlay" id="preview-overlay">
  <div class="preview-container" id="preview-container">
    <img id="preview-img">
    <div class="tip-marker" id="tip-marker" style="display:none"></div>
  </div>
  <div class="preview-info" id="preview-info">Click on the <em>cursor tip</em> to annotate (saves immediately). ESC to close.</div>
</div>
<div class="keyboard-hint">J/K: next/prev | 1: OK | 2: Wrong | 3: Confusing | 4: Del | Space: annotate tip</div>

<script>
function toggleInfo() {
  var bar = document.getElementById('info-bar');
  bar.classList.toggle('collapsed');
  localStorage.setItem('validation_info', bar.classList.contains('collapsed') ? '0' : '1');
}
// Restore info bar state
if (localStorage.getItem('validation_info') === '0') {
  document.getElementById('info-bar').classList.add('collapsed');
}

var currentPage = 0;
var currentFilter = 'all';
var currentSort = 'priority';
var currentLabel = 'all';
var currentSource = 'all';
var totalPages = 1;
var selectedRow = -1;
var currentSamples = [];

async function loadSamples() {
  var url = '/api/validation/samples?page=' + currentPage +
    '&sort=' + currentSort +
    '&filter=' + currentFilter +
    '&label=' + currentLabel +
    '&source_filter=' + currentSource;
  var res = await fetch(url);
  var data = await res.json();
  totalPages = Math.ceil(data.total / data.page_size) || 1;
  document.getElementById('page-info').textContent =
    'Page ' + (currentPage + 1) + '/' + totalPages + ' (' + data.total + ' total)';

  // Update filter counts in tabs
  var fc = data.filter_counts || {};
  ['all','unreviewed','reviewed','claude','disagreement'].forEach(function(k) {
    var el = document.getElementById('cnt-' + k);
    if (el) el.textContent = '(' + (fc[k] || 0) + ')';
  });

  // Populate source dropdown (once)
  var srcSel = document.getElementById('source-select');
  if (fc.sources && srcSel.options.length <= 1) {
    Object.keys(fc.sources).sort().forEach(function(src) {
      var opt = document.createElement('option');
      opt.value = src;
      opt.textContent = src + ' (' + fc.sources[src] + ')';
      srcSel.appendChild(opt);
    });
  }

  currentSamples = data.samples;
  var tbody = document.getElementById('samples-body');
  tbody.replaceChildren();
  selectedRow = -1;

  data.samples.forEach(function(s, idx) {
    var tr = document.createElement('tr');
    if (s.review) tr.classList.add('reviewed');
    tr.id = 'row-' + s.filename;
    tr.dataset.idx = idx;

    // Patch image
    var tdImg = document.createElement('td');
    var img = document.createElement('img');
    img.className = 'patch-img';
    img.src = '/api/validation/patch/' + encodeURIComponent(s.filename);
    img.onclick = function() { showPreview(s.filename); };
    tdImg.appendChild(img);
    tr.appendChild(tdImg);

    // Label
    var tdLabel = document.createElement('td');
    var labelSpan = document.createElement('span');
    labelSpan.className = s.label === 'pos' ? 'label-pos' : 'label-neg';
    labelSpan.textContent = s.label;
    tdLabel.appendChild(labelSpan);
    if (s.review) {
      var reviewSpan = document.createElement('span');
      reviewSpan.style.cssText = 'color:#4CAF50;font-size:10px;margin-left:4px';
      reviewSpan.textContent = '[' + s.review.action + ']';
      tdLabel.appendChild(reviewSpan);
    }
    if (s.cnn_disagrees) {
      var dBadge = document.createElement('span');
      dBadge.className = 'disagree-badge';
      dBadge.textContent = 'DISAGREE';
      tdLabel.appendChild(dBadge);
    }
    tr.appendChild(tdLabel);

    // Cursor type
    var tdType = document.createElement('td');
    var ctype = s.cursor_type || '';
    if (ctype) {
      var typeSpan = document.createElement('span');
      var typeClass = 'cursor-type ';
      if (ctype === 'arrow') typeClass += 'arrow';
      else if (ctype === 'hand') typeClass += 'hand';
      else if (ctype === 'ibeam' || ctype === 'I-beam') typeClass += 'ibeam';
      else typeClass += 'unknown';
      typeSpan.className = typeClass;
      typeSpan.textContent = ctype;
      tdType.appendChild(typeSpan);
    } else {
      tdType.style.color = '#333';
      tdType.textContent = '-';
    }
    tr.appendChild(tdType);

    // Source
    var tdSource = document.createElement('td');
    var srcSpan = document.createElement('span');
    var srcClass = 'source';
    if (s.source && s.source.indexOf('claude') >= 0) srcClass += ' claude';
    else if (s.source && s.source.indexOf('ground_truth') >= 0) srcClass += ' ground-truth';
    srcSpan.className = srcClass;
    srcSpan.textContent = s.source || '?';
    tdSource.appendChild(srcSpan);
    if (s.is_confusing) {
      var badge = document.createElement('span');
      badge.className = 'confusing-badge';
      badge.textContent = '\u26A0';
      tdSource.appendChild(badge);
    }
    tr.appendChild(tdSource);

    // CNN Confidence
    var tdConf = document.createElement('td');
    var bar = document.createElement('div');
    bar.className = 'conf-bar';
    var fill = document.createElement('div');
    var confVal = s.cnn_confidence != null ? s.cnn_confidence : 0;
    fill.className = 'conf-fill ' + (confVal > 0.7 ? 'conf-high' : confVal > 0.4 ? 'conf-mid' : 'conf-low');
    fill.style.width = Math.round(confVal * 100) + '%';
    bar.appendChild(fill);
    tdConf.appendChild(bar);
    tdConf.appendChild(document.createTextNode(s.cnn_confidence != null ? s.cnn_confidence.toFixed(3) : '?'));
    tr.appendChild(tdConf);

    // Priority
    var tdPri = document.createElement('td');
    tdPri.style.color = '#666';
    tdPri.textContent = s.priority.toFixed(1);
    tr.appendChild(tdPri);

    // Quick actions (legacy)
    var tdAct = document.createElement('td');
    tdAct.className = 'actions-cell';
    ['OK:correct:btn-ok', 'X:wrong:btn-wrong', 'Del:delete:btn-del'].forEach(function(spec) {
      var parts = spec.split(':');
      var btn = document.createElement('button');
      btn.className = 'btn ' + parts[2];
      btn.textContent = parts[0];
      btn.addEventListener('click', function() { reviewSample(s.filename, parts[1]); });
      tdAct.appendChild(btn);
    });
    tr.appendChild(tdAct);

    // Feature tags — one cell per tag column
    if (window.tagSchema) {
      Object.keys(window.tagSchema).forEach(function(col) {
        var tdTag = document.createElement('td');
        tdTag.className = 'tag-cell';
        var currentVal = (s.tags && s.tags[col]) || '';
        var schema = window.tagSchema[col];
        schema.values.forEach(function(val) {
          var chip = document.createElement('span');
          chip.className = 'tag-chip' + (currentVal === val ? ' active' : '');
          chip.textContent = val;
          chip.style.borderColor = schema.color;
          if (currentVal === val) chip.style.background = schema.color + '30';
          chip.addEventListener('click', function() {
            // Toggle: click active chip to deselect, otherwise select
            var newVal = currentVal === val ? '' : val;
            setTag(s.filename, col, newVal);
            // Update UI immediately
            tdTag.querySelectorAll('.tag-chip').forEach(function(c) {
              c.classList.remove('active');
              c.style.background = '';
            });
            if (newVal) {
              chip.classList.add('active');
              chip.style.background = schema.color + '30';
            }
            // Update in-memory
            if (!s.tags) s.tags = {};
            s.tags[col] = newVal;
          });
          tdTag.appendChild(chip);
        });
        tr.appendChild(tdTag);
      });
    }

    tbody.appendChild(tr);
  });
}

async function loadStats() {
  var res = await fetch('/api/validation/stats');
  var data = await res.json();
  var bar = document.getElementById('stats-bar');
  bar.replaceChildren();
  var items = [
    data.pos_count + ' pos', data.neg_count + ' neg',
    data.review_count + ' reviewed', 'Model: ' + data.model_version
  ];
  var sources = data.source_distribution || {};
  Object.keys(sources).forEach(function(k) { items.push(k + ': ' + sources[k]); });
  items.forEach(function(txt) {
    var span = document.createElement('span');
    span.textContent = txt;
    bar.appendChild(span);
  });
}

async function setTag(filename, column, value) {
  var tags = {};
  tags[column] = value;
  var res = await fetch('/api/validation/tag', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filename: filename, tags: tags})
  });
  if (res.ok) {
    showToast(column + '=' + (value || '(cleared)') + ' → ' + filename);
  }
}

async function loadTagSchema() {
  var res = await fetch('/api/validation/tag_schema');
  var data = await res.json();
  window.tagSchema = data.schema;
  // Add column headers to the table
  var thead = document.querySelector('#samples-table thead tr');
  if (thead) {
    Object.keys(data.schema).forEach(function(col) {
      var th = document.createElement('th');
      th.textContent = col;
      th.style.color = data.schema[col].color;
      th.style.fontSize = '0.65rem';
      th.style.textTransform = 'uppercase';
      thead.appendChild(th);
    });
  }
}

async function reviewSample(filename, action) {
  var res = await fetch('/api/validation/review', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filename: filename, action: action})
  });
  if (res.ok) {
    var row = document.getElementById('row-' + filename);
    if (row) row.classList.add('reviewed');
    showToast(action + ': ' + filename);
  }
}

async function retrain() {
  if (!confirm('Start retraining TinyCursorNet?')) return;
  var res = await fetch('/api/validation/retrain', {method: 'POST'});
  var data = await res.json();
  showToast('Training started (PID: ' + data.pid + ')');
}

function setFilter(f) {
  currentFilter = f;
  document.querySelectorAll('.filter-btn').forEach(function(b) {
    b.classList.toggle('active', b.dataset.filter === f);
  });
  currentPage = 0;
  loadSamples();
}

function setSort(s) {
  currentSort = s;
  document.querySelector('.sort-select').value = s;
  currentPage = 0;
  loadSamples();
}

function setLabel(l) { currentLabel = l; currentPage = 0; loadSamples(); }
function setSource(s) { currentSource = s; currentPage = 0; loadSamples(); }

function nextPage() { if (currentPage < totalPages - 1) { currentPage++; loadSamples(); } }
function prevPage() { if (currentPage > 0) { currentPage--; loadSamples(); } }

var previewFilename = null;

function showPreview(filename) {
  previewFilename = filename;
  var marker = document.getElementById('tip-marker');
  marker.style.display = 'none';
  document.getElementById('preview-img').src = '/api/validation/patch/' + encodeURIComponent(filename);
  document.getElementById('preview-overlay').classList.add('active');
  document.getElementById('preview-info').textContent = 'Click on the cursor tip to annotate. ESC to close.';
}

// Click-to-annotate: click on enlarged patch to mark cursor tip position
document.getElementById('preview-img').addEventListener('click', async function(e) {
  if (!previewFilename) return;
  var img = e.target;
  var rect = img.getBoundingClientRect();
  // Click position relative to rendered image (512x512)
  var renderX = e.clientX - rect.left;
  var renderY = e.clientY - rect.top;
  // Scale to actual 64x64 patch coordinates
  var scaleX = 64 / rect.width;
  var scaleY = 64 / rect.height;
  var tipX = Math.round(renderX * scaleX);
  var tipY = Math.round(renderY * scaleY);
  tipX = Math.max(0, Math.min(63, tipX));
  tipY = Math.max(0, Math.min(63, tipY));

  // Show yellow marker at click position
  var marker = document.getElementById('tip-marker');
  marker.style.left = renderX + 'px';
  marker.style.top = renderY + 'px';
  marker.style.display = 'block';

  // Save annotation
  var res = await fetch('/api/validation/review', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({filename: previewFilename, action: 'annotate_tip', tip_x: tipX, tip_y: tipY})
  });
  if (res.ok) {
    document.getElementById('preview-info').textContent =
      'Tip annotated at (' + tipX + ', ' + tipY + ') — saved. Click again to adjust, ESC to close.';
    showToast('Tip: (' + tipX + ',' + tipY + ') → ' + previewFilename);
  }
});

function showToast(msg) {
  var t = document.getElementById('toast');
  t.textContent = msg;
  t.style.display = 'block';
  setTimeout(function() { t.style.display = 'none'; }, 2000);
}

// Keyboard shortcuts for fast tagging
document.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  var rows = document.querySelectorAll('#samples-body tr');
  if (e.key === 'j' || e.key === 'ArrowDown') {
    e.preventDefault();
    if (selectedRow < rows.length - 1) {
      if (selectedRow >= 0) rows[selectedRow].style.outline = '';
      selectedRow++;
      rows[selectedRow].style.outline = '1px solid #ff4444';
      rows[selectedRow].scrollIntoView({block: 'nearest'});
    }
  } else if (e.key === 'k' || e.key === 'ArrowUp') {
    e.preventDefault();
    if (selectedRow > 0) {
      rows[selectedRow].style.outline = '';
      selectedRow--;
      rows[selectedRow].style.outline = '1px solid #ff4444';
      rows[selectedRow].scrollIntoView({block: 'nearest'});
    }
  } else if (selectedRow >= 0 && selectedRow < currentSamples.length) {
    var fn = currentSamples[selectedRow].filename;
    if (e.key === '1') reviewSample(fn, 'correct');
    else if (e.key === '2') reviewSample(fn, 'wrong');
    else if (e.key === '3') reviewSample(fn, 'confusing');
    else if (e.key === '4') reviewSample(fn, 'delete');
    else if (e.key === ' ') { e.preventDefault(); showPreview(fn); }
  }
  if (e.key === 'Escape') document.getElementById('preview-overlay').classList.remove('active');
});

// Load tag schema first, then samples (schema needed to render tag columns)
loadTagSchema().then(function() { loadSamples(); });
loadStats();
</script>
</body>
</html>"""


# ── Profiler HTML ──────────────────────────────────────────────────

PROFILER_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>KVM Pipeline Profiler</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'JetBrains Mono', 'Fira Code', monospace; background: #0a0a0f; color: #c8c8d0; }
.header { background: #101018; padding: 0.6rem 1.5rem; border-bottom: 2px solid #3498db;
    display: flex; justify-content: space-between; align-items: center; }
.header h1 { color: #3498db; font-size: 1.1rem; letter-spacing: 1px; }
.header a { color: #888; text-decoration: none; font-size: 0.75rem; margin-left: 1rem; }
.header a:hover { color: #3498db; }
.main { padding: 1rem; max-width: 1400px; margin: 0 auto; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
.row.full { grid-template-columns: 1fr; }
.row.triple { grid-template-columns: 1fr 1fr 1fr; }
.card { background: #101018; border: 1px solid #1a1a25; border-radius: 8px; padding: 1rem; }
.card-title { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 2px; color: #3498db; margin-bottom: 0.8rem; }
.utc-bar { display: flex; gap: 2rem; align-items: center; margin-bottom: 1rem;
    background: #101018; border: 1px solid #1a1a25; border-radius: 8px; padding: 0.8rem 1rem; }
.utc-lbl { color: #555; font-size: 0.6rem; text-transform: uppercase; }
.utc-val { color: #2ecc71; font-size: 1.1rem; font-variant-numeric: tabular-nums; }
.utc-delta { color: #f1c40f; font-size: 1.1rem; }
/* Waterfall */
.wf-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.wf-label { width: 120px; font-size: 0.75rem; color: #888; text-align: right; flex-shrink: 0; }
.wf-bar-wrap { flex: 1; height: 28px; background: #0a0a12; border-radius: 4px; overflow: hidden; }
.wf-bar { height: 100%; border-radius: 4px; transition: width 0.2s; min-width: 2px; }
.wf-bar.decode { background: #3498db; } .wf-bar.track { background: #2ecc71; }
.wf-bar.overlay { background: #9b59b6; } .wf-bar.encode { background: #e67e22; }
.wf-bar.total { background: #e74c3c; } .wf-bar.http { background: #f1c40f; }
.wf-bar.render { background: #1abc9c; }
.wf-val { width: 80px; font-size: 0.8rem; color: #aaa; font-variant-numeric: tabular-nums; }
.wf-sep { border-top: 1px dashed #333; margin: 6px 0; }
/* Test panel */
.prof-btn { padding: 6px 16px; border: 1px solid #333; border-radius: 4px;
    background: #1a1a25; color: #ccc; font-family: inherit; font-size: 0.8rem; cursor: pointer; }
.prof-btn:hover { background: #252530; }
.prof-btn.primary { border-color: #3498db; color: #3498db; }
.prof-btn.danger { border-color: #e74c3c; color: #e74c3c; }
.prof-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.results-table { width: 100%; font-size: 0.75rem; border-collapse: collapse; margin-top: 0.6rem; }
.results-table th { text-align: right; color: #555; font-weight: normal; padding: 4px 8px; border-bottom: 1px solid #1a1a25; }
.results-table th:first-child { text-align: left; }
.results-table td { text-align: right; padding: 4px 8px; font-variant-numeric: tabular-nums; }
.results-table td:first-child { text-align: left; color: #888; }
/* Metrics */
.metric-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 0.8rem; }
.metric-label { color: #666; }
.metric-val { font-weight: 600; color: #ccc; font-variant-numeric: tabular-nums; }
.metric-val.good { color: #2ecc71; } .metric-val.warn { color: #f1c40f; } .metric-val.bad { color: #e74c3c; }
/* Sparkline */
.spark-canvas { width: 100%; height: 80px; background: #0a0a12; border-radius: 4px; }
/* Frame preview */
.preview-img { max-width: 100%; border-radius: 4px; background: #000; display: block; }
.preview-overlay { font-size: 0.7rem; color: #888; margin-top: 4px; }
</style>
</head>
<body>
<div class="header">
    <div style="display:flex; align-items:center;">
        <h1>PIPELINE PROFILER</h1>
        <a href="/">Dashboard</a>
        <a href="/concepts">Concepts</a>
    </div>
    <div style="font-size:0.75rem; color:#555;" id="prof-status">Connecting...</div>
</div>
<div class="main">
    <!-- UTC Clock Bar -->
    <div class="utc-bar">
        <div><div class="utc-lbl">Browser UTC</div><div class="utc-val" id="browser-utc">--:--:--.---</div></div>
        <div><div class="utc-lbl">Server UTC</div><div class="utc-val" id="server-utc">--:--:--.---</div></div>
        <div><div class="utc-lbl">Clock Delta</div><div class="utc-delta" id="utc-delta">--ms</div></div>
        <div style="margin-left:auto;"><div class="utc-lbl">Daemon FPS</div><div class="metric-val" id="daemon-fps">--</div></div>
        <div><div class="utc-lbl">Blobs</div><div class="metric-val" id="blob-count">--</div></div>
        <div><div class="utc-lbl">Frame #</div><div class="metric-val" id="frame-count" style="color:#555">--</div></div>
    </div>

    <!-- Pipeline Waterfall (full width) -->
    <div class="row full">
        <div class="card">
            <div class="card-title">Pipeline Waterfall (per frame)</div>
            <div class="wf-row"><span class="wf-label">JPEG Decode</span><div class="wf-bar-wrap"><div class="wf-bar decode" id="bar-decode"></div></div><span class="wf-val" id="val-decode">&mdash;</span></div>
            <div class="wf-row"><span class="wf-label">Sil. Track</span><div class="wf-bar-wrap"><div class="wf-bar track" id="bar-track"></div></div><span class="wf-val" id="val-track">&mdash;</span></div>
            <div class="wf-row"><span class="wf-label">Overlay Draw</span><div class="wf-bar-wrap"><div class="wf-bar overlay" id="bar-overlay"></div></div><span class="wf-val" id="val-overlay">&mdash;</span></div>
            <div class="wf-row"><span class="wf-label">JPEG Encode</span><div class="wf-bar-wrap"><div class="wf-bar encode" id="bar-encode"></div></div><span class="wf-val" id="val-encode">&mdash;</span></div>
            <div class="wf-sep"></div>
            <div class="wf-row"><span class="wf-label">Frame Total</span><div class="wf-bar-wrap"><div class="wf-bar total" id="bar-total"></div></div><span class="wf-val" id="val-total">&mdash;</span></div>
            <div class="wf-row"><span class="wf-label">HTTP Round-Trip</span><div class="wf-bar-wrap"><div class="wf-bar http" id="bar-http"></div></div><span class="wf-val" id="val-http">&mdash;</span></div>
        </div>
    </div>

    <!-- 3-Minute Test + Live Frame Preview -->
    <div class="row">
        <div class="card">
            <div class="card-title">3-Minute Profiling Test</div>
            <div style="display:flex; gap:0.5rem; margin-bottom:0.6rem;">
                <button class="prof-btn primary" id="btn-start" onclick="startTest()">Start 3m Test</button>
                <button class="prof-btn danger" id="btn-stop" onclick="stopTest()" disabled>Stop</button>
            </div>
            <div id="test-status" style="font-size:0.8rem; color:#555; margin-bottom:0.3rem;">Idle</div>
            <div id="test-progress" style="font-size:0.75rem; color:#444; margin-bottom:0.5rem;"></div>
            <table class="results-table">
                <thead><tr><th>Stage</th><th>Min</th><th>Avg</th><th>Max</th><th>P95</th></tr></thead>
                <tbody id="results-body"><tr><td colspan="5" style="color:#333; text-align:center;">No test data</td></tr></tbody>
            </table>
        </div>
        <div class="card">
            <div class="card-title">Live Frame Preview</div>
            <img id="preview-frame" class="preview-img" alt="Preview">
            <div class="preview-overlay">
                Cursor: <span id="pv-cursor">(?, ?)</span> |
                Method: <span id="pv-method">--</span> |
                Age: <span id="pv-age">--</span>s
            </div>
        </div>
    </div>

    <!-- Historical Sparkline + Backend Metrics + OCR -->
    <div class="row triple">
        <div class="card">
            <div class="card-title">Frame Total (last 100 frames)</div>
            <canvas id="spark-canvas" class="spark-canvas"></canvas>
            <div style="display:flex; justify-content:space-between; font-size:0.65rem; color:#444; margin-top:4px;">
                <span id="spark-min">min: --</span>
                <span id="spark-avg">avg: --</span>
                <span id="spark-max">max: --</span>
            </div>
        </div>
        <div class="card">
            <div class="card-title">Backend Metrics</div>
            <div class="metric-row"><span class="metric-label">Tracking Method</span><span class="metric-val" id="m-method">--</span></div>
            <div class="metric-row"><span class="metric-label">Tracking Hz</span><span class="metric-val" id="m-hz">--</span></div>
            <div class="metric-row"><span class="metric-label">Confidence</span><span class="metric-val" id="m-conf">--</span></div>
            <div class="metric-row"><span class="metric-label">Misses</span><span class="metric-val" id="m-misses">--</span></div>
            <div class="metric-row"><span class="metric-label">Cursor Age</span><span class="metric-val" id="m-age">--</span></div>
            <div class="metric-row"><span class="metric-label">Validated</span><span class="metric-val" id="m-validated">--</span></div>
            <div class="metric-row"><span class="metric-label">Daemon Timestamp</span><span class="metric-val" id="m-ts" style="font-size:0.65rem; color:#555">--</span></div>
        </div>
        <div class="card">
            <div class="card-title">UTC Latency Measurement</div>
            <div style="font-size:0.7rem; color:#555; margin-bottom:0.6rem;">
                Open a UTC ms clock on the target machine, then measure.
                Claude Vision OCR reads the displayed time.
            </div>
            <div style="display:flex; gap:0.5rem; align-items:center; margin-bottom:0.5rem;">
                <button class="prof-btn primary" id="btn-measure" onclick="measureLatency()">Measure</button>
                <span id="measure-status" style="font-size:0.75rem; color:#555;"></span>
            </div>
            <div id="measure-result" style="font-size:0.75rem; white-space:pre-wrap;"></div>
        </div>
    </div>
</div>

<script>
var serverUtc = '';
var httpRt = 0;
var barMax = 50;
var history = [];
var MAX_HISTORY = 100;
var testRunning = false;
var testPollId = null;
var previewSeq = 0;

// UTC clock
function tickClock() {
    document.getElementById('browser-utc').textContent = new Date().toISOString().slice(11, 23);
    if (serverUtc) {
        var d = Date.now() - new Date(serverUtc).getTime();
        document.getElementById('utc-delta').textContent = d + 'ms';
    }
}
setInterval(tickClock, 100);

function setBar(name, ms) {
    var b = document.getElementById('bar-' + name);
    var v = document.getElementById('val-' + name);
    if (!b || !v) return;
    b.style.width = Math.min(100, (ms / barMax) * 100) + '%';
    v.textContent = ms.toFixed(1) + 'ms';
}

// Main poll
async function poll() {
    try {
        var t0 = Date.now();
        var r = await fetch('/api/profiler');
        httpRt = Date.now() - t0;
        var d = await r.json();
        serverUtc = d.utc;
        document.getElementById('server-utc').textContent = d.utc.slice(11, 23);
        document.getElementById('prof-status').textContent = 'Live';
        document.getElementById('prof-status').style.color = '#2ecc71';

        setBar('decode', d.pipeline.jpeg_decode_ms);
        setBar('track', d.pipeline.silhouette_ms);
        setBar('overlay', d.pipeline.overlay_draw_ms);
        setBar('encode', d.pipeline.jpeg_encode_ms);
        setBar('total', d.pipeline.frame_total_ms);
        setBar('http', httpRt);

        // Daemon stats in header bar
        document.getElementById('daemon-fps').textContent = d.daemon.fps.toFixed(1);
        document.getElementById('blob-count').textContent = d.daemon.blob_count;
        document.getElementById('frame-count').textContent = d.daemon.frame_count;

        // Backend metrics
        document.getElementById('m-method').textContent = d.tracking.method;
        document.getElementById('m-hz').textContent = d.tracking.hz;
        var conf = d.tracking.confidence;
        var confEl = document.getElementById('m-conf');
        confEl.textContent = (conf * 100).toFixed(1) + '%';
        confEl.className = 'metric-val ' + (conf > 0.7 ? 'good' : conf > 0.3 ? 'warn' : 'bad');
        document.getElementById('m-misses').textContent = d.tracking.misses;
        document.getElementById('m-age').textContent = d.tracking.cursor_age_s + 's';
        document.getElementById('m-validated').textContent = d.cursor.validated ? 'Yes' : 'No';
        document.getElementById('m-ts').textContent = d.daemon.timestamp_ns;

        // Cursor position for preview
        document.getElementById('pv-cursor').textContent = '(' + d.cursor.x + ', ' + d.cursor.y + ')';
        document.getElementById('pv-method').textContent = d.cursor.method;
        document.getElementById('pv-age').textContent = d.cursor.age_s;

        // Sparkline history
        history.push(d.pipeline.frame_total_ms);
        if (history.length > MAX_HISTORY) history.shift();
        drawSparkline();
    } catch (e) {
        document.getElementById('prof-status').textContent = 'Disconnected';
        document.getElementById('prof-status').style.color = '#e74c3c';
    }
}
setInterval(poll, 200);

// Preview frame at 2Hz
function updatePreview() {
    document.getElementById('preview-frame').src = '/api/frame?' + previewSeq++;
}
setInterval(updatePreview, 500);

// Sparkline
function drawSparkline() {
    var canvas = document.getElementById('spark-canvas');
    var ctx = canvas.getContext('2d');
    var W = canvas.offsetWidth, H = canvas.offsetHeight;
    canvas.width = W * (window.devicePixelRatio || 1);
    canvas.height = H * (window.devicePixelRatio || 1);
    ctx.scale(window.devicePixelRatio || 1, window.devicePixelRatio || 1);

    ctx.clearRect(0, 0, W, H);
    if (history.length < 2) return;

    var min = Math.min.apply(null, history);
    var max = Math.max.apply(null, history);
    var avg = history.reduce(function(a,b){return a+b;}, 0) / history.length;
    var range = Math.max(max - min, 1);

    document.getElementById('spark-min').textContent = 'min: ' + min.toFixed(1) + 'ms';
    document.getElementById('spark-avg').textContent = 'avg: ' + avg.toFixed(1) + 'ms';
    document.getElementById('spark-max').textContent = 'max: ' + max.toFixed(1) + 'ms';

    // Average line
    var avgY = H - ((avg - min) / range) * (H - 10) - 5;
    ctx.strokeStyle = '#333';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(0, avgY); ctx.lineTo(W, avgY); ctx.stroke();
    ctx.setLineDash([]);

    // Data line
    ctx.strokeStyle = '#e74c3c';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (var i = 0; i < history.length; i++) {
        var x = (i / (MAX_HISTORY - 1)) * W;
        var y = H - ((history[i] - min) / range) * (H - 10) - 5;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fill under curve
    ctx.lineTo((history.length - 1) / (MAX_HISTORY - 1) * W, H);
    ctx.lineTo(0, H);
    ctx.closePath();
    ctx.fillStyle = 'rgba(231, 76, 60, 0.1)';
    ctx.fill();
}

// 3-minute test
var STAGE_NAMES = {
    jpeg_decode_ms: 'JPEG Decode', silhouette_ms: 'Sil. Track',
    overlay_draw_ms: 'Overlay Draw', jpeg_encode_ms: 'JPEG Encode',
    frame_total_ms: 'Frame Total'
};

async function startTest() {
    await fetch('/api/profiler/start', {method:'POST'});
    testRunning = true;
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-stop').disabled = false;
    document.getElementById('test-status').textContent = 'Running...';
    document.getElementById('test-status').style.color = '#2ecc71';
    testPollId = setInterval(pollResults, 1000);
}

async function stopTest() {
    await fetch('/api/profiler/stop', {method:'POST'});
    testRunning = false;
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-stop').disabled = true;
    document.getElementById('test-status').textContent = 'Complete';
    document.getElementById('test-status').style.color = '#f1c40f';
    if (testPollId) { clearInterval(testPollId); testPollId = null; }
    pollResults();
}

async function pollResults() {
    try {
        var r = await fetch('/api/profiler/results');
        var d = await r.json();
        if (d.status === 'no_data') return;
        document.getElementById('test-progress').textContent =
            'Samples: ' + d.samples + ' | ' + d.elapsed_s + 's / ' + d.duration_s + 's';
        if (d.status === 'complete' && testRunning) {
            testRunning = false;
            document.getElementById('btn-start').disabled = false;
            document.getElementById('btn-stop').disabled = true;
            document.getElementById('test-status').textContent = 'Complete';
            document.getElementById('test-status').style.color = '#f1c40f';
            if (testPollId) { clearInterval(testPollId); testPollId = null; }
        }
        if (d.stages) {
            var tbody = document.getElementById('results-body');
            while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
            var keys = Object.keys(STAGE_NAMES);
            for (var i = 0; i < keys.length; i++) {
                if (d.stages[keys[i]]) {
                    var s = d.stages[keys[i]];
                    var tr = document.createElement('tr');
                    [STAGE_NAMES[keys[i]], s.min.toFixed(1), s.avg.toFixed(1), s.max.toFixed(1), s.p95.toFixed(1)].forEach(function(v) {
                        var td = document.createElement('td');
                        td.textContent = v;
                        tr.appendChild(td);
                    });
                    tbody.appendChild(tr);
                }
            }
        }
    } catch(e) {}
}

// OCR Latency measurement
async function measureLatency() {
    var btn = document.getElementById('btn-measure');
    var st = document.getElementById('measure-status');
    var res = document.getElementById('measure-result');
    btn.disabled = true;
    st.textContent = 'Capturing + OCR...';
    st.style.color = '#f1c40f';
    res.textContent = '';
    try {
        var r = await fetch('/api/profiler/measure_latency', {method:'POST'});
        var d = await r.json();
        btn.disabled = false;
        if (d.error) { st.textContent = 'Error'; st.style.color = '#e74c3c'; res.textContent = d.error; return; }
        st.textContent = 'Done'; st.style.color = '#2ecc71';
        var lines = [];
        lines.push('Capture: ' + d.capture_ms.toFixed(1) + 'ms');
        lines.push('Inference: ' + d.inference_ms.toFixed(0) + 'ms');
        lines.push('Local UTC: ' + d.capture_utc);
        lines.push('Screen UTC: ' + (d.displayed_utc || 'not detected'));
        if (d.delta_ms !== null && d.delta_ms !== undefined) lines.push('Delta: ' + d.delta_ms + 'ms');
        if (d.ocr_result && d.ocr_result.notes) lines.push('Notes: ' + d.ocr_result.notes);
        res.textContent = lines.join('\n');
    } catch(e) { btn.disabled = false; st.textContent = 'Failed'; st.style.color = '#e74c3c'; }
}

poll();
</script>
</body>
</html>"""


# ── Concepts HTML ──────────────────────────────────────────────────

_CONCEPTS_PATH = Path(__file__).parent.parent.parent / "docs" / "STAR_TREK_COMPUTER_CONCEPTS.md"
_CONCEPTS_TEXT = ""
if _CONCEPTS_PATH.exists():
    _raw = _CONCEPTS_PATH.read_text()
    _CONCEPTS_TEXT = _raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
else:
    _CONCEPTS_TEXT = "Concepts file not found at " + str(_CONCEPTS_PATH)

CONCEPTS_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Star Trek Computer Concepts</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'JetBrains Mono', 'Fira Code', monospace; background: #0a0a0f; color: #c8c8d0; line-height: 1.6; }
.header { background: #101018; padding: 0.6rem 1.5rem; border-bottom: 2px solid #ff4444;
    display: flex; justify-content: space-between; align-items: center; }
.header h1 { color: #ff4444; font-size: 1.1rem; letter-spacing: 1px; }
.header a { color: #888; text-decoration: none; font-size: 0.75rem; margin-left: 1rem; }
.header a:hover { color: #ff4444; }
.content { max-width: 960px; margin: 2rem auto; padding: 0 2rem; }
pre { white-space: pre-wrap; word-wrap: break-word; font-size: 0.82rem; line-height: 1.7; }
</style>
</head>
<body>
<div class="header">
    <div style="display:flex; align-items:center;">
        <h1>STAR TREK COMPUTER &mdash; CONCEPTS</h1>
        <a href="/">Dashboard</a>
        <a href="/profiler">Profiler</a>
    </div>
</div>
<div class="content"><pre>""" + _CONCEPTS_TEXT + r"""</pre></div>
</body>
</html>"""


# ── Startup ─────────────────────────────────────────────────────────

def init_hardware():
    global mouse, sensor, tracker, recognizer, collector, daemon_client, claude_detector, sil_tracker, cmd_movement, yolo_detector

    print("Initializing hardware...")
    mouse = ESP32Mouse('/dev/ttyUSB0', screen_width=1920, screen_height=1080)

    # Check for C cursor daemon before opening V4L2
    daemon_client = DaemonClient()
    if daemon_client.is_running():
        sensor = None  # Daemon owns /dev/video0
        print("  C daemon detected — sensor owned by daemon")
    else:
        sensor = HDMISensor('/dev/video0')

    mouse.set_position(6, 15)
    print("  Mouse position: " + str(mouse.get_position()))
    print("  Mouse status: " + mouse.status())

    mouse.start_heartbeat(interval=0.1)
    print("  Heartbeat started (100ms)")

    # Anti-sleep jitter is handled by jitter_loop() — not ESP32Mouse's built-in thread.
    # This allows us to piggyback correlation probes on jitter when confidence is low.
    print("  Anti-sleep jitter started (30s)")  # managed by jitter_loop()

    tracker = CursorTracker(mouse, sensor, check_interval=999.0, stale_threshold=9999.0)
    tracker.set_position(6, 15, method="manual")
    # Tracker background thread disabled — motion loop owns the capture device
    # Use /api/probe to manually detect cursor when needed
    print("  Cursor tracker initialized (manual probe via /api/probe)")

    # CNN cursor recognition pipeline
    recognizer = CursorRecognizer()
    collector = CursorSampleCollector()
    state.cnn_model_version = recognizer.model_version
    state.sample_count_pos = collector.positive_count
    state.sample_count_neg = collector.negative_count
    print("  Cursor recognizer: " + recognizer.model_version
          + " (" + str(collector.total_samples) + " samples)")

    # Phase 2: Silhouette tracker (fast ROI-based cursor following)
    sil_tracker = SilhouetteTracker(recognizer)
    print("  Silhouette tracker initialized (ROI: " + str(sil_tracker.roi_size) + "px)")

    # Phase 3: Commanded movement (unified jitter + probe + Lissajous)
    cmd_movement = CommandedMovement(
        mouse, sil_tracker, sensor=sensor, daemon_client=daemon_client)
    print("  Commanded movement initialized (jitter→probe→lissajous)")

    # YOLO full-screen cursor detector (GPU-accelerated)
    yolo_detector = YOLOCursorDetector(conf_threshold=0.3)
    if yolo_detector.load():
        print("  YOLO cursor detector initialized ("
              + str(yolo_detector.model_path) + ")")
    else:
        print("  YOLO cursor detector: no model loaded (will use pretrained)")
        yolo_detector = None

    # Claude Vision cursor detector (uses subscription, no API key)
    claude_detector = ClaudeVisionCursorDetector(model="haiku")
    print("  Claude Vision detector initialized (haiku)")

    threading.Thread(target=motion_detection_loop, daemon=True).start()
    print("  Motion detection loop started")

    threading.Thread(target=jitter_loop, daemon=True).start()
    threading.Thread(target=pi_keyboard_monitor_loop, daemon=True).start()
    threading.Thread(target=cursor_validation_loop, daemon=True).start()
    print("  Cursor validation loop started (30s cycle)")

    print("\nDashboard: http://localhost:8766")


if __name__ == "__main__":
    import uvicorn
    init_hardware()
    uvicorn.run(app, host="0.0.0.0", port=8766)
