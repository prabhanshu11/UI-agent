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
from fastapi.responses import HTMLResponse, Response, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import httpx
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
from src.hardware.silhouette_tracker import SilhouetteTracker, CandidateBlob
from src.hardware.commanded_movement import CommandedMovement
from src.hardware.yolo_detector import YOLOCursorDetector
from src.hardware.experience_logger import ExperienceLogger
from src.agents.storyline_logger import StorylineLogger

def utc_now_ms() -> str:
    """UTC timestamp with millisecond precision, e.g. '2026-02-20T14:30:05.123Z'."""
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'


app = FastAPI(title="KVM Control Dashboard")

# ── Static files ────────────────────────────────────────────────────
_STATIC_DIR = Path(__file__).parent / "static"
_PAGES_DIR = _STATIC_DIR / "pages"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


def _serve_page(name: str) -> HTMLResponse:
    """Serve an HTML page from static/pages/."""
    return HTMLResponse((_PAGES_DIR / name).read_text())


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
    # Constant velocity — precise px/s control via HID physics
    "V0":     {"desc": "Stationary",       "velocity_px_s": 0,    "duration_s": 3.0},
    "V100":   {"desc": "Slow 100px/s",     "velocity_px_s": 100,  "duration_s": 3.0},
    "V300":   {"desc": "Medium 300px/s",    "velocity_px_s": 300,  "duration_s": 2.0},
    "V500":   {"desc": "Medium 500px/s",    "velocity_px_s": 500,  "duration_s": 2.0},
    "V1K":    {"desc": "Fast 1000px/s",     "velocity_px_s": 1000, "duration_s": 1.0},
    "V2K":    {"desc": "Very fast 2Kpx/s",  "velocity_px_s": 2000, "duration_s": 0.5},
    "V3K":    {"desc": "Extreme 3Kpx/s",    "velocity_px_s": 3000, "duration_s": 0.3},
    "V5K":    {"desc": "Stress 5Kpx/s",     "velocity_px_s": 5000, "duration_s": 0.2},
    # Acceleration ramps
    "A_UP":   {"desc": "0→2000 in 0.5s",    "velocity_px_s": 0,    "accel_px_s2": 4000,  "duration_s": 0.5},
    "A_DOWN": {"desc": "2000→0 in 0.5s",    "velocity_px_s": 2000, "accel_px_s2": -4000, "duration_s": 0.5},
    "A_BURST":{"desc": "Burst 2K for 0.2s", "velocity_px_s": 2000, "duration_s": 0.2},
    # Replay actual loss events
    "REPLAY": {"desc": "Replay loss event",  "replay": True},
    # Diagonal
    "D500":   {"desc": "Diagonal 500px/s",  "velocity_px_s": 500,  "duration_s": 2.0, "diagonal": True},
    "D2K":    {"desc": "Diagonal 2Kpx/s",   "velocity_px_s": 2000, "duration_s": 0.5, "diagonal": True},
    # Jitter (micro-shake equivalent)
    "JITTER": {"desc": "±3px jitter",       "velocity_px_s": 0,    "jitter_px": 3,    "duration_s": 3.0},
}


def _generate_hid_reports(velocity_px_s: float, duration_s: float,
                          direction: tuple = (1, 0), accel: float = 0,
                          jitter_px: int = 0) -> list[tuple[int, int]]:
    """Generate list of (dx, dy) HID reports for target velocity.

    At 50Hz (20ms reports): dx_per_report = velocity * 0.02.
    Tracks fractional remainder for sub-pixel precision.
    HID values clamped to [-127, 127] per report.
    """
    import random
    reports = []
    dt = 0.02  # 20ms per report (50Hz USB polling)
    t = 0.0
    rem_x, rem_y = 0.0, 0.0

    while t < duration_s:
        v = velocity_px_s + accel * t
        if v < 0:
            v = 0

        if jitter_px > 0 and v == 0:
            # Random ±jitter displacement
            raw_x = random.randint(-jitter_px, jitter_px)
            raw_y = random.randint(-jitter_px, jitter_px)
        else:
            disp = v * dt
            raw_x = disp * direction[0] + rem_x
            raw_y = disp * direction[1] + rem_y

        dx = int(max(-127, min(127, round(raw_x))))
        dy = int(max(-127, min(127, round(raw_y))))
        rem_x = raw_x - dx
        rem_y = raw_y - dy
        reports.append((dx, dy))
        t += dt

    return reports


def _replay_loss_event_reports() -> list[tuple[int, int]]:
    """Extract velocity profile from the latest loss event and generate HID reports.

    Reads timeline.jsonl from the most recent loss event directory,
    extracts position deltas around the trigger point, and converts
    them to HID reports that reproduce the movement pattern.
    """
    if not _LOSS_EVENT_DIR.exists():
        return []

    # Find latest event directory
    event_dirs = sorted(_LOSS_EVENT_DIR.iterdir(), reverse=True)
    event_dirs = [d for d in event_dirs if d.is_dir() and (d / "timeline.jsonl").exists()]
    if not event_dirs:
        return []

    timeline_path = event_dirs[0] / "timeline.jsonl"
    entries = []
    with open(timeline_path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if len(entries) < 2:
        return []

    # Extract position deltas and timing around the trigger (t_offset_s near 0)
    reports = []
    for i in range(1, len(entries)):
        e0, e1 = entries[i - 1], entries[i]
        dt = e1.get("t_offset_s", 0) - e0.get("t_offset_s", 0)
        if dt <= 0:
            dt = 0.2  # fallback 5Hz
        ddx = e1.get("cx", 0) - e0.get("cx", 0)
        ddy = e1.get("cy", 0) - e0.get("cy", 0)
        # Convert to HID reports at 50Hz within this time window
        n_reports = max(1, int(dt / 0.02))
        per_dx = ddx / n_reports
        per_dy = ddy / n_reports
        rx, ry = 0.0, 0.0
        for _ in range(n_reports):
            rx += per_dx
            ry += per_dy
            hdx = int(max(-127, min(127, round(rx))))
            hdy = int(max(-127, min(127, round(ry))))
            rx -= hdx
            ry -= hdy
            reports.append((hdx, hdy))

    return reports


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
_LOSS_HISTORY_SIZE = 8  # ~1.6s at 5Hz
_loss_history: deque = deque(maxlen=_LOSS_HISTORY_SIZE)
_loss_events_log: list[dict] = []  # in-memory index of captured events

# Per-frame state log — compact JSONL for post-hoc debugging
_STATE_LOG_DIR = Path(__file__).parent.parent.parent / "data"
_STATE_LOG_PATH = _STATE_LOG_DIR / "state_log.jsonl"
_STATE_LOG_MAX_BYTES = 50 * 1024 * 1024  # 50MB rotation
_state_log_bytes = 0  # Track size to avoid stat() every frame


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

# CNN model comparison for velocity tests (loaded at startup if models exist)
_vtest_recognizers: dict[str, CursorRecognizer] = {}  # version -> recognizer

# Sustained low-confidence tracker for jitter-based re-acquisition
_low_confidence_since: float = 0.0  # monotonic time when CNN confidence first went <0.7
_vision_lockout_until: float = 0.0  # monotonic time until which motion blobs can't override position

# CNN-validated position: where CNN last confirmed cursor presence.
# cursor_validated auto-revokes when tracked position drifts >100px from this.
_cnn_validated_x: int = 0
_cnn_validated_y: int = 0
_CNN_DRIFT_INVALIDATE_PX = 100  # auto-invalidate threshold

# Urgent CNN recheck: when cursor_validated becomes False unexpectedly,
# reduce CNN interval to 2s until re-validated or re-acquired.
_cnn_urgent_recheck: bool = False
_stale_recovery_attempts: int = 0  # consecutive probe cycles where CNN still fails
_VISION_LOCKOUT_S: float = 30.0     # how long Vision position is protected from motion override
_YOLO_LOCKOUT_S: float = 5.0       # how long a high-confidence YOLO position is protected
_yolo_lockout_until: float = 0.0   # monotonic time until which YOLO position is protected
_CNN_LOCKOUT_S: float = 10.0       # protect CNN-validated position from motion overrides
_cnn_lockout_until: float = 0.0    # monotonic time until which CNN position is protected

# Passive mode: observe-only, no mouse movements at all.
# Silhouette tracking, CNN, YOLO still run for data capture, but
# no jitter, probes, calibrate_to_corner, or any HID mouse reports.
_passive_mode: bool = False

# Mouse passthrough: pipe user's mouse through ESP32 pebble
_passthrough_active: bool = False
_passthrough_dx: int = 0  # Accumulated delta (batched by frontend at 50Hz)
_passthrough_dy: int = 0
_passthrough_lock = threading.Lock()

# Passthrough experience recording
_passthrough_session: Optional[dict] = None  # Active recording session
_PT_SCREENSHOT_INTERVAL_S = 5.0
_PT_CNN_SAMPLE_INTERVAL_S = 10.0

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
            # BUT exempt blobs near the VALIDATED cursor position — the cursor
            # itself creates persistent motion during continuous movement.
            # Radius 200px (was 400) + require CNN validation to prevent
            # noise blobs (e.g. UTC clock milliseconds) from self-reinforcing
            # as the "cursor position".
            filtered_blobs = []
            for b in cursor_blobs:
                gx = min(_NOISE_GRID_W - 1, b.centroid[0] // _NOISE_CELL_PX)
                gy = min(_NOISE_GRID_H - 1, b.centroid[1] // _NOISE_CELL_PX)
                if _noise_grid[gy][gx] >= _NOISE_THRESHOLD:
                    # Only exempt if cursor position is CNN-validated and close
                    if (pos and (now - pos.timestamp < 5) and
                            state.cursor_validated):
                        blob_dist = ((b.centroid[0] - pos.x) ** 2 +
                                     (b.centroid[1] - pos.y) ** 2) ** 0.5
                        if blob_dist < 200:
                            filtered_blobs.append(b)
                            continue
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
                    # Velocity-scaled distance threshold: at high speed,
                    # cursor moves further between frames
                    _vel_est = 0.0
                    if len(velocity_history) >= 2:
                        _recent = velocity_history[-3:]
                        _dt = _recent[-1][2] - _recent[0][2]
                        if _dt > 0:
                            _avg_d = sum(
                                (v[0]**2 + v[1]**2)**0.5 for v in _recent
                            ) / len(_recent)
                            _vel_est = _avg_d / (_dt / len(_recent))
                    blob_dist_limit = min(800, 300 + _vel_est * 0.3)
                    if dist < blob_dist_limit:
                        # Silhouette confidence gate: if tracker is confident,
                        # reject blobs too far from silhouette position
                        accept = True
                        sil_gate_dist = min(300, 100 + _vel_est * 0.15)
                        if (sil_tracker and state.silhouette_confidence > 0.6
                                and state.silhouette_latency_ms < 50):
                            sil_dist = ((best.centroid[0] - state.cursor_x) ** 2 +
                                        (best.centroid[1] - state.cursor_y) ** 2) ** 0.5
                            if sil_dist > sil_gate_dist:
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

                        # CNN lockout: protect CNN-validated position from
                        # random blobs — only accept within 80px
                        if accept and now < _cnn_lockout_until:
                            blob_dist_from_cnn = (
                                (best.centroid[0] - _cnn_validated_x) ** 2 +
                                (best.centroid[1] - _cnn_validated_y) ** 2) ** 0.5
                            if blob_dist_from_cnn > 80:
                                accept = False

                        # Urgent recheck lockout: don't let random blobs
                        # refresh position when CNN has flagged it wrong.
                        # Let position go stale so calibrate_to_corner fires.
                        if accept and _cnn_urgent_recheck:
                            accept = False

                        if accept:
                            primary_blob = best
                            bx, by = best.centroid
                            # Update velocity history
                            dx = bx - pos.x
                            dy = by - pos.y
                            velocity_history.append((dx, dy, now))
                            if len(velocity_history) > MAX_VELOCITY_SAMPLES * 2:
                                velocity_history = velocity_history[-MAX_VELOCITY_SAMPLES:]
                            # Feed velocity to silhouette tracker for ROI scaling
                            if sil_tracker and len(velocity_history) >= 2:
                                recent = velocity_history[-3:]
                                dt_sum = recent[-1][2] - recent[0][2]
                                if dt_sum > 0:
                                    avg_dx = sum(v[0] for v in recent) / len(recent)
                                    avg_dy = sum(v[1] for v in recent) / len(recent)
                                    vel = ((avg_dx ** 2 + avg_dy ** 2) ** 0.5) / (dt_sum / len(recent))
                                    sil_tracker.set_velocity(vel)
                            if tracker:
                                tracker.set_position(bx, by, method="motion_track")
                            pos = tracker.position
                            if sil_tracker:
                                sil_tracker.reset()
                else:
                    # No known position — use Vision as anchor if recent,
                    # otherwise pick blob closest to typical cursor size
                    if _cnn_urgent_recheck:
                        pass  # Don't pick random blobs during recovery
                    elif now < _vision_lockout_until:
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

            # ── ESP32 dead-reckoning prior: velocity-gated fallback.
            # Only use ESP32 estimate when:
            #   1. Velocity test is running with commanded velocity >= 50 px/s
            #      (at V0/stationary, silhouette is more accurate than ESP32)
            #   2. No primary blob found (motion detector can't see cursor)
            #   3. Position is stale (>0.2s since last update)
            # This is the middle ground: V0 stays untouched (silhouette),
            # V100+ gets ESP32 help when silhouette falls behind.
            if mouse and _vtest_running and primary_blob is None:
                commanded_vel = _vtest_progress.get("velocity_px_s", 0)
                if commanded_vel >= 50:
                    esp_x = mouse.estimated_x
                    esp_y = mouse.estimated_y
                    pos = tracker.position if tracker else None
                    pos_stale = pos is None or (now - pos.timestamp > 0.2)
                    if pos_stale:
                        if tracker:
                            tracker.set_position(esp_x, esp_y, method="esp32_dead_reckoning")
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

                        # ── YOLO detection (crop pipeline + full-frame fallback) ──
                        # YOLO expects BGR (OpenCV native), not RGB
                        if yolo_detector and yolo_detector.loaded:
                            best_yolo = None
                            pos_now = tracker.position if tracker else None

                            # Crop pipeline: silhouette finds candidates → YOLO validates crops
                            if sil_tracker and pos_now:
                                candidates = sil_tracker.get_candidates(
                                    frame_rgb, pos_now.x, pos_now.y)
                                for cand in candidates[:5]:  # max 5 crops
                                    crop_dets = yolo_detector.detect_crop(
                                        frame, cand.bbox)
                                    if crop_dets:
                                        best_yolo = crop_dets[0]
                                        break

                            # Full-frame fallback: discovery mode when no position
                            # or crop pipeline found nothing
                            if best_yolo is None:
                                yolo_dets = yolo_detector.detect(frame)
                                if yolo_dets:
                                    if pos_now and len(yolo_dets) > 1:
                                        yolo_dets.sort(key=lambda d: (
                                            (d.cx - pos_now.x) ** 2 +
                                            (d.cy - pos_now.y) ** 2))
                                    best_yolo = yolo_dets[0]

                            if best_yolo:
                                state.yolo_active = True
                                state.yolo_x = int(best_yolo.cx)
                                state.yolo_y = int(best_yolo.cy)
                                state.yolo_confidence = float(best_yolo.confidence)
                                state.yolo_bbox = (int(best_yolo.x1), int(best_yolo.y1),
                                                   int(best_yolo.x2), int(best_yolo.y2))
                                state.yolo_inference_ms = float(best_yolo.inference_ms)
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
                                elif pos and best_yolo.confidence > 0.7:
                                    drift = ((best_yolo.cx - pos.x) ** 2 +
                                             (best_yolo.cy - pos.y) ** 2) ** 0.5
                                    if drift > 80:
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
                                yolo_close = (
                                    state.yolo_active
                                    and (now - state.yolo_timestamp) < _YOLO_LOCKOUT_S
                                    and ((track_result.x - state.yolo_x) ** 2 +
                                         (track_result.y - state.yolo_y) ** 2) ** 0.5 < 60
                                )
                                sil_trusted = (
                                    state.cursor_validated or
                                    state.cnn_confidence > 0.8 or      # live CNN high = trust
                                    track_result.method == "motion_roi" or  # motion always trusted
                                    track_result.method == "jitter_confirm" or  # jitter-proven
                                    vision_close or
                                    yolo_close                          # YOLO agrees
                                )
                                # Extra guard: template-only result that jumped far
                                # from current position is likely a spurious match.
                                # Even if cursor_validated, the CNN validated at
                                # the OLD position, not at this new far-away spot.
                                if (sil_trusted and
                                        track_result.method == "template" and pos):
                                    sil_jump = ((track_result.x - pos.x) ** 2 +
                                                (track_result.y - pos.y) ** 2) ** 0.5
                                    if sil_jump > 150:
                                        sil_trusted = False
                                if sil_trusted and tracker:
                                    tracker.set_position(
                                        track_result.x, track_result.y,
                                        method="sil_" + track_result.method)
                                    pos = tracker.position
                                elif tracker:
                                    # Gated but found — only refresh timestamp if
                                    # CNN has validated this position. When not
                                    # validated, let position go stale so recovery
                                    # kicks in (calibrate_to_corner + re-scan).
                                    if state.cursor_validated:
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
                                    # But skip if urgent recheck — let CNN loop
                                    # handle recovery via calibrate_to_corner
                                    if not _cnn_urgent_recheck:
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

            # ── YOLO detection (crop pipeline + full-frame fallback) ──
            if yolo_detector and yolo_detector.loaded:
                best_yolo = None

                # Crop pipeline: silhouette candidates → YOLO crop validation
                if sil_tracker and pos:
                    candidates = sil_tracker.get_candidates(frame, pos.x, pos.y)
                    for cand in candidates[:5]:
                        crop_dets = yolo_detector.detect_crop(frame, cand.bbox)
                        if crop_dets:
                            best_yolo = crop_dets[0]
                            break

                # Full-frame fallback for discovery
                if best_yolo is None:
                    yolo_dets = yolo_detector.detect(frame)
                    if yolo_dets:
                        if pos and len(yolo_dets) > 1:
                            yolo_dets.sort(key=lambda d: (
                                (d.cx - pos.x) ** 2 + (d.cy - pos.y) ** 2))
                        best_yolo = yolo_dets[0]

                if best_yolo:
                    state.yolo_active = True
                    state.yolo_x = int(best_yolo.cx)
                    state.yolo_y = int(best_yolo.cy)
                    state.yolo_confidence = float(best_yolo.confidence)
                    state.yolo_bbox = (int(best_yolo.x1), int(best_yolo.y1),
                                       int(best_yolo.x2), int(best_yolo.y2))
                    state.yolo_inference_ms = float(best_yolo.inference_ms)
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

                        if best and not _cnn_urgent_recheck:
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

def _log_state_line(entry: dict):
    """Append compact per-frame state to rotating JSONL.

    ~200 bytes/line x 5Hz = ~1KB/s = ~3.5MB/hour.
    50MB rotation = ~14 hours of data. Enough for any debugging session.
    """
    global _state_log_bytes
    compact = {
        "t": entry.get("utc", ""),
        "cx": entry.get("cx"), "cy": entry.get("cy"),
        "m": entry.get("method", ""),
        "cnn": round(entry.get("cnn_conf", 0), 3),
        "sil": round(entry.get("sil_conf", 0), 3),
        "sm": entry.get("sil_method", ""),
        "v": entry.get("validated", False),
        "ya": entry.get("yolo_active", False),
        "yc": round(entry.get("yolo_conf", 0), 3) if entry.get("yolo_active") else 0,
        "bc": entry.get("blob_count", 0),
    }
    try:
        line = json.dumps(compact, separators=(",", ":")) + "\n"
        line_bytes = len(line.encode())

        # Rotate if over limit
        if _state_log_bytes > _STATE_LOG_MAX_BYTES:
            rotated = _STATE_LOG_PATH.with_suffix(".jsonl.1")
            try:
                _STATE_LOG_PATH.rename(rotated)
            except OSError:
                pass
            _state_log_bytes = 0

        with open(_STATE_LOG_PATH, "a") as f:
            f.write(line)
        _state_log_bytes += line_bytes
    except Exception:
        pass  # Never crash the motion loop for logging


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

    # Append compact state to persistent JSONL log
    _log_state_line(entry)

    # Auto-invalidate CNN validation if position drifted
    _check_cnn_drift()

    # Check for loss event trigger
    _check_loss_trigger(now)


def _check_cnn_drift():
    """Auto-invalidate cursor_validated if position drifted from CNN-validated spot.

    Called after any position update. Uses velocity-aware threshold:
    - At 0 px/s: 100px (tight, catches teleports)
    - At 1000 px/s: 400px cap (allows fast cursor movement)

    Also provides fast-path re-validation: if live CNN confidence is >0.8
    and position hasn't moved far, re-validate immediately instead of
    waiting for the 30s CNN validation loop.
    """
    global _cnn_urgent_recheck, _cnn_lockout_until
    if not state.cursor_validated:
        # Fast-path re-validation: CNN confidence is high but validated=False
        # (e.g. after YOLO moved position). Re-validate if close to old spot.
        if (state.cnn_confidence > 0.8
                and _cnn_validated_x > 0 and _cnn_validated_y > 0):
            reval_drift = ((state.cursor_x - _cnn_validated_x) ** 2 +
                           (state.cursor_y - _cnn_validated_y) ** 2) ** 0.5
            if reval_drift < 50:
                state.cursor_validated = True
                _cnn_urgent_recheck = False
        return
    drift = ((state.cursor_x - _cnn_validated_x) ** 2 +
             (state.cursor_y - _cnn_validated_y) ** 2) ** 0.5
    # Velocity-aware threshold: allow more drift when cursor is moving fast
    velocity = _estimate_velocity()
    drift_threshold = max(100, min(400, 100 + velocity * 0.3))
    if drift > drift_threshold:
        state.cursor_validated = False
        _cnn_urgent_recheck = True
        _cnn_lockout_until = 0.0  # Clear CNN lockout too
        # Clear vision lockout — the lockout position is likely wrong too
        global _vision_lockout_until
        _vision_lockout_until = 0.0


def _estimate_velocity() -> float:
    """Estimate cursor velocity (px/s) from last 3 frames in loss history."""
    if len(_loss_history) < 2:
        return 0.0
    # Use up to last 3 frames for smoothing
    recent = list(_loss_history)[-3:]
    velocities = []
    for i in range(1, len(recent)):
        dt = recent[i]["t"] - recent[i - 1]["t"]
        if dt <= 0:
            continue
        ddx = recent[i]["x"] - recent[i - 1]["x"]
        ddy = recent[i]["y"] - recent[i - 1]["y"]
        dist = (ddx * ddx + ddy * ddy) ** 0.5
        velocities.append(dist / dt)
    return sum(velocities) / len(velocities) if velocities else 0.0


def _check_loss_trigger(now: float):
    """Detect confidence transitions that indicate cursor loss.

    Uses multi-frame consensus to avoid false triggers:
    - Position jump: velocity-proportional threshold (400 + vel*0.5, capped at 1200)
    - CNN drop: 3 of last 4 frames below 0.4, previously above 0.65
    - Silhouette drop: 3 of last 4 frames below 0.25
    - Validation lost: 2 consecutive unvalidated frames
    """
    global _loss_event_active, _loss_event_cooldown

    if _loss_event_active or now < _loss_event_cooldown:
        # Still append to history for velocity tracking even during cooldown
        _loss_history.append({
            "t": now, "x": state.cursor_x, "y": state.cursor_y,
            "cnn": state.cnn_confidence, "sil": state.silhouette_confidence,
            "validated": state.cursor_validated, "cnn_ms": state.cnn_inference_ms,
        })
        return

    # Append current frame to history
    _loss_history.append({
        "t": now, "x": state.cursor_x, "y": state.cursor_y,
        "cnn": state.cnn_confidence, "sil": state.silhouette_confidence,
        "validated": state.cursor_validated, "cnn_ms": state.cnn_inference_ms,
    })

    if len(_loss_history) < 2:
        return

    trigger = None
    hist = list(_loss_history)
    prev = hist[-2]

    # ── Position jump — velocity-proportional threshold ──
    vel = _estimate_velocity()
    jump_threshold = min(1200, 400 + vel * 0.5)
    ddx = state.cursor_x - prev["x"]
    ddy = state.cursor_y - prev["y"]
    jump = (ddx * ddx + ddy * ddy) ** 0.5
    if jump > jump_threshold and (prev["x"] > 0 or prev["y"] > 0):
        trigger = f"position_jump_{int(jump)}px_thr{int(jump_threshold)}"

    # ── CNN drop — 3-of-4 consensus ──
    if not trigger and len(hist) >= 5:
        last4 = hist[-4:]
        # Only count frames where CNN was actually running
        low_count = sum(1 for h in last4 if h["cnn"] < 0.4 and h["cnn_ms"] > 0)
        # Check that CNN was previously high (before the window)
        before_window = [h for h in hist[:-4] if h["cnn_ms"] > 0]
        was_high = any(h["cnn"] > 0.65 for h in before_window) if before_window else False
        if low_count >= 3 and was_high:
            trigger = "cnn_drop"

    # ── Silhouette drop — 3-of-4 consensus ──
    if not trigger and len(hist) >= 5:
        last4 = hist[-4:]
        low_sil = sum(1 for h in last4 if h["sil"] < 0.25)
        before_window = hist[:-4]
        was_sil_high = any(h["sil"] > 0.7 for h in before_window) if before_window else False
        if low_sil >= 3 and was_sil_high:
            trigger = "silhouette_drop"

    # ── Validation lost — 2-frame confirmation ──
    if not trigger and len(hist) >= 3:
        if (hist[-3]["validated"] and
                not hist[-2]["validated"] and
                not hist[-1]["validated"]):
            trigger = "validation_lost"

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

        # Pause jitter/probes during passthrough — user controls the mouse.
        # Heartbeat stays on (keeps BLE alive), but jitter would cause
        # visible micro-jerks while user is actively moving the cursor.
        if _passthrough_active:
            state.jitter_count += 1
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

        # Passive mode: skip ALL mouse movements, only observe
        if _passive_mode:
            state.jitter_count += 1
            continue

        # When urgent recheck is active, CNN loop owns recovery.
        # Jitter probes find random blobs and re-lock to wrong positions,
        # resetting counters and preventing calibrate_to_corner.
        if _cnn_urgent_recheck:
            # Just do anti-sleep jitter, no probing
            try:
                px = config.jitter_pixels
                if sil_tracker:
                    sil_tracker.expect_motion(amplitude_px=px)
                mouse._send_raw(px, 0)
                time.sleep(0.15)  # Wait for motion to appear in frame
                mouse._send_raw(-px, 0)
            except Exception:
                pass
            state.jitter_count += 1
            continue

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
                    if sil_tracker:
                        sil_tracker.expect_motion(amplitude_px=px)
                    mouse._send_raw(px, 0)
                    time.sleep(0.15)
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
            # Normal anti-sleep jitter: ±Npx — ALSO a discovery opportunity.
            # Capture before/after, full-frame diff → only the cursor moves.
            _jitter_discover(use_daemon)

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


def _jitter_discover(use_daemon):
    """Every jitter is a discovery opportunity.

    Instead of blindly sending ±3px, capture frames before and after
    the jitter, full-frame diff to find where the motion appeared.
    Only the cursor moves from commanded jitter — everything else stays still.

    If candidates found, YOLO crop-validates. If confirmed → ground truth.
    Even without YOLO, a single small blob from commanded motion is high
    confidence cursor discovery.

    This is a lightweight version of _jitter_reacquire() — same principle
    (commanded motion → observe → correlate), smaller amplitude.
    """
    global _yolo_lockout_until
    px = config.jitter_pixels

    def _capture():
        if use_daemon and daemon_client:
            jpeg = daemon_client.read_jpeg()
            if jpeg:
                return cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        elif sensor:
            return sensor.capture(settle_frames=1)
        return None

    try:
        # 1. Capture BEFORE jitter
        frame_before = _capture()
        if frame_before is None:
            # Fallback: just send jitter without observation
            mouse._send_raw(px, 0)
            time.sleep(0.15)
            mouse._send_raw(-px, 0)
            return

        # 2. Signal silhouette tracker, send jitter
        if sil_tracker:
            sil_tracker.expect_motion(amplitude_px=px)
        mouse._send_raw(px, 0)
        time.sleep(0.20)  # 200ms for daemon JPEG buffer to refresh

        # 3. Capture AFTER jitter
        frame_after = _capture()

        # 4. Undo jitter
        mouse._send_raw(-px, 0)

        if frame_after is None:
            return

        # 5. Full-frame diff — low threshold for ±3px motion
        blobs = extract_motion_blobs(
            frame_before, frame_after,
            threshold=8,          # Low: catch subtle cursor edge changes
            min_pixels=3,         # Small: ±3px moves very few pixels
            max_pixels=500,       # Cursor-sized only (not UI redraws)
        )

        if not blobs:
            return

        # 6. Filter: reject known-noisy cells (toolbar, clock animations)
        clean = []
        for b in blobs:
            gx = min(_NOISE_GRID_W - 1, b.centroid[0] // _NOISE_CELL_PX)
            gy = min(_NOISE_GRID_H - 1, b.centroid[1] // _NOISE_CELL_PX)
            if _noise_grid[gy][gx] >= _NOISE_THRESHOLD:
                continue
            clean.append(b)
        candidates = clean or blobs  # Fall back to unfiltered if all rejected

        # 7. Score candidates — prefer small, cursor-shaped blobs
        scored = []
        for b in candidates:
            score = 1.0
            # Prefer cursor-sized blobs (10-100px)
            if b.pixel_count < 10:
                score *= 0.5
            elif b.pixel_count > 200:
                score *= 0.3
            # Shape scoring via recognizer
            if recognizer and recognizer.cursor_template is not None:
                try:
                    bx, by = b.centroid
                    half = 32  # PATCH_SIZE
                    h, w = frame_after.shape[:2]
                    if half <= bx < w - half and half <= by < h - half:
                        gray = cv2.cvtColor(frame_after, cv2.COLOR_RGB2GRAY)
                        patch = gray[by - half//2:by + half//2,
                                     bx - half//2:bx + half//2]
                        if patch.shape == (32, 32):
                            shape_score = recognizer.score_blob_shape(patch)
                            score *= (0.5 + shape_score)
                except Exception:
                    pass
            scored.append((score, b))
        scored.sort(key=lambda x: x[0], reverse=True)

        if not scored:
            return

        # 8. YOLO crop-validation of top candidate(s)
        best_score, best_blob = scored[0]
        bx, by = best_blob.centroid
        yolo_confirmed = False

        if yolo_detector and yolo_detector.loaded:
            # Crop 300x300 around candidate and run YOLO
            h, w = frame_after.shape[:2]
            crop_half = 150
            cx0 = max(0, bx - crop_half)
            cy0 = max(0, by - crop_half)
            cx1 = min(w, bx + crop_half)
            cy1 = min(h, by + crop_half)
            crop = frame_after[cy0:cy1, cx0:cx1]
            if crop.shape[0] > 50 and crop.shape[1] > 50:
                detections = yolo_detector.detect(crop)
                if detections:
                    # YOLO found cursor in the crop — adjust coords to full-frame
                    det = max(detections, key=lambda d: d.confidence)
                    bx = cx0 + int(det.cx)
                    by = cy0 + int(det.cy)
                    yolo_confirmed = True
                    state.yolo_active = True
                    state.yolo_x = bx
                    state.yolo_y = by
                    state.yolo_confidence = det.confidence
                    state.yolo_timestamp = time.monotonic()
                    _yolo_lockout_until = time.monotonic() + _YOLO_LOCKOUT_S
                    print(f"[jitter_discover] YOLO confirmed at ({bx},{by}) "
                          f"conf={det.confidence:.2f}")

        # 9. Decide if we trust this enough to update tracker
        #    - YOLO confirmed: definite ground truth
        #    - Single small blob from commanded motion: high confidence
        #    - Multiple blobs: less certain, pick best but be cautious
        n_candidates = len(scored)
        if yolo_confirmed:
            confidence = 0.95
            method = "jitter_yolo"
        elif n_candidates == 1 and best_blob.pixel_count < 200:
            confidence = 0.85  # Single small blob from commanded motion
            method = "jitter_discover"
        elif n_candidates <= 3 and best_score > 0.5:
            confidence = 0.7
            method = "jitter_discover"
        else:
            # Too many candidates — noisy screen, can't isolate cursor
            return

        # Check if this position is far from current tracked position
        cur_drift = ((bx - state.cursor_x) ** 2 +
                     (by - state.cursor_y) ** 2) ** 0.5

        if cur_drift < 30:
            # Close to current position — just confirms existing track.
            # Let silhouette's jitter_confirm handle this via expect_motion.
            return

        # 10. Update tracker with discovered position
        print(f"[jitter_discover] {method}: ({bx},{by}) "
              f"n_blobs={n_candidates} drift={cur_drift:.0f}px "
              f"blob_px={best_blob.pixel_count} score={best_score:.2f}")

        if tracker:
            tracker.set_position(bx, by, method=method)
        state.cursor_x = bx
        state.cursor_y = by
        state.cursor_method = method
        state.cursor_age_s = 0.0
        state.cursor_validated = False  # Need CNN to confirm
        if sil_tracker:
            sil_tracker.reset()

        # Collect training data at discovered position
        if collector:
            collector.collect_from_locate(
                frame_after, bx, by,
                correlation=confidence,
                num_negatives=2,
                source="jitter_discover",
            )
            state.sample_count_pos = collector.positive_count
            state.sample_count_neg = collector.negative_count

    except Exception as e:
        # Never crash jitter loop — fall back to simple jitter
        try:
            mouse._send_raw(px, 0)
            time.sleep(0.15)
            mouse._send_raw(-px, 0)
        except Exception:
            pass


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
    global _low_confidence_since, _cnn_validated_x, _cnn_validated_y, _cnn_urgent_recheck, _stale_recovery_attempts, _cnn_lockout_until
    while True:
        # Use shorter interval when:
        # - urgent recheck (position drifted from CNN-validated) → 2s
        # - never validated (startup / never found cursor) → 5s
        # - validated and stable → normal interval (30s)
        if _cnn_urgent_recheck:
            interval = 0.5  # Fast recheck — CNN confidence may already be high
        elif not state.cursor_validated and _low_confidence_since == 0:
            interval = 5.0  # First-time search
        else:
            interval = config.cnn_interval_s
        time.sleep(interval)
        if not recognizer or not tracker:
            continue

        use_daemon = daemon_client and daemon_client.is_running()
        if not use_daemon and not sensor:
            continue

        try:
            pos = tracker.position
            now = time.monotonic()
            # Faster stale threshold during urgent recheck (CNN drift detected)
            stale_threshold = 10.0 if _cnn_urgent_recheck else 60.0
            position_stale = (not pos) or (now - pos.timestamp > stale_threshold)

            if position_stale:
                # Position unknown or very stale.
                # Recovery chain (no-mouse-first):
                #   1. YOLO (if recent detection)
                #   2. CNN grid scan (Phase 1: find cursor anywhere on screen)
                #   3. Commanded movement / jitter probe (needs mouse)
                #   4. calibrate_to_corner (last resort, needs mouse)
                reacquired = False

                # 1. YOLO shortcut
                yolo_age = now - state.yolo_timestamp if state.yolo_timestamp > 0 else 999
                if (yolo_detector and state.yolo_active
                        and state.yolo_confidence > 0.5 and yolo_age < 5.0):
                    tracker.set_position(state.yolo_x, state.yolo_y, method="yolo")
                    state.cursor_x = state.yolo_x
                    state.cursor_y = state.yolo_y
                    state.cursor_method = "yolo"
                    state.cursor_age_s = 0.0
                    reacquired = True

                # 2. CNN grid scan — Phase 1 visual detection (no mouse needed)
                if not reacquired and recognizer:
                    try:
                        if use_daemon:
                            jpeg = daemon_client.read_jpeg()
                            if jpeg:
                                scan_frame = cv2.imdecode(
                                    np.frombuffer(jpeg, dtype=np.uint8),
                                    cv2.IMREAD_COLOR)
                            else:
                                scan_frame = None
                        else:
                            scan_frame = sensor.capture(settle_frames=1)

                        if scan_frame is not None:
                            t_scan = time.monotonic()
                            hits = recognizer.grid_scan(
                                scan_frame, stride=64, threshold=0.85)
                            scan_ms = (time.monotonic() - t_scan) * 1000

                            if hits:
                                gx, gy, gconf = hits[0]
                                print(f"[cnn_loop] Grid scan found cursor at "
                                      f"({gx},{gy}) conf={gconf:.3f} "
                                      f"({len(hits)} hits, {scan_ms:.0f}ms)")
                                tracker.set_position(gx, gy, method="grid_scan")
                                state.cursor_x = gx
                                state.cursor_y = gy
                                state.cursor_method = "grid_scan"
                                state.cursor_age_s = 0.0
                                reacquired = True
                            else:
                                print(f"[cnn_loop] Grid scan: no hits "
                                      f"({scan_ms:.0f}ms)")
                    except Exception as e:
                        print(f"[cnn_loop] Grid scan error: {e}")

                # 3. Mouse-based recovery (skip in passive/passthrough mode)
                if not _passive_mode and not _passthrough_active:
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
                # Track consecutive stale-recovery cycles
                _stale_recovery_attempts += 1
                # Calibrate to known corner: immediately if urgent + probes failed,
                # or after 3 cycles otherwise
                calibrate_threshold = 1 if _cnn_urgent_recheck else 3
                if not _passive_mode and not _passthrough_active and _stale_recovery_attempts >= calibrate_threshold and mouse:
                    try:
                        print(f"[cnn_loop] {_stale_recovery_attempts} stale recoveries"
                              " failed — calibrating to corner")
                        mouse.calibrate_to_corner('top_left')
                        cx_target = mouse.screen_width // 2
                        cy_target = mouse.screen_height // 2
                        mouse.move(cx_target, cy_target)
                        if tracker:
                            tracker.set_position(
                                mouse.estimated_x, mouse.estimated_y,
                                method="calibrate_recovery")
                        if sil_tracker:
                            sil_tracker.reset()
                        reacquired = True
                        _stale_recovery_attempts = 0
                        print(f"[cnn_loop] Calibrated to ({mouse.estimated_x}, "
                              f"{mouse.estimated_y})")
                    except Exception as e:
                        print(f"[cnn_loop] Calibrate recovery failed: {e}")
                if reacquired:
                    # Do NOT reset _low_confidence_since — only CNN > 0.7
                    # resets it. Probing a new position doesn't prove the
                    # cursor is actually there.
                    state.cursor_validated = False
                    state.cnn_confidence = 0.0
                    if sil_tracker:
                        sil_tracker.reset()
                else:
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
                _cnn_urgent_recheck = False  # No longer urgent
                _stale_recovery_attempts = 0  # Recovery succeeded
                _cnn_lockout_until = time.monotonic() + _CNN_LOCKOUT_S
                recognizer.update_cursor_template(patch)
                state.cursor_validated = True
                _cnn_validated_x = cx
                _cnn_validated_y = cy
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
                # If CNN has been failing for >5s, escalate to urgent mode.
                # This disables jitter probes, motion_track, and reduces
                # stale threshold so calibrate_to_corner fires faster.
                low_conf_elapsed = now - _low_confidence_since
                if low_conf_elapsed > 5.0 and not _cnn_urgent_recheck:
                    _cnn_urgent_recheck = True
                    _vision_lockout_until = 0.0  # Clear stale lockout
                    print(f"[cnn_loop] CNN failing for {low_conf_elapsed:.0f}s "
                          f"→ urgent recheck mode")
                # Re-acquisition: only probe when we're NOT in urgent
                # recheck mode. When _cnn_urgent_recheck is True, we know
                # the position drifted — probing from wrong position just
                # finds random blobs and keeps the timestamp fresh,
                # preventing the stale-position calibrate recovery.
                if not _cnn_urgent_recheck and not _passive_mode:
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
    return _serve_page("dashboard.html")


@app.get("/profiler")
async def profiler_page():
    return _serve_page("profiler.html")


@app.get("/concepts")
async def concepts_page():
    return _serve_page("concepts.html")


@app.get("/models")
async def models_page():
    return _serve_page("models.html")


@app.get("/api/concepts_text")
async def concepts_text():
    """Return raw concepts markdown text."""
    p = Path(__file__).parent.parent.parent / "docs" / "STAR_TREK_COMPUTER_CONCEPTS.md"
    if p.exists():
        return Response(content=p.read_text(), media_type="text/plain")
    return Response(content="Concepts file not found", media_type="text/plain")


@app.get("/api/models")
async def get_models():
    """Aggregate model metadata for the /models page."""
    data_dir = Path(__file__).parent.parent.parent / "data"

    # CNN models from training_log.jsonl
    cnn_models = []
    active_cnn = state.cnn_model_version if state else "none"
    log_path = data_dir / "cursor_models" / "training_log.jsonl"
    if log_path.exists():
        for line in log_path.read_text().strip().split('\n'):
            if line.strip():
                try:
                    cnn_models.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # YOLO models
    yolo_models = []
    yolo_dir = data_dir / "yolo_models"
    for pt_file in sorted(yolo_dir.glob("cursor_*.pt")) if yolo_dir.exists() else []:
        entry = {
            "name": pt_file.stem,
            "size_mb": pt_file.stat().st_size / (1024 * 1024),
            "metrics": {},
        }
        # Try to find training metrics in runs/
        for run_dir in yolo_dir.glob("runs/*/" + pt_file.stem.replace("_best", "")):
            csv_path = run_dir / "results.csv"
            if csv_path.exists():
                lines = csv_path.read_text().strip().split('\n')
                if len(lines) > 1:
                    headers = [h.strip() for h in lines[0].split(',')]
                    last = [v.strip() for v in lines[-1].split(',')]
                    d = dict(zip(headers, last))
                    try:
                        entry["metrics"] = {
                            "precision": float(d.get("metrics/precision(B)", 0)),
                            "recall": float(d.get("metrics/recall(B)", 0)),
                            "map50": float(d.get("metrics/mAP50(B)", 0)),
                        }
                    except (ValueError, KeyError):
                        pass
        yolo_models.append(entry)

    # Also check runs/detect/ for v2 style
    detect_dir = yolo_dir / "runs" / "detect" if yolo_dir.exists() else None
    if detect_dir and detect_dir.exists():
        for run_dir in sorted(detect_dir.iterdir()):
            best = run_dir / "weights" / "best.pt"
            if best.exists():
                name = run_dir.name
                if not any(y["name"] == name for y in yolo_models):
                    entry = {"name": name, "size_mb": best.stat().st_size / (1024 * 1024), "metrics": {}}
                    csv_path = run_dir / "results.csv"
                    if csv_path.exists():
                        lines = csv_path.read_text().strip().split('\n')
                        if len(lines) > 1:
                            headers = [h.strip() for h in lines[0].split(',')]
                            last = [v.strip() for v in lines[-1].split(',')]
                            d = dict(zip(headers, last))
                            try:
                                entry["metrics"] = {
                                    "precision": float(d.get("metrics/precision(B)", 0)),
                                    "recall": float(d.get("metrics/recall(B)", 0)),
                                    "map50": float(d.get("metrics/mAP50(B)", 0)),
                                }
                            except (ValueError, KeyError):
                                pass
                    yolo_models.append(entry)

    # Sample stats
    sample_stats = {"total": 0, "positive": 0, "negative": 0, "reviewed": 0, "sources": {}}
    index_path = data_dir / "cursor_samples" / "index.jsonl"
    if index_path.exists():
        for line in index_path.read_text().strip().split('\n'):
            if not line.strip():
                continue
            try:
                s = json.loads(line)
                sample_stats["total"] += 1
                if s.get("label") == "pos":
                    sample_stats["positive"] += 1
                else:
                    sample_stats["negative"] += 1
                src = s.get("source", "unknown")
                sample_stats["sources"][src] = sample_stats["sources"].get(src, 0) + 1
            except json.JSONDecodeError:
                pass
    reviews_path = data_dir / "cursor_samples" / "reviews.jsonl"
    if reviews_path.exists():
        sample_stats["reviewed"] = sum(1 for l in reviews_path.read_text().strip().split('\n') if l.strip())

    # Velocity test runs
    velocity_runs = []
    vtest_dir = data_dir / "velocity_test"
    if vtest_dir.exists():
        for run_dir in sorted(vtest_dir.iterdir()):
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                entry = {"run_id": run_dir.name.replace("run_", ""), "summary": {}}
                summary_path = run_dir / "summary.json"
                if summary_path.exists():
                    try:
                        entry["summary"] = json.loads(summary_path.read_text())
                    except json.JSONDecodeError:
                        pass
                velocity_runs.append(entry)

    # Loss event count
    loss_dir = data_dir / "loss_events"
    loss_count = sum(1 for d in loss_dir.iterdir() if d.is_dir()) if loss_dir.exists() else 0

    return {
        "cnn_models": cnn_models,
        "cnn_count": len(cnn_models),
        "active_cnn": active_cnn,
        "yolo_models": yolo_models,
        "yolo_count": len(yolo_models),
        "sample_stats": sample_stats,
        "total_samples": sample_stats["total"],
        "velocity_runs": velocity_runs,
        "velocity_run_count": len(velocity_runs),
        "loss_event_count": loss_count,
    }


@app.get("/api/frame")
async def get_frame():
    with state.lock:
        jpeg = state.frame_overlay
    if jpeg:
        return Response(content=jpeg, media_type="image/jpeg",
                        headers={"X-Frame-UTC": utc_now_ms()})
    return {"error": "No frame yet"}


@app.get("/api/frame_raw")
async def get_frame_raw():
    """Raw JPEG frame without overlays (for client-side canvas rendering)."""
    with state.lock:
        raw = state.frame_raw
    if raw is not None:
        bgr = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(content=buf.tobytes(), media_type="image/jpeg",
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
            "bbox": [int(v) for v in state.yolo_bbox],
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
        "passthrough": {
            "active": _passthrough_active,
            "session": {
                "session_id": _passthrough_session["session_id"],
                "label": _passthrough_session["label"],
                "duration_s": round(time.monotonic() - _passthrough_session["start_time"], 1),
                "events": _passthrough_session["event_count"],
                "screenshots": _passthrough_session["screenshot_count"],
                "cnn_samples": _passthrough_session["cnn_sample_count"],
            } if _passthrough_session else None,
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

    # Last resort: calibrate to known corner + move to center
    try:
        mouse.calibrate_to_corner('top_left')
        cx_target = mouse.screen_width // 2
        cy_target = mouse.screen_height // 2
        mouse.move(cx_target, cy_target)
        tracker.set_position(
            mouse.estimated_x, mouse.estimated_y,
            method="calibrate_recovery")
        if sil_tracker:
            sil_tracker.reset()
        state.cursor_validated = False
        return {
            "cursor": {"x": mouse.estimated_x, "y": mouse.estimated_y},
            "method": "calibrate_recovery",
        }
    except Exception as e:
        return {"error": f"All recovery methods failed (calibrate: {e})"}


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


@app.get("/api/passive")
async def get_passive():
    """Check passive mode status."""
    return {"passive_mode": _passive_mode}


@app.post("/api/passive")
async def set_passive(request: Request):
    """Toggle passive mode. Body: {"enabled": true/false}"""
    global _passive_mode
    data = await request.json()
    _passive_mode = bool(data.get("enabled", False))
    print(f"[api] Passive mode {'enabled' if _passive_mode else 'disabled'}")
    return {"passive_mode": _passive_mode}


@app.get("/api/grid_scan")
async def api_grid_scan(stride: int = 64, threshold: float = 0.6):
    """Run CNN grid scan on current frame. Phase 1 visual detection.

    Returns top cursor candidates with position and confidence.
    No mouse movement required.
    """
    if not recognizer:
        return {"error": "No recognizer available"}

    use_daemon = daemon_client and daemon_client.is_running()
    try:
        if use_daemon:
            jpeg = daemon_client.read_jpeg()
            if not jpeg:
                return {"error": "No frame available"}
            frame = cv2.imdecode(
                np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        elif sensor:
            frame = sensor.capture(settle_frames=1)
        else:
            return {"error": "No capture source"}

        if frame is None:
            return {"error": "Frame capture failed"}

        t0 = time.monotonic()
        hits = recognizer.grid_scan(frame, stride=stride, threshold=threshold)
        scan_ms = (time.monotonic() - t0) * 1000

        return {
            "hits": [{"x": x, "y": y, "confidence": round(c, 4)} for x, y, c in hits],
            "scan_ms": round(scan_ms, 1),
            "stride": stride,
            "threshold": threshold,
            "frame_shape": list(frame.shape[:2]),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/yolo_detect_now")
async def yolo_detect_now():
    """Manual YOLO full-frame detection. Updates tracker as ground truth.

    Unlike the continuous YOLO in the motion loop (which uses crop pipeline),
    this always runs full-frame and treats the result as ground truth —
    updating tracker position and saving CNN training samples.
    """
    global _yolo_lockout_until
    if not yolo_detector or not yolo_detector.loaded:
        return {"error": "YOLO detector not loaded"}

    # Grab current frame
    use_daemon = daemon_client and daemon_client.is_running()
    try:
        if use_daemon:
            jpeg = daemon_client.read_jpeg()
            if not jpeg:
                return {"error": "No frame from daemon"}
            frame = cv2.imdecode(
                np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        elif sensor:
            frame = sensor.capture(settle_frames=1)
        else:
            with state.lock:
                frame = state.frame_raw.copy() if state.frame_raw is not None else None

        if frame is None:
            return {"error": "No frame available"}

        # Full-frame YOLO detection
        t0 = time.monotonic()
        detections = yolo_detector.detect(frame)
        dt_ms = (time.monotonic() - t0) * 1000

        if detections:
            best = max(detections, key=lambda d: d.confidence)
            with state.lock:
                state.yolo_active = True
                state.yolo_x = int(best.cx)
                state.yolo_y = int(best.cy)
                state.yolo_confidence = best.confidence
                state.yolo_inference_ms = dt_ms
                state.yolo_detection_count = len(detections)
                state.yolo_timestamp = time.monotonic()
                state.yolo_bbox = (int(best.x1), int(best.y1), int(best.x2), int(best.y2))

            # Update tracker as ground truth
            if tracker:
                tracker.set_position(state.yolo_x, state.yolo_y, method="yolo_manual")
            _yolo_lockout_until = time.monotonic() + _YOLO_LOCKOUT_S

            # Collect CNN training sample at confirmed position
            if collector:
                collector.collect_from_locate(
                    frame, state.yolo_x, state.yolo_y,
                    correlation=state.yolo_confidence,
                    num_negatives=2, source="yolo_ground_truth",
                )

            print(f"[yolo_manual] Detected cursor at ({state.yolo_x},{state.yolo_y}) "
                  f"conf={state.yolo_confidence:.3f} in {dt_ms:.1f}ms "
                  f"({len(detections)} total detections)")

            return {
                "active": True,
                "x": state.yolo_x, "y": state.yolo_y,
                "confidence": state.yolo_confidence,
                "inference_ms": round(dt_ms, 1),
                "count": len(detections),
                "bbox": [int(best.x1), int(best.y1), int(best.x2), int(best.y2)],
                "all_detections": [
                    {"x": int(d.cx), "y": int(d.cy), "conf": round(d.confidence, 3),
                     "bbox": [int(d.x1), int(d.y1), int(d.x2), int(d.y2)]}
                    for d in detections
                ],
            }
        else:
            with state.lock:
                state.yolo_active = False
                state.yolo_detection_count = 0
            print(f"[yolo_manual] No detections in {dt_ms:.1f}ms")
            return {"active": False, "inference_ms": round(dt_ms, 1), "count": 0}

    except Exception as e:
        return {"error": str(e)}


# ── Passthrough Experience Helpers ────────────────────────────────

def _passthrough_read_frame() -> tuple[bytes | None, np.ndarray | None]:
    """Read JPEG bytes from daemon, optionally decode to numpy for CNN."""
    if not daemon_client or not daemon_client.is_running():
        return None, None
    try:
        jpeg = daemon_client.read_jpeg()
    except Exception:
        return None, None
    if jpeg is None:
        return None, None
    # Decode for CNN sample collection
    arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return jpeg, arr


def _passthrough_screenshot_loop():
    """Background thread: periodic screenshots + CNN samples during passthrough."""
    import os
    last_cnn_time = time.monotonic()
    while _passthrough_active and _passthrough_session:
        time.sleep(_PT_SCREENSHOT_INTERVAL_S)
        if not _passthrough_active or not _passthrough_session:
            break
        sess = _passthrough_session
        jpeg, frame = _passthrough_read_frame()
        if jpeg is None:
            continue
        # Save raw JPEG bytes directly (no re-encode)
        ts = datetime.now().strftime("%H%M%S_%f")[:-3]
        shot_path = os.path.join(sess["screenshots_dir"], f"pt_{ts}.jpg")
        try:
            with open(shot_path, "wb") as f:
                f.write(jpeg)
            sess["screenshot_count"] += 1
            sess["logger"].log("screenshot", details={"path": shot_path})
        except Exception:
            pass
        # CNN gold sample every _PT_CNN_SAMPLE_INTERVAL_S
        now = time.monotonic()
        if collector and mouse and frame is not None and (now - last_cnn_time) >= _PT_CNN_SAMPLE_INTERVAL_S:
            try:
                collector.collect_from_locate(
                    frame, mouse.estimated_x, mouse.estimated_y,
                    correlation=0.95, source="passthrough_gold")
                sess["cnn_sample_count"] += 1
                last_cnn_time = now
            except Exception:
                pass


def _passthrough_capture_click(pos: tuple[int, int]):
    """Immediate screenshot + CNN sample on click (highest-value moment)."""
    if not _passthrough_session:
        return
    import os
    sess = _passthrough_session
    jpeg, frame = _passthrough_read_frame()
    if jpeg is None:
        return
    # Save raw JPEG bytes directly
    ts = datetime.now().strftime("%H%M%S_%f")[:-3]
    shot_path = os.path.join(sess["screenshots_dir"], f"click_{ts}.jpg")
    try:
        with open(shot_path, "wb") as f:
            f.write(jpeg)
        sess["screenshot_count"] += 1
        sess["logger"].log("click_screenshot", position=pos, details={"path": shot_path})
    except Exception:
        pass
    # CNN sample — user clicked exactly where cursor is
    if collector and frame is not None:
        try:
            collector.collect_from_locate(
                frame, pos[0], pos[1],
                correlation=0.99, source="passthrough_click")
            sess["cnn_sample_count"] += 1
        except Exception:
            pass


# ── Input Passthrough ─────────────────────────────────────────────
# Pipes user's mouse/keyboard through ESP32 pebble device + Pi Zero.
# Frontend sends batched (dx,dy) at 50Hz via Pointer Lock API.

@app.post("/api/passthrough/move")
async def passthrough_move(request: Request):
    """Accept batched mouse delta from frontend Pointer Lock."""
    if not _passthrough_active:
        return {"error": "passthrough not active"}
    if not mouse:
        return {"error": "mouse not connected"}
    data = await request.json()
    dx = int(data.get("dx", 0))
    dy = int(data.get("dy", 0))
    if dx == 0 and dy == 0:
        return {"ok": True}
    # Send directly — ESP32 HID_MAX is 127, chunk if larger
    try:
        while abs(dx) > 0 or abs(dy) > 0:
            chunk_x = max(-127, min(127, dx))
            chunk_y = max(-127, min(127, dy))
            mouse._send_raw(chunk_x, chunk_y)
            dx -= chunk_x
            dy -= chunk_y
    except Exception as e:
        return {"error": str(e)}
    # Sync dashboard state from ESP32 dead-reckoning
    with state.lock:
        state.cursor_x = mouse.estimated_x
        state.cursor_y = mouse.estimated_y
        state.cursor_method = "passthrough"
        state.cursor_age_s = 0.0
    if sil_tracker:
        sil_tracker.reset()
    if _passthrough_session:
        _passthrough_session["logger"].log("mouse_move",
            position=(mouse.estimated_x, mouse.estimated_y),
            details={"dx": int(data.get("dx", 0)), "dy": int(data.get("dy", 0))})
        _passthrough_session["event_count"] += 1
    return {"ok": True}


@app.post("/api/passthrough/click")
async def passthrough_click(request: Request):
    """Forward mouse click through ESP32."""
    if not _passthrough_active:
        return {"error": "passthrough not active"}
    if not mouse:
        return {"error": "mouse not connected"}
    data = await request.json()
    button = data.get("button", "left")
    if button not in ("left", "right", "middle"):
        return {"error": "invalid button"}
    try:
        mouse.click(button)
    except Exception as e:
        return {"error": str(e)}
    # Refresh cursor age on click
    with state.lock:
        state.cursor_age_s = 0.0
    if _passthrough_session:
        pos = (mouse.estimated_x, mouse.estimated_y)
        _passthrough_session["logger"].log("click",
            position=pos, details={"button": button})
        _passthrough_session["event_count"] += 1
        # Click = highest-value moment — trigger immediate screenshot + CNN sample
        threading.Thread(
            target=_passthrough_capture_click,
            args=(pos,),
            daemon=True,
        ).start()
    return {"ok": True}


@app.post("/api/passthrough/scroll")
async def passthrough_scroll(request: Request):
    """Forward scroll wheel through ESP32."""
    if not _passthrough_active:
        return {"error": "passthrough not active"}
    if not mouse:
        return {"error": "mouse not connected"}
    data = await request.json()
    clicks = int(data.get("clicks", 0))
    if clicks == 0:
        return {"ok": True}
    try:
        mouse.scroll(clicks)
    except Exception as e:
        return {"error": str(e)}
    if _passthrough_session:
        _passthrough_session["logger"].log("scroll",
            position=(mouse.estimated_x, mouse.estimated_y),
            details={"clicks": clicks})
        _passthrough_session["event_count"] += 1
    return {"ok": True}


_PI_KEYBOARD_URL = "http://10.55.0.2:8081/api/keyboard/type"
_pi_http_client: Optional[httpx.AsyncClient] = None


@app.post("/api/passthrough/key")
async def passthrough_key(request: Request):
    """Forward keypress to Windows via Pi Zero keyboard API."""
    global _pi_http_client
    if not _passthrough_active:
        return {"error": "passthrough not active"}
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return {"ok": True}
    if _pi_http_client is None:
        _pi_http_client = httpx.AsyncClient(timeout=3.0)
    try:
        resp = await _pi_http_client.post(
            _PI_KEYBOARD_URL,
            json={"text": text},
        )
        resp.raise_for_status()
    except Exception as e:
        return {"error": f"Pi keyboard: {e}"}
    if _passthrough_session:
        _passthrough_session["logger"].log("key",
            position=(mouse.estimated_x if mouse else 0, mouse.estimated_y if mouse else 0),
            details={"text": text})
        _passthrough_session["event_count"] += 1
    return {"ok": True}


@app.post("/api/passthrough/toggle")
async def passthrough_toggle(request: Request):
    """Toggle input passthrough mode with experience recording."""
    global _passthrough_active, _passthrough_session
    try:
        data = await request.json()
    except Exception:
        data = {}
    label = data.get("label", "passthrough")

    _passthrough_active = not _passthrough_active

    if _passthrough_active:
        # Snapshot cursor state at activation
        start_x = mouse.estimated_x if mouse else 0
        start_y = mouse.estimated_y if mouse else 0
        print(f"[passthrough] ACTIVE — starting at ({start_x}, {start_y})")

        # Start experience recording
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path("logs/experiences/passthrough")
        screenshots_dir = session_dir / f"{session_id}_{label}" / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        storyline = StorylineLogger(
            experience_dir=session_dir / f"{session_id}_{label}",
            session_id=session_id,
        )
        storyline.start_experience(f"passthrough_{label}", attributes={
            "label": label,
            "start_x": start_x,
            "start_y": start_y,
        })
        exp_logger = ExperienceLogger(
            log_dir=str(session_dir),
            storyline_logger=storyline,
        )
        exp_logger.log("passthrough_start", position=(start_x, start_y), details={
            "label": label,
            "session_id": session_id,
        })
        _passthrough_session = {
            "logger": exp_logger,
            "storyline": storyline,
            "session_id": session_id,
            "label": label,
            "start_time": time.monotonic(),
            "event_count": 0,
            "screenshot_count": 0,
            "cnn_sample_count": 0,
            "screenshots_dir": str(screenshots_dir),
        }
        # Start screenshot capture thread
        threading.Thread(
            target=_passthrough_screenshot_loop,
            daemon=True,
            name="pt-screenshots",
        ).start()
    else:
        # Flush final state and trigger CNN validation
        if mouse:
            with state.lock:
                state.cursor_x = mouse.estimated_x
                state.cursor_y = mouse.estimated_y
                state.cursor_method = "passthrough_end"
                state.cursor_age_s = 0.0
                state.cursor_validated = False

        # End experience recording
        if _passthrough_session:
            sess = _passthrough_session
            duration = time.monotonic() - sess["start_time"]
            sess["logger"].log("passthrough_end", position=(state.cursor_x, state.cursor_y), details={
                "duration_s": round(duration, 1),
                "event_count": sess["event_count"],
                "screenshot_count": sess["screenshot_count"],
                "cnn_sample_count": sess["cnn_sample_count"],
            })
            sess["storyline"].save_narrative()
            sess["logger"].close()
            print(f"[passthrough] Session saved: {sess['event_count']} events, "
                  f"{sess['screenshot_count']} screenshots, "
                  f"{sess['cnn_sample_count']} CNN samples, "
                  f"{round(duration, 1)}s")
            _passthrough_session = None

        print(f"[passthrough] INACTIVE — final pos ({state.cursor_x}, {state.cursor_y})")
    return {"active": _passthrough_active}


@app.get("/api/passthrough/status")
async def passthrough_status():
    """Get passthrough state."""
    return {"active": _passthrough_active}


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
        "passive_mode": _passive_mode,
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
            global _low_confidence_since, _vision_lockout_until, _cnn_validated_x, _cnn_validated_y, _cnn_urgent_recheck, _cnn_lockout_until
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
                _cnn_validated_x = result.cursor_x
                _cnn_validated_y = result.cursor_y
                _cnn_urgent_recheck = False
                _stale_recovery_attempts = 0
                _cnn_lockout_until = time.monotonic() + _CNN_LOCKOUT_S
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
            "bbox": [int(v) for v in state.yolo_bbox],
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
_CURSOR_MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "cursor_models"

# Secondary model for A/B comparison in validation UI
_comparison_model: dict = {"version": None, "model": None}
_YOLO_CACHE_PATH = _VALIDATION_SAMPLES_DIR / "yolo_cache.jsonl"
_YOLO_MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "yolo_models"


def _load_yolo_cache() -> dict:
    """Load YOLO enrichment cache, keyed by filename."""
    cache = {}
    if not _YOLO_CACHE_PATH.exists():
        return cache
    with open(_YOLO_CACHE_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    cache[entry["filename"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue
    return cache


@app.get("/validation")
async def validation_page():
    """Validation GUI for reviewing CNN training samples."""
    return _serve_page("validation.html")


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

    # Load YOLO cache
    yolo_cache = _load_yolo_cache()

    # Score all entries (CNN inference for priority)
    model = recognizer._cnn_model if recognizer else None
    import torch

    scored_entries = []
    for entry in entries:
        filename = entry.get("filename", "")
        review = reviewed.get(filename)
        is_reviewed = review is not None

        cnn_conf = None
        comp_conf = None
        comp_model = _comparison_model["model"]
        if model is not None or comp_model is not None:
            patch_path = _VALIDATION_SAMPLES_DIR / filename
            if patch_path.exists():
                patch = cv2.imread(str(patch_path), cv2.IMREAD_GRAYSCALE)
                if patch is not None and patch.shape == (64, 64):
                    tensor = torch.from_numpy(
                        patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                    with torch.no_grad():
                        if model is not None:
                            cnn_conf = model(tensor).item()
                        if comp_model is not None:
                            comp_conf = comp_model(tensor).item()

        priority = 5.0
        entry_label = entry.get("label", "")
        entry_source = entry.get("source", "")
        is_confusing = entry.get("is_confusing", 0)
        needs_review = entry.get("needs_review", False)

        # Normalize label (some entries use 1/0 instead of pos/neg)
        if entry_label == 1 or entry_label == "1":
            entry_label = "pos"
        elif entry_label == 0 or entry_label == "0":
            entry_label = "neg"

        cnn_disagrees = False
        if is_reviewed:
            priority = 99.0
        elif needs_review:
            # Human-captured samples get highest priority (ground truth)
            priority = -1.0
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

        # Derive origin type from source
        origin_type = "sensor"
        if "ground_truth" in entry_source or "claude" in entry_source:
            origin_type = "ground_truth"
        elif "live_tag" in entry_source:
            origin_type = "sensor"

        # CNN prediction labels
        cnn_predicted = None
        if cnn_conf is not None:
            cnn_predicted = "pos" if cnn_conf > 0.5 else "neg"
        comp_predicted = None
        if comp_conf is not None:
            comp_predicted = "pos" if comp_conf > 0.5 else "neg"
        models_disagree = (cnn_predicted is not None and comp_predicted is not None
                           and cnn_predicted != comp_predicted)

        has_context = (_VALIDATION_SAMPLES_DIR / f"ctx_{filename}").exists()

        # YOLO enrichment data
        yolo_entry = yolo_cache.get(filename, {})
        yolo_detected = yolo_entry.get("yolo_detected", False) if yolo_entry else False
        yolo_conf = yolo_entry.get("yolo_conf")
        yolo_disagrees = False
        if yolo_entry:
            yolo_says_cursor = yolo_detected
            label_says_cursor = entry_label == "pos"
            if yolo_entry.get("enrichment_source"):  # only if enriched
                yolo_disagrees = yolo_says_cursor != label_says_cursor

        scored_entries.append({
            **entry,
            "label": entry_label,  # normalized
            "cnn_confidence": round(cnn_conf, 3) if cnn_conf is not None else None,
            "cnn_predicted": cnn_predicted,
            "comparison_confidence": round(comp_conf, 3) if comp_conf is not None else None,
            "comp_predicted": comp_predicted,
            "models_disagree": models_disagree,
            "origin_type": origin_type,
            "priority": round(priority, 3),
            "review": review,
            "cnn_disagrees": cnn_disagrees,
            "needs_review": needs_review,
            "tags": file_tags.get(filename, {}),
            "has_context": has_context,
            "yolo_detected": yolo_detected,
            "yolo_conf": yolo_conf,
            "yolo_disagrees": yolo_disagrees,
            "yolo_enriched": bool(yolo_entry.get("enrichment_source")),
            "yolo_source": yolo_entry.get("enrichment_source"),
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
        "needs_review": sum(1 for e in scored_entries if e.get("needs_review")),
        "pos": sum(1 for e in scored_entries if e["label"] == "pos"),
        "neg": sum(1 for e in scored_entries if e["label"] == "neg"),
        "yolo_detected": sum(1 for e in scored_entries if e.get("yolo_detected")),
        "yolo_disagrees": sum(1 for e in scored_entries if e.get("yolo_disagrees")),
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
    elif filter == "needs_review":
        filtered = [e for e in filtered if e.get("needs_review")]
    elif filter == "yolo_detected":
        filtered = [e for e in filtered if e.get("yolo_detected")]
    elif filter == "yolo_disagrees":
        filtered = [e for e in filtered if e.get("yolo_disagrees")]

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
    elif sort == "yolo_conf_asc":
        filtered.sort(key=lambda e: e.get("yolo_conf") or 0)
    elif sort == "yolo_conf_desc":
        filtered.sort(key=lambda e: e.get("yolo_conf") or 0, reverse=True)

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
        "runtime_model": recognizer.model_version if recognizer else "none",
        "comparison_model": _comparison_model["version"],
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


@app.get("/api/validation/context/{filename}")
async def get_validation_context(filename: str):
    """Serve a 256x256 context crop for the validation UI.

    Context crops are saved as ctx_<filename> alongside the 64x64 patches
    for high-value sources (claude_vision, ground_truth, lissajous).
    """
    if "/" in filename or ".." in filename:
        return {"error": "Invalid filename"}
    ctx_path = _VALIDATION_SAMPLES_DIR / f"ctx_{filename}"
    if not ctx_path.exists():
        return Response(status_code=404)
    return Response(
        content=ctx_path.read_bytes(),
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


_CUSTOM_SCHEMA_PATH = _VALIDATION_SAMPLES_DIR / "custom_schema.jsonl"


def _load_full_schema() -> dict:
    """Build merged schema: defaults + custom additions + discovered from reviews."""
    schema = {
        "quality": {"values": ["ok", "wrong", "unsure"], "color": "#3498db"},
        "confusing": {"values": ["yes", "no"], "color": "#e67e22"},
        "cursor_type": {"values": ["arrow", "hand", "ibeam", "busy", "crosshair", "excel_cross", "move", "unknown"], "color": "#9b59b6"},
        "visibility": {"values": ["full", "partial", "hidden", "offscreen"], "color": "#2ecc71"},
        "background": {"values": ["clean", "noisy", "animated", "textured", "text"], "color": "#e74c3c"},
        "application": {"values": ["excel", "ms_teams", "win_desktop_env", "win_explorer"], "color": "#1abc9c"},
    }

    # Merge custom schema additions (append-only log)
    if _CUSTOM_SCHEMA_PATH.exists():
        for line in _CUSTOM_SCHEMA_PATH.read_text().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                action = entry.get("action")
                col = entry.get("column", "")
                if action == "add_column" and col:
                    if col not in schema:
                        schema[col] = {
                            "values": entry.get("values", []),
                            "color": entry.get("color", "#888"),
                        }
                elif action == "add_value" and col and col in schema:
                    val = entry.get("value", "")
                    if val and val not in schema[col]["values"]:
                        schema[col]["values"].append(val)
                elif action == "add_value" and col and col not in schema:
                    # Auto-create column if adding value to non-existent column
                    schema[col] = {"values": [entry.get("value", "")], "color": "#888"}
            except Exception:
                pass

    # Discover extra columns/values from existing reviews
    if _VALIDATION_REVIEWS_PATH.exists():
        for line in _VALIDATION_REVIEWS_PATH.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get("action") == "tag" and "tags" in r:
                    for k, v in r["tags"].items():
                        if k not in schema:
                            schema[k] = {"values": [], "color": "#888"}
                        if v and v not in schema[k]["values"]:
                            schema[k]["values"].append(v)
            except Exception:
                pass

    return schema


@app.get("/api/validation/tag_schema")
async def get_tag_schema():
    """Return the current tag columns and their possible values."""
    return {"schema": _load_full_schema()}


@app.post("/api/validation/schema/add_column")
async def add_schema_column(request: Request):
    """Add a new tag column to the schema.

    Body: {"column": "scene_type", "values": ["static", "scrolling"], "color": "#e67e22"}
    """
    body = await request.json()
    col = body.get("column", "").strip()
    values = body.get("values", [])
    color = body.get("color", "#888")

    if not col:
        return {"error": "column name required"}
    if not isinstance(values, list) or len(values) == 0:
        return {"error": "at least one value required"}

    record = {
        "action": "add_column",
        "column": col,
        "values": values,
        "color": color,
        "timestamp": time.time(),
    }
    with open(_CUSTOM_SCHEMA_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    return {"ok": True, "column": col, "values": values}


@app.post("/api/validation/schema/add_value")
async def add_schema_value(request: Request):
    """Add a new value to an existing tag column.

    Body: {"column": "background", "value": "gradient"}
    """
    body = await request.json()
    col = body.get("column", "").strip()
    val = body.get("value", "").strip()

    if not col or not val:
        return {"error": "column and value required"}

    record = {
        "action": "add_value",
        "column": col,
        "value": val,
        "timestamp": time.time(),
    }
    with open(_CUSTOM_SCHEMA_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")

    return {"ok": True, "column": col, "value": val}


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


@app.get("/api/validation/models")
async def get_validation_models():
    """List available CNN model versions with metadata from training_log.jsonl."""
    models = []
    log_path = _CURSOR_MODELS_DIR / "training_log.jsonl"
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                models.append({
                    "version": entry["model_file"].replace(".pt", ""),
                    "file": entry["model_file"],
                    "val_acc": entry.get("val_acc"),
                    "val_loss": entry.get("best_val_loss"),
                    "train_samples": entry.get("train_samples"),
                    "epochs": entry.get("epochs_trained"),
                    "major": entry.get("major"),
                    "notes": entry.get("notes"),
                    "timestamp": entry.get("timestamp"),
                })
            except (json.JSONDecodeError, KeyError):
                continue

    runtime_version = recognizer.model_version if recognizer else "none"
    comp_version = _comparison_model["version"]

    return {
        "models": models,
        "runtime": runtime_version,
        "comparison": comp_version,
    }


@app.post("/api/validation/set_comparison")
async def set_comparison_model(request: Request):
    """Load a secondary CNN model for A/B comparison.

    Body: {"version": "v003_20260220"} or {"version": null} to clear.
    """
    global _comparison_model
    data = await request.json()
    version = data.get("version")

    if not version:
        _comparison_model = {"version": None, "model": None}
        return {"ok": True, "version": None}

    model_file = _CURSOR_MODELS_DIR / f"{version}.pt"
    if not model_file.exists():
        return {"error": f"Model file not found: {version}.pt"}

    try:
        from src.hardware.cursor_cnn import TinyCursorNet
        import torch
        model = TinyCursorNet()
        model.load_state_dict(torch.load(str(model_file), map_location="cpu", weights_only=True))
        model.eval()
        _comparison_model = {"version": version, "model": model}
        return {"ok": True, "version": version}
    except Exception as e:
        return {"error": f"Failed to load model: {e}"}


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


# ── YOLO Enrichment + Batch Operations ───────────────────────────────

_yolo_enrich_proc = None  # Track running enrichment subprocess


@app.post("/api/validation/yolo_enrich")
async def trigger_yolo_enrichment():
    """Trigger YOLO batch enrichment as background subprocess."""
    global _yolo_enrich_proc
    import subprocess

    if _yolo_enrich_proc and _yolo_enrich_proc.poll() is None:
        return {"status": "already_running", "pid": _yolo_enrich_proc.pid}

    project_root = Path(__file__).parent.parent.parent
    script = project_root / "scripts" / "yolo_enrich_samples.py"
    venv_python = project_root / ".venv" / "bin" / "python"

    _yolo_enrich_proc = subprocess.Popen(
        [str(venv_python), str(script)],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    return {"status": "started", "pid": _yolo_enrich_proc.pid}


@app.get("/api/validation/yolo_status")
async def get_yolo_status():
    """Return YOLO model version, cache stats."""
    # Find current YOLO model
    yolo_model_name = None
    yolo_models = sorted(_YOLO_MODELS_DIR.glob("cursor_*.pt"), reverse=True)
    if yolo_models:
        yolo_model_name = yolo_models[0].stem

    # Cache stats
    cache = _load_yolo_cache()
    total_cached = len(cache)
    detected_count = sum(1 for v in cache.values() if v.get("yolo_detected"))
    enrichment_sources = {}
    for v in cache.values():
        src = v.get("enrichment_source", "none")
        enrichment_sources[src] = enrichment_sources.get(src, 0) + 1

    # Count total samples
    total_samples = 0
    index_path = _VALIDATION_SAMPLES_DIR / "index.jsonl"
    if index_path.exists():
        with open(index_path) as f:
            total_samples = sum(1 for line in f if line.strip())

    enrichment_running = (_yolo_enrich_proc is not None and
                          _yolo_enrich_proc.poll() is None)

    return {
        "yolo_model": yolo_model_name,
        "total_cached": total_cached,
        "total_samples": total_samples,
        "detected_count": detected_count,
        "enrichment_sources": enrichment_sources,
        "enrichment_running": enrichment_running,
        "pct_enriched": round(total_cached / total_samples * 100, 1)
                        if total_samples > 0 else 0,
    }


@app.post("/api/validation/batch_review")
async def batch_review(request: Request):
    """Batch review multiple samples at once.

    Body: {"filenames": ["pos_001.png", "neg_002.png"], "action": "correct"}
    """
    data = await request.json()
    filenames = data.get("filenames", [])
    action = data.get("action")

    valid_actions = ("correct", "wrong", "confusing", "delete")
    if not filenames or action not in valid_actions:
        return {"error": f"Need filenames list and action ({'/'.join(valid_actions)})"}

    count = 0
    with open(_VALIDATION_REVIEWS_PATH, "a") as f:
        for fn in filenames:
            review = {
                "filename": fn,
                "action": action,
                "timestamp": time.time(),
            }
            if action == "wrong":
                current_label = "pos" if fn.startswith("pos_") else "neg"
                review["corrected_label"] = "neg" if current_label == "pos" else "pos"
            if action == "confusing":
                review["is_confusing_to_human"] = True
            f.write(json.dumps(review) + "\n")
            count += 1

    return {"ok": True, "count": count, "action": action}


@app.post("/api/validation/batch_tag")
async def batch_tag(request: Request):
    """Batch tag multiple samples at once.

    Body: {"filenames": ["pos_001.png", "neg_002.png"], "tags": {"cursor_type": "arrow"}}
    """
    data = await request.json()
    filenames = data.get("filenames", [])
    tags = data.get("tags", {})

    if not filenames or not tags:
        return {"error": "Need filenames list and tags dict"}

    count = 0
    with open(_VALIDATION_REVIEWS_PATH, "a") as f:
        for fn in filenames:
            record = {
                "filename": fn,
                "action": "tag",
                "tags": tags,
                "timestamp": time.time(),
            }
            f.write(json.dumps(record) + "\n")
            count += 1

    return {"ok": True, "count": count}


@app.get("/api/validation/suggest/{filename}")
async def suggest_tags(filename: str):
    """Return YOLO + CNN analysis and suggested tags for a sample."""
    if "/" in filename or ".." in filename:
        return {"error": "Invalid filename"}

    # Load YOLO cache
    yolo_cache = _load_yolo_cache()
    yolo_data = yolo_cache.get(filename, {})

    # Get CNN confidence
    cnn_conf = None
    cnn_predicted = None
    model = recognizer._cnn_model if recognizer else None
    if model is not None:
        import torch
        patch_path = _VALIDATION_SAMPLES_DIR / filename
        if patch_path.exists():
            patch = cv2.imread(str(patch_path), cv2.IMREAD_GRAYSCALE)
            if patch is not None and patch.shape == (64, 64):
                tensor = torch.from_numpy(
                    patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
                with torch.no_grad():
                    cnn_conf = model(tensor).item()
                    cnn_predicted = "pos" if cnn_conf > 0.5 else "neg"

    # Build suggestions based on available data
    suggestions = {}
    yolo_conf = yolo_data.get("yolo_conf")

    if yolo_data.get("yolo_detected"):
        # Suggest cursor_type based on bbox aspect ratio
        bbox = yolo_data.get("yolo_bbox")
        if bbox:
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            aspect = w / h if h > 0 else 1.0
            if aspect > 1.3:
                suggestions["cursor_type"] = "hand"
            elif aspect < 0.7:
                suggestions["cursor_type"] = "ibeam"
            else:
                suggestions["cursor_type"] = "arrow"

        # Suggest visibility based on confidence
        if yolo_conf and yolo_conf > 0.7:
            suggestions["visibility"] = "full"
        elif yolo_conf and yolo_conf > 0.4:
            suggestions["visibility"] = "partial"

    return {
        "yolo": {
            "detected": yolo_data.get("yolo_detected", False),
            "conf": yolo_conf,
            "bbox": yolo_data.get("yolo_bbox"),
            "source": yolo_data.get("enrichment_source"),
            "model": yolo_data.get("yolo_model"),
        },
        "cnn": {
            "conf": round(cnn_conf, 4) if cnn_conf is not None else None,
            "predicted": cnn_predicted,
        },
        "suggestions": suggestions,
    }


@app.get("/api/validation/heartbeat")
async def validation_heartbeat():
    """Lightweight heartbeat for auto-refresh polling.

    Returns model version, sample count, YOLO stats for change detection.
    """
    # Model version
    model_version = recognizer.model_version if recognizer else "none"

    # Quick sample count
    sample_count = 0
    index_path = _VALIDATION_SAMPLES_DIR / "index.jsonl"
    if index_path.exists():
        with open(index_path) as f:
            sample_count = sum(1 for line in f if line.strip())

    # YOLO cache count
    yolo_cache_count = 0
    if _YOLO_CACHE_PATH.exists():
        with open(_YOLO_CACHE_PATH) as f:
            yolo_cache_count = sum(1 for line in f if line.strip())

    # YOLO model name
    yolo_model = None
    yolo_models = sorted(_YOLO_MODELS_DIR.glob("cursor_*.pt"), reverse=True)
    if yolo_models:
        yolo_model = yolo_models[0].stem

    enrichment_running = (_yolo_enrich_proc is not None and
                          _yolo_enrich_proc.poll() is None)

    return {
        "model_version": model_version,
        "sample_count": sample_count,
        "yolo_model": yolo_model,
        "yolo_cache_count": yolo_cache_count,
        "enrichment_running": enrichment_running,
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
                      run_dir: Path,
                      velocity_px_s: float = 0.0,
                      run_yolo: bool = False) -> dict:
    """Capture full sensor state for one velocity test step.

    Args:
        velocity_px_s: Target velocity for this step (for recording).
        run_yolo: If True, run YOLO detection on this frame for ground truth.

    Returns a dict suitable for JSONL logging.
    """
    now_mono = time.monotonic()

    # Save frame for CNN training
    frame_name = f"frame_{profile_id}_{direction}_{step:03d}.jpg"
    frame_path = run_dir / "frames" / frame_name
    frame_np = None
    if daemon_client and daemon_client.is_running():
        jpeg = daemon_client.read_jpeg()
        if jpeg:
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            frame_path.write_bytes(jpeg)
            # Decode for CNN comparison
            frame_np = cv2.imdecode(
                np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    tracked_x = state.cursor_x
    tracked_y = state.cursor_y
    error_px = ((tracked_x - commanded_total_x) ** 2 +
                (tracked_y - commanded_total_y) ** 2) ** 0.5

    # CNN model comparison: evaluate all loaded models on the tracked position patch
    cnn_comparison = {}
    if frame_np is not None and recognizer:
        patch = extract_gray_patch(frame_np, tracked_x, tracked_y)
        if patch is not None:
            # Primary recognizer
            cnn_comparison[recognizer.model_version] = round(
                recognizer.classify_patch(patch), 4)
            # Comparison recognizers
            for ver, rec in _vtest_recognizers.items():
                cnn_comparison[ver] = round(rec.classify_patch(patch), 4)

    # YOLO ground truth (at key frames)
    yolo_gt = None
    if run_yolo and frame_np is not None and yolo_detector is not None:
        dets = yolo_detector.detect(frame_np)
        if dets:
            best = max(dets, key=lambda d: d.confidence)
            yolo_gt = {
                "x": int(best.cx), "y": int(best.cy),
                "confidence": round(float(best.confidence), 4),
                "inference_ms": round(float(best.inference_ms), 2),
            }

    result = {
        "utc": utc_now_ms(),
        "monotonic": now_mono,
        "profile": profile_id,
        "step": step,
        "direction": direction,
        "velocity_px_s": round(velocity_px_s, 1),
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
        "cnn_comparison": cnn_comparison,
        "daemon": {
            "fps": round(state.fps, 1),
            "blob_count": state.blob_count,
        },
        "yolo": {
            "active": state.yolo_active,
            "x": state.yolo_x,
            "y": state.yolo_y,
            "confidence": round(state.yolo_confidence, 3),
            "inference_ms": round(state.yolo_inference_ms, 1),
            "detection_count": state.yolo_detection_count,
        },
        "esp32": {
            "x": mouse.estimated_x if mouse else 0,
            "y": mouse.estimated_y if mouse else 0,
            "soft_bounds_px": mouse.soft_bounds_px if mouse else 0,
        },
        "noise_grid": noise_grid_features(),
        "error_px": round(error_px, 1),
        "label": _auto_label(error_px),
        "frame": frame_name,
    }
    if yolo_gt:
        result["yolo_ground_truth"] = yolo_gt
    return result


def _run_velocity_test(profiles: list[str]):
    """Background thread: run velocity test profiles sequentially.

    New velocity-based approach:
    - Generates precise HID reports from velocity/duration specs
    - Sends at 50Hz (20ms per report) for accurate velocity control
    - Captures snapshots at ~5Hz (every 4th report) to match ring buffer rate
    - Tests in two directions: RIGHT then DOWN (diagonal profiles go both)
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

    REPORT_INTERVAL_S = 0.02  # 50Hz HID reports
    SNAPSHOT_EVERY_N = 4      # Capture snapshot every 4th report (~12.5Hz)
    SOFT_BOUNDS_PX = 150      # Keep cursor 150px from screen edges

    try:
        # Enable soft bounds for the duration of the test
        if mouse:
            mouse.set_soft_bounds(SOFT_BOUNDS_PX)
            print(f"[vtest] Soft bounds enabled: {SOFT_BOUNDS_PX}px margin")
        for pid in profiles:
            if not _vtest_running:
                break
            if pid not in VELOCITY_PROFILES:
                continue

            prof = VELOCITY_PROFILES[pid]

            # Handle REPLAY profile separately
            if prof.get("replay"):
                reports = _replay_loss_event_reports()
                if not reports:
                    print(f"[vtest] REPLAY: no loss events to replay")
                    continue
                directions = [("replay", reports)]
            else:
                velocity = prof.get("velocity_px_s", 0)
                duration = prof.get("duration_s", 1.0)
                accel = prof.get("accel_px_s2", 0)
                jitter = prof.get("jitter_px", 0)
                is_diagonal = prof.get("diagonal", False)

                if is_diagonal:
                    # Diagonal: split velocity equally between X and Y
                    norm = 2 ** 0.5
                    direction_vec = (1 / norm, 1 / norm)
                    reports = _generate_hid_reports(
                        velocity, duration, direction=direction_vec,
                        accel=accel, jitter_px=jitter)
                    directions = [("diagonal", reports)]
                else:
                    # Generate reports for RIGHT and DOWN
                    reports_right = _generate_hid_reports(
                        velocity, duration, direction=(1, 0),
                        accel=accel, jitter_px=jitter)
                    reports_down = _generate_hid_reports(
                        velocity, duration, direction=(0, 1),
                        accel=accel, jitter_px=jitter)
                    directions = [("right", reports_right), ("down", reports_down)]

            for direction, reports in directions:
                if not _vtest_running:
                    break

                total_reports = len(reports)
                _vtest_progress = {
                    "profile": pid,
                    "direction": direction,
                    "step": 0,
                    "total": total_reports,
                    "phase": "testing",
                    "velocity_px_s": prof.get("velocity_px_s", 0),
                }

                # Record starting position
                start_esp_x = mouse.estimated_x if mouse else 0
                start_esp_y = mouse.estimated_y if mouse else 0
                cum_dx, cum_dy = 0.0, 0.0
                bounds_hit_logged = False

                for i, (dx, dy) in enumerate(reports):
                    if not _vtest_running:
                        break

                    # Send HID report
                    if mouse and (dx != 0 or dy != 0):
                        actual_dx, actual_dy = mouse.move(
                            dx, dy, respect_bounds=True)
                        cum_dx += actual_dx
                        cum_dy += actual_dy
                        # Log first time soft bounds clamps movement
                        if not bounds_hit_logged and (actual_dx != dx or actual_dy != dy):
                            bounds_hit_logged = True
                            print(f"[vtest] {pid}_{direction}: soft bounds hit at "
                                  f"step {i}/{total_reports}, "
                                  f"ESP ({mouse.estimated_x},{mouse.estimated_y})")

                    # Sleep for one HID report interval
                    time.sleep(REPORT_INTERVAL_S)

                    # Capture snapshot at reduced rate
                    if i % SNAPSHOT_EVERY_N == 0 or i == total_reports - 1:
                        commanded_x = start_esp_x + cum_dx
                        commanded_y = start_esp_y + cum_dy
                        # Compute instantaneous velocity for this report
                        inst_vel = ((dx ** 2 + dy ** 2) ** 0.5) / REPORT_INTERVAL_S
                        # Run YOLO on last frame of each profile for ground truth
                        run_yolo = (i == total_reports - 1)
                        snapshot = _capture_snapshot(
                            pid, i, direction, dx, dy,
                            commanded_x, commanded_y, run_dir,
                            velocity_px_s=inst_vel,
                            run_yolo=run_yolo,
                        )
                        _vtest_results.append(snapshot)

                        with open(log_path, "a") as f:
                            f.write(json.dumps(snapshot) + "\n")

                    _vtest_progress["step"] = i + 1

                # Return to start position after direction run
                if mouse and (cum_dx != 0 or cum_dy != 0):
                    mouse.move(int(-cum_dx), int(-cum_dy))
                    time.sleep(0.5)  # Settle after return

                # Log per-direction summary
                dir_results = [r for r in _vtest_results
                               if r["profile"] == pid and r["direction"] == direction]
                if dir_results:
                    acc = sum(1 for r in dir_results if r["label"] == "accurate")
                    lost = sum(1 for r in dir_results if r["label"] == "lost")
                    methods = set(r["tracked"]["method"] for r in dir_results)
                    errors = [r["error_px"] for r in dir_results]
                    print(f"[vtest] {pid}_{direction}: "
                          f"acc={acc}/{len(dir_results)} "
                          f"({100*acc/len(dir_results):.0f}%), "
                          f"lost={lost}, "
                          f"err_mean={sum(errors)/len(errors):.1f}, "
                          f"err_max={max(errors):.1f}, "
                          f"methods={methods}")

            # Recalibrate between profiles: move to known corner
            # then position at center for next profile
            if mouse:
                _vtest_progress["phase"] = "recalibrating"
                esp_before = (mouse.estimated_x, mouse.estimated_y)
                mouse.calibrate_to_corner('top_left')
                time.sleep(0.3)
                # Move to center of screen for next test
                mouse.move(mouse.screen_width // 2,
                           mouse.screen_height // 2,
                           respect_bounds=False)
                time.sleep(0.5)  # Let tracker settle
                esp_after = (mouse.estimated_x, mouse.estimated_y)

                # Log recalibration event
                recal_event = {
                    "type": "recalibration",
                    "utc": utc_now_ms(),
                    "profile_completed": pid,
                    "esp_before": {"x": esp_before[0], "y": esp_before[1]},
                    "esp_after": {"x": esp_after[0], "y": esp_after[1]},
                    "tracked_after": {"x": state.cursor_x, "y": state.cursor_y},
                    "drift_px": round(
                        ((esp_after[0] - state.cursor_x) ** 2 +
                         (esp_after[1] - state.cursor_y) ** 2) ** 0.5, 1),
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(recal_event) + "\n")
                print(f"[vtest] Recalibrated after {pid}: "
                      f"ESP ({esp_after[0]},{esp_after[1]}) "
                      f"tracked ({state.cursor_x},{state.cursor_y}) "
                      f"drift={recal_event['drift_px']}px")

        # Compute summary stats per profile
        _vtest_summary = _compute_vtest_summary(_vtest_results)

        # Save summary
        with open(run_dir / "summary.json", "w") as f:
            json.dump(_vtest_summary, f, indent=2)

        # Print final summary table
        print(f"\n[vtest] ═══ VELOCITY TEST COMPLETE ═══")
        print(f"[vtest] {'Profile':<16} {'Acc%':>6} {'ErrMean':>8} {'ErrMax':>8} {'Methods'}")
        print(f"[vtest] {'─'*60}")
        for key in sorted(_vtest_summary.keys()):
            v = _vtest_summary[key]
            print(f"[vtest] {key:<16} {v['accuracy_pct']:>5.1f}% "
                  f"{v['error_mean']:>8.1f} {v['error_max']:>8.1f} "
                  f"{v['methods']}")

    except Exception as e:
        import traceback
        traceback.print_exc()
        _vtest_progress["phase"] = f"error: {e}"
    finally:
        _vtest_running = False
        _vtest_progress["phase"] = "done"
        # Disable soft bounds after test
        if mouse:
            mouse.set_soft_bounds(0)
            print(f"[vtest] Soft bounds disabled")


def _compute_vtest_summary(results: list[dict]) -> dict:
    """Aggregate velocity test results per profile + direction.

    Includes CNN model comparison stats: mean confidence per model version
    at each velocity level.
    """
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
        velocities = [r.get("velocity_px_s", 0) for r in records]
        n = len(records)
        label_counts = {
            "accurate": labels.count("accurate"),
            "drift": labels.count("drift"),
            "wrong": labels.count("wrong"),
            "lost": labels.count("lost"),
        }

        # CNN comparison: mean confidence per model version
        cnn_means = {}
        all_cnn = defaultdict(list)
        for r in records:
            for ver, conf in r.get("cnn_comparison", {}).items():
                all_cnn[ver].append(conf)
        for ver, confs in all_cnn.items():
            cnn_means[ver] = round(statistics.mean(confs), 4) if confs else 0

        summary[key] = {
            "count": n,
            "velocity_px_s_mean": round(statistics.mean(velocities), 1) if velocities else 0,
            "velocity_px_s_max": round(max(velocities), 1) if velocities else 0,
            "error_mean": round(statistics.mean(errors), 1) if errors else 0,
            "error_max": round(max(errors), 1) if errors else 0,
            "error_min": round(min(errors), 1) if errors else 0,
            "error_p95": round(sorted(errors)[int(n * 0.95)] if n >= 2 else errors[0], 1) if errors else 0,
            "labels": label_counts,
            "accuracy_pct": round(100 * label_counts["accurate"] / n, 1) if n > 0 else 0,
            "methods": list(set(r["tracked"]["method"] for r in records)),
            "cnn_comparison": cnn_means,
        }

    return summary


@app.post("/api/velocity_test")
async def start_velocity_test(request: Request):
    """Start velocity/acceleration test in background thread.

    POST body (optional): {"profiles": ["V0","V100","V500","V1K","A_UP","D500",...]}
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
    return _serve_page("loss-events.html")


# ── YOLO Gallery API ─────────────────────────────────────────────────

@app.get("/api/yolo/frames")
async def yolo_frames_list():
    """List all loss event frames available for YOLO detection."""
    events_dir = Path(__file__).parent.parent.parent / "data" / "loss_events"
    if not events_dir.exists():
        return {"events": [], "total_frames": 0}
    result = []
    total = 0
    for edir in sorted(events_dir.iterdir(), reverse=True):
        if not edir.is_dir():
            continue
        frames_dir = edir / "frames"
        if not frames_dir.exists():
            continue
        fnames = sorted(f.name for f in frames_dir.iterdir() if f.suffix in (".jpg", ".png"))
        if not fnames:
            continue
        # Read manifest for metadata
        meta = {}
        mf = edir / "manifest.json"
        if mf.exists():
            try:
                meta = json.loads(mf.read_text())
            except Exception:
                pass
        result.append({
            "event_id": edir.name,
            "frames": fnames,
            "count": len(fnames),
            "trigger": meta.get("trigger", "unknown"),
            "timestamp": meta.get("trigger_time", ""),
        })
        total += len(fnames)
    return {"events": result, "total_frames": total}


@app.get("/api/yolo/detect/{event_id}/{frame_name:path}")
async def yolo_detect_frame(event_id: str, frame_name: str):
    """Run YOLO on a loss-event frame, return annotated JPEG."""
    global yolo_detector
    frame_path = Path(__file__).parent.parent.parent / "data" / "loss_events" / event_id / "frames" / frame_name
    if not frame_path.exists():
        return Response(status_code=404, content="Frame not found")
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return Response(status_code=500, content="Failed to read frame")
    # Run YOLO
    dets = []
    if yolo_detector is not None:
        dets = yolo_detector.detect(frame)
    # Draw detections on frame
    for d in dets:
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), (255, 255, 0), 2)
        label = f"cursor {d.confidence:.2f}"
        cv2.putText(frame, label, (d.x1, d.y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.circle(frame, (d.cx, d.cy), 5, (0, 255, 0), -1)
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/yolo/detect_json/{event_id}/{frame_name:path}")
async def yolo_detect_json(event_id: str, frame_name: str):
    """Run YOLO on a loss-event frame, return JSON results."""
    global yolo_detector
    frame_path = Path(__file__).parent.parent.parent / "data" / "loss_events" / event_id / "frames" / frame_name
    if not frame_path.exists():
        return {"error": "Frame not found"}
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return {"error": "Failed to read frame"}
    # Read timeline for tracked position
    tracked_x, tracked_y = None, None
    tl_path = frame_path.parent.parent / "timeline.jsonl"
    if tl_path.exists():
        for line in tl_path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("frame", "").endswith(frame_name):
                tracked_x = entry.get("cursor", {}).get("x")
                tracked_y = entry.get("cursor", {}).get("y")
                break
    dets = []
    if yolo_detector is not None:
        dets = yolo_detector.detect(frame)
    return {
        "frame": frame_name,
        "event_id": event_id,
        "detections": [
            {"x1": int(d.x1), "y1": int(d.y1), "x2": int(d.x2), "y2": int(d.y2),
             "cx": int(d.cx), "cy": int(d.cy), "confidence": round(float(d.confidence), 4),
             "inference_ms": round(float(d.inference_ms), 2)}
            for d in dets
        ],
        "tracked_position": {"x": tracked_x, "y": tracked_y} if tracked_x is not None else None,
        "yolo_loaded": yolo_detector is not None and yolo_detector.loaded,
    }


@app.get("/yolo", response_class=HTMLResponse)
async def yolo_gallery_page():
    """YOLO detection gallery — browse loss event frames with YOLO results."""
    return _serve_page("yolo.html")


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

    # Load all available CNN model versions for velocity test comparison
    _model_dir = Path(__file__).parent.parent.parent / "data" / "cursor_models"
    if _model_dir.exists():
        for mf in sorted(_model_dir.glob("v*_*.pt")):
            version = mf.stem  # e.g. "v005_20260220"
            if version != recognizer.model_version:
                try:
                    r = CursorRecognizer(model_file=mf)
                    _vtest_recognizers[version] = r
                except Exception:
                    pass
        if _vtest_recognizers:
            print("  Velocity test comparison models: "
                  + ", ".join(sorted(_vtest_recognizers.keys())))

    # Phase 2: Silhouette tracker (fast ROI-based cursor following)
    sil_tracker = SilhouetteTracker(recognizer)
    print("  Silhouette tracker initialized (ROI: " + str(sil_tracker.roi_size) + "px)")

    # Phase 3: Commanded movement (unified jitter + probe + Lissajous)
    cmd_movement = CommandedMovement(
        mouse, sil_tracker, sensor=sensor, daemon_client=daemon_client)
    print("  Commanded movement initialized (jitter→probe→lissajous)")

    # YOLO full-screen cursor detector (GPU-accelerated)
    yolo_detector = YOLOCursorDetector(conf_threshold=0.15)
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
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="KVM Dashboard")
    parser.add_argument("--passive", action="store_true",
                        help="Passive mode: observe only, no mouse movements")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    if args.passive:
        _passive_mode = True
        print("*** PASSIVE MODE: no mouse movements will be sent ***")

    init_hardware()

    # Init state log size tracker (for rotation)
    if _STATE_LOG_PATH.exists():
        _state_log_bytes = _STATE_LOG_PATH.stat().st_size
        print(f"  State log: {_state_log_bytes / 1024 / 1024:.1f}MB "
              f"({_STATE_LOG_PATH})")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
