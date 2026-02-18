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
from datetime import datetime
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
    error: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

state = MotionState()


@dataclass
class DashboardConfig:
    """Runtime-tunable parameters exposed via /api/config."""
    frame_fps: float = 1.0          # Overlayed video render rate (Hz)
    cnn_interval_s: float = 30.0    # CNN validation cycle (seconds)
    jitter_interval_s: float = 30.0 # Anti-sleep jitter interval
    jitter_pixels: int = 3          # Jitter amplitude (px)
    lock: threading.Lock = field(default_factory=threading.Lock)

config = DashboardConfig()


mouse: Optional[ESP32Mouse] = None
sensor: Optional[HDMISensor] = None
tracker: Optional[CursorTracker] = None
recognizer: Optional[CursorRecognizer] = None
collector: Optional[CursorSampleCollector] = None
daemon_client: Optional[DaemonClient] = None


# ── Motion detection service (background thread) ───────────────────

def draw_blob_overlay(frame: np.ndarray, blobs: list[MotionBlob],
                      cursor_pos: tuple[int, int],
                      pointer_blob_idx: int = -1) -> np.ndarray:
    """Draw motion blob rectangles and cursor crosshair on frame.

    Args:
        frame: RGB frame to draw on.
        blobs: Motion blobs to highlight.
        cursor_pos: (x, y) of known cursor position.
        pointer_blob_idx: Index of the blob identified as cursor (-1 if none).
            This blob gets green coloring; all others get blue.
    """
    overlay = frame.copy()

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

    # Cursor position and HUD info are shown by the browser-side CSS dot
    # and sidebar panels respectively — no need to draw on the video frame.

    return overlay


def frame_to_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _daemon_motion_loop():
    """Read blobs + frames from C daemon shm. Cursor ID stays in Python.

    Two update rates:
    - State (blobs, cursor dot): ~20Hz — smooth CSS cursor overlay
    - Frame (JPEG decode + draw overlays + re-encode): config.frame_fps (default 1Hz)

    Jitter correlation: when the anti-sleep jitter fires (every 30s),
    the resulting motion blob is matched to the cursor's last known
    position. This keeps tracker.position continuously maintained
    without any extra probe movements.
    """
    last_frame_time = 0.0

    while True:
        try:
            ds = daemon_client.read_state()
            if ds is None:
                time.sleep(0.05)
                continue

            now = time.monotonic()

            # ── Continuous cursor tracking from daemon blobs ────────
            pos = tracker.position if tracker else None
            cursor_blobs = [b for b in ds.blobs if 5 <= b.pixel_count <= 500]
            if cursor_blobs:
                if pos and (now - pos.timestamp < 10):
                    # Position fresh — pick blob closest to last known
                    best = min(cursor_blobs, key=lambda b:
                        ((b.centroid[0] - pos.x) ** 2 +
                         (b.centroid[1] - pos.y) ** 2) ** 0.5)
                    dist = ((best.centroid[0] - pos.x) ** 2 +
                            (best.centroid[1] - pos.y) ** 2) ** 0.5
                    if dist < 200:
                        bx, by = best.centroid
                        if tracker:
                            tracker.set_position(bx, by, method="motion_track")
                        pos = tracker.position
                else:
                    # Position stale (>10s) — pick smallest as initial guess
                    best = min(cursor_blobs, key=lambda b: b.pixel_count)
                    bx, by = best.centroid
                    if tracker:
                        tracker.set_position(bx, by, method="motion_track")
                    pos = tracker.position

            # ── Update shared state (fast, ~20Hz) ──────────────────
            cursor_pos = (0, 0)
            with state.lock:
                state.blobs = ds.blobs
                state.blob_count = ds.blob_count
                state.fps = ds.fps
                state.frame_count = ds.frame_count
                if pos:
                    state.cursor_x = pos.x
                    state.cursor_y = pos.y
                    state.cursor_age_s = now - pos.timestamp
                    state.cursor_method = pos.method
                    cursor_pos = (pos.x, pos.y)

            # ── Render overlayed frame (slow, config.frame_fps) ────
            frame_interval = 1.0 / max(0.1, config.frame_fps)
            if now - last_frame_time >= frame_interval:
                jpeg = daemon_client.read_jpeg()
                if jpeg:
                    # Decode → draw overlays → re-encode
                    frame = cv2.imdecode(
                        np.frombuffer(jpeg, dtype=np.uint8),
                        cv2.IMREAD_COLOR)
                    if frame is not None:
                        # cv2 decodes to BGR; draw_blob_overlay expects RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        overlay = draw_blob_overlay(
                            frame_rgb, ds.blobs, cursor_pos)
                        overlay_jpeg = frame_to_jpeg(overlay)
                        with state.lock:
                            state.frame_overlay = overlay_jpeg
                            state.frame_raw = frame_rgb
                    last_frame_time = now

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
    """Original Python motion detection loop (fallback when daemon not running)."""
    prev_frame = None
    frame_times = []

    while True:
        try:
            t0 = time.monotonic()
            frame = sensor.capture(settle_frames=1)
            state.frame_count += 1

            blobs = []
            if prev_frame is not None:
                blobs = extract_motion_blobs(
                    prev_frame, frame,
                    threshold=15, min_pixels=5, max_pixels=3000,
                )

            pos = tracker.position if tracker else None
            cursor_pos = (0, 0)
            if pos:
                cursor_pos = (pos.x, pos.y)
                state.cursor_x = pos.x
                state.cursor_y = pos.y
                state.cursor_method = pos.method
                state.cursor_age_s = time.monotonic() - pos.timestamp

            # Auto-track: update cursor position from cursor-sized blobs
            pointer_blob_idx = -1
            if blobs:
                cursor_blobs = [b for b in blobs if 5 <= b.pixel_count <= 500]
                position_stale = not pos or (time.monotonic() - pos.timestamp > 10)
                if cursor_blobs:
                    best = None
                    bx, by = 0, 0

                    # Shape-informed scoring: if we have a cursor template,
                    # score blobs by shape match instead of just size
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
                            # Pick highest shape score
                            if scored and scored[0][1] > 0.5:
                                best = scored[0][0]
                                bx, by = best.centroid
                                state.cursor_method = "shape_track"
                        else:
                            # Among high-scoring blobs, pick closest to known position
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

                    # Fallback: original size-based heuristic
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
                        # Track which blob is the pointer for color coding
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

            overlay = draw_blob_overlay(frame, blobs, cursor_pos, pointer_blob_idx)
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
            elapsed = time.monotonic() - t0
            time.sleep(max(0, 0.05 - elapsed))

        except Exception as e:
            state.error = str(e)
            time.sleep(1)


def jitter_monitor_loop():
    while True:
        try:
            state.mouse_connected = mouse.status() == "CONNECTED" if mouse else False
            state.heartbeat_active = (
                hasattr(mouse, '_heartbeat_stop') and
                not mouse._heartbeat_stop.is_set()
            ) if mouse else False
            state.anti_sleep_active = (
                hasattr(mouse, '_anti_sleep_stop') and
                not mouse._anti_sleep_stop.is_set()
            ) if mouse else False
        except Exception:
            pass
        time.sleep(2)


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

    # Move RIGHT
    mouse._send_raw(amp, 0)
    time.sleep(0.15)
    frame_b = _capture()

    # Move DOWN
    mouse._send_raw(0, amp)
    time.sleep(0.15)
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
    best_match = None
    best_dist = 9999
    for ba in blobs_ab:
        for bc in blobs_bc:
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
       - confidence > 0.7: confirmed, update template
       - confidence < 0.7: dual-probe re-acquire

    The dual-probe re-acquire moves cursor right then down, captures 3
    frames, and matches blobs across both directions. Only the real cursor
    follows both movements — UI noise doesn't.
    """
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
                # Position unknown or very stale — skip CNN, go straight
                # to jitter re-acquire. This is the startup path and the
                # recovery path when cursor is lost.
                result = _jitter_reacquire(use_daemon)
                if result:
                    state.cursor_validated = False  # Need CNN to confirm
                    state.cnn_confidence = 0.0
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
                # Cursor confirmed at known position
                recognizer.update_cursor_template(patch)
                state.cursor_validated = True
                if collector:
                    collector.collect_from_tracking(
                        frame, cx, cy, confidence=confidence)
                    state.sample_count_pos = collector.positive_count
                    state.sample_count_neg = collector.negative_count
            else:
                # CNN says cursor NOT here — jitter re-acquire
                state.cursor_validated = False
                _jitter_reacquire(use_daemon)

        except Exception:
            pass  # Non-critical — will retry next cycle


# ── API endpoints ───────────────────────────────────────────────────

@app.get("/")
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/api/frame")
async def get_frame():
    with state.lock:
        jpeg = state.frame_overlay
    if jpeg:
        return Response(content=jpeg, media_type="image/jpeg")
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
        },
        "hardware": {
            "mouse_connected": state.mouse_connected,
            "heartbeat_active": state.heartbeat_active,
            "anti_sleep_active": state.anti_sleep_active,
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
        "fps": round(state.fps, 1),
        "frame_count": state.frame_count,
        "error": state.error,
        "timestamp": datetime.now().isoformat(),
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
        mouse._send_raw(amplitude, 0)
        time.sleep(0.15)
        if use_daemon:
            jpeg2 = daemon_client.read_jpeg()
            frame_after = _cv2.imdecode(
                _np.frombuffer(jpeg2, dtype=_np.uint8), _cv2.IMREAD_COLOR) if jpeg2 else frame_before
        else:
            frame_after = sensor.capture(settle_frames=1)
        mouse._send_raw(-amplitude, 0)

        # Use cursor shape filter to distinguish cursor from UI changes
        blobs = extract_motion_blobs(
            frame_before, frame_after,
            threshold=25, min_pixels=5, max_pixels=500,
            cursor_shape_filter=True,
        )

        # Also get unfiltered for comparison
        all_blobs = extract_motion_blobs(
            frame_before, frame_after,
            threshold=25, min_pixels=5, max_pixels=3000,
        )

        if not blobs:
            # Fallback: use smallest blob near expected area (cursor is small)
            small_blobs = [b for b in all_blobs if b.pixel_count < 200]
            small_blobs.sort(key=lambda b: b.pixel_count)
            if small_blobs:
                blobs = small_blobs

        if not blobs:
            return {"error": "No cursor detected", "total_blobs": len(all_blobs)}

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
        # Restart jitter with new interval
        if mouse:
            mouse.stop_anti_sleep()
            mouse.start_anti_sleep(
                interval=config.jitter_interval_s,
                pixels=config.jitter_pixels)
    if "jitter_pixels" in data:
        config.jitter_pixels = max(1, min(30, int(data["jitter_pixels"])))
        # Restart jitter with new amplitude
        if mouse:
            mouse.stop_anti_sleep()
            mouse.start_anti_sleep(
                interval=config.jitter_interval_s,
                pixels=config.jitter_pixels)

    return await get_config()


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
        .feed { background: #000; position: relative; overflow: hidden; }
        .feed img { width: 100%; height: 100%; object-fit: contain; }
        .cursor-dot {
            position: absolute; width: 24px; height: 24px;
            border-radius: 50%; pointer-events: none;
            background: radial-gradient(circle, #fff 20%, #00ff00 50%, transparent 70%);
            box-shadow: 0 0 12px #00ff00, 0 0 24px #00ff0088;
            transform: translate(-50%, -50%);
            animation: cursor-pulse 1.5s ease-in-out infinite;
            z-index: 10; display: none;
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
        <h1>KVM MOTION DETECTOR</h1>
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
            <img id="live-frame" src="" alt="Loading...">
            <div class="cursor-dot" id="cursor-dot"></div>
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
                    <div class="tier t1" id="tier-1">T1 Micro</div>
                    <div class="tier t2" id="tier-2">T2 Macro</div>
                    <div class="tier t3" id="tier-3">T3 Lissajous</div>
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
                <div class="panel-title">Anti-Sleep Jitter</div>
                <div class="hw-row">
                    <span class="hw-label">Status</span>
                    <span class="hw-value" id="jitter-status">—</span>
                </div>
                <div class="jitter-bar" id="jitter-bar">
                    <div class="jitter-pulse"></div>
                </div>
                <div style="font-size:0.65rem; color:#444; margin-top:3px;">
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

                document.getElementById('fps-display').textContent = s.fps + ' FPS';

                setDot('dot-mouse', s.hardware.mouse_connected ? 'green' : 'red');
                setDot('dot-heartbeat', s.hardware.heartbeat_active ? 'green' : 'gray');
                setDot('dot-jitter', s.hardware.anti_sleep_active ? 'green' : 'gray');
                setDot('dot-keyboard',
                    s.pi_keyboard === true ? 'green' :
                    s.pi_keyboard === false ? 'yellow' : 'gray');

                document.getElementById('cursor-coords').textContent =
                    '(' + s.cursor.x + ', ' + s.cursor.y + ')';
                document.getElementById('cursor-method').textContent = s.cursor.method;
                document.getElementById('cursor-age').textContent = s.cursor.age_s;
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
                if (s.hardware.anti_sleep_active) {
                    jitterStatus.textContent = 'Active';
                    jitterStatus.className = 'hw-value ok';
                    jitterBar.classList.add('active');
                } else {
                    jitterStatus.textContent = 'Inactive';
                    jitterStatus.className = 'hw-value warn';
                    jitterBar.classList.remove('active');
                }

                setHW('hw-mouse', s.hardware.mouse_connected, 'Connected', 'Disconnected');
                setHW('hw-heartbeat', s.hardware.heartbeat_active, 'Active (100ms)', 'Inactive');
                setHW('hw-keyboard',
                    s.pi_keyboard === true, 'BT Connected',
                    s.pi_keyboard === false ? 'Discoverable' : 'Unreachable');
                document.getElementById('hw-frames').textContent = s.frame_count;

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
            var dot = document.getElementById('cursor-dot');
            var img = document.getElementById('live-frame');
            var container = document.getElementById('feed-container');
            if (!img.naturalWidth || cursorX <= 0 && cursorY <= 0) {
                dot.style.display = 'none';
                return;
            }
            // Map 1920x1080 cursor coords to the displayed image position
            var containerRect = container.getBoundingClientRect();
            var imgRect = img.getBoundingClientRect();
            var scaleX = imgRect.width / 1920;
            var scaleY = imgRect.height / 1080;
            var offsetX = imgRect.left - containerRect.left;
            var offsetY = imgRect.top - containerRect.top;
            dot.style.left = (offsetX + cursorX * scaleX) + 'px';
            dot.style.top = (offsetY + cursorY * scaleY) + 'px';
            dot.style.display = 'block';
        }

        // State updates fast (smooth cursor dot), frame updates slow (1fps)
        // Frame is rendered at config.frame_fps (default 1Hz) so polling
        // faster just wastes bandwidth returning the same cached JPEG.
        function scheduleFrame() {
            updateFrame();
            setTimeout(scheduleFrame, 1000);  // 1fps — matches default frame_fps
        }
        setInterval(updateState, 200);  // 5Hz — smooth cursor dot movement
        scheduleFrame();
        updateState();
    </script>
</body>
</html>
"""


# ── Startup ─────────────────────────────────────────────────────────

def init_hardware():
    global mouse, sensor, tracker, recognizer, collector, daemon_client

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

    mouse.start_anti_sleep(interval=30.0, pixels=3)
    print("  Anti-sleep jitter started (30s)")

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

    threading.Thread(target=motion_detection_loop, daemon=True).start()
    print("  Motion detection loop started")

    threading.Thread(target=jitter_monitor_loop, daemon=True).start()
    threading.Thread(target=pi_keyboard_monitor_loop, daemon=True).start()
    threading.Thread(target=cursor_validation_loop, daemon=True).start()
    print("  Cursor validation loop started (30s cycle)")

    print("\nDashboard: http://localhost:8766")


if __name__ == "__main__":
    import uvicorn
    init_hardware()
    uvicorn.run(app, host="0.0.0.0", port=8766)
