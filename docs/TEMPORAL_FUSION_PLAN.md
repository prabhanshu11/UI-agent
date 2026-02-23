# Temporal Fusion Plan — Blob Dataset + Temporal YOLO + Motion Trail + Position Fusion

> **Status:** Plan (not implemented)
> **Created:** 2026-02-20
> **Depends on:** `BLOB_DATASET_PLAN.md`, `CURSOR_DETECTION_V2_PLAN.md`

---

## Problem

Cursor detection is unreliable — CNN and YOLO overlays "don't make sense", models don't
use motion data, and cursor loss recovery is 31% (68.6% never recover). Four independent
improvements address this:

| Track | Summary | Language |
|-------|---------|----------|
| **A** | Blob Dataset — persist per-frame blob data as JSONL | C |
| **B** | Motion Trail — shared utility for trail-informed detection | Python |
| **C** | Temporal YOLO — dual-frame inference for velocity vectors | Python |
| **D** | Position Fusion — retro CNN, confidence gate, YOLO anchor | Python |

These are independent tracks that can be implemented in any order.

---

## Track A: Blob Dataset — C Daemon JSONL Logging

**Full design:** `BLOB_DATASET_PLAN.md`

**File:** `daemon/cursor_daemon.c` (+ recompile)

### Changes Summary

| Location | Change |
|----------|--------|
| Config struct (~line 50) | Add `blob_log_dir`, `no_blob_log` fields |
| Defaults (~line 60) | `.blob_log_dir = NULL, .no_blob_log = 0` |
| `print_usage()` (~line 680) | Add `--blob-log-dir PATH`, `--no-blob-log` docs |
| `parse_args()` (~line 694) | Add `'b'`/`'B'` options to `long_opts[]` and switch |
| After `g_ring` (~line 86) | Add `BlobSnapshot` struct + `g_blob_ring[64]` + atomic head/tail |
| After `now_ns()` (~line 101) | Add `realtime_ns()` using `CLOCK_REALTIME` (wall-clock for dataset) |
| After `enqueue_frame()` (~line 574) | Add `enqueue_blob_snapshot()` — memcpy + atomic store, ~40ns |
| After `g_shm->frame_count++` (~line 944) | Call `enqueue_blob_snapshot()` if `!cfg.no_blob_log` |
| After recording thread code (~line 560) | Add `blob_log_thread()` — drains ring at 100Hz, writes JSONL |
| `main()` after rec thread start (~line 841) | `pthread_create(&blob_tid, NULL, blob_log_thread, NULL)` |
| `main()` default dir setup (~line 742) | Set default `blob_log_dir` to `data/blob_log/` |
| Cleanup (~line 974) | `pthread_join(blob_tid, NULL)` |

### Ring Buffer

```c
#define BLOB_RING_SIZE 64
typedef struct {
    uint64_t timestamp_ns;    // CLOCK_REALTIME
    uint64_t frame_number;
    int      blob_count;
    ShmBlob  blobs[MAX_BLOBS]; // 32 * 24 = 768 bytes
} BlobSnapshot;               // ~792 bytes * 64 = ~50 KB total

static BlobSnapshot g_blob_ring[BLOB_RING_SIZE];
static _Atomic uint32_t g_blob_ring_head = 0;
static _Atomic uint32_t g_blob_ring_tail = 0;
```

### JSONL Schema

```json
{"ts":1771587600123456789,"fn":45021,"bc":3,"b":[{"cx":512,"cy":340,"a":0.785,"px":87,"bx":[500,330,524,350]}]}
```

File rotation: every `CHUNK_DURATION_S` (300s), same cadence as MKV chunks.
Output: `data/blob_log/blobs_YYYYMMDD_HHMMSS.jsonl`

### Portability Note

Use `PRIu64` from `<inttypes.h>` for `uint64_t` format specifiers instead of `%lu`.

---

## Track B: Motion Trail — Shared Utility

**New file:** `src/hardware/motion_trail.py`

Central abstraction consumed by CNN, YOLO, tracker, and dashboard.

### API

```python
@dataclass
class MotionTrailPoint:
    x: int; y: int
    timestamp: float           # time.monotonic()
    velocity_x: float          # px/s, computed from consecutive points
    velocity_y: float
    pixel_count: int
    confidence: float          # from CNN/YOLO, 0.0 if unknown
    source: str                # "daemon_blob", "yolo", "cnn_verify", "dead_reckoning"

class MotionTrail:
    """Thread-safe ring buffer (~120 points = ~2-4s of history)."""

    def append(x, y, pixel_count, confidence, source)  # auto-computes velocity
    def get_recent(max_age_s=2.0) -> list[MotionTrailPoint]
    def velocity_estimate(window_s=0.3) -> tuple[float, float]  # (vx, vy) avg
    def speed_estimate(window_s=0.3) -> float
    def extrapolate(dt=0.1) -> tuple[int, int]  # predicted position
```

### Integration Points (in `kvm_dashboard.py`)

1. On primary blob accepted in `_daemon_motion_loop()`: `trail.append(bx, by, source="daemon_blob")`
2. On YOLO detection: `trail.append(det.cx, det.cy, confidence=det.confidence, source="yolo")`
3. On dead reckoning: `trail.append(mouse.estimated_x, mouse.estimated_y, source="dead_reckoning")`
4. Replace inline `velocity_history` list with `trail.velocity_estimate()`
5. Feed `sil_tracker.set_velocity()` from `trail.speed_estimate()`
6. Expose in `/api/state` for sub-panel visualization

---

## Track C: Temporal YOLO — Dual-Frame Velocity Detection

**New file:** `src/hardware/temporal_yolo.py`

### Architecture

```
Frame Buffer (deque, 16 slots = 3.2s at 5Hz)
    |
    +-- Current frame --> YOLO detect() --> detections_now
    |                                          |
    +-- Lagged frame (1s ago) -> YOLO detect() -> detections_lag
                                                   |
                                    Match by proximity + class_id
                                                   |
                                    velocity = displacement / time_delta
```

### Key Classes

```python
class FrameBuffer:
    """deque[TimestampedFrame] with maxlen=16. Push every decoded frame."""
    def push(frame, timestamp, frame_number)
    def get_lagged(target_lag=1.0, tolerance=0.3) -> Optional[TimestampedFrame]

class TemporalYOLODetector:
    """Wraps YOLOCursorDetector, adds frame buffer + velocity computation."""
    def push_frame(frame, timestamp)  # call on every decoded frame
    def detect_with_velocity(frame, timestamp) -> TemporalResult
    # Internally: detect current + detect lagged -> match -> compute velocity

@dataclass
class TemporalDetection(YOLODetection):
    vx: float; vy: float; speed: float; direction: float
    has_velocity: bool; lag_cx: int; lag_cy: int; time_delta: float
```

### Integration in `kvm_dashboard.py`

1. Initialize: `temporal_yolo = TemporalYOLODetector(yolo_detector, lag_seconds=1.0)`
2. Push every decoded frame: `temporal_yolo.push_frame(frame, time.monotonic())`
3. Replace full-frame YOLO fallback with `temporal_yolo.detect_with_velocity()`
4. Add `temporal` dict to `/api/state`: `{vx, vy, speed, direction, has_velocity, lag_s, inference_ms}`

### Performance Budget

| Metric | Value |
|--------|-------|
| Single YOLO inference | ~15-30ms (RTX 2060 SUPER) |
| Dual YOLO (sequential, shared weights) | ~30-60ms |
| Frame decode rate | 5Hz = 200ms between frames |
| Dual inference fits in budget | 60ms << 200ms |
| Frame buffer memory (16 frames) | ~95MB (reduce to 8 -> ~48MB if tight) |

---

## Track D: Position Fusion Enhancements

**Full design:** `CURSOR_DETECTION_V2_PLAN.md`

**Files:** `src/hardware/cursor_tracker.py`, `src/hardware/position_fusion.py`

### D1. Retro CNN Recovery

After CNN EMA drops to concern zone (0.3-0.8), try `_retro_cnn_recover()` BEFORE probe
movements. Runs `classify_patch()` at last-known position. If conf > 0.8, position
confirmed — no mouse movement needed.

**Impact:** Recovery rate 31% -> ~63% (43/71 non-recovered events had retro_cnn > 0.8).

### D2. Trail-Guided CNN Verification

Modify `_cnn_verify()`: instead of extracting 64x64 patch at stored `pos.x, pos.y`,
use `trail.extrapolate(0.0)` to predict where cursor IS NOW based on recent motion.
Extract patch at predicted position.

### D3. YOLO Trail-Informed Scoring

Modify `YOLOAnchor.check()` to score YOLO detections by consistency with motion trail:
- Detection near trail prediction -> confidence boost (1.5x)
- Detection far from prediction during stillness -> confidence reduction (0.6x)
- Pass `MotionTrail` to `YOLOAnchor.check()` for trail-predicted position

### D4. Confidence Gate

Replace age-based 3.0s stale threshold with fused confidence:
- **High (>0.8):** Normal — CNN confirms, trail fresh, YOLO corroborates
- **Medium (0.3-0.8):** Try retro CNN, increase YOLO scan rate to 0.5s
- **Low (<0.3):** Immediate escalation — no 10s wait

### D5. Dead Reckoning Integration

`ESP32Mouse._send_raw()` already accumulates `estimated_x/y`. Connect it:
- In `_track_loop()`, feed DR position to trail with `source="dead_reckoning", confidence=0.3`
- When YOLO/CNN corrects position, call `mouse.set_position()` to resync DR

---

## Implementation Order

### Phase 1: Foundation (no behavior change)
1. **Track A:** C daemon blob JSONL logging (independent, pure C)
2. **Track B:** Create `motion_trail.py` with `MotionTrail` class + unit tests

### Phase 2: Wire the trail (populate, not yet consume for decisions)
3. In `kvm_dashboard.py`: instantiate `MotionTrail`, feed from blobs + YOLO + DR
4. Replace inline `velocity_history` with `trail.velocity_estimate()`
5. Expose trail in `/api/state` for sub-panel

### Phase 3: Trail-informed detection
6. **Track D1:** Retro CNN recovery in `cursor_tracker.py`
7. **Track D2:** Trail-guided `_cnn_verify()` — patch at predicted position
8. **Track D3:** YOLO trail scoring in `position_fusion.py`

### Phase 4: Temporal YOLO
9. **Track C:** Create `temporal_yolo.py` with `TemporalYOLODetector`
10. Integrate into `kvm_dashboard.py` (push frames, replace full-frame fallback)
11. Add temporal velocity data to `/api/state`

### Phase 5: Confidence gate
12. **Track D4:** Replace stale threshold with fused confidence in `_track_loop()`
13. **Track D5:** Dead reckoning integration
14. Wire `YOLOAnchor` into `CursorTracker` as optional parameter

---

## Files Summary

| File | Action | Track |
|------|--------|-------|
| `daemon/cursor_daemon.c` | Modify: blob ring buffer + blob_log_thread + CLI flags | A |
| `src/hardware/motion_trail.py` | **New**: MotionTrail shared utility | B |
| `src/hardware/temporal_yolo.py` | **New**: TemporalYOLODetector + FrameBuffer | C |
| `src/hardware/cursor_tracker.py` | Modify: retro CNN, trail-guided verify, confidence gate | D |
| `src/hardware/position_fusion.py` | Modify: YOLOAnchor accepts trail, trail scoring | D |
| `src/web/kvm_dashboard.py` | Modify: instantiate trail + temporal YOLO, feed data, expose in API | B,C,D |

---

## Verification

| Track | How to verify |
|-------|---------------|
| Blob JSONL | Restart daemon -> `ls data/blob_log/` -> JSONL files appear, growing at ~1.2 KB/s idle |
| Motion Trail | `curl localhost:8766/api/state \| jq .trail` -> shows recent points with velocity |
| Temporal YOLO | Model inference sub-panel shows velocity arrows; `/api/state` has `temporal.speed > 0` during movement |
| Retro CNN | Loss events directory shows fewer entries (recovery rate improves) |
| Confidence Gate | Cursor recovery time drops from 8.6s mean to <2s |
