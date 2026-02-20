# Cursor Detection V2 — Position Fusion Engine

**Date:** 2026-02-20
**Based on:** Analysis of 102 loss events from `data/loss_events/ANALYSIS_REPORT.md`
**Status:** Plan (not yet implemented)

---

## Executive Summary

The current cursor tracking system loses the cursor in 0.32 seconds and takes 8.6 seconds to recover (when it recovers at all — 68.6% of events never recover). The root cause is architectural: the system is **reactive** (wait 10s, then probe) instead of **continuous** (always know where the cursor is).

This plan proposes a Position Fusion Engine that eliminates the 10-second stale gap entirely.

---

## Part 1: Quick Wins (Small Code Changes, Big Impact)

These can be done independently without architectural changes.

### 1.1 Use Retro CNN for Recovery

**File:** `src/hardware/cursor_tracker.py`
**Impact:** Recovery rate 31% → ~63%

**Finding:** 100% of recovered events had `live_cnn > 0.8` at recovery. But 43/71 non-recovered events had `retro_cnn > 0.8` — the cursor IS visible, the system just doesn't use retro CNN for recovery decisions.

**Change:** In the tracking loop, after capturing a frame for probe verification, also run `cursor_recognizer.classify_patch()` on the frame at the last known position. If confidence > 0.8, accept as confirmed position.

```python
# In _probe_micro() or a new _cnn_verify() method:
patch = extract_patch(frame, pos.x, pos.y, size=64)
conf = self.recognizer.classify_patch(patch)
if conf > 0.8:
    self._position.timestamp = time.monotonic()
    return True  # Position confirmed by CNN without any mouse movement
```

### 1.2 Multi-Signal Consensus Before Loss Declaration

**File:** `src/hardware/cursor_tracker.py`
**Impact:** Eliminates false `cnn_drop` alarms (6+ events)

**Finding:** In several events, live CNN dropped 0.85→0.35 in a single frame while the cursor was stationary AND silhouette confidence was 1.0. The system declared loss despite overwhelming evidence the cursor was still there.

**Change:** Require at least 2 of 3 signals to agree before declaring loss:
- Live CNN confidence
- Silhouette tracker confidence
- Retro CNN confidence (run on demand)

```python
# Don't declare lost if silhouette is confident
if silhouette_conf > 0.8 and retro_cnn_conf > 0.7:
    # Override — cursor is still here despite live CNN drop
    self._position.timestamp = time.monotonic()
    continue
```

### 1.3 Fix (6,15) Ghost Position

**File:** `src/hardware/cursor_tracker.py` (or wherever `set_position` is called initially)
**Impact:** Eliminates 11% of phantom events

**Finding:** 11 events had last-known position at exactly (6, 15) with `method="manual"` — clearly an uninitialized default, not a real detection. The system then "jumps" 1500-2000px when it actually finds the cursor.

**Change:** Guard against invalid initial positions:
```python
def set_position(self, x, y, method="manual"):
    if x < 20 and y < 20 and method == "manual":
        return  # Reject likely-uninitialized position
    ...
```

### 1.4 Temporal Smoothing on Live CNN

**File:** `src/hardware/cursor_cnn.py` or tracking loop
**Impact:** Prevents single-frame CNN drops from triggering false loss

**Finding:** Live CNN drops from 0.85 to 0.0 in one frame, which is too volatile for a loss trigger.

**Change:** Apply exponential moving average:
```python
self._cnn_ema = 0.7 * self._cnn_ema + 0.3 * raw_cnn_confidence
# Use self._cnn_ema for loss decisions, not raw value
```

---

## Part 2: Position Fusion Engine (Architectural Change)

### 2.1 Overview

Replace the current reactive tracking loop with a continuous position fusion system:

```
┌─────────────────────────────────────────────────────────────┐
│                  POSITION FUSION ENGINE                      │
│                                                              │
│  4 Input Streams (parallel, different rates):               │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Dead Reckoning│  │ CNN Patch    │  │ Silhouette   │      │
│  │ from sent     │  │ at predicted │  │ Tracker ROI  │      │
│  │ HID commands  │  │ position     │  │              │      │
│  │ (50 Hz)       │  │ (30 Hz, <2ms)│  │ (20 Hz)      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                  │                  │              │
│         └──────────┬───────┴──────────┬───────┘              │
│                    │                  │                       │
│              ┌─────▼──────┐   ┌──────▼───────┐              │
│              │ POSITION   │   │ YOLO         │              │
│              │ ESTIMATE   │◄──│ Ground Truth │              │
│              │ (x, y, θ)  │   │ (1-2 Hz)     │              │
│              └─────┬──────┘   └──────────────┘              │
│                    │                                         │
│              ┌─────▼──────────────────────┐                 │
│              │ CONFIDENCE GATE            │                 │
│              │ High (>0.8): normal ops    │                 │
│              │ Med (0.3-0.8): YOLO scan   │                 │
│              │ Low (<0.3): immediate      │                 │
│              │   escalation (no 10s wait) │                 │
│              └────────────────────────────┘                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Dead Reckoning from Sent Commands

**File:** `src/hardware/esp32_mouse.py`

Every `_send_raw(dx, dy)` call already knows the exact displacement. Accumulate it:

```python
def _send_raw(self, dx, dy):
    dx = max(-self.HID_MAX, min(self.HID_MAX, dx))
    dy = max(-self.HID_MAX, min(self.HID_MAX, dy))
    self.ser.write(f'MOUSE:{dx},{dy}\n'.encode())
    self.ser.flush()

    # Dead reckoning update
    self.estimated_x += dx
    self.estimated_y += dy
    self._dr_timestamp = time.monotonic()

    time.sleep(0.02)
```

**Limitation:** This only tracks OUR commands. If the user moves the mouse physically on the Windows laptop, we don't see it. This is why YOLO ground truth correction is essential — it catches physical mouse drift.

**Note:** `estimated_x/y` already exist in `ESP32Mouse.__init__()` but are never updated by `_send_raw()`. This is the first fix.

### 2.3 Continuous CNN Verification

**New method in tracking loop**

Instead of only running CNN during loss events, run it continuously on every captured frame:

```python
def _continuous_verify(self, frame):
    """Run CNN at predicted position every frame. <2ms overhead."""
    pos = self.position
    if pos is None:
        return

    patch = extract_64x64_patch(frame, pos.x, pos.y)
    confidence = self.cnn.predict(patch)

    if confidence > 0.8:
        # Confirmed — update timestamp
        pos.timestamp = time.monotonic()
        pos.confidence = confidence
    elif confidence < 0.3:
        # Immediate concern — trigger YOLO scan, don't wait 10s
        self._request_yolo_scan()
```

**Cost:** <2ms per frame at 30fps = 60ms/s total. Negligible.

### 2.4 YOLO Ground Truth Anchor

**File:** New `src/hardware/position_fusion.py`

Run YOLO on full 1920x1080 frames periodically as absolute ground truth:

```python
class YOLOAnchor:
    """Periodic YOLO full-frame scan for position correction."""

    def __init__(self, yolo_detector, interval=1.0):
        self.yolo = yolo_detector
        self.interval = interval  # seconds between scans
        self._last_scan = 0

    def check(self, frame, predicted_pos):
        """Run YOLO if interval elapsed. Returns corrected position or None."""
        now = time.monotonic()
        if now - self._last_scan < self.interval:
            return None

        self._last_scan = now
        detections = self.yolo.detect(frame)

        if not detections:
            return None

        # Find detection closest to predicted position
        best = min(detections, key=lambda d:
            ((d.cx - predicted_pos.x)**2 + (d.cy - predicted_pos.y)**2)**0.5)

        dist = ((best.cx - predicted_pos.x)**2 +
                (best.cy - predicted_pos.y)**2)**0.5

        if dist < 200:  # Reasonable drift correction
            return (best.cx, best.cy, best.confidence)
        elif dist > 200:
            # Large discrepancy — cursor may have been moved by user
            # Trust YOLO if confidence is high
            if best.confidence > 0.7:
                return (best.cx, best.cy, best.confidence)

        return None
```

**YOLO inference time:** ~15-30ms on GPU (YOLOv8n). At 1-2 Hz, this is 15-60ms/s. Affordable.

### 2.5 Velocity-Aware ROI

**File:** `src/hardware/silhouette_tracker.py`

Currently ROI expands only after 2 consecutive misses. Instead, scale ROI with velocity:

```python
def _compute_roi_size(self, velocity_px_s):
    """Scale ROI with cursor velocity."""
    # At 0 velocity: 200px ROI (tight)
    # At 500 px/s: 400px ROI
    # At 1000+ px/s: 800px ROI (max)
    base = self.ROI_MIN  # 200
    velocity_bonus = min(600, velocity_px_s * 0.6)
    return min(self.ROI_MAX, int(base + velocity_bonus))
```

**Why this matters:** At 1000px/s, the cursor exits a 200px box in 0.2 seconds. A 800px box buys 0.8 seconds — enough for the silhouette tracker to keep up.

### 2.6 Immediate Escalation (No 10-Second Wait)

**File:** `src/hardware/cursor_tracker.py`

Replace the stale_threshold approach with confidence-based escalation:

```python
def _track_loop(self):
    while not self._stop_event.is_set():
        frame = self._capture()

        # Always run CNN (continuous, <2ms)
        cnn_conf = self._continuous_verify(frame)

        # Always run silhouette tracker
        sil_result = self.silhouette.track(frame, pos.x, pos.y)

        # Periodically run YOLO (every 1-2s)
        yolo_correction = self.yolo_anchor.check(frame, pos)
        if yolo_correction:
            self.set_position(*yolo_correction[:2], method="yolo_anchor")

        # Confidence gate (replaces stale_threshold)
        fused_confidence = self._fuse_confidence(cnn_conf, sil_result, yolo_correction)

        if fused_confidence > 0.8:
            pass  # All good
        elif fused_confidence > 0.3:
            # Medium confidence — increase YOLO scan rate
            self.yolo_anchor.interval = 0.5  # Scan every 500ms
        else:
            # Low confidence — IMMEDIATE escalation
            self.yolo_anchor.interval = 0.2  # Scan every 200ms
            if not self._yolo_found_it:
                self._probe_micro()  # Try micro-shake
                if not found:
                    self._escalate_lissajous()  # Skip macro, go straight to Lissajous
```

### 2.7 Dual-YOLO Architecture (User Request)

The user specifically asked for "2 instances of custom YOLO, both with motion data and jitter pulse timings feeding in."

**Instance 1: YOLO Anchor** (described above)
- Runs on full 1920x1080 frames at 1-2 Hz
- Purpose: ground truth position correction
- Model: `cursor_v2_best.pt` (fine-tuned for cursor detection)

**Instance 2: YOLO Crop Validator**
- Runs on 256x256 crops around candidates from silhouette tracker
- Purpose: validate blob candidates (is this blob actually a cursor?)
- Model: same `cursor_v2_best.pt` but on smaller input
- Already partially exists in `silhouette_tracker.get_candidates()` + `yolo_detector.detect_crop()`

**Motion data integration:**
Both instances receive supplementary context:
- Velocity vector from dead reckoning (which direction is cursor moving?)
- Jitter pulse timing (when did we last send a known displacement?)
- This helps distinguish cursor movement from UI animations: if YOLO detects something moving AND it's at the predicted dead-reckoning position AND it moved in sync with our jitter pulse → high confidence it's the cursor

### 2.8 CNN + Silhouette Merger (User Request)

The user asked to merge TinyCursorNet + silhouette tracker into a single CPU-runnable model. **Preserve existing code** (user: "Don't delete its code").

**Approach:** Create a combined `CursorVerifier` class that wraps both:

```python
class CursorVerifier:
    """Unified CPU-fast cursor verification (CNN + silhouette).

    Runs both TinyCursorNet and template matching in a single call.
    Returns fused confidence score. Total latency: <3ms on CPU.

    This is the fast fallback that runs continuously.
    Original TinyCursorNet and SilhouetteTracker code is preserved
    in their respective files.
    """

    def __init__(self, cnn_model, recognizer):
        self.cnn = cnn_model          # TinyCursorNet (14K params, <2ms)
        self.recognizer = recognizer  # Template matcher

    def verify(self, frame, x, y, prev_frame=None):
        """Combined verification at (x, y). Returns (confidence, method)."""
        # CNN check: extract 64x64 patch
        patch = extract_patch(frame, x, y, 64)
        cnn_conf = self.cnn.predict(patch)

        # Template check: match cursor templates in ROI
        roi = extract_roi(frame, x, y, 200)
        template_conf = self.recognizer.match_templates(roi)

        # Motion check (if previous frame available)
        motion_conf = 0.0
        if prev_frame is not None:
            motion_conf = self._check_motion(frame, prev_frame, x, y)

        # Fuse: weighted combination
        fused = 0.5 * cnn_conf + 0.3 * template_conf + 0.2 * motion_conf
        method = "cnn" if cnn_conf > template_conf else "template"

        return fused, method
```

---

## Part 3: Implementation Order

### Phase A: Quick Wins (can be done in ~1 hour)
1. Fix (6,15) ghost position guard
2. Add retro CNN recovery signal
3. Add temporal smoothing on live CNN
4. Add multi-signal consensus check

### Phase B: Dead Reckoning + Continuous CNN (~2-3 hours)
1. Update `ESP32Mouse._send_raw()` to accumulate estimated position
2. Add `_continuous_verify()` to tracking loop
3. Remove 10-second stale threshold, replace with confidence gate

### Phase C: YOLO Ground Truth Anchor (~2-3 hours)
1. Create `YOLOAnchor` class in new `position_fusion.py`
2. Integrate into tracking loop at 1-2 Hz
3. Add drift correction logic

### Phase D: Velocity-Aware ROI (~1 hour)
1. Modify `SilhouetteTracker._compute_roi_size()` to scale with velocity
2. Pass velocity from tracker to silhouette

### Phase E: CNN + Silhouette Merger (~2 hours)
1. Create `CursorVerifier` class (preserves original code)
2. Wire into tracking loop as the primary fast-path verifier

### Phase F: Dual-YOLO with Motion Context (~3 hours)
1. YOLO crop validator for blob candidates
2. Jitter pulse timing correlation
3. Motion-aware detection scoring

---

## Key Constants (Derived from 102-Event Analysis)

| Constant | Current | Proposed | Rationale |
|----------|---------|----------|-----------|
| `stale_threshold` | 10.0s | **removed** | Replace with confidence gate |
| `check_interval` | 2.0s | **0.033s** (30 Hz) | Continuous verification |
| `ROI_MIN` | 200px | 200px | Keep for stationary |
| `ROI at 1000px/s` | 200px (static) | **800px** | Velocity-proportional |
| `MISS_EXPAND_AFTER` | 2 misses | **1 miss** | Faster expansion |
| `CNN loss threshold` | implicit | **<0.3 (EMA)** | Smoothed, not raw |
| `Recovery signal` | live_cnn only | **max(live, retro) > 0.8** | Use retro CNN |
| `YOLO scan interval` | never | **1.0s normal, 0.2s urgent** | Periodic anchor |
| `Ghost position guard` | none | **reject x<20 && y<20 && method=manual** | Fix init bug |

---

## Expected Impact

| Metric | Current | After Quick Wins | After Full Fusion |
|--------|---------|-----------------|-------------------|
| Recovery rate | 31% | ~63% | ~90%+ |
| Time to recover | 8.6s mean | ~4s | <1s |
| False loss events | ~6% | ~0% | ~0% |
| Ghost position events | 11% | 0% | 0% |
| Position awareness | Periodic (2s) | Continuous (30Hz) | Continuous (50Hz) |
