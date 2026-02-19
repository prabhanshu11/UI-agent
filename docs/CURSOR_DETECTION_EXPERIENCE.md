# Cursor Detection Experience — Systematic Testing Plan

> **Read this at every session start.** This document IS the plan.
> It survives context cleanup. It is the single source of truth for
> cursor detection development.

## What is an "Experience"?

Any development work that uses Claude Vision (`/api/claude_detect`) to validate
cursor detection changes is an **experience**. Experiences produce:
- Logged sensor data at every step
- Ground truth from Claude Vision
- Failure analysis with screen timestamps
- Algorithm improvements derived from real data

## Current Architecture (after commit fad0373)

```
                     ┌─────────────────────────────┐
                     │     Claude Vision (Agent)    │
                     │   Ground Truth (~10s/call)   │
                     │   Establishes general pos    │
                     └──────────┬──────────────────┘
                                │ high/medium → set_position
                                ▼
┌──────────────────────────────────────────────────────────┐
│                   POSITION STATE                          │
│  cursor_x, cursor_y, method, age_s, cursor_validated     │
└───┬──────────┬───────────┬───────────┬──────────────────┘
    │          │           │           │
    ▼          ▼           ▼           ▼
┌────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐
│ Motion │ │Silhouette│ │   CNN   │ │ Jitter Loop  │
│ Blobs  │ │ Tracker  │ │Validate │ │ (anti-sleep  │
│ (20Hz) │ │  (5Hz)   │ │ (30s)   │ │  + reacquire)│
└────────┘ └──────────┘ └─────────┘ └──────────────┘
```

### The Problem

These components don't work as a mesh. They work as independent silos:
- Motion blobs lock onto UI animations (clocks, tickers)
- Silhouette tracker false-matches on toolbar icons (99.7% confidence)
- CNN false-validates wrong positions (90%+ on non-cursor areas)
- Jitter never fires because CNN always says "looks good"
- Claude Vision is the only reliable source but takes 10s per call

### The Goal

Make the silhouette tracker the **primary real-time tracker**, with all other
components feeding into it. The silhouette tracker operates at video-frame
rate on a lower-quality stream and should be the workhorse — not CNN, not
motion blobs in isolation.

## The Silhouette-First Model

### Phase 1: Establish Position (Claude Vision)

Claude Vision is slow (~10s) but reliable. Use it to:
1. Find cursor on a still screen
2. Set the initial position
3. Provide ground truth for all subsequent tracking

```python
# Step 1: Run Claude Vision
result = await claude_detector.detect(frame, est_x, est_y)
# Step 2: Lock silhouette tracker onto Vision position
sil_tracker.reset()  # Reset ROI to tight 200px around Vision position
tracker.set_position(result.cursor_x, result.cursor_y, method="claude_vision")
```

### Phase 2: Stick with Silhouette (Real-Time)

Once position is established, silhouette tracker follows it in real-time:
- **Template matching**: Stationary cursor — match learned template in 200px ROI
- **Motion ROI**: Moving cursor — frame-diff in ROI, shape-scored blobs
- **Adaptive ROI**: Expand 200→400→600→800px on consecutive misses

The silhouette tracker is the DEFAULT continuous tracker. Its output:
- Updates cursor position at ~5Hz (frame decode rate)
- Feeds into jitter re-acquisition (if silhouette loses cursor)
- Should work on BOTH the high-quality 5Hz decode AND a potential low-quality
  higher-FPS secondary feed

### Phase 3: Validate with CNN (Periodic)

CNN runs every 30s as a CHECK on silhouette, not as the authority:
- If CNN > 0.7: silhouette is trusted, update template
- If CNN < 0.7: DON'T immediately re-acquire. Instead:
  1. Flag position as "unvalidated"
  2. Let silhouette continue tracking (it might be right, CNN wrong)
  3. After 30s of sustained unvalidated: jitter probes
  4. If jitter also fails: Claude Vision re-establishes position

### Phase 4: Jitter Re-Acquisition (Recovery)

When confidence has been low for one full jitter cycle:
1. `_jitter_reacquire()` fires dual-direction probe (15px RIGHT, then DOWN)
2. Captures 3 frames, correlates blobs across both directions
3. Only the real cursor follows both mouse movements
4. Result: new position → silhouette re-locks → CNN validates next cycle

**Current Bug:** CNN false-validates wrong positions → `_low_confidence_since`
never accumulates → jitter never fires. The silhouette tracker's own output
(ROI expanding, misses accumulating) should ALSO trigger jitter, not just CNN.

**Fix needed:** Jitter should fire when EITHER:
- CNN confidence < 0.7 for 30s (current, broken by false positives), OR
- Silhouette misses > threshold (e.g., ROI expanded to 800px), OR
- Silhouette confidence < 0.3 for sustained period

## Systematic Testing Protocol

### Test Setup

1. Dashboard running at http://localhost:8766/
2. Target machine showing a known screen (world clock is fine)
3. Claude Vision available as ground truth
4. Logging enabled for all sensor data

### Test Sequence

Each test follows this protocol:

```
1. ESTABLISH: Claude Vision detects cursor → ground truth position
2. LOG: Full sensor stack snapshot (see "What to Log" below)
3. ACT: Move cursor with known velocity/acceleration pattern
4. LOG: Sensor snapshots at ~200ms intervals during movement
5. STOP: Cursor stationary → wait for all trackers to converge
6. VERIFY: Claude Vision detects again → compare with tracker position
7. LOG: Final snapshot + delta analysis
```

### Velocity/Acceleration Test Matrix

| Test | Movement | Speed | Duration | Purpose |
|------|----------|-------|----------|---------|
| V0 | Stationary | 0 | 10s | Baseline: all trackers on still cursor |
| V1 | Slow diagonal | ~2px/frame | 5s | Silhouette should follow easily |
| V2 | Medium horizontal | ~10px/frame | 3s | Test velocity prediction |
| V3 | Fast sweep | ~50px/frame | 2s | Silhouette will lose → jitter recovery |
| V4 | Snap to corner | Instant | — | Position jump: all trackers should re-acquire |
| A1 | Accelerating | 0→30px/frame | 3s | Test adaptive ROI expansion |
| A2 | Decelerate+stop | 30→0px/frame | 3s | Test re-lock after fast movement |
| A3 | Jitter (tiny) | ±1-2px | 10s | Silhouette should hold, noise filter stable |

### What to Log at Each Step

```json
{
  "test_id": "V1_20260220_014530",
  "step": 3,
  "utc": "2026-02-20T01:45:30.123Z",
  "screen_utc": "2026-02-19T20:45:30.456 UTC",
  "screen_utc_source": "world_clock_ocr",
  "tracker": {
    "x": 490, "y": 465,
    "method": "sil_template",
    "age_s": 0.1
  },
  "vision_ground_truth": {
    "x": 488, "y": 462,
    "confidence": "high",
    "latency_ms": 8500
  },
  "silhouette": {
    "active": true,
    "method": "template",
    "confidence": 0.87,
    "roi_size": 200,
    "misses": 0,
    "hz": 4.8,
    "latency_ms": 2.1
  },
  "cnn": {
    "confidence": 0.82,
    "validated": true,
    "inference_ms": 18.0
  },
  "motion_blobs": [
    {"centroid": [490, 465], "size": 28, "angle": 0.3}
  ],
  "noise_grid_active_cells": 3,
  "jitter": {
    "count": 5,
    "low_confidence_s": 0,
    "reacquire_pending": false
  },
  "delta_tracker_vs_vision": 3.6,
  "frame_snapshot_path": "data/experience/V1_step3.jpg",
  "overlay_snapshot_path": "data/experience/V1_step3_overlay.jpg",
  "notes": "Cursor moving slowly right, silhouette tracking well"
}
```

### The UTC Screen Time Correlation

The target screen shows a world clock with UTC time to millisecond precision.
Each logged snapshot includes:
- **utc** — the dashboard server's UTC time (Python `datetime.now(timezone.utc)`)
- **screen_utc** — OCR'd from the HDMI feed (the clock on the Windows screen)
- **Delta** — difference between the two = total pipeline latency

This correlation lets us:
1. Match logged events to specific frames in recorded video
2. Measure capture→display latency precisely
3. Create training data: "at screen time T, cursor was at (x,y)"

### Using Test Data to Improve Algorithms

After each test run:

1. **Plot trajectory**: tracker position vs ground truth position over time
2. **Identify failure points**: where did delta exceed 30px?
3. **Categorize failures**:
   - "Silhouette lost during fast movement" → tune ROI expansion rate
   - "Motion blobs picked up clock animation" → noise grid missed it
   - "CNN false-validated wrong position" → add to negative training set
   - "Jitter failed to re-acquire" → adjust probe amplitude or direction
4. **Generate training data**: frame patches at confirmed positions → CNN training
5. **Record video segments**: high-quality HDMI capture during test → replay analysis

## Implementation: /api/experience Endpoint

Add to `kvm_dashboard.py`:

```python
@app.post("/api/experience/log")
async def log_experience(request: Request):
    """Log a sensor snapshot with optional user note.

    POST body: {"note": "cursor is at approximately (500, 400)", "test_id": "V1"}
    Captures full sensor stack and writes to data/experience/log.jsonl
    """
    # Capture ALL current state
    # Write to JSONL file
    # Return snapshot for confirmation
```

```python
@app.post("/api/experience/start_test")
async def start_test(request: Request):
    """Start a named test sequence. Logs snapshots at configured interval."""

@app.post("/api/experience/stop_test")
async def stop_test():
    """Stop current test. Write summary with trajectory analysis."""
```

## How to Use This Document

### At Session Start
1. Read this document
2. Read CURSOR_FIX_PLAN.md for current status
3. `git log --oneline -10` to see what's been committed
4. `curl http://localhost:8766/api/state` to see current tracker state

### During Development
1. Before changing algorithm: run Claude Vision to establish ground truth
2. Make the change
3. Run a test from the matrix above
4. Log results
5. Commit with descriptive message

### At Session End
1. Update CURSOR_FIX_PLAN.md with findings
2. Commit any uncommitted work
3. Note which tests passed/failed in this document

## Current Status

### Session 2026-02-20 Findings

**CNN false positives are real but trainable.** The 14K-parameter TinyCursorNet
gives 90%+ confidence on toolbar icons. This breaks the recovery chain because
CNN validation prevents `_low_confidence_since` from accumulating. But the model
can be retrained with better negative samples (toolbar icons labeled as non-cursor).

**Silhouette tracker is the right primary tracker.** It runs at frame rate,
uses template matching + motion detection in a small ROI, and is lightweight.
The current issue is that it's gated behind CNN validation — when CNN is wrong,
silhouette is wrong too. Fix: make silhouette the PRIMARY tracker and CNN the
secondary validator.

**Jitter re-acquisition needs silhouette input.** Currently only fires on
sustained low CNN confidence. Should ALSO fire when silhouette loses the cursor
(ROI expanded to max, high miss count). This bypasses the CNN false positive
problem.

**Noise grid works but needs time to warm up.** The inverted-start approach
(presume all regions noisy) blocks initial false blobs but takes ~0.4s to clear
legitimate regions. This is acceptable.

## Next Steps (ordered by priority)

1. Wire silhouette misses into jitter trigger (bypass CNN false positive problem)
2. Add /api/experience/log endpoint for sensor snapshot logging
3. Run V0 test (stationary baseline) with full logging
4. Run V1-V3 tests (velocity series) to identify failure modes
5. Use logged data to tune silhouette ROI expansion + noise grid parameters
6. Retrain CNN with negative samples from false-positive positions
