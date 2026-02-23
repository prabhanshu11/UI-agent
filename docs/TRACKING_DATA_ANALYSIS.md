# Tracking Data Analysis — Week of Feb 19-23, 2026

> Generated 2026-02-23 from state_log.jsonl (103K entries), 7 velocity test runs (1,693 frames), 268 loss events, and blob logs.

## Executive Summary

**The tracker is frozen 99.99% of the time.** Out of 103,649 state log entries spanning 8.9 hours, there were only **9 position changes**. The motion detector (C daemon) IS detecting blobs when the cursor moves, but the blob-to-position pipeline in `kvm_dashboard.py` has 5 filtering gates that reject nearly all blob activity.

One velocity test run (`run_20260220_105008`) shows near-perfect tracking at all speeds (100% accuracy at 300 px/s) — but it uses `esp32_dead_reckoning`, bypassing visual detection entirely.

## Key Findings

### 1. State Log: Position Frozen for Hours

| Period | Duration | Position | Entries | Method |
|--------|----------|----------|---------|--------|
| 00:45 - 08:06 | 7.4 hours | (280, 960) | 87,479 | sil_template |
| 08:06 - 08:13 | 7 minutes | (810, 298) | 1,155 | sil_template |
| 08:13:48-58 | 11 seconds | 7 changes | 27 | sil_motion_roi + sil_template |
| 08:13:59 - 09:38 | 1.4 hours | (577, 621) | 14,982 | sil_template |

- **Position changes: 9 out of 103,649 entries (0.009%)**
- **YOLO active: 0% (never fired)**
- **YOLO detections: 0**
- **Validated: 0%**
- Methods: 100% sil_template (7 sil_motion_roi during the 11s burst)
- CNN confidence: bimodal at 0.225 and 0.837 (no middle ground)

### 2. The 11-Second Movement Episode (08:13:48-58)

The only period of active tracking. Pattern:
1. `sil_motion_roi` fires → small 8px correction (confidence 1.0)
2. `sil_template` re-locks → large 138-216px jump (confidence 0.45-0.56)
3. Alternates 3 times, then freezes again at (577, 621)

This proves the tracker CAN detect motion (sil_motion_roi) and re-acquire (sil_template), but only briefly before freezing.

### 3. Velocity Tests: ESP32 Dead Reckoning vs Visual Tracking

| Velocity | esp32_dead_reckoning (run_105008) | motion_track (run_105428) | sil_template (run_104358) |
|----------|-----------------------------------|---------------------------|---------------------------|
| V0 (still) | 45px error, 38% acc | 38px error, 59% acc | 0px error, 100% acc |
| V100 | 3px error, **100% acc** | 141px error, 10% acc | 139px error, 8% acc |
| V300 | 4-12px error, **100/85% acc** | 294px error, 4% acc | 331px error, 4% acc |
| V500 | 10-11px error, **65-85% acc** | 170-510px error, 4% acc | 510px error, 4% acc |
| V1000 | 17px error, **22-36% acc** | 536px error, 0% acc | 651px error, 0% acc |

**The gap is enormous.** Dead reckoning tracks the cursor within 17px even at 1000 px/s. Visual tracking loses it at 100 px/s.

### 4. Blob Detection Works, Pipeline Doesn't

In velocity test run_105428 at V300:
- Step 0: 1 blob detected, tracked position frozen at (1303, 172)
- Step 96: **12 blobs, 3 active cells** — position still (1303, 172)
- Step 99: **32 blobs, 17 active cells** — position barely moved to (1368, 175)

**The C daemon blob detector sees the cursor moving. The Python pipeline rejects the blobs.**

### 5. Loss Events: 81% Never Recover

- **268 total loss events** (Feb 19-23)
- **Recovery rate: 19%** (52/268)
- **Stuck-at-loss: 21%** (56/268) — cursor was moving, tracker froze at zero velocity
- Median duration: 24.9 seconds
- Median max velocity at loss: 1,306 px/s
- Top triggers: `validation_lost` (93), `cnn_drop` (61), `position_jump` (114)
- Top edge cases: peak_acceleration (253), peak_velocity (247), high_accel (215)

### 6. Aggregate: Velocity → Error Transfer Function

From all velocity test runs combined:

| Commanded Velocity | Mean Tracking Error | Accuracy |
|--------------------|--------------------:|----------|
| 0 px/s (stationary) | 123.8 px | 34.0% |
| 100 px/s | 99.5 px | 40.2% |
| 300 px/s | 274.3 px | 32.0% |
| 500 px/s | 286.7 px | 27.5% |
| 1000 px/s | 385.7 px | 9.5% |

This includes both working (dead reckoning) and broken (visual) runs. Visual-only accuracy at any speed > 0 is effectively 0-18%.

## Root Cause: 5 Filtering Gates in blob→position Pipeline

Located in `src/web/kvm_dashboard.py` motion detection loop:

| Gate | Line | Filter | Impact |
|------|------|--------|--------|
| 1 | ~604-605 | Blob size: rejects > 200px | Fast cursor = bigger blobs → rejected |
| 2 | ~610 | Aspect ratio: rejects < 0.2 | I-beam cursors, thin blobs → rejected |
| 3 | ~613 | BBox: rejects > 80px wide/tall | Fast cursor trail > 80px → rejected |
| 4 | ~642-651 | Persistent motion → "animation" | Cursor regions flagged as noise. Needs CNN validation to exempt → circular: no blobs = no validation = no exemption |
| 5 | ~658-659 | YOLO lockout: suppresses ALL blobs | When YOLO active, ALL blob processing disabled |

**Circular dependency in Gate 4:** Blobs update position → CNN validates → exempts from noise filter → but blobs can't update position without passing the noise filter first.

**Silhouette ROI too small:** The ROI (200px) is static when velocity is never fed (because blobs are filtered out). At 300 px/s, cursor exits ROI in <0.67 frames.

## Calibration Opportunity

The velocity test data contains **1,693 frames** with both:
- `commanded.total_x/total_y` — ESP32's intended cursor position (from HID deltas)
- `tracked.x/y` — visual tracker's observed position

This is exactly the calibration data needed. The delta between commanded and tracked = the combined effect of Windows mouse acceleration + tracking latency.

Run `105008` (esp32_dead_reckoning) shows the Windows acceleration error alone (3-17px). Runs `104358`/`105428` show the visual tracking error layered on top.

To derive calibration:
1. From `run_105008`: commanded vs tracked → Windows acceleration curve
2. From `run_105428`: commanded vs tracked → total error (Windows + visual tracker)
3. Subtract: total - Windows = pure visual tracking error per velocity

## Actionable Fixes (Ranked by Impact)

### Critical: Unblock Blob→Position Pipeline

1. **Relax blob size gates**: Increase max blob size from 200 to 2000, max bbox from 80 to 200. Fast-moving cursors create larger blobs.

2. **Break the circular dependency in Gate 4**: Exempt blobs that are within `motion_trail` proximity from the persistent motion filter, instead of requiring CNN validation.

3. **Feed velocity to silhouette tracker unconditionally**: Even when blobs are filtered, compute velocity from the blob centroid deltas and feed it to `sil_tracker.set_velocity()`. This expands the ROI for fast motion.

4. **Remove YOLO lockout for blobs**: YOLO lockout should only suppress YOLO re-queries, not silence blob processing entirely.

### High: Use Existing Calibration Data

5. **Extract calibration from velocity tests**: Parse `run_105008` to build the Windows acceleration transfer function. Load into `MouseCalibrator` without re-running the sweep.

6. **Log commanded + tracked together in state_log**: Add `commanded_x/y` field when ESP32 is active, enabling continuous calibration data collection.

### Medium: Continuous Diagnostics

7. **Log blob acceptance/rejection per gate**: Add counters for each gate's reject rate. Surface in `/api/state` so we can see which gate kills the most blobs.

8. **Add position-change frequency to daily reports**: Alert when position changes < 1/minute (indicates frozen tracker).
