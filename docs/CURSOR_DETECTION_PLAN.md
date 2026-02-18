# Fix Cursor Detection — Record-Then-Analyze + Optimized Sweep

## Current Changes (already applied, not yet committed)

`cursor_locator.py` has been rewritten from the broken 3-thread live pipeline to the proven sequential record-then-analyze approach. This is ready to test.

## This Plan: Two Additional Improvements

### A) Cursor Shape Detection (Robustness)

Enhance `extract_motion_blobs()` or add a post-filter to validate that a motion blob looks like a cursor arrow. The Windows default cursor is a white arrow with black outline (~20x30px). By checking the shape/structure of motion blobs, we can:

1. **Filter false positives** — UI animations (spinners, blinking text cursors) produce motion blobs but don't look like an arrow
2. **Handle different cursor colors** — Windows cursors can be white, black, or inverted. Instead of matching a specific color, match the **shape**: a roughly triangular/arrow-shaped blob with sharp edges

**Approach**: After extracting motion blobs, check the blob's shape characteristics:
- **Aspect ratio**: Cursor blob is taller than wide (~0.6-0.7 width/height ratio)
- **Triangularity**: High ratio of bounding box area to actual pixel count (arrow is not a filled rectangle)
- **Edge sharpness**: Cursor has crisp anti-aliased edges, not the soft blur of motion artifacts

This is a **filter on existing blobs**, not a separate detection pass. Added to `cursor_detector.py:extract_motion_blobs()` as an optional `cursor_shape_filter=True` parameter.

### B) Optimized Sweep Parameters

Reduce sweep time from 10s to 5s while covering **5x the area of 2 x 1080p monitors** — enough to account for 4K monitors, rotated displays, and worst-case cursor start positions.

**Current parameters** (too slow, sized for ~1.5x of 2x1080p):
```
amp_x = 3000  → 6000px sweep width
amp_y = 1500  → 3000px sweep height
omega_x = π/8 → 16s period (only covers 62% in 10s)
max_duration = 10s → 500 HID reports
```

**Math**: Base = 2 x 1080p monitors (3840 x 1080 = 4.1M px²). 5x that area accounts for scenarios like:
- Two 4K monitors (2x resolution = 4x area per monitor)
- One horizontal + one vertical monitor (portrait mode increases height)
- Cursor stuck at worst corner of the farthest monitor from the HDMI capture

**Worst case scenario**: Cursor is at the bottom-right corner of a vertical (portrait) second monitor, while the HDMI captures the leftmost horizontal monitor. The Lissajous must sweep from that far corner all the way across to the captured screen. With 5x area (√5 ≈ 2.24x per dimension), the sweep extends far enough to reach any corner.

**Proposed parameters**:
```
amp_x = 4300  → 8600px sweep width   (3840 * √5 ≈ 8589)
amp_y = 1200  → 2400px sweep height  (1080 * √5 ≈ 2415)
omega_x = π/5 → 10s period (half-period in 5s = full sweep one direction)
omega_y = π√2/5 → 7.1s period (irrational ratio preserved)
max_duration = 5s → 250 HID reports
Area = 20.6M px² ≈ 5.0x base (4.1M)
```

**Why 5s is enough**: A half-period traverses the full amplitude range (center → +max → center → -max). Even if cursor starts at the worst corner of a vertical+horizontal arrangement, the Lissajous visits the HDMI-captured 1920x1080 zone within one half-cycle.

**HID scaling**: `_max_displacement()` auto-normalizes to fit ±127 — always valid regardless of amplitude.

## Files to Modify

| File | Change |
|------|--------|
| `src/hardware/cursor_detector.py` | Update `LissajousCurve` defaults + add shape filter to `extract_motion_blobs` |
| `src/hardware/cursor_locator.py` | Update `max_duration` default from 10.0 to 5.0 |

## Verification

1. `uv run python -m src.hardware.cursor_locator` — cursor detected with correlation > 0.8 in ~6-7s total (5s sweep + 1-2s analysis)
2. `shake()` — visible wiggle on HDMI screen
3. Run `pair_k380.py` end-to-end

## Context

### The Full Story (Feb 16-18, 15+ sessions, 2 machines)

**The goal**: Full KVM control of a corporate Windows laptop from Linux desktop — keyboard, mouse, and video.

**The architecture**:
```
Desktop (Arch Linux)
├── USB HDMI capture card ← Windows laptop HDMI out → /dev/video0
├── USB serial → ESP32 (/dev/ttyUSB0) → BLE mouse → Windows ("M350 Pebble")
└── USB data cable → Pi Zero 2W (10.55.0.2:8081) → BT keyboard → Windows ("Keyboard K380")
```

**Chapter 1 — Laptop KVM Build (Feb 16, 4 sessions)**:
Built via laptop SSH'ing into desktop. ESP32 flashed with Logitech VID/PID, Pi Zero deployed with MAC spoofing + HID profile. ESP32 mouse worked immediately. Pi keyboard kept failing: shows under "Other Devices" (not "Input"), no battery icon, disconnects immediately. 8+ hours of BT HID debugging across sessions `682a8bf6`, `47a50355`, `f2a58db1`, `6395c7ec`.

**Chapter 2 — Desktop Takeover + Reboot (Feb 16 ~14:37)**:
Session `ec41fe43` (5h52m): Laptop forced reboot. User clarified two separate projects: (1) Windows VM streaming to website, (2) corporate laptop control via AI agent. Built sessions catalog (`utilities/claude-sessions-catalog.md`), KVM agent prompt (`esp32-bt-hid/docs/KVM_AGENT_PROMPT.md`). Rebooted both machines for kernel 6.18.7 (USB was broken on 6.17.9).

**Chapter 3 — Try 1: Manual K380 Pairing Attempts (Feb 16 ~20:59)**:
Sessions `ff04450c`, `26d0ad6e`, `84906f1d`. Star Trek Computer dashboard started. K380 paired but: "Other devices", no battery, immediate disconnect. User frustrated after 8 hours of loops: *"same error as before... why don't you operate the mouse, use display as feedback, do pairing and removing after desired result is not achieved."* — Birth of autonomous UI control concept.

**Chapter 4 — "DID YOU LOCATE THE POINTER?" (Feb 16 ~23:07)**:
Session `2c216942`. Agent tried driving Windows UI via ESP32 mouse without knowing where the cursor was. User interrupted: **"DID YOU LOCATE THE POINTER?"** Then described the calculus curve vision:
> *"create a motion detection utility which moves the mouse, records video at good frame rate, processes the frame to find motion. Cursor should move in known curves across any continuous screen surface area."*
> *"the pointer will follow mathematical curves which are sneaky to find their way across any continuous screen surface area. We'll use calculus to find where the motion blur is making those same curves."*

**Chapter 5 — Lissajous Cursor Detection (Feb 16 23:28)**:
Session `47d0c44b` (10h9m). Built `cursor_detector.py` (Lissajous curve + curvature matching) and `cursor_locator.py` (live detection with early termination). User said: *"now find my pointer and continue with fixing the keyboard autonomously using mouse and video."*

**Chapter 6 — Star Trek Experience Philosophy (Feb 17 09:38)**:
Session `911b4a7d` (12h47m). User defined two experience types:
1. **"Getting control"** — Lissajous sweep finds cursor (like camera swerve algorithm from earlier repo)
2. **"IT Work"** — Navigating UI declaratively: *"like when the Star Trek computer or Skynet gains access to one terminal, it actively tries to use known ways to gain more access as the need arises for certain tunnel on a planet to be explored."*

Key directive: *"make sure entire end-to-end experience is logged, and understood in a deeper than SWE coder way."*

**Chapter 7 — Navigation System Built (Feb 17 22:25)**:
Session `3095ea66`. Built full navigation module: NavigationStep types, WindowsNavigator (perceive→act→verify + parallel hypothesis recovery via Claude Agent SDK), OTel-integrated StorylineLogger, `pair_k380.py` experience script. Committed as `fffe09f`.

**Chapter 8 — `/new-experience` Skill (Feb 17-18)**:
This session. Template + skill for generating future experience scripts. Committed and pushed to local-bootstrapping.

### This Is Try 2

**Try 1** (Chapter 3): Manual pairing attempts through the agent clicking blindly — failed because cursor position was unknown, no systematic navigation, no logging.

**Try 2** (this plan): Full autonomous navigation with:
- Lissajous curve cursor detection (knows where the pointer is)
- LLM vision for finding UI elements on HDMI-captured screen
- Declarative 9-step navigation with verify-after-click
- Hypothesis recovery system (if a click misses, parallel Claude agents diagnose why)
- Full OTel + StorylineLogger narrative recording

### Key Sessions for Resume

| Session | Topic | Resume |
|---------|-------|--------|
| `47d0c44b` | Lissajous cursor detection | `claude --resume 47d0c44b` |
| `911b4a7d` | Experience philosophy + research | `claude --resume 911b4a7d` |
| `3095ea66` | Navigation system + pair_k380.py | `claude --resume 3095ea66` |
| `ec41fe43` | Session catalog + KVM context | `claude --resume ec41fe43` |
| `2c216942` | Calculus curve origin | `claude --resume 2c216942` |

## What Broke — Root Cause Analysis

### The successful detection (session `47d0c44b`, Feb 17 09:22)

The cursor was detected at (1842, 15) with **0.92 correlation**. The approach was:
1. Record full 5s video while ESP32 runs Lissajous sweep
2. Stop recording
3. Load ALL frames offline (`load_video_frames`)
4. Run `extract_motion_blobs` + `match_curve_segment` on the full set
5. Detection in <0.01s once blobs were extracted (56s for 346 blob extractions)

**This worked** because it analyzed the complete video, including the later frames where the cursor had entered the HDMI-captured screen.

### The rewrite (commit `5dd8115`, Feb 17 05:04)

Replaced the offline approach with a "live detection" pipeline:
- 3 concurrent threads: RECORD (ffmpeg), MOVE (Lissajous), ANALYZE (periodic)
- `_analyze_loop` runs every 1 second, loads frames from growing MKV file
- Early termination on first match

### Why the live pipeline fails (our debug run, Feb 18)

1. At the 5.3s check, `load_video_frames` returns 117 frames
2. Only 1 of the first 50 frames had motion blobs (cursor hasn't reached visible screen yet)
3. The Lissajous curve covers a 6000×3000 virtual desktop — cursor enters HDMI screen only after ~3-4s
4. Frames 117-193 (t≈3.9-6.4s) DO have cursor motion — **76 consecutive frames with blobs**
5. But the analyze loop only ran ONCE, saw mostly empty frames, and then **the timeout at 15.9s terminated everything**
6. It never got a chance to re-analyze with the later frames

**The bug**: The analyze loop should have run multiple iterations (every 1s), but it only ran once at 5.3s. The likely cause: `load_video_frames` on a growing file is too slow (it took ~10s in the previous session), so by the time it finishes the first analysis, the 12s timeout has nearly expired. The second analysis attempt hits the timeout.

### What the user told me (that I should have known)

> "I saw the cursor moving across both the screens"

The mouse IS working. The detection pipeline is what's broken — not the mouse, not the BLE connection. I wasted time trying to change the mouse reconnect logic instead of reading the session history first.

## The Fix

**File**: `~/Programs/UI-agent/src/hardware/cursor_locator.py`

Change `locate()` back to the approach that actually worked: **record first, analyze second** (not concurrent).

### Current (broken) flow:
```
start_recording()  ←──── concurrent with ────→  _analyze_loop()
_execute_curve()   ←──── concurrent with ────→  (tries to analyze growing file)
```

### Fixed flow:
```
start_recording()          # Start ffmpeg
_execute_curve()           # Run full Lissajous sweep (5-10s) — blocking
stop_recording()           # Finish clean MKV file
frames = load_video_frames()  # Load all frames at once — no growing file issue
extract_motion_blobs()     # Run on complete frame set
match_curve_segment()      # Find the Lissajous signature
```

This is simpler, already proven to work, and avoids the "analyzing a growing file" problem entirely.

### Specific changes to `cursor_locator.py`:

1. **`locate()`**: Replace the 3-thread pipeline with sequential record→sweep→stop→analyze
2. Keep `max_duration` parameter (controls sweep length, default 10s)
3. Keep verbose output
4. Keep `CursorPosition` return type — no API change
5. The `_analyze_loop` method gets replaced with a single `_analyze_offline` method

### What NOT to change:
- `cursor_detector.py` — the math is correct (0.92 correlation proves it)
- `esp32_mouse.py` — mouse is working fine
- `hdmi_sensor.py` — recording and frame loading work
- `navigator.py` — the `_establish_control` flow is fine once locate() works
- `verified_click()` — keep as is
- `calibrate()` — keep as is
- `shake()` — keep as is

## Key Files

| File | Action | Role |
|------|--------|------|
| `~/Programs/UI-agent/src/hardware/cursor_locator.py` | **MODIFY** | Fix `locate()` to use record-then-analyze |
| `~/Programs/UI-agent/src/hardware/cursor_detector.py` | No change | Math is proven correct |
| `~/Programs/UI-agent/src/hardware/esp32_mouse.py` | No change | Working fine |

## Verification

1. Run `locate()` standalone: `uv run python src/hardware/cursor_locator.py`
2. Cursor should be detected with correlation > 0.8
3. `shake()` should produce visible wiggle on HDMI screen
4. Then run `pair_k380.py` end-to-end

## Meta-Lesson: Experience Preservation

This failure happened because:
1. Session `47d0c44b` proved the offline approach works (0.92 correlation)
2. The rewrite (commit `5dd8115`) changed to a live pipeline that was never tested end-to-end
3. This session ran the untested live pipeline, it failed, and I blamed the mouse instead of reading the history

**What should happen**: When an approach is proven to work, the working code path must be preserved or the replacement must be tested before commit. The conversation history IS the experience log — reading it before running is not optional.

## Future: Lightweight Cursor Tracking Daemon

Discussed in session `47d0c44b` — a continuous low-res, high-fps monitor that:
- Runs in background after initial `locate()` succeeds
- Monitors cursor position via periodic small probe movements
- If cursor is lost (no motion detected), triggers `shake()` + re-detect
- Enables "always know where the cursor is" without full Lissajous sweep each time

**NOT part of this fix** — this is a post-locate() enhancement. First we need `locate()` itself working reliably again. The daemon would wrap around a working `locate()` as a persistence layer.
