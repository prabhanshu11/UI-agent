# Cursor Detection — Complete Knowledge Base

> Compiled from sessions 47d0c44b, 911b4a7d, 2c216942, 3095ea66, f0870570
> (Feb 16-18, 2026). Preserved before context compaction.

## 1. The Problem

We control a Windows laptop from a Linux desktop via:
- ESP32 BLE mouse (relative HID reports via `/dev/ttyUSB0`)
- HDMI capture card (sees one 1920x1080 monitor via `/dev/video0`)

The Windows machine may have **multiple monitors in any arrangement** — we don't know which one the HDMI card captures, and we don't know where the cursor is. The cursor could be on a completely different screen.

## 2. The Lissajous Curve Approach

### Origin (session 2c216942, "DID YOU LOCATE THE POINTER?")

User vision:
> *"create a motion detection utility which moves the mouse, records video at good frame rate, processes the frame to find motion. Cursor should move in known curves across any continuous screen surface area."*
> *"the pointer will follow mathematical curves which are sneaky to find their way across any continuous screen surface area. We'll use calculus to find where the motion blur is making those same curves."*

### Why Lissajous (session 47d0c44b)

A Lissajous curve with **irrational frequency ratio** (omega_y/omega_x = sqrt(2)) is:
- **Space-filling**: visits every region of the virtual desktop
- **Smooth**: no discontinuities, produces continuous motion blur
- **Analytically known**: x(t) = A*sin(w1*t), y(t) = A*sin(w2*t) — we know EXACTLY where the cursor should be at time t
- **Unique curvature**: at any point, the local direction + curvature signature is unique (non-repeating due to irrational ratio)

### Current Parameters (optimized Feb 18)
```python
amp_x = 4300   # Half-width: covers 8600px (5x area of 2x1080p)
amp_y = 1200   # Half-height: covers 2400px
omega_x = pi/5       # Period = 10s, half-cycle in 5s
omega_y = pi*sqrt(2)/5  # Irrational ratio preserved
max_duration = 5.0   # 250 HID reports at 50Hz
```

Sized for **5x the area of two 1080p monitors** — accounts for:
- Two 4K monitors (2x resolution = 4x area)
- One horizontal + one vertical (portrait) monitor
- Cursor starting at the worst possible corner

## 3. Solving the Entire Cursor Motion from a Few Frames

### The Mathematical Insight

Since we **know the curve equation analytically**, we don't need to observe the cursor for the entire sweep. We only need **8+ consecutive frames** where the cursor is visible on the HDMI-captured screen.

The algorithm (`match_curve_segment` in `cursor_detector.py`):

1. **Extract motion blobs** from consecutive frame pairs (frame differencing)
2. **Build trajectory**: sequence of (centroid_x, centroid_y, angle) across frames
3. **Find consecutive runs** of frames with detected motion (min 8 frames)
4. **For each run**, compute the observed direction angles between centroids
5. **Slide a window** along the Lissajous time parameter `t`:
   - At each `t`, compute expected angles from the curve equation
   - Compare expected vs observed angles (circular correlation)
6. **Best match** gives us `t_start` and `t_end` — the exact segment of the curve

From `t_end`, we know:
- **Where the cursor IS right now** (the centroid of the last matched blob)
- **Where the cursor WAS** at any earlier time (just evaluate curve(t) for t < t_end)
- **Where the cursor WILL BE** if we keep sweeping (evaluate curve(t) for t > t_end)

### Why This Works

The Lissajous curve's direction angle `theta(t) = atan2(A_y*w_y*cos(w_y*t), A_x*w_x*cos(w_x*t))` is **unique at every t** (because w_y/w_x is irrational). So matching just 8 consecutive angle observations against the expected sequence uniquely identifies the time parameter.

**Proven**: Session 47d0c44b achieved **0.92 correlation** detecting cursor at (1842, 15) from 76 consecutive blob frames out of ~200 total frames.

### Translation Check

UI animations (spinners, blinking cursors) produce motion blobs too, but they are **stationary** — their centroids don't translate between frames. Real cursor motion produces blobs that **move across the screen** following the Lissajous path. The `translation_threshold=3.0` pixels/frame filter eliminates false positives from animations.

## 4. Cursor Shape Detection

### The Shape Filter (added Feb 18)

Motion blobs can be scored by how "cursor-like" their shape is:

| Blob Type | Fill Ratio | Aspect | Behavior |
|-----------|-----------|--------|----------|
| Cursor arrow | 0.3-0.5 | 0.5-0.8 | Translates with Lissajous |
| Text cursor (I-beam) | 0.8+ | < 0.15 | Stationary blink |
| Spinner | 0.7+ | > 0.8 | Stationary rotation |
| Screen transition | varies | varies | >100px, fills region |

`_cursor_shape_score()` in `cursor_detector.py` assigns 0.0-1.0 based on:
- **Fill ratio**: pixel_count / bounding_box_area (arrow ~ 0.3-0.5)
- **Aspect ratio**: min(w,h)/max(w,h) (filters ultra-thin I-beams)
- **Size**: 15-50px typical for cursor at 1080p

### Handling Different Cursor Colors

Windows cursors can be white (default), black, or inverted. The shape filter doesn't depend on color — it works on the **motion blob geometry** from frame differencing. Whatever color the cursor is, moving it produces a motion blob with the arrow's characteristic shape.

## 5. Three-Tier Cursor Tracking Daemon

`cursor_tracker.py` — the lightweight always-on position monitor:

```
Tier 1: Micro-shake (+-3px)
  - Invisible to user, 40ms at half-res
  - Confirms cursor is still at known position
  - Runs every 2s when position is stale (>10s old)

Tier 2: Macro-shake (+-15px)
  - Slightly visible, catches small drift
  - Updates position to actual blob centroid
  - Only if Tier 1 fails

Tier 3: Lissajous re-detect
  - Full 5s sweep, expensive
  - Only if Tier 2 also fails (cursor truly lost)
  - Resets position from scratch
```

All operations run at **960x540** (downsampled) for speed — blob extraction drops from ~160ms/pair at full-res to ~40ms/pair at half-res.

## 6. Record-Then-Analyze vs Live Pipeline

### What Works: Record-Then-Analyze (Proven Feb 17, session 47d0c44b)

```
start_recording()          # ffmpeg captures to MKV
_execute_curve()           # Blocking Lissajous sweep (5s)
stop_recording()           # Clean MKV file with proper trailer
load_video_frames()        # Decode ALL frames at once
extract_motion_blobs()     # Frame-pair diffs
match_curve_segment()      # Sliding window correlation
```

**Result**: 0.92 correlation, detected at (1842, 15), analysis took <1s after frames loaded.

### What Failed: Live Pipeline (Commit 5dd8115, never tested)

3 concurrent threads: RECORD + MOVE + ANALYZE. Failed because:
- `load_video_frames()` on a **growing MKV** (no container trailer yet) is unreliable
- First analysis at 5.3s returned 117 frames, but cursor hadn't entered screen yet
- `load_video_frames` took ~10s, consuming the entire timeout
- Analyze loop only ran ONCE instead of the intended periodic checks

**Feb 18 test**: Reverted to record-then-analyze. Cursor detected at (5, 11) with 0.81 correlation in ~6 seconds total.

### Future: C Daemon (branch `future/c-cursor-daemon`)

Using libav directly (instead of ffmpeg subprocess) would enable true real-time analysis:
- Read frames from V4L2 device in-place (no file I/O)
- Frame differencing in C with potential SIMD
- Expose cursor position via shared memory
- Sub-millisecond per-frame processing

Design doc: `docs/C_CURSOR_DAEMON_DESIGN.md`

## 7. Logging and Observability

### OTel Integration (session 911b4a7d)

Each "experience" (like K380 pairing) is an OTel **trace**. Each navigation step is a **span** within that trace. The `StorylineLogger` writes multi-track narrative:

| Track | Purpose |
|-------|---------|
| MAIN | High-level story arc |
| KNOWLEDGE | What the agent learned |
| PERCEPTION | What was seen on screen |
| TECHNICAL | Debug data (blob counts, correlations) |
| AGENT | Agent decision reasoning |
| USER | Human-visible events |

### Experience Philosophy (session 911b4a7d)

User defined two experience types:
1. **"Getting Control"** — Lissajous sweep = the computer learning to find its own hand on screen. Like the camera swerve algorithm.
2. **"IT Work"** — Navigating Windows UI declaratively = *"like when the Star Trek computer or Skynet gains access to one terminal, it actively tries to use known ways to gain more access."*

Key directive: *"make sure entire end-to-end experience is logged, and understood in a deeper than SWE coder way."*

## 8. Key Files

| File | Purpose |
|------|---------|
| `src/hardware/cursor_detector.py` | Lissajous curve math, motion blob extraction, curvature matching |
| `src/hardware/cursor_locator.py` | Record-then-analyze pipeline, calibration, verified_click, shake |
| `src/hardware/cursor_tracker.py` | Three-tier background daemon (micro/macro/lissajous) |
| `src/hardware/esp32_mouse.py` | Serial HID mouse control, heartbeat, position tracking |
| `src/hardware/hdmi_sensor.py` | V4L2 frame capture, video recording, frame loading |
| `src/navigation/navigator.py` | WindowsNavigator: perceive-act-verify loop with LLM vision |
| `src/navigation/types.py` | NavigationStep, StepAction, StepResult |
| `src/vision/llm_vision.py` | OpenRouter LLM vision for UI element detection |
| `scripts/pair_k380.py` | 9-step K380 Bluetooth pairing experience |

## 9. Key Sessions

| Session | Duration | Topic | Resume |
|---------|----------|-------|--------|
| `2c216942` | 21m | Calculus curve origin ("DID YOU LOCATE THE POINTER?") | `claude --resume 2c216942` |
| `47d0c44b` | 10h9m | Lissajous implementation, 0.92 detection proven | `claude --resume 47d0c44b` |
| `911b4a7d` | 12h47m | Star Trek philosophy, OTel, navigation design | `claude --resume 911b4a7d` |
| `3095ea66` | 59m | Navigation module + pair_k380.py built | `claude --resume 3095ea66` |
| `f0870570` | ongoing | Fix detection, optimize sweep, shape filter | Current session |

## 10. Meta-Lesson: Experience Preservation

This knowledge base exists because:
1. Session `47d0c44b` proved the offline approach works (0.92 correlation)
2. A rewrite (commit `5dd8115`) changed to a live pipeline that was **never tested**
3. When the untested pipeline failed, the agent blamed the mouse instead of reading history
4. The user had to say: *"current situation reveals that you are not able to recreate experiences effectively... Once a good thing is created, it should not be forgotten."*

**Rule**: When an approach is proven to work, the working code path must be preserved or the replacement must be tested before commit. Conversation history IS the experience log.
