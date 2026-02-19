# Star Trek Computer — Consolidated Research

> Master document consolidating all research from 40+ Claude Code sessions (Jan 4 – Feb 19, 2026).
> Sources: `docs/*.md`, `~/.claude/plans/*.md`, JSONL session extractions, git history, experience narratives.
> Session index: `STAR_TREK_COMPUTER_SESSIONS.md` (chronological catalog with resume commands).

### Concept Structure

The sections below are organized by implementation area, but the true conceptual hierarchy is:

```
Ship Computer (§9) — orchestrator
├── ASMP (§8) — generalized actuator-sensor protocol
│   ├── Cursor Awareness (§3-6, §10) — one system, not four separate ones
│   │   The Lissajous, shape filter, three-tier recovery, C daemon, dashboard,
│   │   and CNN validation are all parts of a single tracking mesh (§1 Mesh).
│   └── Camera Awareness — Tapo C210 (separate repo, same ASMP pattern)
├── Navigation (§7) — uses Cursor Awareness for "IT Work" experience type
├── Hardware (§2) — physical actuators and sensors
└── Experience System (§7 OTel, §11 lessons) — logging and skill preservation
```

**Key insight**: Sections 3-6 and 10 describe ONE system (Cursor Awareness) from different angles.
They are kept separate below for detailed reference, but when building or debugging, treat
them as a mesh — changes to the C daemon affect the dashboard, CNN validation affects
three-tier recovery, and the Lissajous is always the last-resort fallback for all of them.

---

## 1. Vision & Philosophy

### The Star Trek Computer Concept

The system is a **Star Trek ship computer** — an AI that has full awareness and control of its environment across multiple machines. It uses multiple sensors (cameras, HDMI capture, audio) and actuators (ESP32 mouse, Pi keyboard, camera PTZ) to perceive and act on the physical world.

> *"like when the Star Trek computer or Skynet gains access to one terminal, it actively tries to use known ways to gain more access as the need arises for certain tunnel on a planet to be explored."*
> — User, session `911b4a7d` (Feb 17)

### Two Experience Types (session `911b4a7d`)

1. **"Getting Control"** — The computer learning to find its own hand on screen. The Lissajous sweep is the cursor discovering itself, like a camera swerve algorithm. The system bootstraps from zero knowledge of cursor position.

2. **"IT Work"** — Navigating Windows UI declaratively. Opening Settings, clicking through menus, pairing Bluetooth devices. The agent observes, decides, acts, verifies — like a human operator but with programmatic precision.

### Key Directives

- *"make sure entire end-to-end experience is logged, and understood in a deeper than SWE coder way"* — Every action is an experience to learn from.
- *"Once a good thing is created, it should not be forgotten"* — Proven approaches must be preserved. Conversation history IS the experience log.
- *"if the position is not known, only the locator utility needs to be used, and not random motions"* — Binding constraint for all agents.
- *"The mouse pointer should not be affected by image recognition at any point"* — Motion detection is the base layer. CNN only validates.

### Mesh Architecture (user-described, session `9545c157`)

The cursor detection algorithms work in a **mesh**, not a strict hierarchy:

```
┌──────────────────┐     ┌─────────────────┐     ┌────────────┐
│ Silhouette Scan  │ ←→  │ CNN Full-Screen  │ ←→  │   Jitter   │
│ (fast, per-frame)│     │ (is cursor here?)│     │ (confirm)  │
└────────┬─────────┘     └────────┬────────┘     └─────┬──────┘
         │                        │                      │
         └────────────┬───────────┘                      │
                      ▼                                   │
              ┌───────────────┐                           │
              │ CNN Small-Patch│ ←─────────────────────────┘
              │ (validate +   │
              │  train data)  │
              └───────────────┘
```

- **Silhouette scan** → Fast check for cursor shape variations anywhere on screen (NOT YET IMPLEMENTED)
- **CNN full-screen** → Determines if cursor is even ON this screen; low confidence = not here (NOT YET IMPLEMENTED)
- **Jitter** → Establish/confirm position when we know cursor IS on screen (IMPLEMENTED — anti-sleep ±3px)
- **CNN small-patch** → Validate final position + create training data (PARTIALLY IMPLEMENTED — CNN confidence but not trainable)

**User's latest request (Feb 19):** Pure visual algorithm that uses computer vision + locally trainable CNN to locate cursor at any cost, providing gold images or parameters like silhouette dimensions of each arrow type. Statistical tracking of those geometries when stationary AND when moving. "Discover + stick."

---

## 2. Hardware Stack

### Physical Setup

```
Desktop (Arch Linux, PRIME A520M-K)
├── USB HDMI capture card ← Windows laptop HDMI out → /dev/video0
│   MacroSilicon MS2109 USB3.0 (534d:2109)
│   1920x1080 MJPEG, ~30fps
│
├── USB serial → ESP32 (/dev/ttyUSB0) → BLE mouse → Windows ("M350 Pebble")
│   Logitech VID/PID spoofing
│   Relative HID reports (dx, dy, buttons, scroll)
│
└── USB data cable → Pi Zero 2W (10.55.0.2:8081) → BT keyboard → Windows ("Keyboard K380")
    MAC spoofing + HID profile
    HTTP API for typing: POST /type {"text": "..."}
```

### Device Paths

| Component | Device | Known Issues |
|-----------|--------|-------------|
| HDMI Capture | `/dev/video0` (desktop), `/dev/video4` (laptop) | USB autosuspend causes black frames. Fix: `echo "on" > /sys/bus/usb/devices/BUS-PORT/power/control` |
| ESP32 Mouse | `/dev/ttyUSB0` | BLE goes stale within ~1s of no activity. Fix: heartbeat (zero-movement reports) or DTR toggle for hard reset |
| Pi Zero KB | `10.55.0.2:8081` | USB gadget network needs manual IP: `ip addr add 10.55.0.1/24 dev enp8s0f3u3u4` |

### Signal Detection Heuristic

| File Size | Meaning |
|-----------|---------|
| > 100KB | Real Windows desktop content |
| 20-100KB | Partial signal or simple wallpaper |
| < 20KB | Black/solid = no HDMI signal |

### HDMI Capture Timing
- Need 2-3 frames to stabilize after connection or movement
- Wait ≥300ms after mouse movement before capturing for verification
- Use `ffmpeg -frames:v 5 -update 1` for stable single-frame capture

---

## 3. Cursor Detection — Lissajous Curve

### Origin (session `2c216942`, Feb 16)

User's vision when the agent tried to click blindly without knowing where the cursor was:

> *"DID YOU LOCATE THE POINTER?"*
> *"create a motion detection utility which moves the mouse, records video at good frame rate, processes the frame to find motion. Cursor should move in known curves across any continuous screen surface area."*
> *"the pointer will follow mathematical curves which are sneaky to find their way across any continuous screen surface area. We'll use calculus to find where the motion blur is making those same curves."*

### Why Lissajous (session `47d0c44b`)

A Lissajous curve with **irrational frequency ratio** (`ω_y/ω_x = √2`):
- **Space-filling**: visits every region of the virtual desktop
- **Smooth**: no discontinuities, produces continuous motion blur
- **Analytically known**: `x(t) = A·sin(ω₁·t)`, `y(t) = A·sin(ω₂·t)`
- **Unique curvature**: at any point, the local direction + curvature is unique (non-repeating due to irrational ratio)

### Current Parameters (optimized Feb 18)

```python
amp_x = 4300     # Half-width: covers 8600px (5x area of 2x1080p)
amp_y = 1200     # Half-height: covers 2400px
omega_x = π/5    # Period = 10s, half-cycle in 5s
omega_y = π√2/5  # Irrational ratio preserved
max_duration = 5.0  # 250 HID reports at 50Hz
```

Sized for **5x the area of two 1080p monitors** — accounts for 4K, portrait, and worst-case cursor positions.

### The Mathematical Insight

Since we know the curve equation analytically, we only need **8+ consecutive frames** where the cursor is visible:

1. **Extract motion blobs** from consecutive frame pairs (frame differencing)
2. **Build trajectory**: sequence of (centroid_x, centroid_y, angle) across frames
3. **Find consecutive runs** of frames with detected motion (min 8 frames)
4. **Slide a window** along the Lissajous time parameter `t`:
   - At each `t`, compute expected angles from the curve equation
   - Compare expected vs observed angles (circular correlation)
5. **Best match** gives us `t_start` and `t_end`

The direction angle `θ(t) = atan2(A_y·ω_y·cos(ω_y·t), A_x·ω_x·cos(ω_x·t))` is **unique at every t** due to the irrational frequency ratio.

### Proven Result

Session `47d0c44b` achieved **0.92 correlation** detecting cursor at (1842, 15) from 76 consecutive blob frames. Analysis took <0.01s after blob extraction.

### Record-Then-Analyze Pipeline (PROVEN)

```
start_recording()          # ffmpeg captures to MKV
_execute_curve()           # Blocking Lissajous sweep (5s)
stop_recording()           # Clean MKV file with proper trailer
load_video_frames()        # Decode ALL frames at once
extract_motion_blobs()     # Frame-pair diffs
match_curve_segment()      # Sliding window correlation
```

### Live Pipeline (FAILED — commit `5dd8115`)

3 concurrent threads failed because `load_video_frames()` on a growing MKV (no container trailer) is unreliable. Analyze loop only ran once, missed the late frames where cursor was visible. **Reverted to record-then-analyze.**

### Key Files

| File | Purpose |
|------|---------|
| `src/hardware/cursor_detector.py` | Lissajous curve math, motion blob extraction, curvature matching, `_cursor_shape_score()` |
| `src/hardware/cursor_locator.py` | Record-then-analyze pipeline, calibration, `verified_click()`, `shake()` |

---

## 4. Cursor Detection — Shape & Silhouette

### What Exists: Shape Filter (`_cursor_shape_score()` in `cursor_detector.py`)

Motion blobs scored by how "cursor-like" their shape is:

| Blob Type | Fill Ratio | Aspect | Behavior |
|-----------|-----------|--------|----------|
| Cursor arrow | 0.3-0.5 | 0.5-0.8 | Translates with movement |
| Text cursor (I-beam) | 0.8+ | < 0.15 | Stationary blink |
| Spinner | 0.7+ | > 0.8 | Stationary rotation |
| Screen transition | varies | varies | >100px, fills region |

**Translation threshold** of 3.0 px/frame eliminates stationary animations.

### What Handles Different Cursor Colors

The shape filter works on **motion blob geometry**, not color. Whatever color the cursor is (white, black, inverted), moving it produces a blob with the arrow's characteristic shape.

### What's Missing: Pure Visual Detection (User Request, Feb 19)

The user wants a system that can find the cursor **without moving it**:
- **Silhouette scan**: Check all cursor shape variations (arrow, hand, I-beam, hourglass, crosshair) across the entire screen
- **Gold images/parameters**: Known dimensions and shapes of each Windows cursor type
- **Statistical tracking**: Track cursor geometries when stationary AND when moving
- **Trainable CNN**: Local model that learns from confirmed cursor positions
- **"Discover + stick"**: Find the cursor visually, then maintain lock

#### Windows Cursor Types (for silhouette library)

| Cursor | Approx Size (1080p) | Shape | Key Features |
|--------|---------------------|-------|-------------|
| Normal Select (arrow) | 12×19 px | Triangle + tail | White fill, black outline, ~0.35 fill ratio |
| Text Select (I-beam) | 5×20 px | Thin vertical | Very low aspect ratio (<0.25) |
| Hand (link) | 15×19 px | Pointing hand | Higher fill ratio (~0.5) |
| Busy (hourglass) | 15×19 px | Rotating ring | Animated, changes shape |
| Resize (arrows) | 19×19 px | Double arrow | Symmetric, ~0.3 fill ratio |
| Crosshair | 19×19 px | Cross shape | Very low fill ratio (~0.15) |
| Move (4-arrow) | 19×19 px | 4 arrows | Low fill ratio |

---

## 5. Cursor Detection — Three-Tier Recovery

### `cursor_tracker.py` — Always-On Position Monitor

```
Tier 1: Micro-shake (±3px)
  - Invisible to user, 40ms at half-res
  - Confirms cursor is still at known position
  - Runs every 2s when position is stale (>10s old)

Tier 2: Macro-shake (±15px)
  - Slightly visible, catches small drift
  - Updates position to actual blob centroid
  - Only if Tier 1 fails

Tier 3: Lissajous re-detect
  - Full 5s sweep, expensive
  - Only if Tier 2 also fails (cursor truly lost)
  - Resets position from scratch
```

All operations run at **960×540** (downsampled) for speed — blob extraction drops from ~160ms/pair at full-res to ~40ms/pair at half-res.

### Adaptive Detection Area (Section 11 of Knowledge Base)

Once cursor is found and stationary, monitor a **200×200px window** around known position (< 5ms instead of 40ms). Expands dynamically:
- Cursor moves beyond jitter threshold (>3px) → area grows to follow trajectory
- Cursor disappears entirely → expand 200→400→800→full frame → Tier 3 Lissajous

---

## 6. Cursor Detection — C Daemon

### Current Implementation

The C daemon (`daemon/cursor-daemon`) runs as a separate process:
- Captures from V4L2 at 30fps (MJPEG decode to Y-plane)
- Frame differencing in C with connected component labeling (CCL)
- Writes to POSIX shared memory at `/dev/shm/cursor-daemon`
- Lock-free sequence counter for torn-read detection
- Double-buffered JPEG (write to inactive, atomically flip)

### Shared Memory Layout (`daemon_client.py`)

```
OFF_MAGIC       = 0      (0xCDA0CDA0)
OFF_SEQ_BEGIN   = 8      (64-bit sequence counter)
OFF_CURSOR_X    = 16     (int32)
OFF_CURSOR_Y    = 20     (int32)
OFF_CURSOR_AGE  = 24     (float)
OFF_BLOB_COUNT  = 32     (int32)
OFF_BLOBS       = 36     (32 × 24 bytes = 768 bytes)
OFF_FPS         = 804    (float)
OFF_SEQ_END     = 832    (64-bit sequence counter)
OFF_JPEG0       = 848    (512KB buffer 0)
OFF_JPEG1       = 525140 (512KB buffer 1)
OFF_JPEG_ACTIVE = 1049432 (which buffer to read)
```

### Known Issue: Stale JPEG Buffer

The daemon's **blob data** (`read_state()`) updates reliably at 30fps. But the **JPEG double-buffer** sometimes stales — multiple reads return identical bytes. This was the root cause of Python probe failures in session `b1625283`: probes read daemon JPEGs, diffed them, got zero blobs because all frames were identical.

**Fix**: Probes must use daemon blob data (`read_state().blobs`), not JPEG diff.

### Cursor Methods (daemon writes these)

| ID | Method | Description |
|----|--------|-------------|
| 0 | unknown | No tracking active |
| 1 | motion_track | Blob closest to last position |
| 2 | shape_track | Shape-filtered blob match |
| 3 | micro_shake | Tier 1: ±3px jitter confirmation |
| 4 | macro_shake | Tier 2: ±15px reacquire |
| 5 | lissajous | Tier 3: full sweep |
| 6 | api_probe | Manual probe endpoint |
| 7 | cnn_reacquire | CNN-assisted reacquire |

---

## 7. Navigation System

### WindowsNavigator (session `3095ea66`)

Perceive-act-verify loop with LLM vision:

```python
class NavigationStep:
    name: str           # "open_bluetooth_settings"
    action: StepAction  # click(x, y), type("text"), scroll(dy)
    verify: str         # "Bluetooth & devices page visible"
    timeout: float      # Max seconds to wait for verification
```

### Parallel Hypothesis Recovery

When a click fails, the system spawns **parallel Claude Agent SDK agents** to diagnose:
- `cursor_drifted.md` — LLM analyzes screenshot: "Where is the mouse cursor? How far from expected?"
- `click_missed.md` — LLM analyzes: "Is the target visible? How far was the cursor?"

### OTel StorylineLogger

Multi-track narrative logging (Agent Zero compatible):

| Track | Purpose |
|-------|---------|
| MAIN | High-level story arc |
| KNOWLEDGE | What the agent learned |
| PERCEPTION | What was seen on screen |
| TECHNICAL | Debug data (blob counts, correlations) |
| AGENT | Agent decision reasoning |
| USER | Human-visible events |

Each "experience" (like K380 pairing) is an OTel **trace**. Each step is a **span**.

---

## 8. Actuator-Sensor Mapping Protocol (ASMP)

### Generalized Protocol (session `dbad911b`, Jan 15)

Any **actuator** (thing that moves) mapped to any **sensor** (thing that observes) through an **API layer**:

```
ACTUATOR ←→ API ←→ KNOWLEDGE/ACTION ←→ AGENT (Claude)
(PTZ cam)   (ONVIF)  (LLM Vision)       (Hooks)
(ESP32 mouse)(Serial) (Frame diff)       (Commands)
(Drone)     (MAVLink) (SLAM)             (Goals)
```

### Hybrid Binary Spatial Search

```
Phase 1: COARSE SWEEP (continuous, low quality)
├── Actuator moves continuously
├── Sensor stream analyzed in real-time
├── Passive snapshots every N ms
├── Agent hooks: detection events
└── If target POSSIBLY detected → Phase 2

Phase 2: BINARY REFINEMENT (stop-and-capture, high quality)
├── Stop at detection region
├── High-quality capture (no motion blur)
├── LLM confirms with high confidence
├── If false positive → binary subdivide space
└── Repeat until found

Phase 3: LOCK & TRACK
├── Maintain position on target
├── Periodic re-confirmation
└── Alert on target movement
```

### Implementations

| Implementation | Actuator | Sensor | Detection |
|---------------|----------|--------|-----------|
| Cursor detection | ESP32 BLE mouse | USB HDMI capture | Frame differencing (deterministic) |
| Camera search | Tapo C210 PTZ | RTSP stream | LLM vision (Gemini Flash for sweep, Claude Sonnet for confirm) |

### Original User Vision (verbatim, Jan 15)

> *"the approach should be such that while the camera is swiping, it is taking snapshots or like a video and that is also being sent to the some sort of API and those are presented to the agent as hooks. And then if it is detected that on the way it something that keyword was found or something, then not even keyword, the CLAUDE agent just gets that thing. And then the CLAUDE agent is asked to just stop."*

---

## 9. Ship Computer / Orchestrator

### Telegram Heartbeat (plan `calm-booping-leaf.md`)

Desktop runs 1-minute heartbeat checking Telegram, dispatching commands, spawning background Claude agents:

| Command | Action | Spawns Agent? |
|---------|--------|--------------|
| `/help` | List commands | No |
| `/status` | System stats, uptime | No |
| `/connect` | Start interactive 5s-poll session | Yes |
| `/agents` | List running agents | No |
| Any text | Spawn background one-shot agent | Yes (max 10 turns, $0.50) |

RPi5 acts as fallback — pings desktop, notifies user if down.

### L1/L2/L3 Agent Hierarchy (`STAR_TREK_COMPUTER_PROMPT.md`)

- **L1** (Strategic Orchestrator): Understands overall task, monitors Knowledge Building dashboard, directs L2
- **L2** (Timeline Summarizer): Creates 5-minute summaries from keyframes
- **L3** (Parallel Screenshot Analyzers): Multiple agents processing keyframes in parallel via Claude Agent SDK

### Modes

| Mode | Screenshot Interval | Change Threshold | Summarization |
|------|-------------------|-----------------|---------------|
| Learning | Every 10s | 5% | All keyframes |
| Doing | Every 30s | 15% | On-demand |
| Meeting | Every 60s | 30% | Batch (5 min) |

---

## 10. KVM Dashboard (`kvm_dashboard.py`)

### What It Does

Web dashboard at `http://localhost:8766` showing:
- Live MJPEG video stream from HDMI capture
- Cursor position overlay (green pulsing dot via CSS)
- Blob rectangles (blue for all, green for pointer)
- Sidebar: blob count, tracker method, CNN confidence, FPS, mouse status

### API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/video_feed` | MJPEG stream |
| `/api/state` | JSON: cursor pos, blobs, fps, mouse status |
| `/api/probe` | Move mouse, detect blob, return position |
| `/api/locate` | Dual-direction probe (120px) |
| `/api/mouse/move` | Send relative mouse movement |
| `/api/mouse/reconnect` | DTR toggle for ESP32 reset |

### Background Loops

1. **Daemon motion loop** (20Hz) — Reads daemon state, updates overlay, tracks cursor during jitter window
2. **Jitter reacquire** (every 30s) — Dual-direction 15px probe to confirm/update cursor position
3. **Anti-sleep** (every 30s) — ±3px jitter to keep Windows from sleeping

### Known Issues (as of Feb 19)

- Daemon JPEG buffer stales intermittently — probes must use blob data, not JPEG diff
- ESP32 BLE disconnects if no heartbeat — needs continuous zero-movement reports
- Continuous blob tracking on busy screens chases video noise — only track during jitter window

---

## 11. Key Lessons Learned

### Experience Preservation (Meta-Lesson)

Session `47d0c44b` proved offline analysis works (0.92 correlation). Commit `5dd8115` rewrote to live pipeline that was **never tested**. When it failed, the agent blamed the mouse instead of reading history. User's response:

> *"current situation reveals that you are not able to recreate experiences effectively... Once a good thing is created, it should not be forgotten."*

**Rule**: When an approach is proven to work, the working code path must be preserved. Conversation history IS the experience log.

### Instruction Binding

Claude reads CLAUDE.md and project docs but doesn't always bind to them. Instructions are treated as "information" not "rules." The fix: read docs → COMMIT to follow them → check inference against constraints → ASK before violating.

### Read Before Code

Before changing any code, read:
1. Git log (recent commits contain solutions already tried)
2. Session history (previous conversations may have solved this)
3. Project docs (CURSOR_DETECTION_KNOWLEDGE.md, etc.)

### Proven vs Untested

| Approach | Status | Evidence |
|----------|--------|----------|
| Record-then-analyze (offline) | PROVEN | 0.92 correlation, session `47d0c44b` |
| Live 3-thread pipeline | FAILED | Growing MKV unreliable, commit `5dd8115` |
| C daemon blob extraction | WORKING | 30fps, reliable blob data via shared memory |
| C daemon JPEG buffer | INTERMITTENT | Sometimes stales, same bytes on multiple reads |
| Dual-direction probe | UNTESTED (mouse was disconnected) | Correct math verified against `/api/locate` |
| Shape filter (`_cursor_shape_score`) | IMPLEMENTED | Fill ratio, aspect ratio scoring |
| Pure visual detection (silhouette) | NOT IMPLEMENTED | User's latest requirement |
| Trainable CNN | NOT IMPLEMENTED | User's latest requirement |

---

## 12. Open Problems

### Pure Visual Cursor Detection (User Request, Feb 19)

> *"even when mouse is not connected there should be a pure visual algorithm, which uses computer vision local trainable cnn at high level to locate cursor at any cost and providing gold images or parameters like silhouette dimensions of each type of arrow that the current arrow could be, and a statistical way to track those geometries when the pointer is stationary AND when it moves. Discover + stick."*

This requires:
1. **Silhouette library**: Gold images / parameter sets for each Windows cursor type
2. **Full-screen scan**: Template matching or sliding window CNN across 1920×1080
3. **Trainable local CNN**: Learns from confirmed cursor positions (self-supervised from jitter confirmations)
4. **Stationary tracking**: Track cursor without movement — use appearance, not motion
5. **Moving tracking**: When cursor moves, statistical tracking of the geometry as it translates

### Integration with Existing Mesh

The pure visual detection fills the "Silhouette scan" and "CNN full-screen" boxes in the mesh architecture (Section 1). It should:
- Run independently of mouse connection
- Provide candidates that jitter + CNN small-patch validate
- Feed training data back to the CNN from confirmed positions
- Degrade gracefully: visual detection → jitter confirmation → Lissajous re-detect

---

## 13. Key Files

| File | Purpose |
|------|---------|
| `src/hardware/cursor_detector.py` | Lissajous curve math, motion blob extraction, curvature matching, shape scoring |
| `src/hardware/cursor_locator.py` | Record-then-analyze pipeline, calibration, verified_click, shake |
| `src/hardware/cursor_tracker.py` | Three-tier background daemon (micro/macro/lissajous) |
| `src/hardware/esp32_mouse.py` | Serial HID mouse control, heartbeat, anti-sleep, position tracking |
| `src/hardware/hdmi_sensor.py` | V4L2 frame capture, video recording, frame loading |
| `src/hardware/daemon_client.py` | Python client for C daemon shared memory |
| `src/web/kvm_dashboard.py` | Web dashboard with video feed, API endpoints, background loops |
| `src/navigation/navigator.py` | WindowsNavigator: perceive-act-verify loop with LLM vision |
| `src/navigation/types.py` | NavigationStep, StepAction, StepResult |
| `src/vision/llm_vision.py` | OpenRouter LLM vision for UI element detection |
| `daemon/cursor-daemon` | C daemon binary (V4L2 capture, frame diff, blob extraction, shared memory) |
| `scripts/pair_k380.py` | 9-step K380 Bluetooth pairing experience |
| `STAR_TREK_COMPUTER_PROMPT.md` | L1/L2/L3 orchestrator agent prompt |
| `agents/CONTROL.md` | Agent control center (current mode, recording settings) |
| `docs/protocols/ACTUATOR_SENSOR_PROTOCOL.md` | Generalized ASMP protocol |
| `docs/protocols/SPATIAL_SEARCH_PROTOCOL_RAW.md` | Original user vision (verbatim) |
