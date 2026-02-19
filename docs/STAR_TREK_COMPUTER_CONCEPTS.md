# Star Trek Computer — True Concept Structure

> This document reorganizes all research around the **actual conceptual architecture**,
> not implementation details. For detailed reference, see `STAR_TREK_COMPUTER_RESEARCH.md`.
> For session history, see `STAR_TREK_COMPUTER_SESSIONS.md`.

---

## The Six Real Concepts

Everything in this project reduces to six concepts. The 13 sections in
RESEARCH.md are implementation details of these six:

```
                        ┌─────────────────────────┐
                        │    SHIP COMPUTER         │
                        │    (Star Trek Vision)    │
                        │                          │
                        │  "The computer on the    │
                        │   Enterprise that knows  │
                        │   everything and can do  │
                        │   anything through any   │
                        │   terminal it touches"   │
                        └────────────┬────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
           ┌────────▼────────┐ ┌────▼─────┐ ┌───────▼────────┐
           │      ASMP       │ │  AGENT   │ │   EXPERIENCE   │
           │ Actuator-Sensor │ │  BRAIN   │ │    SYSTEM      │
           │ Mapping Protocol│ │ (Claude) │ │ (memory +      │
           │                 │ │          │ │  narrative)     │
           │ "Any actuator   │ │ "LLM     │ │                │
           │  mapped to any  │ │  vision  │ │ "Once a good   │
           │  sensor through │ │  + tools │ │  thing is made │
           │  an API layer"  │ │  + hooks"│ │  it should not │
           └───────┬─────────┘ └────┬─────┘ │  be forgotten" │
                   │                │       └───────┬────────┘
          ┌────────┴────────┐       │               │
          │                 │       │               │
   ┌──────▼──────┐  ┌──────▼──────┐│       ┌───────▼────────┐
   │   CURSOR    │  │   CAMERA    ││       │  OTel Traces   │
   │  AWARENESS  │  │  AWARENESS  ││       │  StorylineLog  │
   │             │  │             ││       │  Conversation  │
   │  ESP32 mouse│  │  Tapo C210  ││       │  History as    │
   │  + HDMI cap │  │  + ONVIF    ││       │  Experience    │
   └──────┬──────┘  └─────────────┘│       └────────────────┘
          │                        │
          │    ┌───────────────┐   │
          └───►│  NAVIGATION   │◄──┘
               │               │
               │ perceive →    │
               │ decide →      │
               │ act →         │
               │ verify        │
               └───────────────┘
```

### Mapping from RESEARCH.md sections

| True Concept | RESEARCH.md Sections | Why they're one concept |
|-------------|---------------------|------------------------|
| Ship Computer | §1 Vision, §9 Orchestrator | The "why" — Star Trek terminal metaphor |
| ASMP | §8 Protocol | The generalizable pattern behind everything |
| Cursor Awareness | §3 Lissajous, §4 Shape, §5 Three-Tier, §6 C Daemon, §10 Dashboard | ALL parts of one tracking mesh — not separate systems |
| Camera Awareness | §8 (Tapo example) | Same ASMP pattern, different actuator/sensor |
| Navigation | §7 Navigator | Consumes Cursor Awareness to do IT Work |
| Experience System | §7 OTel, §11 Lessons, §12 Open Problems | Memory that persists across sessions |
| (Hardware) | §2, §13 | Physical layer — not a concept, a dependency |

---

## Concept 1: Ship Computer

**What it is:** An AI that gains awareness and control of its environment, terminal
by terminal, exactly like the Enterprise computer.

**Two experience types:**

```
┌─────────────────────────────────────────────────────────────┐
│                    SHIP COMPUTER                            │
│                                                             │
│  Experience Type 1: "GETTING CONTROL"                       │
│  ─────────────────────────────────────                      │
│  The computer learning to find its own hand on screen.      │
│  Lissajous sweep = camera swerve algorithm.                 │
│  Bootstrap from ZERO knowledge of cursor position.          │
│                                                             │
│  Experience Type 2: "IT WORK"                               │
│  ─────────────────────────────                              │
│  Navigating Windows UI declaratively.                       │
│  Open Settings → Bluetooth → Pair K380.                     │
│  Observe → Decide → Act → Verify.                           │
│                                                             │
│  ┌─────────┐     ┌─────────┐     ┌─────────┐               │
│  │ Telegram │────►│ Spawner │────►│ Claude  │               │
│  │ Command  │     │ (L1)    │     │ Agents  │               │
│  └─────────┘     └────┬────┘     │ (L2/L3) │               │
│                       │          └─────────┘               │
│  L1: Strategic Orchestrator (goals, modes)                  │
│  L2: Timeline Summarizer (5-min keyframe batches)           │
│  L3: Parallel Screenshot Analyzers (Agent SDK)              │
│                                                             │
│  Modes: Learning (10s) │ Doing (30s) │ Meeting (60s)        │
└─────────────────────────────────────────────────────────────┘
```

**Key user quotes:**
- *"like when the Star Trek computer or Skynet gains access to one terminal"*
- *"make sure entire end-to-end experience is logged, and understood in a deeper than SWE coder way"*

---

## Concept 2: ASMP (The Unifying Pattern)

**What it is:** Any actuator mapped to any sensor through an API layer, with
an agent providing intelligence. This is the ONE pattern that Cursor Awareness,
Camera Awareness, and future capabilities all implement.

```
┌─────────────────────────────────────────────────────────────┐
│                 ASMP — THE PATTERN                          │
│                                                             │
│   ACTUATOR ←──→ API ←──→ KNOWLEDGE ←──→ AGENT             │
│                                                             │
│   Instance 1: Cursor                                        │
│   ┌──────────┐  ┌────────┐  ┌───────────┐  ┌──────────┐   │
│   │ ESP32    │  │ Serial │  │ Frame     │  │ Claude   │   │
│   │ BLE Mouse│──│ /USB   │──│ Diff +   │──│ Vision + │   │
│   │          │  │        │  │ Blob CCL  │  │ Commands │   │
│   └──────────┘  └────────┘  └───────────┘  └──────────┘   │
│                                                             │
│   Instance 2: Camera                                        │
│   ┌──────────┐  ┌────────┐  ┌───────────┐  ┌──────────┐   │
│   │ Tapo     │  │ ONVIF  │  │ LLM Vision│  │ Claude   │   │
│   │ C210 PTZ │──│ RTSP   │──│ (Gemini   │──│ Agent    │   │
│   │          │  │        │  │  + Sonnet) │  │ + Hooks  │   │
│   └──────────┘  └────────┘  └───────────┘  └──────────┘   │
│                                                             │
│   Future Instance: Drone                                    │
│   ┌──────────┐  ┌────────┐  ┌───────────┐  ┌──────────┐   │
│   │ Drone    │  │MAVLink │  │ SLAM      │  │ Claude   │   │
│   │ Motors   │──│        │──│           │──│ Goals    │   │
│   └──────────┘  └────────┘  └───────────┘  └──────────┘   │
│                                                             │
│   THREE-PHASE SEARCH (all instances follow this):           │
│                                                             │
│   Phase 1: COARSE SWEEP ───► Phase 2: BINARY REFINE        │
│   (continuous, low quality)   (stop-and-capture, confirm)   │
│                                        │                    │
│                               Phase 3: LOCK & TRACK         │
│                               (maintain, re-confirm)        │
└─────────────────────────────────────────────────────────────┘
```

---

## Concept 3: Cursor Awareness (The Tracking Mesh)

**What it is:** ONE system with multiple algorithms that work as a mesh, not a
hierarchy. This is the single biggest insight from consolidation — RESEARCH.md
§3, §4, §5, §6, and §10 all describe different faces of this one system.

```
┌─────────────────────────────────────────────────────────────┐
│              CURSOR AWARENESS — THE MESH                    │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 DISCOVERY LAYER                       │   │
│  │                                                      │   │
│  │  Lissajous Sweep (5s, √2 freq ratio)                │   │
│  │  ┌─────────────────────────────────────────┐         │   │
│  │  │ x(t) = A·sin(ω₁·t)                     │         │   │
│  │  │ y(t) = A·sin(ω₂·t)  where ω₂/ω₁ = √2  │         │   │
│  │  │                                         │         │   │
│  │  │ Space-filling, analytically known,      │         │   │
│  │  │ unique curvature at every t.            │         │   │
│  │  │ Match observed angles → find cursor.    │         │   │
│  │  │ PROVEN: 0.92 correlation.               │         │   │
│  │  └─────────────────────────────────────────┘         │   │
│  │                                                      │   │
│  │  Silhouette Scan (NOT YET BUILT)                     │   │
│  │  Template matching for cursor shapes WITHOUT moving  │   │
│  │                                                      │   │
│  │  CNN Full-Screen (NOT YET BUILT)                     │   │
│  │  "Is cursor even ON this screen?" binary classifier  │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 TRACKING LAYER                        │   │
│  │                                                      │   │
│  │  C Daemon (30fps V4L2 → frame diff → blob CCL)       │   │
│  │  ├── Shared memory IPC at /dev/shm/cursor-daemon     │   │
│  │  ├── Lock-free sequence counter                      │   │
│  │  └── Double-buffered JPEG (known: stales sometimes)  │   │
│  │                                                      │   │
│  │  Shape Filter (_cursor_shape_score)                   │   │
│  │  ├── Fill ratio 0.3-0.5 = arrow cursor               │   │
│  │  ├── Aspect ratio 0.5-0.8 = normal arrow             │   │
│  │  └── Translation > 3px/frame = real movement          │   │
│  │                                                      │   │
│  │  Motion-Based Position (blob centroid nearest to last)│   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 VALIDATION LAYER                      │   │
│  │                                                      │   │
│  │  TinyCursorNet CNN (14,337 params, ~2ms inference)   │   │
│  │  ├── Validates motion-detected position               │   │
│  │  ├── NOT a replacement for motion detection           │   │
│  │  └── Self-supervised from jitter confirmations        │   │
│  │                                                      │   │
│  │  Dual-Direction Probe (120px jitter + diff)           │   │
│  │  ├── Move known amount → measure blob displacement    │   │
│  │  └── Both directions cancel out noise                 │   │
│  │                                                      │   │
│  │  Jitter Correlation (anti-sleep ±3px → match blob)    │   │
│  │  └── Free validation every 30s from anti-sleep        │   │
│  └──────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 RECOVERY LAYER                        │   │
│  │                                                      │   │
│  │  Three-Tier Escalation:                               │   │
│  │                                                      │   │
│  │  Tier 1: Micro-shake (±3px, 40ms, invisible)         │   │
│  │     │    "Are you still there?"                       │   │
│  │     │    Runs every 2s when position stale >10s       │   │
│  │     ▼ fail                                           │   │
│  │  Tier 2: Macro-shake (±15px, slightly visible)       │   │
│  │     │    "Lost you nearby — searching close range"    │   │
│  │     │    Updates position to actual blob centroid     │   │
│  │     ▼ fail                                           │   │
│  │  Tier 3: Lissajous re-detect (5s full sweep)         │   │
│  │          "Completely lost — starting from scratch"    │   │
│  │          Resets position from zero                    │   │
│  │                                                      │   │
│  │  Adaptive area: 200px → 400 → 800 → full → Tier 3   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                 INFRASTRUCTURE                        │   │
│  │                                                      │   │
│  │  KVM Dashboard (port 8766)                            │   │
│  │  ├── Live MJPEG stream with blob overlays             │   │
│  │  ├── CSS cursor dot (green pulsing, position mapped)  │   │
│  │  ├── Sidebar: blobs, tracker tier, CNN, hardware      │   │
│  │  └── API: /api/state, /api/probe, /api/locate         │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘

MESH connections (any node can feed any other):

  Silhouette ←──→ CNN Full ←──→ Jitter ←──→ CNN Patch
       │              │            │             │
       └──────────────┴────────────┴─────────────┘
                    All feed into
                  TRACKING LAYER
```

---

## Concept 4: Camera Awareness (Life Tracking)

**What it is:** The same ASMP pattern applied to the Tapo C210 camera —
eventually tracking everything in the user's physical space.

```
┌─────────────────────────────────────────────────────────────┐
│              CAMERA AWARENESS — LIFE TRACKING               │
│                                                             │
│  CURRENT STATE:                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Tapo C210 Camera                                    │    │
│  │ ├── ONVIF PTZ (pan, tilt, zoom over network)        │    │
│  │ ├── RTSP live stream (1080p)                        │    │
│  │ ├── Controlled via Android emulator (ADB)           │    │
│  │ │   └── BlueStacks / Tapo app automation            │    │
│  │ └── ASMP Phase 1: Coarse sweep via PTZ              │    │
│  │     Phase 2: Stop-and-capture via LLM vision        │    │
│  │     Phase 3: Lock on target and track               │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  VISION — "TRACK EVERYTHING IN MY LIFE":                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                                                     │    │
│  │  Camera at desk ─────┐                              │    │
│  │  Camera at door ─────┼───► Ship Computer            │    │
│  │  Camera at kitchen ──┘    │                         │    │
│  │                           ├── Person tracking       │    │
│  │                           ├── Activity recognition  │    │
│  │                           ├── Object state          │    │
│  │                           │   (door open/closed,    │    │
│  │                           │    lights on/off,       │    │
│  │                           │    person at desk)      │    │
│  │                           └── Proactive actions     │    │
│  │                               (dim lights when      │    │
│  │                                leaving, alert if    │    │
│  │                                forgot something)    │    │
│  │                                                     │    │
│  │  Same ASMP: actuator=PTZ, sensor=RTSP,             │    │
│  │  knowledge=LLM vision, agent=Claude                 │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## Concept 5: Navigation (The IT Work Loop)

**What it is:** Perceive-act-verify loop that uses Cursor Awareness to
actually DO things on Windows. This is "Experience Type 2" — IT Work.

```
┌─────────────────────────────────────────────────────────────┐
│               NAVIGATION — PERCEIVE-ACT-VERIFY              │
│                                                             │
│  WindowsNavigator loop:                                     │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐              │
│  │ PERCEIVE │───►│  DECIDE  │───►│   ACT    │              │
│  │          │    │          │    │          │              │
│  │ Screenshot│    │ LLM says │    │ Click at │              │
│  │ via HDMI  │    │ "click   │    │ (x, y)  │              │
│  │ capture   │    │  Settings│    │ via ESP32│              │
│  │           │    │  button" │    │ mouse   │              │
│  └──────────┘    └──────────┘    └────┬─────┘              │
│       ▲                               │                     │
│       │          ┌──────────┐         │                     │
│       └──────────│  VERIFY  │◄────────┘                     │
│                  │          │                                │
│                  │ Take new │                                │
│                  │ screenshot│                               │
│                  │ "Did the │                                │
│                  │  Settings│                                │
│                  │  page    │                                │
│                  │  open?"  │                                │
│                  └────┬─────┘                                │
│                       │                                      │
│              ┌────────▼────────┐                             │
│              │ HYPOTHESIS      │                             │
│              │ RECOVERY        │                             │
│              │ (on failure)    │                             │
│              │                 │                             │
│              │ Parallel agents:│                             │
│              │ ├── cursor_drifted.md — "Where IS cursor?"   │
│              │ ├── click_missed.md  — "Was target visible?" │
│              │ └── Spawned via Agent SDK                    │
│              └─────────────────┘                             │
│                                                             │
│  Each step is a NavigationStep:                              │
│  ┌──────────────────────────────────────────┐               │
│  │ name: "open_bluetooth_settings"          │               │
│  │ action: click(x=1200, y=400)             │               │
│  │ verify: "Bluetooth & devices page"       │               │
│  │ timeout: 5.0                             │               │
│  └──────────────────────────────────────────┘               │
│                                                             │
│  Depends on Cursor Awareness for:                           │
│  - Knowing WHERE the cursor is before clicking              │
│  - Confirming the cursor MOVED to the target                │
│  - Recovering when click misses (cursor drift detection)    │
└─────────────────────────────────────────────────────────────┘
```

---

## Concept 6: Experience System (Memory That Persists)

**What it is:** The system that makes this "Star Trek" instead of just
"automation." Every action is an experience. Every experience is logged.
Every log can be replayed and learned from.

```
┌─────────────────────────────────────────────────────────────┐
│             EXPERIENCE SYSTEM — "NEVER FORGET"              │
│                                                             │
│  OTel StorylineLogger:                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ Each "experience" (e.g., K380 Pairing) = OTel TRACE │    │
│  │ Each step = OTel SPAN                               │    │
│  │                                                     │    │
│  │ Tracks: MAIN │ KNOWLEDGE │ PERCEPTION │ TECHNICAL   │    │
│  │         AGENT │ USER                                │    │
│  │                                                     │    │
│  │ Multi-track narrative:                              │    │
│  │ ├── MAIN: "Pairing K380 keyboard"                   │    │
│  │ ├── KNOWLEDGE: "Windows PIN entry requires 6 digits"│    │
│  │ ├── PERCEPTION: "Bluetooth settings page visible"   │    │
│  │ ├── TECHNICAL: "blob_count=12, correlation=0.92"    │    │
│  │ ├── AGENT: "Chose to click 'Add device' button"     │    │
│  │ └── USER: "Pairing succeeded"                       │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Conversation History = Experience Log:                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 40+ Claude Code sessions (Jan 4 – Feb 19, 2026)     │    │
│  │ ├── Stored in datalake (SQLite FTS5)                │    │
│  │ ├── Searchable: find-conversations "query"          │    │
│  │ ├── Resumable: claude --resume SESSION_ID           │    │
│  │ └── Key insight: PROVEN approaches must be          │    │
│  │     preserved. "Once a good thing is created,       │    │
│  │     it should not be forgotten."                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Meta-Lessons (learned the hard way):                       │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ 1. Read git log BEFORE coding (solutions exist)     │    │
│  │ 2. Proven code paths must be preserved              │    │
│  │ 3. Having information ≠ using it (instruction bind) │    │
│  │ 4. Motion detection is base layer; CNN only         │    │
│  │    validates — "mouse should not be affected by     │    │
│  │    image recognition at any point"                  │    │
│  │ 5. If position unknown, ONLY use locator utility    │    │
│  │    — never random motions                           │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## The Hardware Layer

Not a concept — a dependency. Physical things that enable the concepts.

```
┌─────────────────────────────────────────────────────────────┐
│                    HARDWARE STACK                            │
│                                                             │
│  Desktop (Arch Linux, PRIME A520M-K)                        │
│  ├── USB HDMI Capture ←── Windows Laptop HDMI Out           │
│  │   MS2109, 1920x1080 MJPEG, ~30fps, /dev/video0          │
│  │                                                          │
│  ├── USB Serial → ESP32 → BLE Mouse → Windows              │
│  │   /dev/ttyUSB0, Logitech M350 Pebble spoof              │
│  │   Relative HID (dx, dy, buttons, scroll)                 │
│  │                                                          │
│  └── USB Data → Pi Zero 2W → BT Keyboard → Windows         │
│      10.55.0.2:8081, K380 MAC spoof                         │
│      HTTP API: POST /type {"text": "..."}                   │
│                                                             │
│  Signal: >100KB = real content, <20KB = no signal           │
│  Timing: 300ms wait after move, 2-3 frames to stabilize    │
│  Gotcha: USB autosuspend causes black frames                │
│  Gotcha: BLE dies within ~1s without heartbeat              │
└─────────────────────────────────────────────────────────────┘
```

---

## Open Frontiers

What's built, what's not, and what comes next:

```
┌─────────────────────────────────────────────────────────────┐
│                    STATUS MAP                                │
│                                                             │
│  BUILT & PROVEN                                             │
│  ═══════════════                                            │
│  [■] Lissajous sweep + offline analysis (0.92 correlation)  │
│  [■] C daemon (30fps V4L2, blob CCL, shared memory)         │
│  [■] Shape filter (fill ratio, aspect ratio, translation)   │
│  [■] Three-tier recovery (micro/macro/lissajous)            │
│  [■] KVM Dashboard (MJPEG, overlays, sidebar)               │
│  [■] Anti-sleep jitter (±3px every 30s)                     │
│  [■] TinyCursorNet CNN (14K params, 2ms inference)          │
│  [■] Dual-direction probe math (verified)                   │
│  [■] Record-then-analyze pipeline                           │
│                                                             │
│  BUILT, ISSUES KNOWN                                        │
│  ═══════════════════                                        │
│  [◧] C daemon JPEG buffer (stales intermittently)           │
│  [◧] ESP32 BLE connection (disconnects, needs heartbeat)    │
│  [◧] Jitter correlation tracking (only during jitter window)│
│                                                             │
│  NOT YET BUILT                                              │
│  ═════════════                                              │
│  [□] Silhouette scan (template match cursor shapes)         │
│  [□] CNN full-screen ("is cursor even on this screen?")     │
│  [□] Trainable local CNN (self-supervised from jitter)      │
│  [□] Pure visual detection without mouse movement           │
│  [□] Statistical cursor geometry tracking                   │
│  [□] Camera Awareness (Tapo C210 ASMP integration)          │
│  [□] Navigation system (perceive-act-verify running)        │
│  [□] Ship Computer orchestrator (Telegram → agent spawner)  │
│  [□] Experience System (OTel traces in production)          │
│  [□] Multi-camera life tracking                             │
│                                                             │
│  FAILED (DO NOT RETRY)                                      │
│  ═════════════════════                                      │
│  [✗] Live 3-thread pipeline (growing MKV unreliable)        │
│  [✗] Continuous blob tracking on busy screens (noise)       │
│  [✗] CNN as replacement for motion detection (too slow)     │
└─────────────────────────────────────────────────────────────┘
```

---

## How It All Connects

The complete data flow from user command to Windows action:

```
User (Telegram)
    │
    ▼
Ship Computer (L1 Orchestrator)
    │
    ├── "Pair the K380 keyboard"
    │
    ▼
Navigation System
    │
    ├── Step 1: Open Settings
    │   ├── PERCEIVE: screenshot via HDMI capture
    │   ├── DECIDE: LLM says "click Settings icon at (1200, 600)"
    │   ├── ACT: ESP32 mouse moves + clicks
    │   │   └── Cursor Awareness confirms position BEFORE click
    │   └── VERIFY: new screenshot, LLM confirms Settings opened
    │
    ├── Step 2: Click Bluetooth
    │   └── (same perceive-decide-act-verify loop)
    │
    ├── Step 3: Click "Add device"
    │   └── If click misses → Hypothesis Recovery
    │       ├── cursor_drifted? (parallel agent checks)
    │       └── Three-tier recovery kicks in
    │
    └── Step N: Enter PIN on K380
        └── Pi Zero keyboard types PIN digits
    │
    ▼
Experience System
    │
    ├── OTel trace: "K380 Pairing" (span per step)
    ├── Narrative: what worked, what failed, what was learned
    └── Training data: confirmed cursor positions → CNN
```
