# Experience Validation: Human Feedback for Agent Reasoning

> **Status:** Plan (not implemented)
> **Created:** 2026-02-20
> **Context:** The system logs rich experience data — timestamped sequences of perception,
> action, and outcome — via `ExperienceLogger` and `StorylineLogger` with OTel traces.
> Passthrough sessions, Lissajous sweeps, and navigation steps are all recorded as
> experiences in `logs/experiences/`. But there is no way for a human to **evaluate**
> these experiences — to say "the system drew the right conclusion here" or "this action
> was correct but the reasoning was wrong." The existing `/validation` page handles
> cursor patch correctness (perception validation). This plan adds **reasoning validation**
> (the entire observe → conclude → decide → act → verify chain).

---

## Problem

The system has two feedback loops, one complete and one missing:

| Feedback Loop | Exists? | What it validates |
|---------------|---------|-------------------|
| Cursor patch validation (`/validation`) | **Yes** | Perception: "Is this a cursor?" — 64x64 patch → CNN label |
| Experience validation | **No** | Reasoning: "Did the system understand the situation? Draw the right conclusion? Take the right action? Head in the right direction?" |

Without experience validation:
- Agent mistakes repeat across sessions (no human signal on reasoning quality)
- Correct approaches can't be distinguished from lucky outcomes
- The system can't learn which *types* of reasoning fail (situational awareness vs action execution)
- The Star Trek Computer's core principle — *"once a good thing is created, it should not be forgotten"* — has no structured way to mark what IS good

### The Reasoning Chain

Each experience involves a chain of decisions. Each link can independently succeed or fail:

```
Perception          Conclusion           Decision            Action             Outcome
(What did the    →  (What did the     →  (What did the    →  (Was it         →  (Did the
 system see?)        system infer?)       system decide?)     executed well?)    goal succeed?)

Examples:
"Cursor at (500,   "This is the        "Click this        "Click sent to     "Bluetooth
 300), hand shape"  Bluetooth toggle"    element"           (502, 298)"        paired ✓"

Things that go wrong independently:
✓ saw cursor      ✗ wrong element     ✓ right decision   ✓ good execution   ✗ failed
✓ saw cursor      ✓ right element     ✗ wrong action     —                  ✗ failed
✓ saw cursor      ✓ right element     ✓ right decision   ✗ click missed     ✗ failed
✗ wrong position  —                   —                  —                  ✗ failed
```

## Design

### New page: `/experiences`

A table-based validation UI under the same navigation bar as `/validation`, listing
all recorded experiences with tagging columns. Similar layout to the cursor validation
page but operating on **experience sessions** rather than 64x64 patches.

### Data Sources

Experiences come from three sources, all under `logs/experiences/`:

| Source | Files | Description |
|--------|-------|-------------|
| Passthrough sessions | `passthrough/*.jsonl` + subdirs | Human-driven mouse/keyboard sessions (K380 pairing, etc.) |
| Standalone sessions | `*.jsonl` (root level) | Agent-driven (Lissajous sweep, cursor detection) |
| Storyline sessions | `storylines/*.jsonl` | Multi-track narrative with OTel spans |

The backend needs a new endpoint to **list and serve** these experiences.

### Tag Dimensions

Seven dimensions, each independently taggable. Designed around the reasoning chain:

#### 1. Situational Awareness (`awareness`)
*"Did the system correctly understand what was happening on screen?"*

| Value | Meaning |
|-------|---------|
| `correct` | System's understanding of the screen state was accurate |
| `partial` | Partially correct — got some elements right but missed others |
| `incorrect` | System misread the situation (wrong window, wrong element, wrong state) |
| `unknown` | Can't tell from the data whether awareness was correct |

#### 2. Conclusion Quality (`conclusion`)
*"Was the inference/conclusion the system drew from its observations valid?"*

| Value | Meaning |
|-------|---------|
| `correct` | The conclusion logically follows from the observations and is factually right |
| `incorrect` | The conclusion is wrong — doesn't match what the observations should have led to |
| `orthogonal` | The conclusion is unrelated to the actual situation — addressing the wrong thing entirely |
| `lucky` | The conclusion happened to be right, but the reasoning was unsound |

#### 3. Action Decision (`decision`)
*"Given what the system concluded, was the chosen action the right one?"*

| Value | Meaning |
|-------|---------|
| `correct` | The right action was chosen for the situation |
| `incorrect` | Wrong action — should have done something different |
| `premature` | Right action but taken too early (before sufficient information) |
| `excessive` | Overkill — a simpler action would have sufficed |

#### 4. Action Execution (`execution`)
*"Was the chosen action executed correctly at the hardware level?"*

| Value | Meaning |
|-------|---------|
| `correct` | Action was executed accurately (click hit target, movement was smooth) |
| `missed` | Action missed — click at wrong coordinates, movement overshot |
| `partial` | Partially executed — started correctly but didn't complete |
| `failed` | Execution failed at hardware level (BLE disconnect, serial timeout) |

#### 5. Direction (`direction`)
*"Is the overall trajectory — across multiple actions — heading toward the goal?"*

| Value | Meaning |
|-------|---------|
| `correct` | System is making progress toward the stated goal |
| `incorrect` | System is moving away from the goal or in a dead end |
| `stalled` | System is not making progress (repeating same actions, stuck in a loop) |
| `exploratory` | System is exploring, not clear yet if direction is correct (acceptable) |

#### 6. Direction Certainty (`certainty`)
*"How confident am I (the human reviewer) that my direction assessment is correct?"*

| Value | Meaning |
|-------|---------|
| `high` | I'm sure about my assessment of the direction |
| `medium` | Fairly confident but could be wrong |
| `low` | Uncertain — might need more context or data to judge |

#### 7. Overall Verdict (`verdict`)
*"Overall, was this experience valuable or problematic?"*

| Value | Meaning |
|-------|---------|
| `valuable` | This experience produced useful behavior or learning |
| `neutral` | Neither particularly good nor bad |
| `problematic` | This experience contains behavior that should be corrected |
| `exemplary` | This is a reference experience — future agents should learn from it |

### Tag Storage

Tags stored as a JSON file alongside each experience:

```
logs/experiences/passthrough/20260220_155709_k380_pairing/
├── screenshots/
├── storylines/
├── 2026-02-20_155709.jsonl          # Raw events
├── narrative.md                      # Generated narrative
└── validation.json                   # NEW — human tags
```

**`validation.json` schema:**
```json
{
  "session_id": "20260220_155709",
  "reviewed_at": "2026-02-20T16:30:00.000Z",
  "reviewer": "human",
  "tags": {
    "awareness": "correct",
    "conclusion": "incorrect",
    "decision": "correct",
    "execution": "correct",
    "direction": "correct",
    "certainty": "medium",
    "verdict": "problematic"
  },
  "notes": "Correctly saw the Bluetooth toggle but concluded it was already off when it was on."
}
```

## Implementation

### Backend: New API Endpoints

**File:** `src/web/kvm_dashboard.py`

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/experiences` | GET | List all experience sessions with metadata |
| `/api/experiences/{session_id}` | GET | Get single experience detail (events, narrative, tags) |
| `/api/experiences/{session_id}/events` | GET | Get raw JSONL events for an experience |
| `/api/experiences/{session_id}/tag` | POST | Save/update validation tags |
| `/api/experiences/{session_id}/narrative` | GET | Get generated narrative markdown |
| `/api/experiences/{session_id}/screenshots` | GET | List screenshots for a session |
| `/api/experiences/tag_schema` | GET | Return available tag dimensions and values |

**`GET /api/experiences` response:**
```json
{
  "sessions": [
    {
      "session_id": "20260220_155709",
      "label": "k380_pairing",
      "type": "passthrough",
      "started_at": "2026-02-20T15:57:09.776Z",
      "duration_s": 9.6,
      "event_count": 3,
      "screenshot_count": 0,
      "has_narrative": true,
      "has_storyline": true,
      "has_validation": false,
      "tags": null,
      "directory": "logs/experiences/passthrough/20260220_155709_k380_pairing"
    }
  ],
  "total": 7,
  "unreviewed": 5,
  "stats": {
    "by_type": {"passthrough": 5, "standalone": 2},
    "by_verdict": {"valuable": 1, "neutral": 0, "problematic": 1, "unreviewed": 5}
  }
}
```

The listing endpoint scans `logs/experiences/` for:
1. `passthrough/` subdirs (each `YYYYMMDD_HHMMSS_label/` is a session)
2. Root-level `.jsonl` files (standalone sessions)
3. Checks for `validation.json` existence in each session dir

### Frontend: New Page

**New files:**
- `src/web/static/pages/experiences.html` — HTML structure
- `src/web/static/css/experiences.css` — Page-specific styles
- `src/web/static/js/experiences.js` — Table logic, tag interaction, event viewer

**Page structure:**

```
┌─────────────────────────────────────────────────────────────┐
│ EXPERIENCE VALIDATION                                        │
│ [Dashboard] [Profiler] [Validation] [Loss Events] [YOLO]    │
│ [Concepts] [Models] [Experiences ← NEW, active]             │
├──────────────┬──────────────────────────────────────────────┤
│ Filters:     │ Stats: 7 total | 5 unreviewed | 1 valuable  │
│ [All] [Unrev]│ [1 problematic]                              │
│ [Passthrough]│                                              │
│ [Standalone] │                                              │
├──┬───────┬───┴──┬──────┬──────┬──────┬──────┬──────┬───────┤
│  │Session│Label │Dur.  │Events│Aware │Concl.│Decis.│ Dir.  │
│  │       │      │      │      │      │      │      │       │
│  │220155 │k380  │9.6s  │ 3    │ ✓    │  ✗   │  ✓   │ →     │
│  │ 42    │pair  │      │      │      │      │      │       │
│  │220155 │test  │3.1s  │ 1    │  —   │  —   │  —   │  —    │
│  │ 54    │      │      │      │      │      │      │       │
├──┴───────┴──────┴──────┴──────┴──────┴──────┴──────┴───────┤
│ DETAIL PANEL (shown when row selected)                       │
│ ┌──────────────────────┬───────────────────────────────────┐ │
│ │ Narrative            │ Tag Panel                         │ │
│ │ (markdown rendered)  │                                   │ │
│ │                      │ Awareness: [correct] [partial]    │ │
│ │ ## Main Storyline    │            [incorrect] [unknown]  │ │
│ │ - Starting K380 pair │                                   │ │
│ │                      │ Conclusion: [correct] [incorrect] │ │
│ │ ## Perception        │             [orthogonal] [lucky]  │ │
│ │ - Lissajous started  │                                   │ │
│ │ - Cursor not found   │ Decision: [correct] [incorrect]   │ │
│ │                      │           [premature] [excessive] │ │
│ │ ## Technical         │                                   │ │
│ │ - BLE reconnect      │ Execution: [correct] [missed]     │ │
│ │ - Heartbeat started  │            [partial] [failed]     │ │
│ │                      │                                   │ │
│ │                      │ Direction: [correct] [incorrect]  │ │
│ │                      │            [stalled] [exploratory]│ │
│ │                      │                                   │ │
│ │                      │ Certainty: [high] [medium] [low]  │ │
│ │                      │                                   │ │
│ │                      │ Verdict: [valuable] [neutral]     │ │
│ │                      │          [problematic] [exemplary]│ │
│ │                      │                                   │ │
│ │                      │ Notes: [________________] [Save]  │ │
│ └──────────────────────┴───────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Key UI behaviors:**
- Click a row to expand the detail panel below it
- Tag chips work like the existing validation page — click to set, click again to clear
- Tags save on click (POST to `/api/experiences/{id}/tag`)
- Narrative panel renders the markdown from `narrative_*.md` or `narrative.md`
- If screenshots exist, show a thumbnail strip in the detail panel
- Filter buttons: All / Unreviewed / Passthrough / Standalone / Problematic / Exemplary
- Sort by: Date (default) / Duration / Event count / Verdict

**Keyboard shortcuts** (following existing validation page pattern):
- `J/K` or arrows: navigate sessions
- `Enter`: expand/collapse detail panel
- `1-7`: quick-tag the 7 dimensions (cycles through values)
- `N`: focus notes field

### Navigation Integration

Add "Experiences" to the nav bar on ALL existing pages. In each page's HTML header:
```html
<a href="/experiences" class="nav-link">Experiences</a>
```

And in `kvm_dashboard.py`, add the route:
```python
@app.get("/experiences")
async def experiences_page():
    return _serve_page("experiences.html")
```

### Connection to Existing Systems

This validation page creates a **feedback signal** that completes the experience loop:

```
Experience recorded (ExperienceLogger + StorylineLogger)
    ↓
Human reviews on /experiences page
    ↓
Tags saved as validation.json alongside experience
    ↓
Future agents read validation.json to:
    - Skip approaches tagged "problematic"
    - Replicate approaches tagged "exemplary"
    - Understand what types of reasoning fail (conclusion vs execution)
    - Calibrate confidence (user's certainty tag)
```

This is the same feedback pattern as the cursor validation page:
```
CNN patch → human review on /validation → retrain CNN
Experience → human review on /experiences → inform future agent decisions
```

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/web/static/pages/experiences.html` | **New** | Experience validation page HTML |
| `src/web/static/css/experiences.css` | **New** | Page-specific styles |
| `src/web/static/js/experiences.js` | **New** | Table logic, tag interaction, detail viewer |
| `src/web/kvm_dashboard.py` | Edit | Add 7 API endpoints + page route + experience scanning logic |
| All existing page HTMLs | Edit | Add "Experiences" nav link |

## Estimated Effort

| Component | Lines |
|-----------|-------|
| Backend (API endpoints, experience scanner) | ~200 |
| HTML | ~100 |
| CSS | ~80 |
| JavaScript | ~300 |
| Nav link updates (6 pages) | ~12 |
| **Total** | ~690 |

## Verification

1. Navigate to `localhost:8766/experiences` — page loads with session list
2. Verify passthrough sessions appear with correct metadata (duration, event count)
3. Click a session row — detail panel expands with narrative + tag chips
4. Click tag chips — verify they toggle and save (check `validation.json` created)
5. Refresh page — verify tags persist
6. Filter by "Unreviewed" — verify only untagged sessions show
7. Check nav link appears on all existing pages (Dashboard, Profiler, Validation, etc.)

## Temporal Calibration — The Core Evidence Problem

### Why this matters

When the system evaluates its own actions as "pass" or "fail," the conclusion is
unreliable without millisecond-precise temporal evidence. The fundamental problem:

```
Did focus loss happen BEFORE the movement? → The movement caused focus loss (action error)
Did focus loss happen AFTER the movement?  → Focus was already lost (perception error)
These look identical unless you have sub-frame timing correlation.
```

The agent's timestamps say "I clicked at T=1234ms" and the dashboard says "blob disappeared
at T=1250ms" — but are those clocks synchronized? What's the latency between the ESP32
sending a HID report and the HDMI capture card showing the result?

### What the detail panel needs

The experience detail panel should include a **profiler-correlated event timeline** — not just
the narrative, but the raw evidence:

```
TIME (ms)     SOURCE          EVENT
─────────────────────────────────────────────────────
0.000         ESP32           move(dx=5, dy=0) sent
16.4          HDMI capture    frame decoded (blob at 502,300)
18.2          Silhouette      ROI match, confidence 0.82
33.1          HDMI capture    next frame (blob at 507,300)
35.0          CNN             inference: 0.91 confidence
46.8          Dashboard       state poll #1234 served
```

This timeline cross-references:
1. **Action timestamps** from ExperienceLogger (when the agent commanded something)
2. **Pipeline profiler data** from the daemon loop (when each frame was processed)
3. **Dashboard state polls** (what the frontend was seeing at each moment)
4. **System clock on Windows** (visible in bottom-right of HDMI capture — the human
   ground truth for "what time was it on the target machine?")

### Clock synchronization

The system has multiple clocks that may drift:

| Clock | Source | Precision | Known Issues |
|-------|--------|-----------|-------------|
| Desktop `CLOCK_REALTIME` | Linux system time | ~1ms | NTP synced |
| Dashboard UTC (`utc_now_ms()`) | Same Linux clock | ~1ms | Same as above |
| C daemon timestamps | `clock_gettime(CLOCK_REALTIME)` | ~1μs | Same clock |
| ESP32 serial | No clock — latency from USB serial | ~2-5ms | USB serial buffer delay |
| Windows system clock | Visible on HDMI capture | ~1s resolution (taskbar) | May drift from Linux clock |
| HDMI capture card | V4L2 frame timestamp | ~33ms (30fps) | Frame arrival, not scene time |

**Key insight:** The Windows taskbar clock visible in the HDMI feed is the one ground-truth
timestamp the human can verify visually. By OCR'ing the taskbar time and comparing it to the
Linux timestamp when that frame was captured, we can establish the **cross-machine clock offset**.

### Integration with existing profiler

The pipeline profiler (`/profiler` page) already measures per-stage latency. The experience
validation timeline should incorporate profiler samples that overlap with the experience's
time window:

```python
# In the GET /api/experiences/{id}/timeline endpoint:
# 1. Load experience events (ExperienceLogger JSONL)
# 2. Load profiler samples within the experience time window
# 3. Load daemon blob data within the time window (if blob JSONL logging exists)
# 4. Merge all three into a unified timeline sorted by timestamp
# 5. Return to frontend for rendering
```

This gives the human reviewer **the actual evidence** to judge whether a "pass" or "fail"
evaluation was correct — not just the agent's narrative, but the raw sensor data.

## Open Questions

- **Event timeline viewer:** Should the detail panel show a scrollable event timeline (like a mini-log viewer) in addition to the narrative? This would show the raw JSONL events with timestamps. More useful for debugging but adds complexity. **Update: Yes — the profiler-correlated timeline described above IS this timeline. It's essential for validating pass/fail conclusions.**
- **Screenshot viewer:** When screenshots exist, should they be shown inline in the detail panel or as a separate lightbox/gallery? Inline thumbnails are simpler.
- **Span-level tagging:** Should individual OTel spans within an experience be independently taggable, or is session-level tagging sufficient for now? Session-level is simpler and probably sufficient for initial version.
- **Export format:** Should validated experiences be exportable as a training dataset (e.g., JSONL of session + tags) for fine-tuning agent behavior? Useful but not MVP.
- **Windows clock OCR:** Should the dashboard automatically OCR the Windows taskbar clock from HDMI frames to compute cross-machine clock offset? This would enable fully automatic temporal calibration but adds an OCR dependency.
- **Profiler sample retention:** Currently profiler samples are only collected during explicit 3-minute profiling runs. Should the daemon continuously log per-frame timing so that ANY experience can be retroactively correlated with pipeline data?
