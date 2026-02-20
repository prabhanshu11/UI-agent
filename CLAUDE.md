# UI-Agent Project — Agent Instructions

> **READ THIS BEFORE WRITING ANY CODE.** This file is BINDING, not informational.
> Also read: `docs/STAR_TREK_COMPUTER_RESEARCH.md` and `docs/STAR_TREK_COMPUTER_CONCEPTS.md`

## Known False Diagnosis — DO NOT ASSUME THIS

**"There are multiple blobs (blinking cursors, notification badges, typing indicators, chat animations) — the tracker picked up one of those instead of the actual cursor displacement."**

This is almost NEVER the actual reason cursor detection fails. If you find yourself writing this explanation, **STOP and ask the user** what they think about this "catch-all" reasoning. It's a pattern where agents blame UI noise instead of investigating the real root cause (usually: wrong code, wrong offsets, broken pipeline, stale data, or incorrect assumptions about what's working).

## Cursor Detection Philosophy — THE MODEL

There must ALWAYS be a fallback that does NOT rely on having control of the mouse or keyboard to locate the cursor. This is the Star Trek Computer model.

### Phase 1: Slow but Reliable Visual Detection (PRIORITY)

A pure visual algorithm that finds the cursor WITHOUT needing mouse/keyboard access.

- **Can be slow** — seconds, not milliseconds
- **Can be computationally expensive** — full-screen scan is fine
- **MUST be reliable** — on a still screen, it finds the cursor
- Uses template matching, CNN, silhouette library of known cursor shapes
- This is the FOUNDATION. Everything else builds on top of it.
- This does NOT exist yet. **Building this is the current priority.**

### Phase 2: Fast Silhouette Tracking (after Phase 1 works)

Once the cursor is located by Phase 1, use a silhouette-following motion detector to STICK with it.

- Measured in frequency (Hz) — how fast can we re-confirm position
- Follows the cursor's visual silhouette as it moves
- Uses motion detection (frame differencing) optimized for the cursor's known shape
- This is the "discover + stick" pattern: Phase 1 discovers, Phase 2 sticks

### Phase 3: Lissajous Curve — Fastest Brute-Force Guaranteed Finder

The Lissajous curve is the **fastest brute-force way** to establish cursor location when we have mouse control. It is NOT the primary detection method (that's Phase 1), but it IS the guaranteed fallback when mouse is available.

1. **Brute-force finder** — sweeps 5x the area of two 1080p monitors (amp_x=4300, amp_y=1200). Mathematically guaranteed to visit every region of the virtual desktop. Works even when:
   - Cursor is on another monitor in a multi-display setup
   - Cursor is hidden behind a window or off-screen
   - Visual detection can't find it (unusual cursor shape, overlay, etc.)
2. **Keeps things alive** — anti-sleep jitter (±3px every 30s), heartbeat, prevents Windows from sleeping, keeps BLE connection alive
3. **Quick re-lock** — when silhouette tracker (Phase 2) loses cursor:
   - Command known Lissajous movement → detect matching pattern
   - Match observed trajectory to expected trajectory → instant re-lock
4. **Works in tandem** with Phase 1 and Phase 2:
   - Phase 1 visual scan is tried first (no mouse needed)
   - If visual scan fails → Lissajous sweep (needs mouse, GUARANTEED)

### The Complete Model

```
FALLBACK (always available, no hardware needed):
  Phase 1: Visual scan → find cursor on still screen → slow but reliable

FAST PATH (when mouse is available):
  Phase 2: Silhouette tracker → stick with cursor → measured in Hz

RECOVERY CHAIN:
  Cursor lost?
    → Phase 1 visual scan (no mouse needed, 2-5s)
    → Lissajous sweep (needs mouse, 5s, GUARANTEED)
        Works even when:
        - Cursor is on another monitor
        - Cursor is hidden or off-screen
        - Multi-display with unknown geometry
        - Visual scan failed for any reason
```

## Key Binding Rules

1. **"The mouse pointer should not be affected by image recognition at any point"** — Motion detection is the base layer for tracking. CNN/visual detection only VALIDATES or DISCOVERS. It never MOVES the cursor.

2. **"If the position is not known, only the locator utility needs to be used, and not random motions"** — No blind clicking. Find the cursor first.

3. **"Once a good thing is created, it should not be forgotten"** — Proven working code paths must be preserved. Check git log before rewriting.

4. **Read the research doc** (`docs/STAR_TREK_COMPUTER_RESEARCH.md`) before making changes. The solution to your problem is probably already documented there.

## Git Discipline

```bash
# ALWAYS do this before writing code:
git log --oneline -20   # What's been tried?
git diff HEAD --stat     # What's uncommitted?
git status               # What's the current state?
```

Check the "Proven vs Untested" table in the research doc (§11) before choosing an approach.

## Disk Usage Warning (2026-02-20)

**cursor-daemon records 720p@30fps H.264 MKV chunks indefinitely with no cleanup.**

| Source | Location | Rate | Cleanup |
|--------|----------|------|---------|
| KVM recordings | `data/recordings/kvm_*.mkv` | 3-20 GB/day | **None** |
| Blob logs (planned) | `data/blob_log/blobs_*.jsonl` | 0.1-1 GB/day | **None** |
| Loss events | `data/loss_events/*/` | 0-9 GB/day | **None** |

- **Free space (2026-02-20):** 247 GB of 446 GB
- **Estimated disk full:** ~March 12, 2026 (~20 days at current idle-screen rate)
- **Heavy use could accelerate to ~8 days**

**Plans:**
- `docs/RECORDING_LIFECYCLE_PLAN.md` — datalake sync, retention engine, dashboard storage tab, archive to external drive. Not yet implemented.
- `docs/BLOB_DATASET_PLAN.md` — Per-frame blob JSONL logging in cursor-daemon (C-side, background thread, lock-free ring buffer). Not yet implemented.
- `docs/DASHBOARD_SUBPANELS_PLAN.md` — 3 sub-panels below HDMI feed: motion blob map, model inference status with live staleness counters, mouse trail visualization (ESP32 + detected + passthrough). Not yet implemented.
- `docs/EXPERIENCE_VALIDATION_PLAN.md` — `/experiences` page for human review of agent reasoning chain: situational awareness, conclusion quality, action decision, execution, direction, certainty, verdict. Tags stored as `validation.json` alongside experience data. Claude Code skill (`/experience-feedback`) queries these tags for live agent feedback. Not yet implemented.

**Stopgap if urgent:**
```bash
find ~/Programs/UI-agent/data/recordings/ -name "kvm_*.mkv" -mtime +7 -delete
find ~/Programs/UI-agent/data/loss_events/ -maxdepth 1 -type d -mtime +3 -exec rm -rf {} +
```
