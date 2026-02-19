# Cursor Detection Fix Plan — Active Experience

> Read this at session start. Status updated after each commit.
> **This is an "experience"** — development using Claude Vision as ground truth.
> Any development work that uses Claude Vision for validation is an experience.

## Problem Statement

The cursor tracker loses position and locks onto UI animations (world clock
milliseconds, tickers) instead of the actual cursor. The tracker reports
positions like (1441, 209) when the cursor is visibly at (490, 465).

## Root Causes Identified

### 1. Persistent animation blobs — FIXED (commit pending)
World clock milliseconds change every frame → motion blobs at same position →
algorithm picks them as cursor when real cursor is stationary.
**Fix:** Noise grid (48x27, 40px cells) tracks persistent motion regions.
Starts FULL (presume noise), decays for quiet regions. Motion blobs in
persistent regions are excluded from cursor scoring.

### 2. Silhouette tracker false-matches — FIXED (commit pending)
The silhouette tracker does template matching in a 200px ROI. When position is
wrong, it template-matches against UI elements (icons, text, toolbar) and
"confirms" wrong position with 0.997 confidence.
**Fix:** CNN confidence gate — silhouette template results only update position
when `cursor_validated=True` OR `method=motion_roi`. Template-only results
are NEVER trusted without CNN backing. Removed the confidence>0.7 bypass
because template matching can be 99.7% confident on wrong elements.

### 3. CNN false positives — DISCOVERED, NOT YET FIXED
**Critical finding:** CNN (TinyCursorNet, 14K params) reports 90%+ confidence
on toolbar areas that are NOT the cursor. When CNN validates a wrong position,
the entire recovery chain breaks:
- _low_confidence_since never gets set
- Jitter re-acquisition never fires
- Silhouette tracker is trusted (CNN validated)
- Everything converges on wrong position
**Needs:** Better training data, larger model, or Claude Vision cross-check.

### 4. Jitter re-acquisition timing bug — FIXED (commit pending)
The validation loop reset `_low_confidence_since` on every re-acquisition
"success" (even wrong positions). The clock never accumulated.
**Fix:** Only CNN > 0.7 resets the clock. Re-acquisition attempts update
position but don't reset the low-confidence timer. Threshold = jitter
interval (30s) so the jitter fires after one full cycle of low confidence.

### 5. Green dot escapes video area — FIXED (commit pending)
JS percentages weren't clamped. Browser resize could push dot outside.
**Fix:** Math.max(0, Math.min(100, pct)) + overflow:hidden on video-area.

## Completed This Session

- [x] Noise grid for persistent blob filtering (inverted start: presume noise)
- [x] jitter_loop() replaces ESP32Mouse anti-sleep (unified control)
- [x] _low_confidence_since tracking (only CNN>0.7 resets)
- [x] Silhouette CNN gate (template-only = untrusted without CNN)
- [x] Green dot clamping + overflow:hidden
- [x] Claude Vision panel on dashboard (position, confidence, delta, age)
- [x] Silhouette "Trusted" indicator on dashboard
- [x] Jitter re-acquisition state on dashboard (pending, countdown)
- [x] CURSOR_FIX_PLAN.md (this file) for context survival

## Still TODO

### Immediate
- [ ] Fix CNN false positives — either retrain or add Claude Vision periodic check
- [ ] Periodic Claude Vision validation (every 2-3 min) as ground truth
- [ ] Log sensor stack + user observations to datalake
- [ ] Claude Code skill for quick datalake entries about cursor position

### Medium Term
- [ ] Phase 1 visual detection (pure, no mouse needed) — see STAR_TREK_COMPUTER_RESEARCH.md
- [ ] Larger CNN model or multi-head attention for better discrimination
- [ ] Noise grid visualization on overlay (show excluded regions)

## Architecture After This Session

```
Motion Blobs (20Hz) ──→ Noise Grid Filter ──→ Velocity Scoring ──→ position
                                                                      │
Silhouette Tracker (5Hz) ──→ CNN Gate ──→ position (only if validated) │
                                                                      │
CNN Validation (30s) ──→ confidence > 0.7 → VALIDATED ────────────────┤
                     └──→ confidence < 0.7 → _low_confidence_since    │
                                              │                       │
Jitter Loop (30s) ──→ if low_conf > 30s ──→ dual-probe reacquire ────┤
                  └──→ else ±3px anti-sleep                           │
                                                                      │
Claude Vision (manual/periodic) ──→ GROUND TRUTH ─────────────────────┘
```

## File to Edit
Only `src/web/kvm_dashboard.py` — monolith, all in one file.

## Key Code Locations (line numbers shift after edits)
- `_daemon_motion_loop()` — motion blob tracking, noise grid, blob scoring
- `cursor_validation_loop()` — CNN validation, _low_confidence_since tracking
- `jitter_loop()` — anti-sleep + opportunistic re-acquisition
- `_jitter_reacquire()` — dual-direction probe (15px RIGHT then DOWN)
- Silhouette tracking in daemon loop: sil_trusted gate
- `/api/claude_detect` — Claude Vision ground truth + state update
- `/api/state` — includes vision{} block for dashboard
- DASHBOARD_HTML / JS: updateCursorDot(), vision panel, trust indicators

## Git Branch
`feature/asmp-cursor-detect`

## Validation Commands
```bash
# Compare tracker vs Claude Vision
curl -s http://localhost:8766/api/claude_detect | python3 -m json.tool
curl -s http://localhost:8766/api/state | python3 -c "
import sys, json; d=json.load(sys.stdin)
print('Tracker:', d['cursor']['x'], d['cursor']['y'])
print('Vision:', d['vision']['x'], d['vision']['y'], d['vision']['confidence'])
"

# Monitor jitter re-acquisition
for i in $(seq 1 10); do
  curl -s http://localhost:8766/api/state | python3 -c "
import sys,json; d=json.load(sys.stdin); j=d['jitter']
print(f\"low_conf={j['low_confidence_s']:.0f}s reacq={j['reacquire_pending']}\")
"
  sleep 5
done
```

## Experience Policy
**Any development work that uses Claude Vision (the /api/claude_detect endpoint
or Claude Agent SDK) to validate cursor detection changes is classified as an
"experience".** Experiences should:
1. Record before/after tracker positions
2. Record Claude Vision ground truth at each step
3. Document what the CNN/silhouette/jitter said vs reality
4. Log to datalake for historical analysis
