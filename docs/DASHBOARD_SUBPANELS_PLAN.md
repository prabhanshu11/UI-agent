# Dashboard Sub-Panels: Blob Map, Model Status, Mouse Trails

> **Status:** Plan (not implemented)
> **Created:** 2026-02-20
> **Context:** The KVM dashboard (`localhost:8766`) currently displays a single HDMI feed canvas
> (left ~70%) and a sidebar (right 320px) with text-based status panels. The sidebar shows blob
> data as a table, model results as individual panels, and has no mouse trail visualization.
> There is no at-a-glance visual summary of motion activity, model health, or mouse input.

---

## Problem

The dashboard's information density is low for the feed area — it's a single video canvas with
all structured data crammed into a scrollable sidebar. Key operational insights require scrolling:

| Information | Currently Available? | Location |
|-------------|---------------------|----------|
| Live HDMI feed | Yes | Feed canvas (left column) |
| Motion blob bounding boxes | Overlays on feed + sidebar table | Feed overlay + sidebar |
| All model inference results + staleness | Scattered across 4+ sidebar panels | Sidebar (scroll required) |
| ESP32 dead-reckoned mouse position | Not exposed in UI | Only in Python memory (`mouse.estimated_x/y`) |
| Mouse passthrough trail | Not visualized | Deltas sent via POST but not tracked |
| Detected vs commanded cursor divergence | Not visualized | Would need ESP32 + detection side-by-side |

Operators need a quick visual scan: "Is motion being detected? Are models running? Where does
the system think the cursor is vs where it actually is?" — without scrolling the sidebar.

## Design

### Layout

Three equal-width panels in a row below the HDMI feed, within the left column. Sidebar spans
both rows unchanged.

```
┌──────────────────────────┬──────────┐
│                          │ Sidebar  │
│    HDMI Feed (main)      │ (scroll) │
│                          │          │
├─────────┬────────┬───────┤          │
│ Panel 1 │Panel 2 │Panel 3│          │
│ Blobs   │Models  │Mouse  │          │
│ Map     │Status  │Trails │          │
└─────────┴────────┴───────┴──────────┘
```

**Panel height:** ~180px fixed (`min-height: 150px; max-height: 200px`).

### Panel 1: Motion Blob Map

A minimap canvas rendering all detected motion blobs at reduced scale.

- **Data source:** `state.blobs[]` from existing `/api/state` (already available, 5Hz poll)
- **Canvas:** Proportional to 1920×1080, fits within sub-panel
- **Drawing:**
  - Screen outline (dark border)
  - Blob bounding boxes: green (primary blob #0), blue (others)
  - Centroid dots: filled circles at blob center of mass
  - "No motion" text when `blob_count == 0`
- **Update rate:** 5Hz (synced to state poll)

This gives operators an instant spatial read of where motion is occurring on the captured
screen, without needing to parse the sidebar blob table.

### Panel 2: Model Inference Status

Compact DOM-based display of all 4 detection models with live staleness counters.

| Model | Data Source | Fields Shown |
|-------|-----------|-------------|
| CNN | `state.cnn` | confidence %, inference_ms, age |
| YOLO | `state.yolo` | confidence %, inference_ms, age_s |
| Silhouette | `state.silhouette` | confidence %, latency_ms, active/Hz |
| Vision (Claude) | `state.vision` | confidence level, latency, age_s |

**Key feature:** "Xs ago" counters that tick every 1 second via `setInterval`, independent of
the 5Hz state poll. This gives a real-time sense of model freshness:
- Green: <5s ago
- Yellow: <30s ago
- Red: >30s ago (or never)

Confidence values are color-coded: green (>70%), yellow (>30%), red (<30%).

### Panel 3: Mouse Trail Visualization

Canvas showing raw movement trails for multiple input sources.

**Trails (ring buffer of last 100 samples):**

| Trail | Color | Source | Description |
|-------|-------|--------|-------------|
| ESP32 | Green (#2ecc71) | `state.esp32.x/y` (new) | Dead-reckoned position — what the ESP32 thinks the cursor is on Windows |
| Detected | Cyan (#00e5ff) | `state.cursor.x/y` | Motion-detected cursor position — where the system actually sees it |
| Passthrough | Orange (#ff6b35) | Frontend delta accumulation | Cumulative position from BT mouse passthrough input |

**Drawing:**
- Trail segments with fading opacity (oldest=0.1, newest=1.0)
- Current position as a solid dot (3px radius)
- Small legend in bottom-left corner
- Canvas proportional to 1920×1080

**Passthrough tracking:** When passthrough mode activates, the cumulative position initializes
from the current ESP32 position. Each `flushPassthroughBatch()` call (50Hz in the frontend)
adds the delta to the accumulator. This shows the raw user input independently from what the
detection pipeline reports.

## Implementation

### Backend changes

**File:** `src/web/kvm_dashboard.py`

Add two fields to `/api/state` response (at ~line 2864 in the return dict):

```python
"esp32": {
    "x": mouse.estimated_x if mouse else 0,
    "y": mouse.estimated_y if mouse else 0,
},
"passthrough_active": _passthrough_active,
```

Both values already exist as module-level variables. `mouse.estimated_x/y` is read at lines
793-794. `_passthrough_active` is a boolean at line 346. No new imports, no locking needed
(CPython GIL makes boolean reads atomic).

### Frontend changes

**Files to modify:**

| File | Changes |
|------|---------|
| `src/web/static/css/dashboard.css` | Grid restructure (add `grid-template-rows: 1fr auto` to `.main`, `grid-row: 1 / -1` to `.sidebar`), reduce canvas max-height by 180px, add `.sub-panels`, `.sub-panel`, `.sub-panel-title`, `.model-row` classes |
| `src/web/static/pages/dashboard.html` | Add `div.sub-panels` between `.feed` and `.sidebar` with 3 panel children (2 canvases + 1 DOM list), add `<script>` for `sub-panels.js` |
| `src/web/static/js/dashboard.js` | 3 hook points: `SubPanels.init()` in `initDashboard()`, `SubPanels.update(s)` in `updateState()`, `SubPanels.recordPassthroughDelta(dx, dy)` in `flushPassthroughBatch()` |

**New file:** `src/web/static/js/sub-panels.js`

IIFE module (`var SubPanels = (function() { ... })();`) following the project's existing
pattern (`CanvasFeed`, `CanvasOverlays`, `Recorder`). Approximately 220 lines containing:

1. **Canvas setup and resize** — `getBoundingClientRect()` for responsive sizing
2. **`drawBlobMap(state)`** — scales blob bbox/centroids from 1920×1080 to canvas dimensions
3. **`updateModelStatus(state)`** — DOM updates for 4 model rows
4. **`tickModelAges()`** — 1-second interval for "Xs ago" counters
5. **Ring buffer helpers** — `pushTrail(trail, x, y)` with 100-entry limit
6. **`updateTrails(state)`** — accumulates ESP32 + detected positions, manages passthrough base
7. **`drawTrails()`** — renders 3 trails with fading opacity + legend
8. **`recordPassthroughDelta(dx, dy)`** — called from dashboard.js at 50Hz

Public API: `init()`, `update(state)`, `recordPassthroughDelta(dx, dy)`.

### CSS grid layout detail

Current:
```css
.main {
    display: grid;
    grid-template-columns: 1fr 320px;
    height: calc(100vh - 50px);
}
```

New:
```css
.main {
    display: grid;
    grid-template-columns: 1fr 320px;
    grid-template-rows: 1fr auto;
    height: calc(100vh - 50px);
}

.sidebar {
    grid-row: 1 / -1;
    grid-column: 2;
}

.sub-panels {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1px;
    background: #1a1a25;
    border-top: 1px solid #1a1a25;
    height: 180px;
}

#canvas-container canvas {
    max-height: calc(100vh - 50px - 180px);
}
```

## Estimated effort

~310 lines of new/changed code across 5 files:

| File | Lines |
|------|-------|
| `kvm_dashboard.py` | +6 |
| `dashboard.css` | +40, ~5 modified |
| `dashboard.html` | +35 |
| `sub-panels.js` (new) | ~220 |
| `dashboard.js` | +9 (3 hooks) |

## Verification

1. `curl localhost:8766/api/state | python3 -m json.tool` — verify `esp32` and `passthrough_active` fields present
2. Open `localhost:8766` in Firefox — 3 panels visible below feed
3. Move something on the captured screen — blob map shows green/blue rectangles
4. Model status rows show "Xs ago" counters that tick live
5. Mouse trail canvas shows trail when cursor is being tracked
6. Sidebar scrolls independently, feed aspect ratio preserved

## Open questions

- **Sub-panel height:** 180px is a guess for a "small Firefox window." May need tuning — could expose a CSS variable or make panels collapsible.
- **Trail decay:** Currently time-based (ring buffer of 100 samples at 5Hz = 20s). Could alternatively fade by distance traveled for more interesting visualization.
- **Passthrough trail accuracy:** The 50Hz delta accumulation drifts from true position over time (no correction). Should it periodically resync from ESP32 dead-reckoned position?
