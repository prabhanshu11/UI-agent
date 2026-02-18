# Mouse Control Learnings

Collected from ESP32 BT HID mouse sessions (Jan–Feb 2026).

## Multi-Monitor Gotcha

### Problem

`ESP32Mouse.calibrate_to_corner()` sends a large negative relative movement
(e.g. -2420, -2420) to park the cursor at the screen edge. On a single-monitor
setup, the cursor reliably lands at (0, 0) of that display.

On multi-monitor Windows, the cursor lands at the **virtual desktop** edge
instead — typically the top-left corner of the **leftmost** monitor. If the
HDMI capture card captures the laptop's built-in display (which may be the
**right** monitor in Windows display arrangement), the cursor is invisible
on the captured screen.

### Root Cause

HID relative mice have no concept of individual monitors. Windows moves the
cursor within the virtual desktop coordinate space:

```
Virtual Desktop (two 1920x1080 monitors, side-by-side):

  (0,0)          (1920,0)          (3840,0)
    ┌──────────────┬──────────────┐
    │  Monitor 1   │  Monitor 2   │
    │  (external)  │  (laptop)    │  ← HDMI captures this one
    │              │              │
    └──────────────┴──────────────┘
  (0,1080)      (1920,1080)     (3840,1080)
```

`calibrate_to_corner()` moves to (0, 0) = top-left of Monitor 1.
The cursor is off-screen relative to HDMI capture of Monitor 2.

### Solution: Frame-Diff Sweep (cursor_locator)

Instead of assuming a fixed screen offset, sweep the cursor across the
virtual desktop and use HDMI frame subtraction to detect when the cursor
enters the captured screen:

1. Park cursor at far-left of virtual desktop (-5000, -3000)
2. Move right in 200px increments
3. After each move, capture two HDMI frames around a small known movement
4. Subtract frames — changed pixels indicate cursor presence
5. When cursor detected, record the X offset and set_position() accordingly

See: `src/hardware/cursor_locator.py`

### Finding the Cursor When Position Is Unknown

> **The locator utility (`cursor_locator.py`) is the ONLY method an agent should use
> to find the cursor when its position is unknown. Never use random motions, blind
> clicks, or guessed positions.** See `CURSOR_DETECTION_KNOWLEDGE.md` Section 0.

Call `CursorLocator.locate()` — it runs a Lissajous sweep that covers any
multi-monitor layout in ~5 seconds and returns a verified cursor position.

### Manual User Workarounds (NOT for agent use)

The following are fallbacks for **manual user intervention only** — an automated
agent must never attempt these:

- **User moves cursor manually**: User physically moves cursor to HDMI-captured
  screen, then calls `mouse.set_position(x, y)` based on observed location
- **Disable second monitor**: User navigates to Display Settings → "PC screen only"
  (requires blind navigation if cursor is on wrong screen)
- **Known offset**: If monitor arrangement is known, user can add offset to all
  coordinates (e.g. `x + 1920` for right-side laptop screen)

## Relative Mouse Position Drift

Over many movements, the estimated position drifts from actual cursor
position. Causes:
- Sub-pixel rounding (HID reports are integers, real movement has fractions)
- Boundary clamping by OS (Windows stops cursor at screen edge)
- Acceleration curves (Windows mouse acceleration changes effective dx/dy)

**Mitigation**: Periodically re-calibrate using frame-diff or corner calibration.
For critical operations (clicking UI elements), use verify-before-click protocol.

## HDMI Capture Timing

The capture card (`/dev/video0`, YUYV format) needs 2-3 frames to stabilize.
Using `ffmpeg -frames:v 3 -update 1` ensures we get the last (most stable)
frame. Single-frame captures often contain transitional artifacts.

Wait at least 300ms after mouse movement before capturing for verification.
