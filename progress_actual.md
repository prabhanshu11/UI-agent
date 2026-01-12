# Internal Progress Tracking (Private)

## Future Plans
- **Optimization**: Rewrite performance-critical components in C++/C/Java once architecture stabilizes.
- **Perception**: 
    - Implement "Stream of Consciousness" log (unified inputs + visuals).
    - Add Audio capture.
    - Add "Log for logs" (meta-retention policy).
- **Architecture**: `spin_once` to eventually support parallel task spawning.

## Internal Notes
- **State of play**: Core Controller verified. Moving to Perception.
- **Perception Strategy**: 
    - Use `mss` for screenshots (fast).
    - Use input hooks to detect "Active" vs "Inactive" states to save storage (don't screenshot 8 hours of sleep).
    - "Key Moments" (Alt-Tab, Activity Resume) should trigger immediate captures.
- **Watch out for**:
    - `keyboard` / `mouse` libraries on Linux often need root. Might need `pynput` or explicit sudo handling.
    - Wayland compatibility for screenshots/inputs.

## Hardware Integration Module (2026-01-12)

### USB HDMI Capture Card - Pibox/MacroSilicon MS2109

**Device Info:**
- **USB ID**: 534d:2109 (MacroSilicon USB3.0 capture)
- **Driver**: uvcvideo (works out of box on Linux)
- **Video Device**: `/dev/video4` (and `/dev/video5` metadata)
- **Max Resolution**: 1920x1080 @ 60fps (MJPEG format)
- **Power**: 500mA (bus-powered)

**Critical Fix - USB Autosuspend:**
- **Issue**: Device triggers USB re-enumeration when accessed, causing target screen to flicker/disconnect repeatedly
- **Solution**: Disable autosuspend for this device:
  ```bash
  echo "on" | sudo tee /sys/bus/usb/devices/1-3/power/control
  ```
  (Replace `1-3` with actual bus-port from `lsusb -t`)

**Working Capture Command:**
```bash
ffmpeg -f v4l2 -input_format mjpeg -video_size 1920x1080 -i /dev/video4 -update 1 -frames:v 1 output.png -y
```

**Signal Detection Logic (2026-01-12 23:12):**

After testing, discovered reliable heuristic for detecting actual screen content:

| Capture | Time | File Size | Status |
|---------|------|-----------|--------|
| #1 | 23:12:01 | 8KB | Black screen (no signal) |
| #2 | 23:12:03 | 8KB | Black screen (no signal) |
| #3 | 23:12:06 | 360KB | ✓ Windows 11 Settings screen |

**Detection Rule:**
- **File size < 20KB**: Solid color (green/black test pattern) = No HDMI signal
- **File size > 20KB**: Actual screen content detected
- **Why**: Solid colors compress to ~8KB PNG, real screens with UI elements are 100KB+

**First Successful Capture:**
- Windows 11 Settings app (System > Display)
- User: Prabhaashu Rajput
- Resolution: 1920x1080, dark theme
- Image quality: Perfect, crisp, full resolution
- Timestamp: 12:52 PM, 1/1/2025 (on Windows laptop)

**Monitoring Script:**
Created `/tmp/monitor_capture.sh` that polls capture card every 2 seconds and auto-detects when proper screen appears using file size heuristic.

**Known Issues from Community:**
1. Audio driver conflict - Linux treats it as USB audio device (needs udev rule)
2. Some manufacturers remove internal radiators to cut costs (reliability issues)
3. Requires kernel 5.8+ for proper stereo audio capture

**References:**
- [Making MS2109 work on Linux](https://www.mjt.me.uk/posts/fixing-missing-macrosilicon-ms2109/)
- [Optimizing USB HDMI grabber](https://ciko.io/posts/cheap_usb_hdmi/)
- [PiKVM Issue #1195](https://github.com/pikvm/pikvm/issues/1195)

**Next Steps:**
- Add signal detection to USBCaptureCard class
- Implement continuous monitoring mode
- Add callback for "signal acquired" event
- Integrate with perception system for multi-screen capture

---

## Datalake Integration (Future Migration)

**Status**: datalake is ready but not yet deployed. All data currently stored locally will migrate to datalake.

**Current Data Storage:**
- Screenshots: In-memory/temp files during execution
- USB HDMI captures: `/tmp/captures/`
- Exploration work: `explorations/` directory

**Migration to Datalake:**
When datalake is deployed, UI-Agent will:
1. **Log all screenshots** to datalake with metadata (timestamp, source, action type)
2. **Log UI actions** (clicks, typing, navigation) with associated screenshots
3. **Log USB capture metadata** (device info, signal detection results, frame rate)
4. **Enable queries** like:
   - "Show me all screenshots from yesterday"
   - "What was I doing on Amazon at 3pm?"
   - "Show me all Windows laptop captures from this week"

**Integration Pattern:**
```python
from datalake import DataLake

dl = DataLake()

# Log UI action with screenshot
dl.log_activity(
    type="ui_action",
    source="ui-agent",
    data={
        "action": "click",
        "target": "submit_button",
        "platform": "web",
        "url": "https://example.com"
    },
    screenshot=screenshot_bytes,
    timestamp=datetime.now()
)

# Query history
screenshots = dl.query(
    "screenshots captured during Amazon checkout",
    time_range=("2026-01-12", "2026-01-13"),
    source="ui-agent"
)
```

**Timeline**: Will implement after datalake deployment and client library is ready.
