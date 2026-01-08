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
