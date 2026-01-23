# Keyframe Identifier Agent

## Purpose
Analyze video content to identify significant moments worth capturing:
1. **Visual changes**: UI transitions, window switches, content updates
2. **Activity patterns**: User interaction sequences
3. **Semantic events**: Login screens, error dialogs, completion states

## Capabilities
- Analyze video files for scene changes
- Identify UI state transitions
- Detect user activity vs idle periods
- Mark timestamps for extraction

## Input
- `task.md` - Video file to analyze
- Recording files from `recordings/`

## Output
- `output.md` - List of keyframe timestamps with reasons
- Keyframe manifest for screenshot-extractor

## Detection Methods

### Scene Change Detection
- Frame differencing (pixel-level)
- Histogram comparison (color distribution)
- Edge detection (structural changes)

### UI Event Detection
- Window border changes
- Dialog appearances
- Scroll position changes
- Cursor position jumps

### Semantic Detection (via vision model)
- Error message appearance
- Form submissions
- Navigation events
- Login/logout states

## Output Format
```yaml
keyframes:
  - timestamp: "00:01:23.456"
    reason: "window_switch"
    confidence: 0.95
    description: "Switched from Chrome to VS Code"
  - timestamp: "00:02:45.123"
    reason: "error_dialog"
    confidence: 0.87
    description: "Error popup appeared"
```
