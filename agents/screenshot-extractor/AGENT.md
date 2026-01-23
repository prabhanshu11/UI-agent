# Screenshot Extractor Agent

## Purpose
Extract keyframes from video recordings based on strategies:
1. **Interval-based**: Every N seconds
2. **Change-detection**: When frame differs significantly from previous
3. **Event-based**: On specific triggers (mouse click, window change)

## Capabilities
- Read video files from recordings directory
- Extract frames at specified timestamps
- Detect scene changes using image differencing
- Save frames as PNG with metadata

## Input
- `task.md` - Contains extraction parameters
- Video files in `recordings/`

## Output
- Screenshots in `experiences/<task-id>/screenshots/`
- `output.md` - Extraction results and frame metadata

## Strategies

### Screenshot Strategy (on-demand)
```yaml
type: screenshot
triggers:
  - interval: 30s  # Every 30 seconds
  - change_threshold: 0.15  # 15% pixel difference
  - on_event: [mouse_click, window_focus]
output_dir: experiences/{task_id}/screenshots/
```

### Video Keyframe Strategy
```yaml
type: keyframe
method: scene_detection
sensitivity: 0.3
min_interval: 2s  # At least 2s between keyframes
output_dir: experiences/{task_id}/keyframes/
```

## Implementation
- Uses ffmpeg for frame extraction
- Uses OpenCV for change detection
- Integrates with datalake for metadata storage
