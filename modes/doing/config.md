# Doing Mode Configuration

## Purpose
Active task execution mode for completing work.

## Behavior
- **Screenshot Frequency**: Every 30 seconds + on change
- **Keyframe Detection**: Active, high sensitivity
- **Summarization**: On-demand (keyframes only)
- **Experience Logging**: Full detail

## Active Agents
- screenshot-extractor: ACTIVE
- keyframe-identifier: ACTIVE
- summarizer: ON_DEMAND

## Recording Settings
- FPS: 20
- Segment: 5 minutes
- Change threshold: 15%

## Experience Capture
```yaml
capture:
  screenshots: true
  keyframes: true
  summaries: on_keyframe
  video_chunks: true
log_level: detailed
```
