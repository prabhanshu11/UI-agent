# Meeting Mode Configuration

## Purpose
Passive observation during meetings/calls.

## Behavior
- **Screenshot Frequency**: Every 60 seconds
- **Keyframe Detection**: Low sensitivity (major changes only)
- **Summarization**: Periodic batches
- **Experience Logging**: Summary only

## Active Agents
- screenshot-extractor: ACTIVE (low freq)
- keyframe-identifier: PASSIVE
- summarizer: BATCH (every 5 min)

## Recording Settings
- FPS: 10 (lower for meetings)
- Segment: 10 minutes
- Change threshold: 30%

## Experience Capture
```yaml
capture:
  screenshots: true
  keyframes: false
  summaries: batch
  video_chunks: true
log_level: summary
```
