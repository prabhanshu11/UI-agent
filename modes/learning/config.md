# Learning Mode Configuration

## Purpose
High-detail capture for learning new skills/tools.

## Behavior
- **Screenshot Frequency**: Every 10 seconds + all changes
- **Keyframe Detection**: Maximum sensitivity
- **Summarization**: Every keyframe + periodic
- **Experience Logging**: Maximum knowledge detail with Agent Zero compatible multi-storyline logging

## Active Agents
- screenshot-extractor: ACTIVE (high freq, 10s interval)
- keyframe-identifier: ACTIVE (high sensitivity, 5% change threshold)
- summarizer: ACTIVE (all keyframes + periodic every 60s)
- storyline-logger: ACTIVE (all narratives)

## Recording Settings
```yaml
fps: 20
segment_minutes: 5
change_threshold: 0.05  # Very sensitive - 5%
min_keyframe_interval: 2  # Allow keyframes every 2 seconds
```

## Experience Capture
```yaml
capture:
  screenshots: true
  keyframes: true
  summaries: every_keyframe
  video_chunks: true
  ocr: true

logging:
  storylines:
    - main: Primary task narrative
    - knowledge: Learnings and insights (CRITICAL for future speed)
    - technical: Commands, code, errors
    - agent: Sub-agent dispatches and returns
    - perception: Screenshots, UI changes, detected elements
    - user: User interactions and feedback

  output_format: jsonl
  narrative_export: markdown
  log_level: maximum
```

## Multi-Storyline Logging (Agent Zero Compatible)
Maintains parallel narratives for comprehensive learning:

1. **Main Storyline**: What is being accomplished
2. **Knowledge Storyline**: Learnings that speed up future tasks
3. **Technical Storyline**: Exact commands and outputs
4. **Agent Storyline**: How sub-agents collaborate
5. **Perception Storyline**: What was seen/detected
6. **User Storyline**: User input and corrections

## Learning Focus
- Capture every UI interaction
- Extract text from screens for searchability
- Build comprehensive procedure documentation
- Enable replay and review
- **Record insights** that can speed up similar tasks
- Track which approaches worked vs failed
- Build reusable command snippets
