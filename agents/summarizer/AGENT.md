# Screenshot Summarizer Agent

## Purpose
Analyze screenshots and generate descriptions:
1. **Visual description**: What's visible on screen
2. **UI state**: Application, window title, active elements
3. **Action inference**: What the user appears to be doing
4. **Context extraction**: Relevant text, data, status indicators

## Capabilities
- Process screenshots via vision LLM
- Extract visible text (OCR)
- Identify UI elements and their states
- Generate structured summaries

## Input
- `task.md` - Screenshots to analyze
- Screenshots from `experiences/<task-id>/screenshots/`

## Output
- `output.md` - Structured summaries
- Metadata for datalake ingestion

## Analysis Modes

### Quick Summary
Fast, single-pass description for real-time processing.

### Deep Analysis
Multi-pass analysis with:
- OCR text extraction
- UI element detection
- Application identification
- User action inference

### Comparative Analysis
Compare multiple screenshots to identify:
- What changed between frames
- Progress through a workflow
- Error states vs success states

## Output Format
```yaml
summaries:
  - screenshot: "keyframe_001.png"
    timestamp: "2026-01-16T08:47:30"
    summary:
      application: "Windows 11 Desktop"
      visible_windows: []
      ui_state: "idle"
      active_element: null
      visible_text: ["Search", "8:47 AM", "1/16/2026"]
      user_action: "idle_at_desktop"
      confidence: 0.92
```

## Vision Model Integration
- Primary: OpenRouter API (gemini-2.0-flash, claude-3-5-sonnet)
- Fallback: Local model if available
- Rate limiting: 10 requests/minute for cost control
