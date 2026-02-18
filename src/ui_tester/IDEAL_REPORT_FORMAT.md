# Ideal UI Test Report Format

What I wish the agent had given me:

## 1. Visual Diff Section

```
BEFORE clicking "Show Thinking":
  - Element .thinking-section: display=none
  - Element .thinking-toggle: text="▸ Show Thinking"

AFTER clicking:
  - Element .thinking-section: display=block
  - Element .thinking-toggle: text="▾ Hide Thinking"
```

## 2. Content Inventory

```
Messages breakdown:
  - 64 with tool_uses > 0, content_text empty
  - 42 with content_thinking, content_text empty
  - 25 with content_text
  - 63 system/context (user messages, no content)

Action: Check if "(no content)" is intentional for tool-only messages
```

## 3. Console Errors (grouped)

```
NETWORK:
  - /favicon.ico 404 (1 occurrence)

JAVASCRIPT:
  - (none)

DEPRECATION:
  - (none)
```

## 4. Interactive Element Status

```
.star (rating):
  - Count: 10
  - Clickable: yes
  - Visual feedback on hover: yes
  - API call on click: /api/rate/session/{id}

.copy-btn:
  - Count: 194
  - Clickable: yes
  - Uses clipboard API

.thinking-toggle:
  - Count: varies (only on messages with thinking)
  - Toggle works: yes
```

## 5. Layout Issues (auto-detected)

```
OVERFLOW:
  - .message-header-right: text truncated when tokens > 999

SCROLL:
  - Page requires 37x viewport to see all content
  - Consider: pagination or virtual scroll

RESPONSIVE:
  - Not tested (headless viewport: 1400x900)
```

## 6. Suggested Fixes (prioritized)

```
1. [CRITICAL] Fix empty content display
   File: templates/session_detail.html
   Line: ~197
   Change: Add conditional for tool_uses and thinking

2. [HIGH] Fix token overflow
   File: templates/session_detail.html
   Add CSS: .token-info { white-space: nowrap }

3. [MEDIUM] Add favicon
   File: web/app.py or static/favicon.ico
```

## Key Insight

The report should tell me **what data is in the database** vs **what's showing in the UI**. The mismatch is where bugs live.
