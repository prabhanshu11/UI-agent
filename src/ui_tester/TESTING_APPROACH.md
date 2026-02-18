# Testing Approach

## What Worked

1. **Get page structure first** - headings, buttons, links, inputs
2. **Scroll full page** - found the 28k pixel height issue
3. **Check console errors** - found favicon 404
4. **Click interactive elements** - verified "Show Thinking" works
5. **Count elements** - 194 copy buttons = 194 messages = data integrity check

## What I Missed Initially

1. **Database query** - Should have checked what's actually stored
   ```sql
   SELECT content_text, content_tool_uses, content_thinking
   FROM claude_messages LIMIT 5
   ```
   Would have revealed tool_uses is stored as int, not JSON

2. **Compare data vs display** - 64 tool-only messages showing "(no content)" was a data-to-UI mismatch

## Recommended Test Flow

```
1. Navigate
2. Screenshot (initial)
3. Get page structure
4. Get console errors
5. Query database for page's data source
6. Compare: what's in DB vs what's rendered
7. Scroll to bottom, screenshot at intervals
8. Click each interactive element type once
9. Verify state changes
10. Generate diff report
```

## Tools That Would Help

- DOM snapshot before/after clicks
- Database query tool (to compare source data)
- Automatic element counting with expected values
- Screenshot diff (before vs after fix)
