# Phase 2 Audit — Immediate Actions

> Generated from multi-day conversation audit (15+ sessions, Feb 19-23, 2026)
> Source: `docs/CONVERSATION_AUDIT.md` (320 lines, full analysis)

---

## Dashboard Verification Results (Live Test)

| Feature | Status | Detail |
|---------|--------|--------|
| SAHI YOLO | RUNNING | 28 tiles, avg 404ms, detections flowing |
| Temporal YOLO | INITIALIZED | Buffer filling (16 frames), no detections yet (YOLO finding nothing to match) |
| `/api/visual_detect` | WORKING | Found cursor at (864,224) via cnn_grid, conf=0.927, 975ms |
| `/api/daily-report` | WORKING | Returns error (no reports yet) — expected, script not run |
| `/api/state` sahi/temporal fields | PRESENT | Both show status dicts |
| ViT-S model | LOADED | cursor_vit_v011 (still training, 179px median error) |
| Cursor tracking | WORKING | Position (810,298) via sil_template, quality=good |

**Key observation:** SAHI is running but finding 0 detections (YOLO active=False, detection_count=0). The cursor IS being tracked via silhouette template matching, but YOLO v4 isn't detecting the cursor in any of the 28 tiles.

### Live Detection Test (Lock Screen — NOT representative)

**CAVEAT:** This test was on the Windows lock screen, which is NOT the environment YOLO v4 was trained on. The model was trained on passthrough sessions with desktop/app usage. The lock screen has a different cursor style and unusual background. YOLO/CNN have been observed working on normal desktop screens in previous sessions.

| Method | Result on Lock Screen | Notes |
|--------|----------------------|-------|
| YOLO v4 | 0 detections | Expected — domain gap from training data |
| SAHI YOLO | 0 detections | Same YOLO model, same gap |
| CNN grid scan | False positive on clock text | CNN may have trained on text-edge patterns |
| Template match | 0.558 at real cursor position | Weak but correct — synthetic arrow partially matches |

**Takeaway:** Need to test on unlocked desktop to get representative results. The lock screen test only shows the model doesn't generalize to unseen screen types, which is a training data coverage issue, not a fundamental model problem.

---

## Top Actions — Ranked by Impact and Urgency

### CRITICAL (Do Now)

#### 1. Test YOLO v4 + SAHI on Unlocked Desktop
- **Impact:** SAHI runs 28-tile inference (404ms/cycle) but 0 detections on lock screen — need to verify on actual desktop where model was trained
- **Effort:** Unlock Windows, retest with `curl localhost:8766/api/visual_detect`
- **Why it matters:** If YOLO works on desktop but not lock screen, that's expected (domain gap). If it also fails on desktop, the model needs retraining with more diverse data

#### 2. Fix Ghost Position (6, 15)
- **Impact:** 11 documented phantom loss events from this uninitialized default
- **Effort:** 5 minutes
- **Action:** In `init_hardware()`, don't hardcode `tracker.set_position(6, 15)` — leave position as None until first real detection
- **Source:** CURSOR_DETECTION_V2_PLAN.md §1.3, conversation audit

#### 3. Validate Phase 2 Code on Unlocked Desktop
- **Impact:** All Phase 2 features (SAHI, temporal YOLO, visual_detect, position fusion) were coded but never tested on a live desktop with cursor moving
- **Effort:** 30 min with active Windows session
- **Action:** Unlock Windows, start passthrough, observe `/api/state` — do SAHI/temporal fields populate? Does recovery cascade work?

### HIGH (Do This Week)

#### 4. Validate visual_detect Cascade in Real Scenarios
- **Impact:** The cascade works (tested: finds cursor via cnn_grid at 975ms) but SAHI and ViT-S legs never fire
- **Effort:** 1-2 hours
- **Action:** Move cursor to different screen positions, test each cascade method independently:
  - `curl localhost:8766/api/visual_detect` — does it find it?
  - Test SAHI alone: check if lowering `conf_threshold` to 0.10 helps
  - Test ViT-S: once crop training converges, reload model

#### 5. Run Daily Analysis Agent Once
- **Impact:** Establishes baseline detection quality metrics
- **Effort:** 5 minutes
- **Action:** `uv run python scripts/daily_analysis_agent.py -p` (uses Claude subscription, no API key per user instruction)
- **Verify:** `data/daily_reports/` has a new `.md` file, `/daily-report` page shows it

#### 6. Test Mouse Calibration
- **Impact:** Directly addresses user's complaint "non-smooth passthrough movement"
- **Effort:** 30 minutes
- **Action:** Run `mouse_calibration.py` Lissajous calibration, then test passthrough with calibration LUT applied
- **Source:** User session `eb59ca0f`, explicit frustration about smoothness

#### 7. Temporal YOLO — Actually Use It
- **Impact:** Currently initialized but never called with detections (YOLO finds nothing)
- **Effort:** Depends on #2
- **Action:** Once YOLO v4 starts detecting, temporal YOLO will automatically produce velocity vectors. The fix is upstream (make YOLO detect first)

### MEDIUM (Do This Sprint)

#### 8. ViT-S Crop Training
- **Impact:** Full-frame mode plateauing at ~180px error (cursor too small at 384x384)
- **Effort:** Queued — will start automatically after current training finishes
- **Expected:** Crop mode should reach <50px since cursor is 20x33px at native resolution

#### 9. Consolidate Plan Documents
- **Impact:** 9+ overlapping plans cause agent confusion — each session reads different subset
- **Effort:** 1 hour
- **Action:** Create single `CURRENT_STATUS.md` with: what works, what doesn't, what to do next. Mark old plans as historical. Reference from CLAUDE.md

#### 10. Experience Validation Page
- **Impact:** Enables human feedback on agent reasoning quality
- **Effort:** 2-3 hours (HTML page + API endpoints)
- **Action:** Implement `docs/EXPERIENCE_VALIDATION_PLAN.md` — `/experiences` page with 7-dimension rating
- **Source:** Direct plan document, experience/storyline loggers already exist

### LOW (Backlog)

#### 11. Blob Dataset Verification (Track A)
- **Action:** Check if C daemon was recompiled with JSONL logging, verify `data/blob_log/` is growing

#### 12. Keyboard Passthrough Special Keys
- **Action:** Pi Zero keyboard currently text-only. Add Enter, Tab, arrow keys, modifiers

#### 13. Ship Computer Orchestrator
- **Action:** The ultimate goal. Blocked on reliable cursor detection + smooth passthrough

---

## The Core Gap (from Audit)

**CLAUDE.md Phase 1 says:** "A pure visual algorithm that finds the cursor WITHOUT mouse/keyboard. This does NOT exist yet. Building this is the current priority."

**Reality:** We have 5+ detection methods (CNN, YOLO, SAHI, ViT-S, Claude Vision) but NONE reliably find the cursor on a still screen. The live test confirms: SAHI runs 28 tiles at 404ms and finds nothing. `visual_detect` only succeeds via CNN grid scan (975ms, not always reliable). The user's frustration from session `4ef589dc` — "CNN and YOLO don't make sense" — remains valid.

**The fix is not adding more methods.** It's making YOLO v4 + SAHI actually detect the cursor. That's action #2 above.

---

## User Quotes to Remember

1. **"The mouse pointer should not be affected by image recognition at any point"** — Detection validates, it never moves
2. **"Once a good thing is created, it should not be forgotten"** — Don't rewrite working code
3. **"CNN and YOLO don't make sense, which one is helping?"** — The core frustration
4. **"the non-smooth movement I face during the mouse passthrough"** — Specific UX pain
5. **"tracking what I do in corporate work sessions more meaningful"** — The end goal
6. **"For the daily agent, use '-p' mode"** — No API key, use subscription
