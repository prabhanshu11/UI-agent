# Conversation Audit — Multi-Day KVM Dashboard Review

> **Generated:** 2026-02-23
> **Sessions reviewed:** 15+ sessions spanning Feb 19 - Feb 23, 2026
> **Total messages scanned:** ~8,000+ across datalake
> **Key sessions:** `eb59ca0f` (60h), `7ae3a46f` (5h), `4ef589dc` (4.5h), `7b93dba4` (9h), `758f8a3f` (24m), `d447d647` (1.5h), `ab35208f` (26h), `fc509cee` (1h), `5df2142c` (53m), `c48d9ab0` (12m), `97c8a008` (7h), `193e944e` (1h)

---

## User's Key Instructions (chronological)

### Foundational Vision (sessions pre-Feb 20, documented in STAR_TREK_COMPUTER_RESEARCH.md)

1. **"like when the Star Trek computer or Skynet gains access to one terminal, it actively tries to use known ways to gain more access as the need arises"** — The system is meant to be an autonomous AI that gains awareness and control, terminal by terminal.

2. **"make sure entire end-to-end experience is logged, and understood in a deeper than SWE coder way"** — Not just SWE debugging logs. The system must understand what it is doing at a narrative level.

3. **"Once a good thing is created, it should not be forgotten"** — Proven working code paths must be preserved. Git history and conversation history ARE the experience log.

4. **"The mouse pointer should not be affected by image recognition at any point"** — Motion detection is the base layer for tracking. CNN/visual detection only VALIDATES or DISCOVERS. It never MOVES the cursor.

5. **"If the position is not known, only the locator utility needs to be used, and not random motions"** — No blind clicking. Find the cursor first.

### Session `7ae3a46f` (Feb 20, Loss Event Detection, ~5 hours)

6. **"Read the previous several messages... and then think about the ones which mention the mouse pass through... now that we are building that capability... First of all, we need to build it more robustly."** — User wanted a thorough context review before implementing passthrough robustness.

### Session `4ef589dc` (Feb 20, YOLO-Primary Cursor Detection, ~4.5 hours)

7. **"Work on blob dataset plan and cursor detection v2 plan. And right now, the detection of real time detection of cursor is still off. From the model's view overlay that you have created, I can see that the span of CNN and YOLO they do not make sense, which one is helping and what is supposed to be better? That's not clear. I was expecting that the CNN would be, at this point, very well trained to stick to the cursor. And I don't see the motion data being correctly training the models to improve over time."** — This is the user's core frustration: the CNN and YOLO overlays visually don't make sense, and models aren't improving with accumulated data.

### Session `eb59ca0f` (Feb 20-22, Robust Input Passthrough, 60+ hours)

8. **"Meanwhile YOLO training runs. Go through the star trek computer research, and pragmatic plan... I'm sure the labels we create from the current YOLO training will find their use somewhere definitely in helping the non-smooth movement I face during the mouse passthrough. That data can only be gotten once we can trust CV based cursor position, so that we can calibrate the hardware mouse."** — User directly states the dependency chain: trustworthy cursor detection -> mouse calibration -> smooth passthrough.

9. **"The motivations were in that conversation and will incorporate the points suggested by the Claude AI in the chat will make the manual work on this project and the overall effectiveness of the project in ultimately being able to track what I do in corporate work sessions more meaningful"** — The end goal is tracking corporate work sessions, not just cursor detection for its own sake.

### Session `ab35208f` (Feb 20-21, Recovery + Continuation)

10. **"I've started the work in terminal 1. For the plan and the context associated with terminal 2, does this document contain some useful info that you can incorporate into your doc"** — User was actively managing multiple parallel agent sessions working on different aspects of the project simultaneously.

### Session `7b93dba4` (Feb 22-23, Phase 1 Vision Foundation, 9 hours)

11. **"Continue with Phase 2."** — After Phase 1 was committed, user explicitly requested continuation to Phase 2 (Dashboard Integration + Model Training + Temporal Fusion).

12. **"yes, queue up the crop run. Fire up the dashboard also and make sure the cursor detection is hooked up to the best logic we have. Go through previous messages of this conversation, mine first, to establish curious points... Do a multiday traversal and make sure cursor and all other features are based on my instructions or your initial plans."** — THIS is the direct instruction that spawned the current audit task.

### Session `c48d9ab0` (Feb 23, Daily Agent)

13. **"Can one work on one specific thing here, without modified other work that is being done. For the daily agent, you don't need api key, use '-p' mode"** — User wants isolated parallel work on specific features without cross-contamination. Also specifies using Claude CLI `-p` mode (subscription, no API key) for the daily agent.

### Session `193e944e` (Feb 22, TAPO Camera)

14. **"Can you continue to work on the Image recognition work through the TAPO camera to log each and everything and all the objects around me."** — The camera awareness system (Concept 4) is actively being worked on in parallel, not just KVM cursor detection.

---

## Unresolved User Requests

### 1. Phase 1 of CLAUDE.md — Pure Visual Detection WITHOUT Mouse (HIGH PRIORITY)

CLAUDE.md states explicitly: **"This does NOT exist yet. Building this is the current priority."**

The user wants a pure visual algorithm that finds the cursor WITHOUT needing mouse/keyboard access. What exists now:
- TinyCursorNet CNN (validates patches, doesn't discover)
- YOLO detector (finds cursor in full frames, but unreliably)
- SAHI YOLO (new, just wired in Phase 2)
- ViT-S regression model (training started, not converged — median error still 201px)
- Claude Vision detector (reliable but slow, 10s per call — exists but too slow for continuous use)

What is MISSING: A reliable, moderate-speed visual detector that can find the cursor on a still screen in 1-3 seconds without ANY mouse movement. This is the stated priority in the binding CLAUDE.md but has never been achieved.

### 2. Non-Smooth Mouse Passthrough Movement

The user explicitly mentioned: **"the non-smooth movement I face during the mouse passthrough"** (session `eb59ca0f`). The hardware mouse calibration system (`src/hardware/mouse_calibration.py`) was created in Phase 1 but has NOT been tested or integrated into the passthrough pipeline. Windows mouse acceleration curves were researched but the calibration lookup table has not been validated.

### 3. "Track Everything in My Life" via Cameras

Session `193e944e`: **"log each and everything and all the objects around me"** — The TAPO camera object detection system was built (ActiveSceneManager with PTZ-aware tracking in `tapo-c210-monitor`), but integration with the Ship Computer orchestrator is not done. There is no unified queryable history across KVM + camera feeds.

### 4. Experience Validation Page (`/experiences`)

`docs/EXPERIENCE_VALIDATION_PLAN.md` describes a human-feedback page for evaluating agent reasoning chains (7 dimensions: awareness, conclusion, decision, execution, direction, certainty, verdict). Status: **Plan only. Not implemented.** No code exists for this page.

### 5. Recording Lifecycle Management (Disk Will Fill)

`docs/RECORDING_LIFECYCLE_PLAN.md` documents that the cursor-daemon records indefinitely with NO cleanup. As of Feb 20, 247GB of 446GB was free, with estimated disk full around March 12. Status: **Plan only. Not implemented.** The stopgap delete commands in CLAUDE.md are the only solution available. Three days have passed since the warning was written.

### 6. Dashboard Sub-Panels (Partial)

`docs/DASHBOARD_SUBPANELS_PLAN.md` describes 3 sub-panels: Motion Blob Map, Model Inference Status, Mouse Trail visualization. Session `758f8a3f` (Phase 2) committed `sub-panels.js` with some sub-panel code, but the original plan called for:
- **Mouse trail visualization showing ESP32 + detected + passthrough** — Not fully implemented
- **Live staleness counters ticking every 1s independently** — Unclear if implemented
- **Color-coded model freshness (green/yellow/red)** — Unclear if implemented

### 7. Daily Analysis Agent

The `scripts/daily_analysis_agent.py` was created in Phase 1 to run claude `-p` for daily cursor detection performance reports. User specified: **"For the daily agent, you don't need api key, use '-p' mode."** The script exists but has NOT been run or tested end-to-end. No daily reports exist in `data/daily_reports/`.

### 8. Blob Dataset (Track A of Temporal Fusion Plan)

C daemon JSONL logging was partially implemented (daemon code modified in the Phase 1 commit `06acf998`), but verification is unclear. The plan calls for blob JSONL files appearing in `data/blob_log/` and growing at ~1.2 KB/s idle. Whether the daemon was recompiled and is actually producing these files needs verification.

---

## Curious Points / Design Decisions

### 1. The "Death Spiral" Pattern

Multiple sessions describe cursor detection as being in a "death spiral":
- `motion_track` picks up random taskbar/UI blobs, NOT the cursor
- CNN never validates (retro_cnn confidence too low on wrong position)
- Triggers reacquisition -> probes move cursor -> more blobs -> repeat

This has been identified and documented multiple times across sessions but the fundamental fix (reliable visual detection without mouse movement) remains unbuilt. Plans keep being created but the core Phase 1 visual detection priority from CLAUDE.md keeps getting deprioritized in favor of more sophisticated approaches (SAHI, ViT-S, Temporal YOLO).

### 2. ViT-S Model Started Training with 201px Median Error

The ViT-S cursor regression model (Phase 1 training) reached 201.3px median error after 32 epochs of 100. On a 1920x1080 screen, 201px error means the model predicts cursor position wrong by about 10% of screen width. This is far from the <5px target stated in the Pragmatic Plan. The training data consists of only 2,382 frames from passthrough sessions.

### 3. Known False Diagnosis Pattern

CLAUDE.md has an explicit warning: agents tend to blame "multiple blobs (blinking cursors, notification badges)" as the reason cursor detection fails, when the REAL root cause is usually "wrong code, wrong offsets, broken pipeline, stale data, or incorrect assumptions about what's working." This is a meta-insight the user learned from repeated agent sessions.

### 4. The 60-Hour Session

Session `eb59ca0f` ran for 60+ hours with 2,605 messages. It was a continuation session that covered multiple compaction cycles. The user was managing 3+ simultaneous Claude Code sessions working on different aspects of the project. This multi-agent parallel development pattern is significant.

### 5. Ghost Position (6, 15)

The position fusion plan documents 11 events where the system's last-known cursor position was exactly (6, 15) with method="manual" — clearly an uninitialized default. This causes phantom loss events and 1500-2000px jumps. Fix proposed in CURSOR_DETECTION_V2_PLAN.md section 1.3. Status: unclear if implemented.

### 6. Multiple Plans, Overlapping Scope

There are now 9+ plan documents in `docs/`:
- BLOB_DATASET_PLAN.md
- CURSOR_DETECTION_PLAN.md
- CURSOR_DETECTION_V2_PLAN.md
- DASHBOARD_SUBPANELS_PLAN.md
- EXPERIENCE_VALIDATION_PLAN.md
- RECORDING_LIFECYCLE_PLAN.md
- TEMPORAL_FUSION_PLAN.md
- CURSOR_DETECTION_EXPERIENCE.md
- Pragmatic_plan_key_to_implement_startrek_computer_assistant__temporal.md

These overlap significantly. The Temporal Fusion Plan references both the Blob Dataset Plan and Cursor Detection V2 Plan. The Pragmatic Plan replaces parts of both. The risk is that agents read one plan but not the others and implement conflicting approaches.

---

## Missing Features (from user conversations)

### From STAR_TREK_COMPUTER_CONCEPTS.md (Concept 1-6)

| Feature | Status | Notes |
|---------|--------|-------|
| Ship Computer Orchestrator (L1) | NOT BUILT | Telegram -> Spawner -> Claude Agents hierarchy |
| Camera Awareness lifecycle | PARTIAL | ActiveSceneManager built, no Ship Computer integration |
| Navigation System (perceive-act-verify) | NOT BUILT | WindowsNavigator concept exists only in docs |
| Experience System (OTel traces in production) | NOT BUILT | ExperienceLogger and StorylineLogger exist in code but no `/experiences` page |
| Silhouette Scan (template match cursor shapes WITHOUT moving) | NOT BUILT | Documented as "NOT YET BUILT" in CONCEPTS.md |
| CNN Full-Screen ("is cursor even on this screen?") | NOT BUILT | Documented as "NOT YET BUILT" in CONCEPTS.md |
| Multi-camera life tracking | NOT BUILT | Single TAPO camera partially working |

### From Temporal Fusion Plan (Track A-D)

| Track | Status | Notes |
|-------|--------|-------|
| A: C daemon blob JSONL logging | PARTIAL | Daemon code modified, needs recompile + verification |
| B: MotionTrail shared utility | COMPLETED | `src/hardware/motion_trail.py` exists (Phase 2 commit) |
| C: Temporal YOLO | COMPLETED | `src/hardware/temporal_yolo.py` created (Phase 2) |
| D1: Retro CNN recovery | COMPLETED | Wired into position_fusion.py (Phase 2) |
| D2: Trail-guided CNN verify | COMPLETED | In position_fusion.py (Phase 2) |
| D3: YOLO trail-informed scoring | COMPLETED | In position_fusion.py (Phase 2) |
| D4: Confidence gate | COMPLETED | In position_fusion.py (Phase 2) |
| D5: Dead reckoning integration | COMPLETED | In position_fusion.py (Phase 2) |

### From User's Direct Requests

| Request | Status | Source |
|---------|--------|--------|
| Mouse calibration for smooth passthrough | CODE EXISTS, NOT TESTED | `src/hardware/mouse_calibration.py`, session `eb59ca0f` |
| "I don't see the motion data correctly training the models" | PARTIALLY ADDRESSED | v4 YOLO trained with CNN-filtered labels, but user hasn't confirmed improvement |
| Corporate work session tracking | NOT BUILT | End goal stated in `eb59ca0f`, no work session tracking exists |
| Keyboard passthrough (Pi Zero) | BASIC ONLY | Text-only via HTTP POST, no special keys (Enter, Tab, arrows) |
| Audio from USB capture | UNKNOWN | User asked "Does audio come out of the USB capture?" in session `004d0e16` — unclear if answered |

---

## Known Pain Points / Bugs

### 1. Cursor Detection "CNN and YOLO don't make sense"

**User quote (session `4ef589dc`):** "From the model's view overlay that you have created, I can see that the span of CNN and YOLO they do not make sense, which one is helping and what is supposed to be better?"

This is the #1 frustration. The overlays show conflicting information — CNN reports high confidence on non-cursor areas, YOLO finds cursors where there are none. The root cause is training data quality: models were trained on frames labeled by the same unreliable systems (motion_track, sil_motion_roi).

### 2. 68.6% of Cursor Loss Events Never Recover

From CURSOR_DETECTION_V2_PLAN.md: The system loses the cursor in 0.32 seconds and takes 8.6 seconds to recover — when it recovers at all. 68.6% of events never recover. The Position Fusion enhancements (Track D) were implemented in Phase 2 but have NOT been tested with the dashboard running.

### 3. Disk Space Filling Up

No recording cleanup exists. CLAUDE.md estimates disk full around March 12, 2026. No implementation has been done despite the plan existing since Feb 20.

### 4. Agent Session Crashes

Session `4ef589dc` crashed with API 500 Internal Server Error. The user had to use conversation recall across sessions to recover context. This is a recurring pattern — long sessions (2000+ messages) become unstable.

### 5. False PASS Verdicts from Agents

`progress_actual.md` documents: "Agent declares things working based on HTTP 200, not visual verification." Agents confirm features work without actually looking at what the dashboard shows on screen. This is the gap between "code compiles" and "feature works as user expects."

### 6. YOLO v4 Still Has Bad Training Data

YOLO v4 was trained filtering by `retrospective_cnn.confidence > 0.80`, but the CNN itself may have false positives. The training data comes from passthrough sessions where ground truth is ESP32 dead-reckoned position (which drifts). This circular dependency means better training data requires solving the very problem the models are meant to solve.

---

## Plans vs Reality

### CLAUDE.md Phase Model

| Phase | CLAUDE.md Says | Reality |
|-------|---------------|---------|
| Phase 1: Slow but Reliable Visual Detection | "This does NOT exist yet. Building this is the current priority." | Multiple systems exist (CNN, YOLO, SAHI, ViT-S, Claude Vision) but NONE reliably find the cursor on a still screen in <5s without mouse |
| Phase 2: Fast Silhouette Tracking | "After Phase 1 works" | Silhouette tracker EXISTS but Phase 1 prerequisite is unmet — it has nothing reliable to bootstrap from |
| Phase 3: Lissajous Guaranteed Finder | "NOT the primary detection method" | Currently the ONLY reliable method but requires mouse control |

### Temporal Fusion Plan (5 Phases)

| Plan Phase | Plan Status | Reality |
|------------|-------------|---------|
| Phase 1: Foundation | "No behavior change" | Blob JSONL partial, MotionTrail created |
| Phase 2: Wire the trail | "Populate, not consume" | Trail wired into dashboard (Phase 2 commit) |
| Phase 3: Trail-informed detection | D1-D3 enhancement | Code created in Phase 2, untested in production |
| Phase 4: Temporal YOLO | Track C | temporal_yolo.py created, untested in production |
| Phase 5: Confidence gate | D4-D5 | Code created in Phase 2, untested in production |

### Pragmatic Plan (Vision Foundation)

| Plan Step | Reality |
|-----------|---------|
| Step 1: SAHI YOLO detector | Created (`sahi_yolo.py`), wired into dashboard |
| Step 2: ViT-S cursor regression | Training started, median error 201px (target <5px) |
| Step 3: Mouse calibration | Code exists (`mouse_calibration.py`), never tested |
| Step 4: Daily analysis agent | Script exists, never run |
| Step 5: Dataset builder | Script exists, produced 2,382 frames |

### The Core Gap

The user's stated end goal is **tracking corporate work sessions** — an AI that can observe and understand what they do on Windows. The current state is: cursor detection still unreliable, mouse passthrough still non-smooth, no work session recording/analysis pipeline, no Navigation system, no Ship Computer orchestrator. The project has sophisticated PLANS and COMPONENTS but the end-to-end pipeline from "user works on Windows" to "system understands and logs what happened" does not exist.

---

## Recommendations

### 1. CRITICAL: Implement Recording Cleanup (days, not weeks)

Disk will fill around March 12. This is a hard deadline. Implement the simplest version of RECORDING_LIFECYCLE_PLAN.md: a cron job or systemd timer that deletes MKV chunks older than 7 days and loss events older than 3 days. This is 20 lines of bash.

### 2. HIGH: Make One Visual Detection Method Actually Work

Instead of adding more detection methods (SAHI, ViT-S, Temporal YOLO), focus on making ONE method reliably find the cursor on a still screen. The most promising candidate is SAHI YOLO with the trained v4 model, because:
- YOLO v4 was trained on CNN-filtered data (best available labels)
- SAHI slices the full 1080p frame into overlapping tiles
- Combined, they should be able to find a small cursor in a full frame

Test this specifically: start the dashboard, move the cursor to a known position, stop the mouse, and verify SAHI YOLO can locate it within 2 seconds. If it cannot, debug why — don't add yet another method.

### 3. HIGH: Validate Phase 2 Code in Production

Tracks C, D1-D5 were all implemented as code in Phase 2 but none have been tested with the dashboard actually running. Before writing more code, start the dashboard with `--passive` mode and verify:
- MotionTrail populates and appears in `/api/state`
- Temporal YOLO produces velocity vectors
- Position fusion confidence gate works
- Retro CNN recovery fires on cursor loss

### 4. MEDIUM: ViT-S Training Needs More Data

2,382 frames is very small for training a ViT-S model. The 201px median error after 32 epochs suggests the model is memorizing rather than learning. Either:
- Collect more passthrough data (run passthrough sessions for 1-2 hours)
- Use the existing 24,202 cursor samples from `data/cursor_samples/index.jsonl`
- Consider unfreezing encoder layers after head converges

### 5. MEDIUM: Test Mouse Calibration

`mouse_calibration.py` addresses the user's specific complaint about non-smooth passthrough movement. It was written but never tested. Run the Lissajous calibration procedure once to build a lookup table, then test passthrough with calibration applied.

### 6. LOW: Consolidate Plan Documents

9+ overlapping plan documents create confusion. Consider marking completed plans as DONE and creating a single CURRENT_PLAN.md that agents read first, with references to detailed plans for specific tracks.

### 7. LOW: Daily Agent Automation

The daily analysis agent could provide valuable feedback on detection quality trends. Set up a systemd timer or cron job to run it daily. But this depends on the dashboard actually producing useful detection data first (see recommendation 2-3).

### 8. FUTURE: The Navigation System

The user's ultimate goal is corporate work session tracking via the Navigation system (perceive-decide-act-verify). Everything before this — cursor detection, passthrough, calibration — is prerequisite infrastructure. Keep this end goal in mind but don't build it until cursor detection is reliable and mouse passthrough is smooth.

---

## Session Index (for reference)

| Session ID | Date | Duration | Topic |
|-----------|------|----------|-------|
| `eb59ca0f` | Feb 20-22 | 60h | Robust Input Passthrough + Experience Recording |
| `7ae3a46f` | Feb 20 | 5h | Loss Event Detection + Velocity Simulation |
| `4ef589dc` | Feb 20 | 4.5h | YOLO-Primary Cursor Detection: Retrain + Restructure |
| `fc509cee` | Feb 20 | 1h | YOLO-Enriched Validation Dashboard |
| `5df2142c` | Feb 20 | 53m | Blob Dataset + Temporal YOLO + Position Fusion + Motion Trail |
| `d447d647` | Feb 20 | 1.5h | KVM UI element / front end |
| `ab35208f` | Feb 20-21 | 26h | Crashed UI-agent session recovery |
| `02a44895` | Feb 21 | 3m | Start KVM dashboard passive mode |
| `9adefc2e` | Feb 21 | 4m | Search KVM dashboard conversations |
| `97c8a008` | Feb 22 | 7h | Active Scene Manager (TAPO camera) |
| `193e944e` | Feb 22 | 1h | Image recognition via TAPO camera |
| `7b93dba4` | Feb 22-23 | 9h | Phase 1: Vision Foundation + Continuous Improvement |
| `758f8a3f` | Feb 23 | 24m | Phase 2: Dashboard Integration + Model Training |
| `c48d9ab0` | Feb 23 | 12m | Daily agent + isolated work |
