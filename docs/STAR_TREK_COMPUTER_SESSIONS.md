# Star Trek Computer — Session Index & Concept Map

> Chronological catalog of all Claude Code sessions related to the Star Trek Computer system.
> Supersedes `windows-automation-sessions-2026-01-16.md`.
> Cross-referenced with `STAR_TREK_COMPUTER_RESEARCH.md` (master research).

---

## Concept Structure

The sessions below are tagged by concept cluster. These clusters emerged from the work itself — some sessions touch multiple clusters because the concepts genuinely overlap.

```
SHIP COMPUTER (orchestrator, Telegram, L1/L2/L3)
│
├── ASMP (Actuator-Sensor Mapping Protocol — the generalization)
│   │
│   ├── CURSOR AWARENESS (ESP32 mouse + HDMI capture)
│   │   ├── Discovery — Lissajous sweep, record-then-analyze
│   │   ├── Tracking — motion detection, shape filter, three-tier recovery
│   │   ├── Validation — CNN, jitter confirmation, dual-direction probe
│   │   ├── Infrastructure — C daemon (30fps), dashboard, shared memory
│   │   └── [Open] Pure Visual — silhouette scan, trainable CNN, "discover+stick"
│   │
│   └── CAMERA AWARENESS (Tapo C210 + ONVIF PTZ — separate repo)
│
├── NAVIGATION (perceive-act-verify, hypothesis recovery, experiences)
│
├── HARDWARE (ESP32 mouse, Pi keyboard, HDMI capture, K380 pairing)
│
├── EXPERIENCE SYSTEM (OTel StorylineLogger, narrative logs, skills)
│
└── INFRASTRUCTURE (datalake, Moonlight, Android emulator, MCP tools)
```

**Key overlaps:**
- Cursor Awareness IS an ASMP implementation — they're the same pattern at different abstraction levels
- Navigation USES Cursor Awareness — every navigation step needs cursor position
- The KVM Dashboard is the observability layer for Cursor Awareness, not a separate concept
- The "Getting Control" experience type = Cursor Discovery; "IT Work" = Navigation
- The C Daemon, shape filter, CNN, and three-tier recovery are all parts of one tracking mesh, not separate systems

---

## Phase 1: Foundation (Jan 4 – Jan 16) — Laptop + Desktop

### Vision & Architecture

| # | Session | Date | Machine | Duration | Topic | Concepts |
|---|---------|------|---------|----------|-------|----------|
| 1 | `9541b458` | Jan 4 | Laptop | — | Initial git/GitHub setup, personal website deploy | INFRASTRUCTURE |
| 2 | `7c76adf9` | Jan 4 | Laptop | — | Early Android emulator for UI automation | CAMERA AWARENESS |
| 3 | `68b19fae` | Jan 5 | Laptop | — | Voice typing (hyprwhspr) troubleshooting | INFRASTRUCTURE |
| 4 | `a75e5c0b` | Jan 8 | Laptop | — | **Star Trek vision planning**, YouTube research on Claude Agent SDK, local voice/image models, "consciousness" for context tracking | SHIP COMPUTER |
| 5 | `4af18670` | Jan 8 | Desktop | — | Windows automation + Moonlight | INFRASTRUCTURE |
| 6 | `1f9f9926` | Jan 8 | Desktop | — | Tapo camera + Android emulator. DO NOT use Aurora Store. Emulator video crashes. | CAMERA AWARENESS |
| 7 | `d0e8df59` | Jan 9 | Desktop | — | Amazon Shopping Agent, Playwright browser automation, address verification | NAVIGATION |
| 8 | `6aae4902` | Jan 10 | Laptop | — | Moonlight/Sunshine remote desktop setup, keyboard shortcuts, `moonlight-toggle` script | INFRASTRUCTURE |
| 9 | `e2c9ece8` | Jan 10 | Desktop | — | Tapo camera continuation. Pan control works. Video goes black after 1 frame. | CAMERA AWARENESS |
| 10 | `29d2d22f` | Jan 11 | Laptop | — | Google Forms via Claude Agent SDK + Selenium | INFRASTRUCTURE |
| 11 | `77e6cee1` | Jan 11 | Laptop | — | ROI calculator interactive doc, OpenRouter + DeepSeek | INFRASTRUCTURE |
| 12 | `b0ca7007` | Jan 11 | Laptop | — | `moonlight-broadcast` command for cross-device streaming control | INFRASTRUCTURE |
| 13 | `1e94d81d` | Jan 11 | Laptop | — | VDI automation discussion, business work automation concept | SHIP COMPUTER |
| 14 | `f3d26b69` | Jan 12 | Laptop | — | **USB capture card integration + UI-Agent architecture planning**. Merging repos. "One computer instance running like Star Trek." | SHIP COMPUTER, HARDWARE |
| 15 | `2da8af04` | Jan 12 | Laptop | — | Repo reorganization. Bought capture card + ESP32 kit. | HARDWARE |
| 16 | `cbbb26c8` | Jan 13 | Laptop | — | Claude Agent SDK usage session | SHIP COMPUTER |
| 17 | `11518f1a` | Jan 13 | Laptop | — | Multi-agent orchestration. Machine status dashboard. Subagents for voice typing + datalake. | SHIP COMPUTER |
| 18 | `6f47a83e` | Jan 14 | Laptop | — | HDMI capture device search | HARDWARE |
| 19 | `cf7a8f47` | Jan 14 | Desktop | — | Windows automation continuation | NAVIGATION |
| 20 | `dbad911b` | Jan 15 | Laptop | — | **Tapo C210 camera automation. ASMP protocol born.** 2D binary search, hybrid capture, agent hooks. | ASMP, CAMERA AWARENESS |

**Resume**: `claude --resume <session-id>`

### Desktop Sessions (brief, Jan 4–16)

These desktop sessions contributed to supporting infrastructure but don't contain primary research:

| Session | Keywords | Topic |
|---------|----------|-------|
| `5d2abcd1` | datalake, widgets | Database setup, Selenium testing agent |
| `5fc36dcc` | tapo, webcam | Tapo scaffolding, USB webcam API |
| `f268982d` | tapo, multi-agent | TP-Link camera, terminal orchestration |
| `01d29e33` | windows | Windows automation |
| `5feeb5c5` | windows | Windows automation |
| `ae6b1c69` | windows | Windows automation |
| `eb6dd91e` | windows | Windows automation |
| `5c9f0d1f` | moonlight, UI-agent | Streaming + UI agent |
| `6cab20c5` | moonlight, UI-agent | Streaming + UI agent |
| `6e25e808` | moonlight, UI-agent | Streaming + UI agent |
| `e1b3ccda` | moonlight, UI-agent | Streaming + UI agent |
| `f2602fec` | moonlight, UI-agent | Streaming + UI agent |
| `1d10fbb2` | esp32, bt hid | ESP32 BT HID |
| `230f8714` | esp32, camera | ESP32 + camera |
| `2974913e` | esp32, camera | ESP32 + camera |
| `570425ed` | esp32, camera | ESP32 + camera |
| `641872c2` | esp32, camera | ESP32 + camera |
| `87da3156` | esp32, camera | ESP32 + camera |
| `c97d52f6` | esp32, camera | ESP32 + camera |
| `d3d60a35` | esp32, camera | ESP32 + camera |
| `ddd15f05` | esp32, camera | ESP32 + camera |
| `4acad5d4` | windows, esp32 | Windows + ESP32 |

### Session Catalog

| # | Session | Date | Machine | Topic |
|---|---------|------|---------|-------|
| 21 | `b9b4f75a` | Jan 16 | Laptop | **Comprehensive conversation recall** — produced `windows-automation-sessions-2026-01-16.md` (43-session catalog) |

---

## Phase 2: KVM Build (Feb 16) — Laptop → Desktop

### Laptop KVM Sessions (ESP32 + Pi Zero + HDMI)

| # | Session | Date | Machine | Duration | Topic | Concepts |
|---|---------|------|---------|----------|-------|----------|
| 22 | `682a8bf6` | Feb 16 | Laptop | — | First KVM build. ESP32 flashed with Logitech VID/PID. Pi Zero deployed with MAC spoofing. | HARDWARE |
| 23 | `47a50355` | Feb 16 | Laptop | — | Main KVM implementation (83MB JSONL). K380 "Other Devices" issue. | HARDWARE |
| 24 | `f2a58db1` | Feb 16 | Laptop | — | K380 pairing debugging continuation. 8+ hours of BT HID loops. | HARDWARE |
| 25 | `6395c7ec` | Feb 16 | Laptop | — | K380 debugging continued. Device class `0x000540` vs `0x002540`. | HARDWARE |

### Desktop Takeover

| # | Session | Date | Machine | Duration | Topic | Concepts |
|---|---------|------|---------|----------|-------|----------|
| 26 | `ec41fe43` | Feb 16 | Desktop | 5h 52m | **Desktop takeover**. Laptop forced reboot. Built sessions catalog. KVM agent prompt. Two projects clarified: (1) Windows VM streaming, (2) corporate laptop control. Rebooted both machines for kernel 6.18.7. | SHIP COMPUTER, HARDWARE |
| 27 | `61c30ea8` | Feb 16 | Desktop | — | Dotfile deployment drift + laptop audio/fingerprint fix. Not KVM-related but happened during same day. | INFRASTRUCTURE |

### First K380 Pairing Attempts (Manual)

| # | Session | Date | Machine | Duration | Topic | Concepts |
|---|---------|------|---------|----------|-------|----------|
| 28 | `ff04450c` | Feb 16 | Desktop | — | **Star Trek Dashboard started** (port 8766). K380 paired but wrong: "Other devices", no battery, immediate disconnect. User frustrated: *"why don't you operate the mouse, use display as feedback"* — birth of autonomous UI control concept. | HARDWARE, CURSOR AWARENESS |
| 29 | `26d0ad6e` | Feb 16 | Desktop | — | K380 keyboard fix. Comprehensive tools inventory. | HARDWARE |
| 30 | `84906f1d` | Feb 16 | Desktop | — | Earlier frame-diff cursor detection + multi-monitor discovery. First `cursor_detector.py` (V1: discrete screenshots, 0.16 confidence false positive from Windows UI animations). | CURSOR AWARENESS |

---

## Phase 3: Cursor Detection (Feb 16–17) — The Breakthrough

### "DID YOU LOCATE THE POINTER?"

| # | Session | Date | Machine | Duration | Topic | Concepts |
|---|---------|------|---------|----------|-------|----------|
| 31 | `2c216942` | Feb 16 | Desktop | 21m | **Birth of Lissajous approach.** Agent tried blind clicking. User: *"DID YOU LOCATE THE POINTER?"* Then described calculus curves: *"the pointer will follow mathematical curves which are sneaky to find their way across any continuous screen surface area."* | CURSOR AWARENESS |

**User directive**: "create a motion detection utility which moves the mouse, records video at good frame rate, processes the frame to find motion."

### Lissajous Implementation

| # | Session | Date | Machine | Duration | Topic | Concepts |
|---|---------|------|---------|----------|-------|----------|
| 32 | `47d0c44b` | Feb 16–17 | Desktop | 10h 9m | **Lissajous implementation**. Built `cursor_detector.py` (Lissajous math + curvature matching) and `cursor_locator.py` (live detection). **0.92 correlation** detecting cursor at (1842, 15). User: *"now find my pointer and continue with fixing the keyboard autonomously using mouse and video."* | CURSOR AWARENESS |

**Key result**: Record-then-analyze pipeline PROVEN. Live pipeline (commit `5dd8115`) FAILED (growing MKV unreliable).

### Star Trek Philosophy + Navigation

| # | Session | Date | Machine | Duration | Topic | Concepts |
|---|---------|------|---------|----------|-------|----------|
| 33 | `911b4a7d` | Feb 17 | Desktop | 12h 47m | **Star Trek experience philosophy**. Defined two experience types: "Getting Control" (Lissajous = finding own hand) and "IT Work" (declarative UI navigation = Skynet accessing a terminal). OTel StorylineLogger designed. *"make sure entire end-to-end experience is logged, and understood in a deeper than SWE coder way."* | SHIP COMPUTER, EXPERIENCE SYSTEM |
| 34 | `3095ea66` | Feb 17 | Desktop | 59m | **Navigation system built.** `WindowsNavigator` (perceive-act-verify), `StorylineLogger` (OTel integration), `pair_k380.py` (9-step experience script). Committed as `fffe09f`. | NAVIGATION, EXPERIENCE SYSTEM |

### Detection Refinement

| # | Session | Date | Machine | Duration | Topic | Concepts |
|---|---------|------|---------|----------|-------|----------|
| 35 | `f0870570` | Feb 17 | Desktop | 2h 15m | **Fix detection pipeline**. Reverted to record-then-analyze. Optimized sweep parameters (amp_x=4300, amp_y=1200). Shape filter added (`_cursor_shape_score`). Three docs created. Commits: `6eea81d`, `375c25e`. | CURSOR AWARENESS |

---

## Phase 4: CNN + Dashboard + Mesh (Feb 18) — The Big Session

| # | Session | Date | Machine | Duration | Topic | Concepts |
|---|---------|------|---------|----------|-------|----------|
| 36 | `85f2fb3f` | Feb 18 | Desktop | 7h 30m | **K380 pairing via Agent SDK**. MCP + Claude Agent SDK architecture (10 hardware tools). Anti-sleep jitter (+/-3px real movement). `k380_agent.py` + `hardware_mcp_server.py` created. ESP32 mouse not actually moving Windows cursor despite CONNECTED. | HARDWARE, NAVIGATION |
| 37 | `b1625283` | Feb 18 | Desktop | 27h 18m | **Major session — CNN pipeline + dashboard rewrite + mesh architecture.** CNN (TinyCursorNet 14K params, ~2ms inference). C daemon built (30fps V4L2, shared memory). Dashboard rewritten with dual-direction probe. 6 commits: `b052a76` through `ef14831`. **User described the mesh architecture** in voice messages: CNN is validation NOT replacement. Silhouette + CNN full-screen + jitter + CNN small-patch work as mesh. "Position comes later." | CURSOR AWARENESS, NAVIGATION |

**Key user voice messages in `b1625283`:**
1. CNN is a validation layer, NOT a replacement for motion detection
2. Motion detector must run at 30 FPS; CNN inference only in validation loop
3. CNN's learned silhouette feeds back to motion detector for shape-informed blob scoring
4. **The mesh**: silhouette scan → CNN full-screen → jitter → CNN small-patch (not a hierarchy)
5. *"I'm trying to create a starter subject computer to which I can talk and it can go out there in the world and discover things and do stuff for me"*

**Git commits in `b1625283`:**

| Commit | Description |
|--------|-------------|
| `b052a76` | CNN cursor recognition pipeline |
| `4b02037` | K380 pairing infrastructure + cursor detection knowledge |
| `0063b2f` | Remove cursor tracking from C daemon — cursor ID stays in Python |
| `0eedb37` | CNN validation fixes, CPU optimization, NVENC GPU encoding |
| `37415ba` | Fix cursor detection: clean overlay, continuous tracking, dual-direction probe |
| `ef14831` | Add secondary cursor tracking for meetings |

### Sync + Datalake

| # | Session | Date | Machine | Duration | Topic | Concepts |
|---|---------|------|---------|----------|-------|----------|
| 38 | `9545c157` | Feb 18 | Desktop | 3h 37m | **"This is gold" — sync + mesh architecture.** Content pasted from `b1625283` to test datalake real-time sync. Datalake Conversation Explorer planned (D3.js tree, SSE). Near-real-time watcher via `inotifywait`. Data loss analysis: 19K user messages empty in FTS5 because `_extract_content()` skips tool_use/tool_result. | INFRASTRUCTURE, EXPERIENCE SYSTEM |
| 39 | `c41d81e9` | Feb 18 | Desktop | 51m | Research continuation, YouTube reference. | CURSOR AWARENESS |

---

## Phase 5: Consolidation (Feb 19) — This Session

| # | Session | Date | Machine | Duration | Topic | Concepts |
|---|---------|------|---------|----------|-------|----------|
| 40 | `b1625283` (cont.) | Feb 19 | Desktop | — | **Research consolidation**. KVM dashboard validation. ESP32 BLE disconnected. 6 uncommitted changes to `kvm_dashboard.py`. User requested pooling all research into "a file or two." Created `STAR_TREK_COMPUTER_RESEARCH.md` + this file. | ALL |

**User's latest directive (Feb 19):** Pure visual cursor detection — silhouette library + trainable CNN + statistical geometry tracking. "Discover + stick."

---

## Supporting Sessions (Feb 16–17) — Not KVM-Specific

These sessions happened concurrently but are primarily about voice typing, system maintenance, or other infrastructure:

| Session | Date | Duration | Topic |
|---------|------|----------|-------|
| `022ef403` | Feb 17 | — | Voice typing full fix: Python 3.14 breakage, Deepgram, BT A2DP/HFP profile switching, xHCI USB crash recovery |
| `aecc2737` | Feb 17 | — | Waybar mic indicator fix, systemd PartOf= discovery, datalake sync gap |
| `4f888b9f` | Feb 17–18 | — | Voice typing mute detection root cause (BT HFP codec init produces digital silence) |

---

## Concept Evolution Timeline

This shows how each concept emerged and developed across sessions:

### Cursor Awareness

```
Jan 15: ASMP born (Tapo camera → generalized to cursor) .................. #20 dbad911b
Feb 16: V1 frame-diff (discrete screenshots, 0.16 false positive) ....... #30 84906f1d
Feb 16: "DID YOU LOCATE THE POINTER?" (calculus curve vision) ............ #31 2c216942
Feb 16: V2 Lissajous (0.92 correlation PROVEN) .......................... #32 47d0c44b
Feb 17: Fix: revert to record-then-analyze, shape filter ................. #35 f0870570
Feb 18: V3 C daemon (30fps V4L2, shared memory, blob extraction) ........ #37 b1625283
Feb 18: V4 CNN validation (TinyCursorNet 14K params) ..................... #37 b1625283
Feb 18: Mesh architecture described (silhouette+CNN+jitter+CNN patch) .... #37 b1625283
Feb 18: Dual-direction probe (right+down, fixes busy-screen noise) ....... #37 b1625283
Feb 18: Secondary cursor for meetings ................................... #37 b1625283
Feb 19: Pure visual detection requested ("discover + stick") ............. #40
```

### Navigation

```
Jan 9:  Amazon Shopping Agent (Playwright, browser automation) ........... #7  d0e8df59
Jan 15: Tapo camera PTZ control via Android emulator ..................... #20 dbad911b
Feb 17: "IT Work" experience type defined ................................ #33 911b4a7d
Feb 17: WindowsNavigator built (perceive-act-verify + hypotheses) ........ #34 3095ea66
Feb 18: MCP tools + Agent SDK architecture (Claude IS the vision) ........ #36 85f2fb3f
```

### Ship Computer

```
Jan 8:  Star Trek vision first articulated .............................. #4  a75e5c0b
Jan 12: "One computer instance running like Star Trek" ................... #14 f3d26b69
Jan 13: Multi-agent orchestration, machine status dashboard .............. #17 11518f1a
Feb 16: Desktop takeover, two projects clarified ......................... #26 ec41fe43
Feb 17: Experience philosophy, "Getting Control" vs "IT Work" ............ #33 911b4a7d
Feb 17: OTel StorylineLogger, L1/L2/L3 hierarchy ........................ #34 3095ea66
```

### Hardware

```
Jan 12: Bought capture card + ESP32 kit ................................. #15 2da8af04
Feb 16: ESP32 flashed, Pi Zero deployed, K380 pairing struggles ......... #22-25
Feb 16: Device class fix (0x000540 → 0x002540 for keyboard) ............. #25 6395c7ec
Feb 18: Anti-sleep: actual +/-3px movement needed (not zero-displacement) #36 85f2fb3f
Feb 18: C daemon: V4L2 capture, NVENC encoding .......................... #37 b1625283
```

---

## Plan Files

Plans represent approved architecture decisions. Cross-referenced with sessions where they were created:

| Plan File | Topic | Created In |
|-----------|-------|------------|
| `calm-booping-leaf.md` | Ship Computer — Telegram heartbeat + Agent SDK spawner | `911b4a7d` |
| `elegant-riding-harp.md` | K380 Autonomous Pairing — MCP + Agent SDK (10 tools) | `85f2fb3f` |
| `moonlit-humming-quill.md` | Mobile VM Access via Web (noVNC → PWA → Moonlight) | — |
| `mossy-greeting-goose.md` | "Locator Only" Rule — binding constraint for cursor detection | `f0870570` |
| `vast-petting-steele.md` | Workspace Bindings + Phase 3 Moonlight Streaming | — |
| `hidden-scribbling-treasure.md` | BLE pairing helper scripts + package verification | — |
| `frolicking-floating-cookie.md` | Datalake Conversation Explorer (D3.js tree, SSE) | `9545c157` |
| `sorted-tinkering-popcorn.md` | KVM Dashboard cursor detection fix | `b1625283` |

---

## Git Commit History (UI-agent, `feature/asmp-cursor-detect`)

Reverse chronological (most recent first):

| Commit | Description | Session |
|--------|-------------|---------|
| `ef14831` | Add secondary cursor tracking for meetings | `b1625283` |
| `37415ba` | Fix cursor detection: clean overlay, continuous tracking, dual-direction probe | `b1625283` |
| `0eedb37` | Fix cursor tracking pipeline + NVENC recording + CPU optimization | `b1625283` |
| `0063b2f` | Remove cursor tracking from C daemon — cursor ID stays in Python | `b1625283` |
| `edd7a4f` | Fix OFF_JPEG_ACTIVE offset mismatch | `b1625283` |
| `53f27b8` | Add daemon binary to gitignore | `b1625283` |
| `06ab504` | Add C cursor daemon (30fps frame diff, blob extraction, shm IPC) | `b1625283` |
| `4b02037` | Add K380 pairing infrastructure + cursor detection knowledge | `b1625283` |
| `b052a76` | Add CNN cursor recognition pipeline | `b1625283` |
| `375c25e` | Add adaptive detection area concept | `f0870570` |
| `6eea81d` | Fix cursor detection: revert to record-then-analyze pipeline | `f0870570` |
| `fffe09f` | Add navigation module + OTel storyline + K380 pairing experience | `3095ea66` |
| `5dd8115` | Rewrite cursor_locator: live detection (FAILED, reverted) | `47d0c44b` |
| `7ac1d44` | Rewrite cursor_detector: Lissajous curve + curvature matching | `47d0c44b` |
| `c6f76a7` | Add video recording control to HDMISensor | — |
| `a88b27b` | Add cursor locate + verify-before-click protocol | — |
| `e949550` | Add frame-diff cursor detector (V1) | `84906f1d` |
| `ec44f24` | Add HDMI capture sensor module | — |
| `e2f801a` | Document ASMP cursor detection application | `dbad911b` |
| `b0b4606` | Document multi-monitor ESP32 mouse gotcha | — |

---

## How to Resume

```bash
# Resume any session
claude --resume <session-id>

# Key sessions to resume for specific work:
claude --resume b1625283   # CNN + dashboard + mesh architecture (most context)
claude --resume 47d0c44b   # Lissajous implementation (0.92 proof)
claude --resume 911b4a7d   # Star Trek philosophy + OTel
claude --resume 3095ea66   # Navigation system + pair_k380.py
claude --resume f0870570   # Detection fix + shape filter
claude --resume 9545c157   # Datalake explorer + mesh explanation
```

**Note**: `--resume` only works on the same machine where the session ran. Laptop sessions cannot be resumed from desktop and vice versa. All Feb 16+ sessions ran on desktop.
