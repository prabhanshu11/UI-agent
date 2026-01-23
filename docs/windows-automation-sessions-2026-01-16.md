# Windows Automation & Related Work - Comprehensive Conversation Discovery

**Date Created**: 2026-01-16
**Source**: Claude Code conversation recall across laptop and desktop sessions

---

## Search Methodology
- **Layer 1 Keywords**: windows, automation, automate, UI-agent, pyautogui, selenium, playwright, rdp, remote desktop, moonlight, sunshine, streaming
- **Layer 2 Keywords** (discovered from Layer 1): capture card, hdmi capture, usb capture, pibox, claude-computer, claude-ui-agent, star trek, lcars, tapo, c210, camera, rtsp, onvif, esp32, bt hid, bluetooth hid, datalake, vdi, citrix, android emulator, adb, appium, agent sdk

## Executive Summary

The Windows automation work spans multiple interconnected projects forming a "Star Trek Computer" vision:
1. **Remote Desktop Control**: Moonlight/Sunshine streaming between desktop and laptop
2. **USB HDMI Capture**: Capturing Windows laptop screen for automation
3. **UI Agent Framework**: Vision-based UI automation using screenshots + AI
4. **Android Emulator**: Controlling mobile apps (Tapo camera, shopping)
5. **ESP32 BT HID**: Hardware keyboard/mouse input to Windows
6. **Claude Agent SDK**: Orchestrating multiple automation agents

---

## LAPTOP CONVERSATIONS (12 sessions)

### Core Windows Automation Sessions

---

#### Session 1: `6aae4902-3e22-4a6c-a3f5-7685b589e1c2`
**Date**: 2026-01-10 15:11:02
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: windows, automation, moonlight, sunshine, UI-agent, esp32, bt hid, datalake

**Summary**: Major Moonlight/Sunshine remote desktop setup session. Configured streaming between desktop and laptop. Discussed keyboard shortcuts, fullscreen mode, focus handling. Sunshine credentials: `prabhanshu:iamapantar`. Worked on keyboard control pass-through issues where laptop shortcuts were triggering on laptop instead of remote desktop.

**Key Discoveries**:
- Moonlight shortcuts: `Alt+Enter` for fullscreen, `Ctrl+Alt+Shift+Q` to quit
- `moonlight-toggle` script for starting/stopping
- Need for robust shortcut handling in fullscreen mode
- Cross-device streaming control via local-bootstrapping

**Resume**: `claude --resume 6aae4902-3e22-4a6c-a3f5-7685b589e1c2`

---

#### Session 2: `f3d26b69-3f4b-4aef-95ce-c1a991e932b8`
**Date**: 2026-01-12 17:18:22
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: windows, automation, capture card, hdmi, UI-agent, vdi, esp32, datalake, agent sdk

**Summary**: Major USB capture card integration and UI-Agent architecture planning. Connected Pibox USB HDMI capture to view Windows laptop screen. Discussed merging multiple repos (UI-agent, claude-computer, usb-webcam-api) into unified architecture. Vision of Star Trek computer assistant.

**Key Concepts Discussed**:
- USB capture card for Windows laptop screen capture
- UI-agent as main repo for all automation
- Directory structure: hardware integrations as modules
- Datalake as central data storage for all projects
- Claude Agent SDK for multi-agent orchestration
- "One computer instance running like Star Trek"

**Resume**: `claude --resume f3d26b69-3f4b-4aef-95ce-c1a991e932b8`

---

#### Session 3: `2da8af04-c41b-4acc-bfd3-0e8d1982e232`
**Date**: 2026-01-12 17:14:54
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: capture card, claude-ui-agent, claude-computer, esp32, agent sdk

**Summary**: Reorganized automation repos and discussed project architecture. Reviewed claude-computer and claude-ui-agent repos. Bought capture card and ESP32 kit for BT HID.

**Resume**: `claude --resume 2da8af04-c41b-4acc-bfd3-0e8d1982e232`

---

#### Session 4: `29d2d22f-bf78-4e9f-a957-b2afc4898dad`
**Date**: 2026-01-11 12:20:21
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: UI-agent, selenium, agent sdk

**Summary**: Creating HTML MCP server and working with Google Forms automation using Claude Agent SDK and Selenium. Filling out forms via UI agent.

**Resume**: `claude --resume 29d2d22f-bf78-4e9f-a957-b2afc4898dad`

---

#### Session 5: `dbad911b-5595-4cc9-bbee-36554a3850e0`
**Date**: 2026-01-15 16:18:16
**Project**: /home/prabhanshu
**Keywords Hit**: tapo, camera, claude-computer, agent sdk

**Summary**: Tapo C210 camera automation using Android emulator. Pan/tilt control through Tapo app. Night vision settings. Camera credentials: `prabhanshu:iamapantar`.

**Key Technical Details**:
- Camera IP discovery via router DHCP or Tapo app
- RTSP/ONVIF access attempted
- Used `/camera-app` skill
- Gemini image model for desk detection

**Resume**: `claude --resume dbad911b-5595-4cc9-bbee-36554a3850e0`

---

#### Session 6: `6f47a83e-bb59-45a4-9856-956cdbf01a83`
**Date**: 2026-01-14 17:59:28
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: windows, automation, capture card, agent sdk

**Summary**: Searching for which device (laptop/desktop) was used for HDMI capture and Windows control via Claude Agent SDK.

**Resume**: `claude --resume 6f47a83e-bb59-45a4-9856-956cdbf01a83`

---

#### Session 7: `77e6cee1-4926-4f7b-8ae3-c30e25b3da49`
**Date**: 2026-01-11 14:21:47
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: automation

**Summary**: Creating interactive document from business cases conversation. Used Loki skill. ROI calculator for solo engineer automating tasks. Used OpenRouter API with DeepSeek model.

**Resume**: `claude --resume 77e6cee1-4926-4f7b-8ae3-c30e25b3da49`

---

#### Session 8: `b9b4f75a-126f-421f-8842-21721f2ee04c`
**Date**: 2026-01-16 02:29:33
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: windows automation

**Summary**: Current session - comprehensive conversation recall for Windows automation work.

**Resume**: `claude --resume b9b4f75a-126f-421f-8842-21721f2ee04c`

---

#### Session 9: `11518f1a-b67f-42f5-88ee-355d305b7746`
**Date**: 2026-01-13 19:48:33
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: moonlight, streaming, datalake, agent sdk

**Summary**: Multi-agent orchestration session. Fixed Claude Code memory issues. Spawned subagents for voice typing and datalake work. Dashboard with machine status indicators. FIDO key authentication workflow.

**Key Architecture**:
- Subagent1: Voice typing on desktop
- Subagent2: Datalake on laptop (primary citizen due to battery backup)
- Control dashboard showing machine status (green/grey for accessible, red/yellow for services)
- Data resilience across machines

**Resume**: `claude --resume 11518f1a-b67f-42f5-88ee-355d305b7746`

---

#### Session 10: `b0ca7007-ad8b-4f93-84c8-cf192a305e6e`
**Date**: 2026-01-11 16:00:03
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: moonlight, streaming

**Summary**: Created `moonlight-broadcast` command for stopping moonlight across all devices. Discussed shortcuts and fullscreen control.

**Resume**: `claude --resume b0ca7007-ad8b-4f93-84c8-cf192a305e6e`

---

#### Session 11: `68b19fae-423d-4390-9557-9c4cc2386382`
**Date**: 2026-01-05 15:19:59
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: streaming

**Summary**: Voice typing (hyprwhspr) troubleshooting. Created notification system with sudo-notify script.

**Resume**: `claude --resume 68b19fae-423d-4390-9557-9c4cc2386382`

---

#### Session 12: `9541b458-e3b5-48fb-bfe6-e1623dd4134d`
**Date**: 2026-01-04 13:38:50
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: streaming

**Summary**: Initial git/GitHub setup. Personal website deployment. GitHub Actions configuration. First exploration of Programs folder structure.

**Resume**: `claude --resume 9541b458-e3b5-48fb-bfe6-e1623dd4134d`

---

### Supporting Sessions (Layer 2 Discoveries)

#### Session 13: `a75e5c0b-2067-4ef0-a923-86e67414b2f8`
**Date**: 2026-01-08 15:19:44
**Project**: /home/prabhanshu
**Keywords Hit**: claude-computer, star trek, lcars

**Summary**: Claude computer / Star Trek vision planning. Researched Claude Agent SDK via YouTube video. Local voice/image models discussion. "Consciousness" for context tracking.

**Resume**: `claude --resume a75e5c0b-2067-4ef0-a923-86e67414b2f8`

---

#### Session 14: `7c76adf9-e989-4b75-b301-8a95bb32f420`
**Date**: 2026-01-04 18:45:36
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: android emulator, adb

**Summary**: Early Android emulator work for UI automation.

**Resume**: `claude --resume 7c76adf9-e989-4b75-b301-8a95bb32f420`

---

#### Session 15: `cbbb26c8-8ef4-4df4-9dd6-fb7d077695e4`
**Date**: 2026-01-13 22:42:56
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: agent sdk

**Summary**: Claude Agent SDK usage session.

**Resume**: `claude --resume cbbb26c8-8ef4-4df4-9dd6-fb7d077695e4`

---

#### Session 16: `1e94d81d-dbf1-4377-af87-583c71ec362e`
**Date**: 2026-01-11 16:04:23
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: vdi, agent sdk

**Summary**: VDI automation discussion. Business work automation concept.

**Resume**: `claude --resume 1e94d81d-dbf1-4377-af87-583c71ec362e`

---

## DESKTOP CONVERSATIONS (27 sessions)

### Key Desktop Sessions

---

#### Desktop Session 1: `1f9f9926-0fe4-4761-abbb-e32543d1ac0d`
**Date**: 2026-01-08 11:30:43
**Project**: /home/prabhanshu
**Keywords Hit**: windows, automation, moonlight, sunshine, capture card, hdmi, tapo, camera, esp32, agent sdk

**Summary**: Major Tapo camera app + Android emulator session. Repeated issues with Aurora Store installs and emulator crashes. Key insight: Use external APK files instead of app stores. Vision model for UI navigation. Created progress_actual.md workflow.

**Key Learnings**:
- DO NOT use Aurora Store - download APK externally
- Emulator crashes due to video rendering (saw feed for split second)
- Need better logging for loop detection
- UI explorations should be documented with screenshots

**Resume**: `claude --resume 1f9f9926-0fe4-4761-abbb-e32543d1ac0d` (run from desktop)

---

#### Desktop Session 2: `d0e8df59-a357-41e5-b725-df80a6075991`
**Date**: 2026-01-09 09:32:25
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: windows, automation, UI-agent, agent sdk

**Summary**: Amazon Shopping Agent development. Created cart management system. Browser-based automation with Playwright. Address verification protocol due to Amazon address bugs.

**Key Features**:
- Web UI for Amazon cart management
- Connectors for Swiggy, Blinkit, Uber Eats (planned)
- Cancelled order tracking
- Address verification on each cart add
- Skill.md for documenting AI navigation learnings

**Resume**: `claude --resume d0e8df59-a357-41e5-b725-df80a6075991` (run from desktop)

---

#### Desktop Session 3: `e2c9ece8-47f3-475b-8bac-52205ced0c3f`
**Date**: 2026-01-10 19:08:54
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: windows, automation, moonlight, sunshine, UI-agent, tapo, camera, esp32, agent sdk

**Summary**: Tapo camera UI automation continuation. Pan control working. Black screen issue - video shows for split second then goes black. App crashes/not responding frequently.

**Key Findings**:
- Pan control WORKS (verified on phone)
- Video feed shows for 1 frame then black
- Hypothesis: Emulator backend not set up for video OR app detecting emulator
- "Cameras" section shows stale screenshot
- Need to test older app version

**Resume**: `claude --resume e2c9ece8-47f3-475b-8bac-52205ced0c3f` (run from desktop)

---

#### Desktop Session 4: `5d2abcd1-f5b6-4df1-ab44-6b33465534b8`
**Date**: (not retrieved, from search)
**Keywords Hit**: capture card, hdmi, UI-agent, claude-computer

**Summary**: Database setup for Claude Code conversations. Datalake integration. Widget development. Selenium-based testing agent. Email logger data model planning.

**Key Architecture**:
- Datalake as central database
- Claude conversations database
- Widget pop-up UI
- Primary services on laptop (battery backup)

**Resume**: `claude --resume 5d2abcd1-f5b6-4df1-ab44-6b33465534b8` (run from desktop)

---

#### Desktop Session 5: `5fc36dcc-a09d-40fa-8e32-2baaed85f52e`
**Date**: (not retrieved, from search)
**Keywords Hit**: moonlight, sunshine, capture card, hdmi, tapo, camera, esp32, agent sdk

**Summary**: Tapo camera scaffolding work. USB webcam API exploration. OpenRouter API for vision intelligence. App login persistence issues.

**Key Technical Work**:
- UI control + capture approach for camera
- Framework for discovering manufacturer changes
- Screenshot intelligence with local GPU
- ADB for keyboard/input issues

**Resume**: `claude --resume 5fc36dcc-a09d-40fa-8e32-2baaed85f52e` (run from desktop)

---

#### Desktop Session 6: `cf7a8f47-7d50-475d-9544-defab46772d8`
**Date**: 2026-01-14 15:17:42
**Project**: /home/prabhanshu/Programs
**Keywords Hit**: windows, automation, moonlight, sunshine

**Summary**: Windows automation continuation.

**Resume**: `claude --resume cf7a8f47-7d50-475d-9544-defab46772d8` (run from desktop)

---

#### Desktop Session 7: `f268982d-5400-4c8e-aa33-412cadee1514`
**Date**: (not retrieved, from search)
**Keywords Hit**: windows, automation, tapo, camera, esp32, agent sdk

**Summary**: TP-Link camera repo continuation. Multi-agent orchestration via terminal windows. UI exploration documentation.

**Resume**: `claude --resume f268982d-5400-4c8e-aa33-412cadee1514` (run from desktop)

---

### Additional Desktop Sessions (Brief)

| Session ID | Date | Keywords |
|------------|------|----------|
| `4af18670-8a99-40bf-9967-2cb2f2af75e0` | 2026-01-08 01:56:19 | windows, automation, moonlight |
| `01d29e33-22ab-43ac-8e39-090a895afc07` | - | windows, automation |
| `5feeb5c5-6317-4c70-b7ac-e9fbea359bac` | - | windows, automation |
| `ae6b1c69-3b43-4a3a-95bc-ea0ee9ba1c91` | - | windows, automation |
| `eb6dd91e-7219-4239-9487-bfe2326197b8` | - | windows, automation |
| `5c9f0d1f-f0e1-4293-a66c-e2e26c6a3e99` | - | moonlight, sunshine, UI-agent |
| `6cab20c5-dbd7-41e2-bd77-eebc552e31e6` | - | moonlight, sunshine, UI-agent |
| `6e25e808-4d7a-4245-a28f-c4ed4adda9ab` | - | moonlight, sunshine, UI-agent |
| `e1b3ccda-8c85-4c3c-bc19-ef50d41dc971` | - | moonlight, sunshine, UI-agent |
| `f2602fec-067e-4524-94b3-de55cbe5ff16` | - | moonlight, sunshine, UI-agent |
| `1d10fbb2-ca25-499b-9184-91f51cc2ef88` | - | esp32, bt hid |
| `230f8714-09c1-4617-8840-4506b376b1cc` | - | esp32, camera |
| `2974913e-d39a-4924-a093-27e03396085b` | - | esp32, camera |
| `570425ed-6932-4283-bb88-ac390fed72de` | - | esp32, camera |
| `641872c2-2ba0-4c1f-8882-13a6729e09f3` | - | esp32, camera |
| `87da3156-3d63-470c-b7a5-7f5f96a1af0a` | - | esp32, camera |
| `c97d52f6-c91f-497e-b405-edccc6e674fc` | - | esp32, camera |
| `d3d60a35-2fdb-4470-b9fa-e10bbad09946` | - | esp32, camera |
| `ddd15f05-5dc5-4f90-804e-5eaf0e18e3e4` | - | esp32, camera |
| `4acad5d4-ada7-464e-99d6-d202a2d65e29` | - | windows, automation, esp32 |

---

## REPO CONNECTIONS

```
                    ┌─────────────────────────────────────────┐
                    │         STAR TREK COMPUTER              │
                    │         (claude-computer)               │
                    └──────────────┬──────────────────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         │                         │                         │
         ▼                         ▼                         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   UI-Agent      │    │   local-        │    │   datalake      │
│   (main repo)   │    │   bootstrapping │    │   (central DB)  │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
    ┌────┴────┐            ┌────┴────┐           ┌────┴────┐
    │ Vision  │            │ Moonlight│           │ Claude  │
    │ Module  │            │ Streaming│           │ History │
    └────┬────┘            └────┬────┘           └─────────┘
         │                      │
┌────────┴────────┐    ┌───────┴───────┐
│ tapo-c210-      │    │ USB HDMI      │
│ monitor         │    │ Capture       │
│ (Android emu)   │    │ (Windows)     │
└─────────────────┘    └───────┬───────┘
                               │
                       ┌───────┴───────┐
                       │ ESP32 BT HID  │
                       │ (Keyboard)    │
                       └───────────────┘
```

---

## KEY TECHNICAL DECISIONS

1. **UI-agent as main repo** for all automation
2. **datalake as central database** for all projects
3. **local-bootstrapping syncs** desktop/laptop configs
4. **Moonlight/Sunshine** for remote desktop streaming
5. **USB HDMI capture** for Windows laptop screen
6. **ESP32 BT HID** for hardware keyboard input
7. **Android emulator** for mobile app automation (Tapo, shopping)
8. **Claude Agent SDK** for multi-agent orchestration

---

## DISCOVERED KEYWORDS (for future searches)

### Layer 1 (Initial)
windows, automation, automate, UI-agent, pyautogui, selenium, playwright, rdp, remote desktop, moonlight, sunshine, streaming

### Layer 2 (Discovered)
capture card, hdmi capture, usb capture, pibox, claude-computer, claude-ui-agent, star trek, lcars, tapo, c210, camera, rtsp, onvif, esp32, bt hid, bluetooth hid, datalake, vdi, citrix, android emulator, adb, appium, agent sdk, shopping agent, aurora store, apk, omarchy-voice-typing, hyprwhspr, local-bootstrapping

### Potential Layer 3 (for deeper search)
openrouter, deepseek, gemini, skill.md, progress_actual.md, UI-exploration, testing bench, widget, dashboard, control panel, jabra, fido key, notifications, ghostty, ram, memory, token optimization
