#!/usr/bin/env python3
"""K380 Bluetooth Pairing Agent — Claude Agent SDK + Stdio MCP Hardware Tools.

Spawns a Claude agent that can see the Windows screen (via HDMI capture),
control the mouse (via ESP32 BT HID), and type (via Pi Zero keyboard API).
The agent uses its own vision to navigate Windows UI and pair the K380.

No OpenRouter, no Gemini, no separate API keys — uses Claude subscription via OAuth.
Hardware is exposed via a stdio MCP server (hardware_mcp_server.py).

Usage:
    cd ~/Programs/UI-agent && uv run python scripts/k380_agent.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Agent SDK refuses to run inside another Claude Code session.
os.environ.pop("CLAUDECODE", None)

PROJECT_ROOT = Path(__file__).parent.parent

from claude_agent_sdk import (
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    query,
)

# ── System Prompt ───────────────────────────────────────────────

SYSTEM_PROMPT = """You are a UI automation agent controlling a Windows 11 laptop.

## Hardware Setup
- You see the Windows screen through an HDMI capture card (1920x1080)
- You control the mouse via an ESP32 Bluetooth HID device
- You can type via a Raspberry Pi Zero 2W Bluetooth keyboard server
- Screenshot is 1280x720, mouse coords are 1920x1080 — multiply screenshot coords by 1.5

## Your Task
Pair the Logitech K380 keyboard to Windows via Bluetooth settings.

## Step-by-Step Approach
1. capture_screenshot to see current state
2. Navigate to Settings > Bluetooth & devices
3. Check if a stale K380 entry exists — if so, remove it first
4. Click "Add device" > "Bluetooth"
5. Wait for K380 to appear in discovery (it should already be in pairing mode)
6. Click K380 to pair
7. Wait for pairing to complete
8. Verify the pairing is correct

## Perceive-Act-Verify Loop
ALWAYS follow this cycle:
1. **Perceive**: capture_screenshot to see the screen
2. **Decide**: Analyze what you see, identify the element to interact with
3. **Act**: mouse_move_to the element, then mouse_click
4. **Wait**: wait 0.5-1.5 seconds for the UI to respond
5. **Verify**: capture_screenshot again to confirm the action worked
6. If it didn't work, try again or adjust approach

## Coordinate Mapping
- Screenshot is 1280x720, mouse grid is 1920x1080
- Element at (X, Y) in screenshot → mouse_move_to(X * 1.5, Y * 1.5)
- Example: element at (400, 300) in screenshot → mouse_move_to(600, 450)

## Pairing Verification Checklist
ALL must be TRUE:
- K380 appears under "Input" section (NOT "Other devices")
- Shows keyboard icon (not generic BT icon)
- Shows battery percentage
- Shows "Connected" status (not just "Paired")
- pi_keyboard_health returns bluetooth_connected: true
- type_text("test123") produces "test123" on screen (verify via screenshot)

## Recovery
- Check pi_keyboard_health for Pi-side status
- Use Bash: ssh pi@10.55.0.2 'journalctl -u bt-keyboard -n 30' for Pi logs
- Remove stale pairing on Windows before re-pairing
- Restart Pi BT: ssh pi@10.55.0.2 'sudo systemctl restart bluetooth bt-keyboard'

## Important
- NEVER skip verification — always screenshot to confirm
- Use locate_cursor if position seems wrong
- Be patient — BT discovery can take 10-30 seconds
- K380 should already be in pairing mode
"""

# ── Main ────────────────────────────────────────────────────────


async def main():
    print("=" * 60)
    print("K380 PAIRING AGENT — Claude Agent SDK + MCP")
    print("=" * 60)

    # Path to the stdio MCP server
    mcp_server_path = str(PROJECT_ROOT / "scripts" / "hardware_mcp_server.py")

    # All hardware MCP tools
    hw_tools = [
        "mcp__hw__capture_screenshot",
        "mcp__hw__mouse_move",
        "mcp__hw__mouse_move_to",
        "mcp__hw__mouse_click",
        "mcp__hw__get_cursor_position",
        "mcp__hw__locate_cursor",
        "mcp__hw__type_text",
        "mcp__hw__pi_keyboard_health",
        "mcp__hw__scroll",
        "mcp__hw__wait",
    ]

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={
            "hw": {
                "command": str(PROJECT_ROOT / ".venv" / "bin" / "python"),
                "args": [mcp_server_path],
            },
        },
        allowed_tools=hw_tools + ["Bash", "Read"],
        permission_mode="bypassPermissions",
        max_turns=50,
        cwd=str(PROJECT_ROOT),
    )

    print(f"\nMCP server: {mcp_server_path}")
    print(f"Tools: {len(hw_tools)} hardware + Bash + Read")
    print("\nLaunching Claude agent...")
    print("The agent will navigate Windows UI to pair the K380 keyboard.")
    print("=" * 60 + "\n")

    try:
        async for message in query(
            prompt="Pair the Logitech K380 keyboard to Windows via Bluetooth. Start by taking a screenshot to see the current state of the screen.",
            options=options,
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"\n{block.text}")
                    elif isinstance(block, ToolUseBlock):
                        # Don't print full screenshot data
                        input_str = str(block.input)
                        if len(input_str) > 200:
                            input_str = input_str[:200] + "..."
                        print(f"\n>> {block.name}({input_str})")
            elif isinstance(message, ResultMessage):
                print(f"\n{'=' * 60}")
                print(f"Agent finished: {message.subtype}")
                print(f"Turns: {message.num_turns}, Duration: {message.duration_ms/1000:.1f}s")
                if message.total_cost_usd:
                    print(f"Cost: ${message.total_cost_usd:.4f}")
                if message.result:
                    print(f"Result: {message.result}")
                print("=" * 60)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\n\nAgent error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
