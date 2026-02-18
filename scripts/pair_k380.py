#!/usr/bin/env python3
"""K380 Bluetooth Pairing Experience.

Navigates Windows 11 Settings to pair a Logitech K380 keyboard via
Bluetooth. Uses the WindowsNavigator perceive→act→verify loop with
recovery hypotheses for resilient autonomous navigation.

Prerequisites:
    - ESP32 BLE HID mouse connected via USB serial
    - HDMI capture card showing the Windows desktop
    - OpenRouter API key set (OPENROUTER_API_KEY env var)
    - K380 keyboard physically available (user puts it in pairing mode
      during step 8)

Usage:
    uv run python scripts/pair_k380.py
    uv run python scripts/pair_k380.py --serial /dev/ttyUSB1 --device /dev/video2
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.storyline_logger import StorylineLogger
from src.hardware.cursor_locator import CursorLocator
from src.hardware.esp32_mouse import ESP32Mouse
from src.hardware.experience_logger import ExperienceLogger
from src.hardware.hdmi_sensor import HDMISensor
from src.navigation import (
    NavigationStep,
    StepAction,
    StepOutcome,
    WindowsNavigator,
)
from src.vision.llm_vision import LLMVision


# ── Declarative navigation steps ─────────────────────────────────

PAIR_K380_STEPS = [
    NavigationStep(
        name="Open Start Menu",
        target="Windows Start button or Windows logo in the taskbar at the bottom left",
        action=StepAction.CLICK,
        verify="Start menu is open showing pinned apps or search bar",
        timeout=5.0,
        retries=3,
        post_delay=1.0,
    ),
    NavigationStep(
        name="Open Settings",
        target="Settings gear icon in the Start menu, or type 'Settings' in the search",
        action=StepAction.CLICK,
        verify="Windows Settings window is open",
        timeout=5.0,
        retries=3,
        post_delay=1.5,
    ),
    NavigationStep(
        name="Navigate to Bluetooth",
        target="'Bluetooth & devices' option in the Settings sidebar or main panel",
        action=StepAction.CLICK,
        verify="Bluetooth & devices settings page is visible",
        timeout=5.0,
        retries=3,
        post_delay=1.0,
    ),
    NavigationStep(
        name="Find stale K380 entry",
        target="'Logitech K380' or 'K380' device entry in the Bluetooth devices list",
        action=StepAction.CLICK,
        verify="K380 device details or options are shown",
        timeout=3.0,
        retries=2,
        optional=True,  # May not exist if first pairing
    ),
    NavigationStep(
        name="Remove stale K380",
        target="'Remove device' button or 'Disconnect' option for the K380",
        action=StepAction.CLICK,
        verify="K380 is no longer in the paired devices list",
        timeout=3.0,
        retries=2,
        post_delay=1.5,
        optional=True,  # Only needed if stale entry exists
    ),
    NavigationStep(
        name="Add Bluetooth device",
        target="'Add device' button with a plus icon on the Bluetooth & devices page",
        action=StepAction.CLICK,
        verify="'Add a device' dialog or wizard is open showing device type options",
        timeout=5.0,
        retries=3,
        post_delay=1.0,
    ),
    NavigationStep(
        name="Select Bluetooth type",
        target="'Bluetooth' option (first option, mice keyboards etc.) in the Add a device dialog",
        action=StepAction.CLICK,
        verify="Device discovery is in progress, showing 'Searching for devices' or a spinning indicator",
        timeout=5.0,
        retries=3,
        post_delay=2.0,
    ),
    NavigationStep(
        name="Select K380 from discovery",
        target="'Logitech K380' or 'K380 Keyboard' entry in the Bluetooth device discovery list",
        action=StepAction.CLICK,
        verify="Pairing in progress, PIN entry prompt, or 'Connecting' status shown",
        timeout=30.0,  # Long timeout — user needs to put K380 in pairing mode
        retries=5,
        pre_delay=2.0,
        post_delay=2.0,
    ),
    NavigationStep(
        name="Confirm pairing",
        target="",  # No click needed — user types PIN on K380
        action=StepAction.WAIT,
        verify="K380 shows as 'Connected' or 'Paired' in Bluetooth devices",
        timeout=30.0,  # User types the PIN code on K380
    ),
]


def main():
    parser = argparse.ArgumentParser(description="K380 Bluetooth Pairing Experience")
    parser.add_argument("--serial", default="/dev/ttyUSB0", help="ESP32 serial port")
    parser.add_argument("--device", default="/dev/video0", help="HDMI capture device")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Vision model for OpenRouter (Gemini 2.5 Flash)")
    parser.add_argument("--log-dir", default="logs/experiences", help="Experience log directory")
    args = parser.parse_args()

    log_dir = PROJECT_ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    print("Initializing hardware...")
    mouse = ESP32Mouse(args.serial, screen_width=1920, screen_height=1080)
    sensor = HDMISensor(args.device)
    locator = CursorLocator(mouse, sensor)
    vision = LLMVision(model=args.model)

    # Initialize logging
    storyline = StorylineLogger(experience_dir=log_dir)
    experience = ExperienceLogger(log_dir=str(log_dir), storyline_logger=storyline)

    # Create navigator
    navigator = WindowsNavigator(
        mouse=mouse,
        sensor=sensor,
        locator=locator,
        vision=vision,
        logger=storyline,
    )

    # Start anti-sleep jitter (real ±3px movement every 30s to prevent Windows sleep)
    print("Starting anti-sleep jitter (±3px every 30s)...")
    mouse.start_heartbeat()
    mouse.start_anti_sleep()

    # Run the experience
    print(f"\nStarting K380 pairing experience...")
    print(f"Steps: {len(PAIR_K380_STEPS)}")
    print(f"Log dir: {log_dir}")
    print(f"Vision model: {args.model}")
    print()

    try:
        results = navigator.run_experience("pair_k380", PAIR_K380_STEPS)

        # Report
        print("\n" + "=" * 60)
        print("EXPERIENCE REPORT")
        print("=" * 60)
        for i, result in enumerate(results):
            status = {
                StepOutcome.SUCCESS: "[OK]",
                StepOutcome.YELLOW_ALERT: "[!!]",
                StepOutcome.ELEMENT_NOT_FOUND: "[--]",
                StepOutcome.CLICK_NO_EFFECT: "[XX]",
                StepOutcome.UNEXPECTED_SCREEN: "[??]",
                StepOutcome.TIMEOUT: "[TT]",
            }.get(result.outcome, "[??]")

            optional = " (optional)" if result.step.optional else ""
            print(f"  {status} Step {i + 1}: {result.step.name}{optional}"
                  f"  [{result.elapsed_s:.1f}s, {result.attempts} attempts]")
            if result.error:
                print(f"       Error: {result.error}")
            if result.hypotheses:
                for h in result.hypotheses:
                    print(f"       Hypothesis: {h['name']} "
                          f"(confidence={h['confidence']:.2f})")

        successes = sum(
            1 for r in results
            if r.outcome in (StepOutcome.SUCCESS, StepOutcome.YELLOW_ALERT)
        )
        print(f"\nResult: {successes}/{len(results)} steps succeeded")

        # Save narrative
        narrative_path = storyline.save_narrative()
        print(f"Narrative saved: {narrative_path}")

    except Exception as e:
        print(f"\nExperience failed: {e}")
        storyline.save_narrative()
        raise

    finally:
        storyline.shutdown()
        experience.close()
        vision.close()
        mouse.close()


if __name__ == "__main__":
    main()
