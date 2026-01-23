#!/usr/bin/env python3
"""L2 Summarizer - Timeline Aggregation and Insight Extraction.

Reads L3 analyzer outputs (individual keyframe descriptions) and produces
5-minute timeline summaries with extracted key insights.

Architecture:
- Reads batch files from agents/analyzers/*.jsonl
- Groups descriptions by 5-minute windows
- Uses Claude to summarize each window into key activities
- Writes REAL insights to storylines/knowledge.jsonl
- Reads task context from agents/CONTROL.md
"""

import os
import sys
import json
import subprocess
import time
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

# Paths
UI_AGENT_DIR = Path(__file__).parent.parent.parent
EXPERIENCES_DIR = UI_AGENT_DIR / "experiences"
ANALYZERS_DIR = UI_AGENT_DIR / "agents" / "analyzers"
CONTROL_FILE = UI_AGENT_DIR / "agents" / "CONTROL.md"
STATE_FILE = UI_AGENT_DIR / "agents" / "summarizer" / "state.json"

# Config
SUMMARY_WINDOW_MINUTES = 5


def get_task_context() -> str:
    """Read current task context from CONTROL.md."""
    if CONTROL_FILE.exists():
        content = CONTROL_FILE.read_text()
        # Extract the task section
        if "## Current Task Context" in content:
            start = content.find("## Current Task Context")
            end = content.find("## Agent Status", start)
            if end == -1:
                end = len(content)
            return content[start:end].strip()
    return "No task context available"


def load_l3_analyses() -> list[dict]:
    """Load all L3 analysis results."""
    analyses = []

    if not ANALYZERS_DIR.exists():
        return analyses

    for batch_file in sorted(ANALYZERS_DIR.glob("*.jsonl")):
        try:
            for line in batch_file.read_text().strip().split('\n'):
                if line:
                    entry = json.loads(line)
                    analyses.append(entry)
        except Exception as e:
            print(f"Error reading {batch_file}: {e}")

    return analyses


def parse_keyframe_time(keyframe_name: str) -> Optional[datetime]:
    """Extract timestamp from keyframe filename.

    Format: keyframe_NNNN_HHMMSS.png
    """
    match = re.search(r'keyframe_\d+_(\d{6})\.png', keyframe_name)
    if match:
        time_str = match.group(1)
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            # Parse HHMMSS
            h = int(time_str[0:2])
            m = int(time_str[2:4])
            s = int(time_str[4:6])
            return datetime.strptime(f"{today} {h:02d}:{m:02d}:{s:02d}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return None


def group_by_time_window(analyses: list[dict], window_minutes: int = 5) -> dict[str, list[dict]]:
    """Group analyses into time windows."""
    windows = defaultdict(list)

    for analysis in analyses:
        kf_time = parse_keyframe_time(analysis.get("keyframe", ""))
        if kf_time:
            # Round down to window start
            window_start = kf_time.replace(
                minute=(kf_time.minute // window_minutes) * window_minutes,
                second=0,
                microsecond=0
            )
            window_key = window_start.strftime("%H:%M")
            windows[window_key].append(analysis)

    return dict(windows)


def get_processed_windows() -> set[str]:
    """Get windows that have already been summarized."""
    processed = set()
    today = datetime.now().strftime("%Y-%m-%d")
    knowledge_file = EXPERIENCES_DIR / today / "storylines" / "knowledge.jsonl"

    if knowledge_file.exists():
        try:
            for line in knowledge_file.read_text().strip().split('\n'):
                if line:
                    entry = json.loads(line)
                    # Check if it's an L2 summary
                    if entry.get("metadata", {}).get("source") == "L2_summarizer":
                        window = entry.get("metadata", {}).get("time_window")
                        if window:
                            processed.add(window)
        except Exception:
            pass

    return processed


def summarize_window(window_key: str, descriptions: list[str], task_context: str) -> Optional[str]:
    """Use Claude to analyze a time window and identify KEY MOMENTS."""
    descriptions_text = "\n".join(f"{i+1}. {d}" for i, d in enumerate(descriptions))

    prompt = f'''You are analyzing a series of screenshots from a user's work session.

CURRENT TASK CONTEXT:
{task_context}

SCREENSHOTS FROM {window_key} (5-minute window) - {len(descriptions)} frames:
{descriptions_text}

Analyze these observations and provide:

1. **KEY MOMENTS** (most important frames - identify by number):
   - Application launches or switches
   - Data entries or form submissions
   - Error messages or warnings
   - Significant UI state changes
   - File operations (open, save, upload, download)

2. **ACTIVITY SUMMARY**: What the user accomplished in this window (2-3 sentences)

3. **WORKFLOW INSIGHT**: Any patterns, inefficiencies, or notable behaviors

Format your response as:
KEY MOMENTS: [list frame numbers and why they're key]
SUMMARY: [your activity summary]
INSIGHT: [workflow observation]'''

    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(UI_AGENT_DIR)
        )

        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()

    except Exception as e:
        print(f"Error summarizing window {window_key}: {e}")

    return None


def write_insight(window_key: str, summary: str, description_count: int):
    """Write insight to knowledge storyline."""
    today = datetime.now().strftime("%Y-%m-%d")
    storylines_dir = EXPERIENCES_DIR / today / "storylines"
    storylines_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "storyline": "knowledge",
        "event_type": "timeline_summary",
        "content": summary,
        "metadata": {
            "source": "L2_summarizer",
            "time_window": window_key,
            "description_count": description_count,
            "applies_to": ["workflow", "activity"],
            "confidence": 0.85
        }
    }

    # Write to knowledge.jsonl
    knowledge_file = storylines_dir / "knowledge.jsonl"
    with open(knowledge_file, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Write to timeline.jsonl
    timeline_file = storylines_dir / "timeline.jsonl"
    with open(timeline_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def update_state(status: str, windows_processed: int):
    """Update agent state file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "status": status,
        "level": "L2",
        "windows_processed": windows_processed,
        "last_update": datetime.now().isoformat(),
        "mode": "timeline_summarization"
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))


def run_continuous():
    """Run continuous summarization."""
    print("=" * 60)
    print("L2 Summarizer - Timeline Aggregation")
    print("=" * 60)
    print(f"Window size: {SUMMARY_WINDOW_MINUTES} minutes")
    print(f"Reading from: {ANALYZERS_DIR}")
    print()

    task_context = get_task_context()
    print(f"Task context loaded: {len(task_context)} chars")
    print()

    windows_processed = 0
    update_state("Running", windows_processed)

    while True:
        try:
            # Load L3 analyses
            analyses = load_l3_analyses()

            if analyses:
                # Group by time window
                windows = group_by_time_window(analyses, SUMMARY_WINDOW_MINUTES)

                # Get already processed
                processed = get_processed_windows()

                # Process new windows
                for window_key in sorted(windows.keys()):
                    if window_key in processed:
                        continue

                    descriptions = [a["description"] for a in windows[window_key] if "description" in a]

                    if len(descriptions) >= 2:  # Need at least 2 descriptions
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Summarizing window {window_key} ({len(descriptions)} frames)")

                        summary = summarize_window(window_key, descriptions, task_context)

                        if summary:
                            write_insight(window_key, summary, len(descriptions))
                            windows_processed += 1
                            print(f"  ✓ {summary[:80]}...")
                            update_state("Running", windows_processed)

            # Wait before next check
            time.sleep(30)

        except KeyboardInterrupt:
            print("\n\nL2 Summarizer stopped.")
            update_state("Stopped", windows_processed)
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)


def run_batch():
    """Run a single batch for testing."""
    print("Running single batch summarization")

    task_context = get_task_context()
    print(f"Task context: {task_context[:200]}...")
    print()

    # Load L3 analyses
    analyses = load_l3_analyses()
    print(f"Found {len(analyses)} L3 analyses")

    if not analyses:
        print("No L3 analyses found. Run L3 analyzer first.")
        return

    # Group by time window
    windows = group_by_time_window(analyses, SUMMARY_WINDOW_MINUTES)
    print(f"Grouped into {len(windows)} time windows")

    # Get already processed
    processed = get_processed_windows()
    print(f"Already processed: {len(processed)} windows")

    # Process unprocessed
    for window_key in sorted(windows.keys()):
        if window_key in processed:
            print(f"\nSkipping {window_key} (already processed)")
            continue

        descriptions = [a["description"] for a in windows[window_key] if "description" in a]

        if len(descriptions) < 2:
            print(f"\nSkipping {window_key} (only {len(descriptions)} descriptions)")
            continue

        print(f"\n[{window_key}] Summarizing {len(descriptions)} frames...")

        summary = summarize_window(window_key, descriptions, task_context)

        if summary:
            write_insight(window_key, summary, len(descriptions))
            print(f"  ✓ Summary: {summary}")
        else:
            print(f"  ✗ Failed to generate summary")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="L2 Summarizer - Timeline Aggregation")
    parser.add_argument("--batch", action="store_true", help="Run single batch")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")

    args = parser.parse_args()

    if args.batch:
        run_batch()
    elif args.continuous:
        run_continuous()
    else:
        # Default: run batch
        run_batch()


if __name__ == "__main__":
    main()
