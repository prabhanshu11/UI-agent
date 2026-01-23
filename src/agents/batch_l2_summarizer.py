#!/usr/bin/env python3
"""Batch L2 Summarizer - Process L3 analyses and identify key moments.

Works with versioned runs from batch_processor.py.

Usage:
    # Summarize a specific run
    python batch_l2_summarizer.py --run run_001

    # Summarize with custom window size
    python batch_l2_summarizer.py --run run_001 --window-minutes 5
"""

import os
import sys
import json
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from collections import defaultdict

# Paths
UI_AGENT_DIR = Path(__file__).parent.parent.parent
RUNS_DIR = UI_AGENT_DIR / "runs"
CONTROL_FILE = UI_AGENT_DIR / "agents" / "CONTROL.md"


def get_task_context() -> str:
    """Read current task context from CONTROL.md."""
    if CONTROL_FILE.exists():
        content = CONTROL_FILE.read_text()
        if "## Current Task Context" in content:
            start = content.find("## Current Task Context")
            end = content.find("## Agent Status", start)
            if end == -1:
                end = len(content)
            return content[start:end].strip()
    return "No task context available"


def parse_keyframe_time(keyframe_name: str) -> Optional[datetime]:
    """Extract timestamp from keyframe filename."""
    match = re.search(r'keyframe_\d+_(\d{6})\.png', keyframe_name)
    if match:
        time_str = match.group(1)
        today = datetime.now().strftime("%Y-%m-%d")
        try:
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
            window_start = kf_time.replace(
                minute=(kf_time.minute // window_minutes) * window_minutes,
                second=0,
                microsecond=0
            )
            window_key = window_start.strftime("%H:%M")
            windows[window_key].append(analysis)

    return dict(windows)


def summarize_window(window_key: str, analyses: list[dict], task_context: str) -> Optional[dict]:
    """Use Claude to analyze a time window and identify KEY MOMENTS."""
    descriptions = [a.get("description", "") for a in analyses]
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

4. **DATA OBSERVED**: Any specific data values, file names, or identifiers visible

Format your response as:
KEY MOMENTS: [list frame numbers and why they're key]
SUMMARY: [your activity summary]
INSIGHT: [workflow observation]
DATA: [specific data values observed]'''

    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(UI_AGENT_DIR)
        )

        if result.returncode == 0 and result.stdout.strip():
            return {
                "time_window": window_key,
                "frame_count": len(descriptions),
                "analysis": result.stdout.strip(),
                "timestamp": datetime.now().isoformat(),
                "keyframe_refs": [a.get("keyframe") for a in analyses]
            }

    except Exception as e:
        print(f"Error summarizing window {window_key}: {e}")

    return None


class BatchL2Summarizer:
    """Process L3 analyses and produce L2 summaries."""

    def __init__(self, run_id: str, window_minutes: int = 5):
        self.run_id = run_id
        self.window_minutes = window_minutes
        self.run_dir = RUNS_DIR / run_id

        if not self.run_dir.exists():
            raise ValueError(f"Run {run_id} not found")

        # Create L2 summaries directory
        self.summaries_dir = self.run_dir / "l2_summaries"
        self.summaries_dir.mkdir(exist_ok=True)

    def load_l3_analyses(self) -> list[dict]:
        """Load all L3 analyses from the run."""
        analyses = []
        l3_dir = self.run_dir / "l3_analyses"

        # Try single file first
        all_file = l3_dir / "all_analyses.jsonl"
        if all_file.exists():
            for line in all_file.read_text().strip().split('\n'):
                if line:
                    analyses.append(json.loads(line))
            return analyses

        # Otherwise load batch files
        for batch_file in sorted(l3_dir.glob("batch_*.jsonl")):
            for line in batch_file.read_text().strip().split('\n'):
                if line:
                    analyses.append(json.loads(line))

        return analyses

    def process(self):
        """Process all L3 analyses and produce L2 summaries."""
        print(f"Processing L2 summaries for run: {self.run_id}")

        task_context = get_task_context()
        print(f"Task context loaded: {len(task_context)} chars")

        # Load L3 analyses
        analyses = self.load_l3_analyses()
        print(f"Loaded {len(analyses)} L3 analyses")

        if not analyses:
            print("No L3 analyses found")
            return

        # Group by time window
        windows = group_by_time_window(analyses, self.window_minutes)
        print(f"Grouped into {len(windows)} time windows")

        summaries = []
        for i, (window_key, window_analyses) in enumerate(sorted(windows.items())):
            if len(window_analyses) < 2:
                print(f"  [{window_key}] Skipping (only {len(window_analyses)} frames)")
                continue

            print(f"  [{window_key}] Summarizing {len(window_analyses)} frames...")

            summary = summarize_window(window_key, window_analyses, task_context)
            if summary:
                summary["run_id"] = self.run_id
                summaries.append(summary)
                print(f"    ✓ Generated summary")

        # Save summaries
        if summaries:
            summary_file = self.summaries_dir / "summaries.jsonl"
            with open(summary_file, "w") as f:
                for s in summaries:
                    f.write(json.dumps(s) + "\n")

            print(f"\nSaved {len(summaries)} summaries to {summary_file}")

            # Also save as readable report
            report_file = self.summaries_dir / "report.md"
            self._generate_report(summaries, report_file)
            print(f"Generated report: {report_file}")

        return summaries

    def _generate_report(self, summaries: list[dict], output_file: Path):
        """Generate a human-readable report from summaries."""
        task_context = get_task_context()

        report = f"""# L2 Analysis Report - {self.run_id}

Generated: {datetime.now().isoformat()}

## Task Context
{task_context}

## Timeline Summary

"""
        for s in summaries:
            report += f"""### {s['time_window']} ({s['frame_count']} frames)

{s['analysis']}

---

"""

        output_file.write_text(report)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch L2 Summarizer")
    parser.add_argument("--run", type=str, required=True, help="Run ID to process")
    parser.add_argument("--window-minutes", type=int, default=5, help="Time window in minutes")

    args = parser.parse_args()

    summarizer = BatchL2Summarizer(args.run, args.window_minutes)
    summarizer.process()


if __name__ == "__main__":
    main()
