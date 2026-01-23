#!/usr/bin/env python3
"""L3 Analyzer - Parallel Screenshot Analysis using Claude Agent SDK.

Uses multiple Claude Agent SDK instances to analyze keyframes in parallel.
Each instance uses the built-in Read tool which presents images visually
to the model, leveraging native vision capability.

Architecture:
- Spawns N parallel Claude subagents (via subprocess)
- Each subagent reads a keyframe with Read tool (vision)
- Each produces a one-sentence description
- Results collected in agents/analyzers/batch_NNN.jsonl
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

# Paths
UI_AGENT_DIR = Path(__file__).parent.parent.parent
EXPERIENCES_DIR = UI_AGENT_DIR / "experiences"
ANALYZERS_DIR = UI_AGENT_DIR / "agents" / "analyzers"
STATE_FILE = UI_AGENT_DIR / "agents" / "keyframe-identifier" / "state.json"

# Config
BATCH_SIZE = 5  # Process 5 keyframes in parallel
MAX_WORKERS = 3  # Max concurrent Claude subagents


def get_unprocessed_keyframes(limit: int = 50) -> list[Path]:
    """Get keyframes that haven't been analyzed yet."""
    today = datetime.now().strftime("%Y-%m-%d")
    keyframes_dir = EXPERIENCES_DIR / today / "keyframes"

    if not keyframes_dir.exists():
        return []

    # Load already processed
    processed = set()
    for batch_file in ANALYZERS_DIR.glob("*.jsonl"):
        try:
            for line in batch_file.read_text().strip().split('\n'):
                if line:
                    entry = json.loads(line)
                    processed.add(entry.get("keyframe"))
        except Exception:
            pass

    # Get unprocessed
    unprocessed = []
    for kf in sorted(keyframes_dir.glob("*.png"), key=lambda p: p.stat().st_mtime):
        if kf.name not in processed:
            unprocessed.append(kf)
            if len(unprocessed) >= limit:
                break

    return unprocessed


def analyze_keyframe_with_claude(keyframe_path: Path) -> Optional[dict]:
    """Analyze a single keyframe using Claude CLI with vision.

    Spawns a Claude CLI process that uses the Read tool to view the image.
    """
    prompt = f'''Read the image file at {keyframe_path} and describe what you see in 2-3 detailed sentences.

Include:
1. WHAT application/window is visible (be specific - app name, window title if readable)
2. WHAT the user is doing (typing, clicking, scrolling, reading, waiting)
3. WHAT data/content is visible (file names, form fields, text content, error messages)
4. WHAT UI state (dialog boxes, menus open, loading indicators, cursor position)

Be specific and factual. Include any readable text. Reply with ONLY the description, nothing else.'''

    try:
        # Run Claude CLI with the prompt
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=120,  # 120 second timeout per image
            cwd=str(UI_AGENT_DIR)
        )

        if result.returncode == 0 and result.stdout.strip():
            description = result.stdout.strip()
            # Clean up any extra output
            lines = description.split('\n')
            # Take the most substantial line as the description
            description = max(lines, key=len) if lines else description

            return {
                "keyframe": keyframe_path.name,
                "path": str(keyframe_path),
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "analyzer": "l3_claude_sdk"
            }
        else:
            print(f"  Error analyzing {keyframe_path.name}: {result.stderr[:200]}")
            return None

    except subprocess.TimeoutExpired:
        print(f"  Timeout analyzing {keyframe_path.name}")
        return None
    except Exception as e:
        print(f"  Exception analyzing {keyframe_path.name}: {e}")
        return None


def process_batch(keyframes: list[Path], batch_num: int) -> list[dict]:
    """Process a batch of keyframes in parallel."""
    results = []

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing batch {batch_num}: {len(keyframes)} keyframes")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_kf = {
            executor.submit(analyze_keyframe_with_claude, kf): kf
            for kf in keyframes
        }

        for future in as_completed(future_to_kf):
            kf = future_to_kf[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"  ✓ {kf.name}: {result['description'][:60]}...")
            except Exception as e:
                print(f"  ✗ {kf.name}: {e}")

    return results


def save_batch(results: list[dict], batch_num: int) -> Path:
    """Save batch results to JSONL file."""
    ANALYZERS_DIR.mkdir(parents=True, exist_ok=True)

    batch_file = ANALYZERS_DIR / f"batch_{batch_num:04d}.jsonl"
    with open(batch_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    return batch_file


def update_state(status: str, processed_count: int):
    """Update agent state file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "status": status,
        "level": "L3",
        "processed_count": processed_count,
        "last_update": datetime.now().isoformat(),
        "mode": "parallel_vision"
    }
    STATE_FILE.write_text(json.dumps(state, indent=2))


def run_continuous():
    """Run continuous analysis of new keyframes."""
    print("=" * 60)
    print("L3 Analyzer - Parallel Vision Analysis")
    print("=" * 60)
    print(f"Using Claude Agent SDK with built-in vision")
    print(f"Batch size: {BATCH_SIZE}, Workers: {MAX_WORKERS}")
    print(f"Output: {ANALYZERS_DIR}")
    print()

    batch_num = len(list(ANALYZERS_DIR.glob("*.jsonl"))) if ANALYZERS_DIR.exists() else 0
    total_processed = 0

    update_state("Running", total_processed)

    while True:
        try:
            # Get unprocessed keyframes
            keyframes = get_unprocessed_keyframes(limit=BATCH_SIZE)

            if keyframes:
                # Process batch
                results = process_batch(keyframes, batch_num)

                if results:
                    # Save results
                    batch_file = save_batch(results, batch_num)
                    total_processed += len(results)
                    batch_num += 1

                    print(f"  Saved {len(results)} analyses to {batch_file.name}")
                    update_state("Running", total_processed)

            else:
                # No new keyframes, wait
                update_state("Waiting", total_processed)
                time.sleep(5)

        except KeyboardInterrupt:
            print("\n\nL3 Analyzer stopped.")
            update_state("Stopped", total_processed)
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)


def run_batch(limit: int = 10):
    """Run a single batch for testing."""
    print(f"Running single batch analysis (limit: {limit})")

    keyframes = get_unprocessed_keyframes(limit=limit)
    if not keyframes:
        print("No unprocessed keyframes found.")
        return

    results = process_batch(keyframes, 0)
    if results:
        batch_file = save_batch(results, int(time.time()))
        print(f"\nSaved {len(results)} analyses to {batch_file}")

        # Print sample
        print("\nSample results:")
        for r in results[:3]:
            print(f"  {r['keyframe']}: {r['description']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="L3 Analyzer - Parallel Vision Analysis")
    parser.add_argument("--batch", type=int, help="Run single batch with N keyframes")
    parser.add_argument("--continuous", action="store_true", help="Run continuously")

    args = parser.parse_args()

    if args.batch:
        run_batch(args.batch)
    elif args.continuous:
        run_continuous()
    else:
        # Default: run batch of 5 to test
        run_batch(5)


if __name__ == "__main__":
    main()
