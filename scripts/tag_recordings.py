"""Auto-tag all KVM recordings using scene scanning.

Processes each MKV recording in data/recordings/, extracts frames at
intervals, runs OCR + activity classification, and produces a structured
tags database in data/recordings/tags.jsonl.

This is the Star Trek Computer's "memory formation" — it watches recordings
and builds an understanding of what happened on the target machine.

Usage:
    uv run python scripts/tag_recordings.py
    uv run python scripts/tag_recordings.py --sample-interval 15  # more detail
    uv run python scripts/tag_recordings.py --limit 10            # first 10 only
    uv run python scripts/tag_recordings.py --recording kvm_20260218_190304.mkv
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.awareness.scene_scanner import scan_recording, RecordingTag

RECORDINGS_DIR = Path(__file__).parent.parent / "data" / "recordings"
TAGS_FILE = RECORDINGS_DIR / "tags.jsonl"


def load_existing_tags() -> dict[str, dict]:
    """Load already-tagged recordings to skip them."""
    existing = {}
    if TAGS_FILE.exists():
        for line in TAGS_FILE.read_text().strip().split("\n"):
            if line.strip():
                tag = json.loads(line)
                existing[tag["filename"]] = tag
    return existing


def tag_all(args):
    recordings = sorted(RECORDINGS_DIR.glob("*.mkv"))
    if args.recording:
        recordings = [RECORDINGS_DIR / args.recording]

    if not recordings:
        print("No recordings found!")
        return

    existing = load_existing_tags() if not args.force else {}

    # Filter already tagged
    to_process = []
    for r in recordings:
        if r.name in existing and not args.force:
            continue
        to_process.append(r)

    if args.limit:
        to_process = to_process[:args.limit]

    print(f"Recordings: {len(recordings)} total, {len(existing)} already tagged, {len(to_process)} to process")
    if not to_process:
        print("Nothing to do!")
        return

    total_start = time.monotonic()
    results: list[RecordingTag] = []

    for i, rec_path in enumerate(to_process):
        print(f"\n[{i+1}/{len(to_process)}] {rec_path.name}...")
        tag = scan_recording(
            rec_path,
            sample_interval_s=args.sample_interval,
        )
        results.append(tag)

        # Print summary
        n_seg = len(tag.segments)
        apps_str = ", ".join(tag.detected_apps[:5]) if tag.detected_apps else "none"
        meeting_str = " [MEETING]" if tag.is_meeting else ""
        print(f"  {tag.duration_s:.0f}s | {n_seg} segments | {tag.dominant_activity}{meeting_str} | apps: {apps_str} | scanned in {tag.scan_time_s:.1f}s")

    # Append to tags file
    with open(TAGS_FILE, "a") as f:
        for tag in results:
            f.write(json.dumps(asdict(tag)) + "\n")

    total_time = time.monotonic() - total_start
    print(f"\n{'='*60}")
    print(f"Tagged {len(results)} recordings in {total_time:.1f}s ({total_time/len(results):.1f}s/recording)")
    print(f"Tags saved to: {TAGS_FILE}")

    # Summary stats
    from collections import Counter
    activities = Counter(t.dominant_activity for t in results)
    meetings = sum(1 for t in results if t.is_meeting)
    print(f"\nActivity breakdown:")
    for act, cnt in activities.most_common():
        print(f"  {act:25s}: {cnt} recordings")
    print(f"Meetings detected: {meetings}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-tag KVM recordings with scene understanding",
    )
    parser.add_argument("--sample-interval", type=float, default=30.0,
                        help="Sample 1 frame per N seconds (default: 30)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N recordings (0=all)")
    parser.add_argument("--recording", type=str, default="",
                        help="Process a specific recording file")
    parser.add_argument("--force", action="store_true",
                        help="Re-tag already tagged recordings")
    args = parser.parse_args()
    tag_all(args)


if __name__ == "__main__":
    main()
