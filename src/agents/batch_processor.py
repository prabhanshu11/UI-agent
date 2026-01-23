#!/usr/bin/env python3
"""Batch Processor - Reprocess all recorded video with versioned runs.

Creates versioned runs that preserve old analysis data while allowing
new algorithm experiments.

Architecture:
- Each run gets a unique run_id (e.g., run_001, run_002)
- All outputs go to runs/{run_id}/ directory
- Configuration is saved with each run for reproducibility
- Old runs are never overwritten

Usage:
    # Create new run with default settings
    python batch_processor.py --new-run

    # Create new run with custom settings
    python batch_processor.py --new-run --change-threshold 0.01 --detail-level high

    # Continue existing run
    python batch_processor.py --continue-run run_001

    # List all runs
    python batch_processor.py --list-runs
"""

import os
import sys
import json
import subprocess
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

# Paths
UI_AGENT_DIR = Path(__file__).parent.parent.parent
RECORDINGS_DIR = UI_AGENT_DIR / "recordings"
EXPERIENCES_DIR = UI_AGENT_DIR / "experiences"
RUNS_DIR = UI_AGENT_DIR / "runs"


class RunConfig:
    """Configuration for a batch processing run."""

    def __init__(
        self,
        run_id: str,
        change_threshold: float = 0.005,
        min_frame_interval: int = 1,
        frame_extract_interval: int = 1,
        l3_detail_level: str = "high",  # low, medium, high
        l2_window_minutes: int = 5,
        l3_batch_size: int = 5,
        l3_max_workers: int = 3,
        description: str = ""
    ):
        self.run_id = run_id
        self.change_threshold = change_threshold
        self.min_frame_interval = min_frame_interval
        self.frame_extract_interval = frame_extract_interval
        self.l3_detail_level = l3_detail_level
        self.l2_window_minutes = l2_window_minutes
        self.l3_batch_size = l3_batch_size
        self.l3_max_workers = l3_max_workers
        self.description = description
        self.created_at = datetime.now().isoformat()
        self.status = "created"

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "change_threshold": self.change_threshold,
            "min_frame_interval": self.min_frame_interval,
            "frame_extract_interval": self.frame_extract_interval,
            "l3_detail_level": self.l3_detail_level,
            "l2_window_minutes": self.l2_window_minutes,
            "l3_batch_size": self.l3_batch_size,
            "l3_max_workers": self.l3_max_workers,
            "description": self.description,
            "created_at": self.created_at,
            "status": self.status
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        config = cls(
            run_id=data["run_id"],
            change_threshold=data.get("change_threshold", 0.005),
            min_frame_interval=data.get("min_frame_interval", 1),
            frame_extract_interval=data.get("frame_extract_interval", 1),
            l3_detail_level=data.get("l3_detail_level", "high"),
            l2_window_minutes=data.get("l2_window_minutes", 5),
            l3_batch_size=data.get("l3_batch_size", 5),
            l3_max_workers=data.get("l3_max_workers", 3),
            description=data.get("description", "")
        )
        config.created_at = data.get("created_at", datetime.now().isoformat())
        config.status = data.get("status", "created")
        return config


class BatchProcessor:
    """Processes all recorded video with versioned output."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.run_dir = RUNS_DIR / config.run_id

        # Create run directory structure
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "keyframes").mkdir(exist_ok=True)
        (self.run_dir / "l3_analyses").mkdir(exist_ok=True)
        (self.run_dir / "l2_summaries").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)

        # State tracking
        self.keyframe_count = 0
        self.last_keyframe_path: Optional[Path] = None
        self.processed_videos: set = set()
        self.l3_analyses: list = []

        # Save config
        self._save_config()

    def _save_config(self):
        """Save run configuration."""
        config_file = self.run_dir / "config.json"
        config_file.write_text(json.dumps(self.config.to_dict(), indent=2))

    def _update_status(self, status: str, stats: dict = None):
        """Update run status."""
        self.config.status = status

        status_file = self.run_dir / "status.json"
        status_data = {
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "keyframe_count": self.keyframe_count,
            "videos_processed": len(self.processed_videos),
            "l3_analyses": len(self.l3_analyses)
        }
        if stats:
            status_data.update(stats)

        status_file.write_text(json.dumps(status_data, indent=2))

    def _log(self, message: str):
        """Log message to run log file."""
        log_file = self.run_dir / "logs" / "processing.log"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
        print(f"[{timestamp}] {message}")

    def calculate_change_percent(self, img1_path: Path, img2_path: Path) -> float:
        """Calculate percentage of pixels changed between two images."""
        if not img1_path.exists() or not img2_path.exists():
            return 1.0

        try:
            result = subprocess.run([
                "compare", "-metric", "AE",
                str(img1_path), str(img2_path),
                "/dev/null"
            ], capture_output=True, text=True, timeout=5)

            diff_pixels = int(result.stderr.strip())

            identify = subprocess.run([
                "identify", "-format", "%w %h", str(img1_path)
            ], capture_output=True, text=True, timeout=3)

            w, h = map(int, identify.stdout.strip().split())
            total_pixels = w * h

            return diff_pixels / total_pixels

        except Exception:
            return 1.0

    def extract_keyframes_from_video(self, video_path: Path) -> list[Path]:
        """Extract keyframes from a video file."""
        keyframes = []

        # Get video duration
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path)
            ], capture_output=True, text=True, timeout=5)

            duration = float(result.stdout.strip())
            if duration < 1:
                return keyframes

        except Exception as e:
            self._log(f"Error getting duration for {video_path.name}: {e}")
            return keyframes

        # Extract frames at regular intervals
        with tempfile.TemporaryDirectory() as tmpdir:
            for t in range(0, int(duration), self.config.frame_extract_interval):
                frame_path = Path(tmpdir) / f"frame_{t:05d}.png"

                result = subprocess.run([
                    "ffmpeg", "-ss", str(t), "-i", str(video_path),
                    "-frames:v", "1", "-y", str(frame_path)
                ], capture_output=True, timeout=10)

                if result.returncode != 0 or not frame_path.exists():
                    continue

                # Check if keyframe (significant change)
                is_keyframe = False
                if self.last_keyframe_path is None:
                    is_keyframe = True
                else:
                    change = self.calculate_change_percent(self.last_keyframe_path, frame_path)
                    if change >= self.config.change_threshold:
                        is_keyframe = True

                if is_keyframe:
                    self.keyframe_count += 1
                    timestamp = f"{t//3600:02d}{(t%3600)//60:02d}{t%60:02d}"
                    kf_name = f"keyframe_{self.keyframe_count:04d}_{timestamp}.png"
                    kf_path = self.run_dir / "keyframes" / kf_name

                    shutil.copy2(frame_path, kf_path)
                    self.last_keyframe_path = kf_path
                    keyframes.append(kf_path)

        return keyframes

    def analyze_keyframe_l3(self, keyframe_path: Path) -> Optional[dict]:
        """Analyze a keyframe using Claude CLI (L3 analysis)."""

        if self.config.l3_detail_level == "low":
            prompt = f'''Read the image file at {keyframe_path} and describe in ONE sentence.
Focus on: application, user action, visible data.
Reply with ONLY the description.'''
        elif self.config.l3_detail_level == "medium":
            prompt = f'''Read the image file at {keyframe_path} and describe in 2 sentences.
Include: application name, user action, visible data/text.
Reply with ONLY the description.'''
        else:  # high
            prompt = f'''Read the image file at {keyframe_path} and describe in 2-3 detailed sentences.

Include:
1. WHAT application/window is visible (be specific - app name, window title)
2. WHAT the user is doing (typing, clicking, scrolling, reading, waiting)
3. WHAT data/content is visible (file names, form fields, text content, errors)
4. WHAT UI state (dialogs, menus, loading indicators, cursor position)

Be specific. Include readable text. Reply with ONLY the description.'''

        try:
            result = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(UI_AGENT_DIR)
            )

            if result.returncode == 0 and result.stdout.strip():
                description = result.stdout.strip()
                lines = description.split('\n')
                description = max(lines, key=len) if lines else description

                return {
                    "keyframe": keyframe_path.name,
                    "path": str(keyframe_path),
                    "description": description,
                    "timestamp": datetime.now().isoformat(),
                    "run_id": self.config.run_id,
                    "detail_level": self.config.l3_detail_level
                }

        except Exception as e:
            self._log(f"Error analyzing {keyframe_path.name}: {e}")

        return None

    def process_keyframes_l3(self, keyframes: list[Path]) -> list[dict]:
        """Process keyframes in parallel with L3 analyzer."""
        results = []

        with ThreadPoolExecutor(max_workers=self.config.l3_max_workers) as executor:
            future_to_kf = {
                executor.submit(self.analyze_keyframe_l3, kf): kf
                for kf in keyframes
            }

            for future in as_completed(future_to_kf):
                kf = future_to_kf[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self._log(f"  L3: {kf.name} -> {result['description'][:60]}...")
                except Exception as e:
                    self._log(f"  L3 error on {kf.name}: {e}")

        return results

    def save_l3_batch(self, results: list[dict], batch_num: int):
        """Save L3 analysis batch."""
        if not results:
            return

        batch_file = self.run_dir / "l3_analyses" / f"batch_{batch_num:04d}.jsonl"
        with open(batch_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        self.l3_analyses.extend(results)

    def process_all_videos(self, date: str = None):
        """Process all recorded videos for a date."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        self._log(f"Starting batch processing for {date}")
        self._log(f"Run ID: {self.config.run_id}")
        self._log(f"Config: threshold={self.config.change_threshold}, detail={self.config.l3_detail_level}")
        self._update_status("running")

        recordings_dir = RECORDINGS_DIR / date
        if not recordings_dir.exists():
            self._log(f"No recordings found for {date}")
            return

        videos = sorted(recordings_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        self._log(f"Found {len(videos)} videos to process")

        batch_num = 0

        for i, video in enumerate(videos):
            if str(video) in self.processed_videos:
                continue

            self._log(f"\n[{i+1}/{len(videos)}] Processing {video.name}")
            self.processed_videos.add(str(video))

            # Extract keyframes
            keyframes = self.extract_keyframes_from_video(video)
            self._log(f"  Extracted {len(keyframes)} keyframes")

            if keyframes:
                # Process in batches
                for j in range(0, len(keyframes), self.config.l3_batch_size):
                    batch = keyframes[j:j + self.config.l3_batch_size]
                    results = self.process_keyframes_l3(batch)
                    self.save_l3_batch(results, batch_num)
                    batch_num += 1

            self._update_status("running", {"current_video": video.name})

        # Final summary
        self._log(f"\nProcessing complete!")
        self._log(f"  Total keyframes: {self.keyframe_count}")
        self._log(f"  Total L3 analyses: {len(self.l3_analyses)}")
        self._log(f"  Videos processed: {len(self.processed_videos)}")

        self._update_status("completed", {
            "total_keyframes": self.keyframe_count,
            "total_l3_analyses": len(self.l3_analyses),
            "videos_processed": len(self.processed_videos)
        })

        # Save all L3 analyses to single file for L2
        all_l3_file = self.run_dir / "l3_analyses" / "all_analyses.jsonl"
        with open(all_l3_file, "w") as f:
            for analysis in self.l3_analyses:
                f.write(json.dumps(analysis) + "\n")

        return self.l3_analyses


def get_next_run_id() -> str:
    """Get the next available run ID."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    existing = [d.name for d in RUNS_DIR.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not existing:
        return "run_001"

    numbers = [int(r.split("_")[1]) for r in existing]
    next_num = max(numbers) + 1
    return f"run_{next_num:03d}"


def list_runs():
    """List all existing runs."""
    if not RUNS_DIR.exists():
        print("No runs found.")
        return

    runs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()])
    if not runs:
        print("No runs found.")
        return

    print(f"\n{'Run ID':<12} {'Status':<12} {'Keyframes':<12} {'L3 Analyses':<12} {'Created':<20}")
    print("-" * 70)

    for run_dir in runs:
        config_file = run_dir / "config.json"
        status_file = run_dir / "status.json"

        if config_file.exists():
            config = json.loads(config_file.read_text())
            status_data = {}
            if status_file.exists():
                status_data = json.loads(status_file.read_text())

            print(f"{config['run_id']:<12} {status_data.get('status', 'unknown'):<12} "
                  f"{status_data.get('keyframe_count', 0):<12} {status_data.get('l3_analyses', 0):<12} "
                  f"{config.get('created_at', 'unknown')[:19]:<20}")

            if config.get('description'):
                print(f"  Description: {config['description']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch Processor - Versioned video reprocessing")
    parser.add_argument("--new-run", action="store_true", help="Create new run")
    parser.add_argument("--continue-run", type=str, help="Continue existing run")
    parser.add_argument("--list-runs", action="store_true", help="List all runs")
    parser.add_argument("--date", type=str, help="Date to process (YYYY-MM-DD)")

    # Config options
    parser.add_argument("--change-threshold", type=float, default=0.005, help="Change threshold (default: 0.005)")
    parser.add_argument("--detail-level", type=str, choices=["low", "medium", "high"], default="high", help="L3 detail level")
    parser.add_argument("--description", type=str, default="", help="Run description")

    args = parser.parse_args()

    if args.list_runs:
        list_runs()
        return

    if args.new_run:
        run_id = get_next_run_id()
        config = RunConfig(
            run_id=run_id,
            change_threshold=args.change_threshold,
            l3_detail_level=args.detail_level,
            description=args.description
        )
        processor = BatchProcessor(config)
        processor.process_all_videos(args.date)

    elif args.continue_run:
        run_dir = RUNS_DIR / args.continue_run
        config_file = run_dir / "config.json"

        if not config_file.exists():
            print(f"Run {args.continue_run} not found")
            return

        config = RunConfig.from_dict(json.loads(config_file.read_text()))
        processor = BatchProcessor(config)
        processor.process_all_videos(args.date)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
