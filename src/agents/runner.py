#!/usr/bin/env python3
"""Agent Runner - Processes video for keyframes and knowledge capture.

Runs in two phases:
1. Retroactive: Process recorded videos from today
2. Live: Continuously process new frames from live feed

Extracts keyframes using change detection and captures knowledge
via the multi-storyline logger.
"""

import os
import sys
import json
import hashlib
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
import tempfile

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.agents.storyline_logger import StorylineLogger, StorylineType

# Paths
UI_AGENT_DIR = Path(__file__).parent.parent.parent
RECORDINGS_DIR = UI_AGENT_DIR / "recordings"
EXPERIENCES_DIR = UI_AGENT_DIR / "experiences"
AGENTS_DIR = UI_AGENT_DIR / "agents"
LIVE_FRAME_PATH = Path("/tmp/live_frame.png")

# Settings - Capture MORE frames, let L2 identify what's truly "key"
CHANGE_THRESHOLD = 0.005  # 0.5% pixel change = capture (very sensitive)
MIN_KEYFRAME_INTERVAL = 1  # Minimum 1 second between frames
FRAME_EXTRACT_INTERVAL = 1  # Extract frame every 1 second from video


def get_image_hash(image_path: Path) -> Optional[str]:
    """Get perceptual hash of image for comparison."""
    if not image_path.exists():
        return None
    try:
        # Use simple file hash for now
        return hashlib.md5(image_path.read_bytes()).hexdigest()
    except Exception:
        return None


def calculate_change_percent(img1_path: Path, img2_path: Path) -> float:
    """Calculate percentage of pixels changed between two images.

    Uses ImageMagick compare for accurate difference calculation.
    """
    if not img1_path.exists() or not img2_path.exists():
        return 1.0  # Assume maximum change if can't compare

    try:
        # Use ImageMagick to calculate difference
        result = subprocess.run([
            "compare", "-metric", "AE",
            str(img1_path), str(img2_path),
            "/dev/null"
        ], capture_output=True, text=True, timeout=5)

        # AE returns number of different pixels
        diff_pixels = int(result.stderr.strip())

        # Get image dimensions
        identify = subprocess.run([
            "identify", "-format", "%w %h", str(img1_path)
        ], capture_output=True, text=True, timeout=3)

        w, h = map(int, identify.stdout.strip().split())
        total_pixels = w * h

        return diff_pixels / total_pixels

    except Exception as e:
        # Fallback: compare file sizes/hashes
        h1 = get_image_hash(img1_path)
        h2 = get_image_hash(img2_path)
        return 0.0 if h1 == h2 else 1.0


class AgentRunner:
    """Runs keyframe extraction and knowledge capture."""

    def __init__(self, mode: str = "learning"):
        self.mode = mode
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.session_id = datetime.now().strftime("%H%M%S")

        # Create experience directory
        self.experience_dir = EXPERIENCES_DIR / self.today
        self.experience_dir.mkdir(parents=True, exist_ok=True)

        # Create keyframes directory
        self.keyframes_dir = self.experience_dir / "keyframes"
        self.keyframes_dir.mkdir(exist_ok=True)

        # Initialize storyline logger
        self.logger = StorylineLogger(
            experience_dir=self.experience_dir,
            session_id=self.session_id
        )

        # Tracking
        self.last_keyframe_path: Optional[Path] = None
        self.last_keyframe_time: float = 0
        self.keyframe_count = 0
        self.processed_videos: set = set()

        # Update agent state
        self._update_state_file()

        # Log startup
        self.logger.log_main(
            "start",
            f"Agent runner started in {mode} mode",
            metadata={"mode": mode, "session_id": self.session_id}
        )

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Agent runner started")
        print(f"  Mode: {mode}")
        print(f"  Experience dir: {self.experience_dir}")
        print(f"  Keyframes dir: {self.keyframes_dir}")

    def _update_state_file(self):
        """Update agent state file for dashboard."""
        state = {
            "status": "Running",
            "mode": self.mode,
            "keyframe_count": self.keyframe_count,
            "last_update": datetime.now().isoformat(),
            "change_threshold": CHANGE_THRESHOLD
        }

        state_file = AGENTS_DIR / "keyframe-identifier" / "state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state, indent=2))

    def check_keyframe(self, frame_path: Path, frame_time: str) -> bool:
        """Check if frame is a keyframe (significant change).

        Returns True if frame should be saved as keyframe.
        """
        now = time.time()

        # Respect minimum interval
        if now - self.last_keyframe_time < MIN_KEYFRAME_INTERVAL:
            return False

        # First frame is always a keyframe
        if self.last_keyframe_path is None or not self.last_keyframe_path.exists():
            return True

        # Calculate change
        change = calculate_change_percent(self.last_keyframe_path, frame_path)

        if change >= CHANGE_THRESHOLD:
            self.logger.log_perception(
                "change_detected",
                f"Significant change detected: {change*100:.1f}% at {frame_time}"
            )
            return True

        return False

    def save_keyframe(self, frame_path: Path, source: str, timestamp: str) -> Path:
        """Save frame as keyframe."""
        self.keyframe_count += 1

        # Generate keyframe filename
        kf_name = f"keyframe_{self.keyframe_count:04d}_{timestamp.replace(':', '')}.png"
        kf_path = self.keyframes_dir / kf_name

        # Copy frame to keyframes
        import shutil
        shutil.copy2(frame_path, kf_path)

        # Update tracking
        self.last_keyframe_path = kf_path
        self.last_keyframe_time = time.time()

        # Log perception
        self.logger.log_perception(
            "keyframe",
            f"Keyframe #{self.keyframe_count} captured from {source}",
            screenshot_path=str(kf_path),
            detected_elements=[{
                "source": source,
                "timestamp": timestamp,
                "keyframe_number": self.keyframe_count
            }]
        )

        # Update state
        self._update_state_file()

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Keyframe #{self.keyframe_count}: {kf_name}")

        return kf_path

    def analyze_keyframe(self, kf_path: Path) -> dict:
        """Analyze keyframe and extract knowledge.

        For now, just logs basic perception. In future, could use
        vision model to describe what's on screen.
        """
        # Basic analysis - file metadata
        stat = kf_path.stat()

        knowledge = {
            "path": str(kf_path),
            "size_kb": stat.st_size / 1024,
            "timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat()
        }

        # Log that we captured this frame
        self.logger.log_knowledge(
            f"Captured screen state at {knowledge['timestamp']} - {kf_path.name}",
            applies_to=["screen_capture", "workflow"],
            confidence=0.6
        )

        return knowledge

    def process_video(self, video_path: Path) -> int:
        """Process a video file, extracting keyframes.

        Returns number of keyframes extracted.
        """
        if str(video_path) in self.processed_videos:
            return 0

        self.processed_videos.add(str(video_path))

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Processing: {video_path.name}")

        self.logger.log_main(
            "processing",
            f"Processing video: {video_path.name}",
            metadata={"video": str(video_path)}
        )

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
                print(f"  Skipping short video ({duration:.1f}s)")
                return 0

        except Exception as e:
            print(f"  Error getting duration: {e}")
            return 0

        keyframes_extracted = 0

        # Extract frames at regular intervals
        with tempfile.TemporaryDirectory() as tmpdir:
            for t in range(0, int(duration), FRAME_EXTRACT_INTERVAL):
                frame_path = Path(tmpdir) / f"frame_{t:05d}.png"

                # Extract frame
                result = subprocess.run([
                    "ffmpeg", "-ss", str(t), "-i", str(video_path),
                    "-frames:v", "1", "-y", str(frame_path)
                ], capture_output=True, timeout=10)

                if result.returncode != 0 or not frame_path.exists():
                    continue

                # Check if keyframe
                timestamp = f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"

                if self.check_keyframe(frame_path, timestamp):
                    kf_path = self.save_keyframe(
                        frame_path,
                        video_path.name,
                        timestamp
                    )
                    self.analyze_keyframe(kf_path)
                    keyframes_extracted += 1

        print(f"  Extracted {keyframes_extracted} keyframes from {video_path.name}")

        self.logger.log_main(
            "completed",
            f"Finished processing {video_path.name}: {keyframes_extracted} keyframes",
            metadata={
                "video": video_path.name,
                "keyframes": keyframes_extracted,
                "duration": duration
            }
        )

        return keyframes_extracted

    def process_recorded_videos(self):
        """Process all recorded videos from today (retroactive)."""
        print("\n" + "="*60)
        print("PHASE 1: Retroactive Processing")
        print("="*60)

        self.logger.log_main(
            "phase_start",
            "Starting retroactive video processing"
        )

        today_dir = RECORDINGS_DIR / self.today
        if not today_dir.exists():
            print("No recordings directory for today")
            return

        # Get all completed videos (not currently being written)
        videos = sorted(today_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)

        # Exclude the current live video (last one being written)
        if videos:
            # Check if last video is still being written
            last_video = videos[-1]
            if (time.time() - last_video.stat().st_mtime) < 10:
                videos = videos[:-1]  # Exclude last (active) video

        print(f"Found {len(videos)} completed videos to process")

        total_keyframes = 0
        for video in videos:
            kf_count = self.process_video(video)
            total_keyframes += kf_count

        print(f"\nRetroactive processing complete: {total_keyframes} total keyframes")

        self.logger.log_main(
            "phase_complete",
            f"Retroactive processing complete: {total_keyframes} keyframes",
            metadata={"total_keyframes": total_keyframes}
        )

        # Log knowledge summary
        self.logger.log_knowledge(
            f"Processed {len(videos)} recordings, extracted {total_keyframes} keyframes",
            applies_to=["video_processing", "keyframe_extraction"],
            confidence=0.9
        )

    def process_live_feed(self):
        """Continuously process live feed frames."""
        print("\n" + "="*60)
        print("PHASE 2: Live Feed Processing")
        print("="*60)

        self.logger.log_main(
            "phase_start",
            "Starting live feed processing"
        )

        print(f"Monitoring: {LIVE_FRAME_PATH}")
        print("Press Ctrl+C to stop\n")

        last_mtime = 0

        while True:
            try:
                if LIVE_FRAME_PATH.exists():
                    mtime = LIVE_FRAME_PATH.stat().st_mtime

                    # New frame available
                    if mtime > last_mtime:
                        last_mtime = mtime

                        frame_time = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")

                        if self.check_keyframe(LIVE_FRAME_PATH, frame_time):
                            kf_path = self.save_keyframe(
                                LIVE_FRAME_PATH,
                                "live_feed",
                                frame_time
                            )
                            self.analyze_keyframe(kf_path)

                # Also check for new video segments
                today_dir = RECORDINGS_DIR / self.today
                if today_dir.exists():
                    for video in today_dir.glob("*.mp4"):
                        if str(video) not in self.processed_videos:
                            # Check if video is complete (not being written)
                            if (time.time() - video.stat().st_mtime) > 10:
                                self.process_video(video)

                time.sleep(0.5)  # Check twice per second

            except KeyboardInterrupt:
                print("\n\nStopping live processing...")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

        # Final summary
        self.logger.log_main(
            "stop",
            f"Agent runner stopped. Total keyframes: {self.keyframe_count}"
        )

        # Save narrative
        narrative_path = self.logger.save_narrative()
        print(f"\nNarrative saved to: {narrative_path}")

    def run(self):
        """Run full processing pipeline."""
        try:
            # Phase 1: Retroactive
            self.process_recorded_videos()

            # Phase 2: Live
            self.process_live_feed()

        except Exception as e:
            self.logger.log_technical(
                "error",
                f"Agent runner error: {e}",
                error=str(e)
            )
            raise


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Agent Runner")
    parser.add_argument("--mode", default="learning", choices=["learning", "doing", "meeting"])
    parser.add_argument("--retroactive-only", action="store_true", help="Only process recorded videos")
    parser.add_argument("--live-only", action="store_true", help="Only process live feed")

    args = parser.parse_args()

    runner = AgentRunner(mode=args.mode)

    if args.retroactive_only:
        runner.process_recorded_videos()
    elif args.live_only:
        runner.process_live_feed()
    else:
        runner.run()


if __name__ == "__main__":
    main()
