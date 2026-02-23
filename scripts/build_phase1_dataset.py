"""Build synchronized Phase 1 training dataset for vision cursor model.

Combines two data sources into a unified (full_frame, cursor_x, cursor_y) dataset:

  1. Passthrough gold + MKV recordings:
     Extract full 1920x1080 frames from MKV at timestamps matching
     passthrough_gold entries in index.jsonl. These are the highest quality
     labels (hardware ground truth from ESP32 mouse passthrough).

  2. YOLO v4 training images:
     Already full 1920x1080 frames with cursor bounding box labels.
     Convert YOLO normalized center (cx, cy) to pixel coordinates.
     Quality: retro-CNN confidence >0.80.

Output:
    data/phase1_dataset/
      frames/           # Full 1920x1080 JPEG frames
      labels.parquet    # Columns: frame_id, cursor_x, cursor_y, timestamp, source, confidence

Usage:
    uv run python scripts/build_phase1_dataset.py                   # Build from both sources
    uv run python scripts/build_phase1_dataset.py --gold-only       # Only passthrough gold
    uv run python scripts/build_phase1_dataset.py --yolo-only       # Only YOLO v4
    uv run python scripts/build_phase1_dataset.py --dry-run         # Show counts without extracting
    uv run python scripts/build_phase1_dataset.py --max-per-mkv 200 # Limit samples per MKV
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
CURSOR_SAMPLES = PROJECT_ROOT / "data" / "cursor_samples" / "index.jsonl"
RECORDINGS_DIR = PROJECT_ROOT / "data" / "recordings"
YOLO_V4_DIR = PROJECT_ROOT / "data" / "yolo_training_v4"
OUTPUT_DIR = PROJECT_ROOT / "data" / "phase1_dataset"

# Target output resolution (passthrough gold coordinates are in this space)
FRAME_W, FRAME_H = 1920, 1080
# MKV recordings may be 720p — frames are upscaled to match coordinate space
JPEG_QUALITY = 92  # Balance between size and quality


def build_mkv_index() -> list[tuple[float, float, str]]:
    """Build sorted (start_ts, end_ts, filename) index using actual MKV durations.

    Uses ffprobe to get real duration of each file, avoiding the gap bug where
    timestamps between recording sessions were incorrectly assigned to MKVs.
    """
    mkvs = sorted(f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".mkv"))
    index = []
    for mkv in mkvs:
        name = mkv.replace("kvm_", "").replace(".mkv", "")
        try:
            dt = datetime.strptime(name, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        start_ts = dt.timestamp()

        # Get actual duration from ffprobe
        mkv_path = os.path.join(str(RECORDINGS_DIR), mkv)
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "json", mkv_path],
                capture_output=True, text=True, timeout=5,
            )
            dur = float(json.loads(result.stdout)["format"]["duration"])
            end_ts = start_ts + dur
        except (subprocess.TimeoutExpired, KeyError, json.JSONDecodeError, ValueError):
            # Fallback: assume 300s (5 min) chunk
            end_ts = start_ts + 300

        index.append((start_ts, end_ts, mkv))
    return index


def find_mkv_for_timestamp(
    ts: float, mkv_index: list[tuple[float, float, str]]
) -> tuple[str, float] | None:
    """Binary search for MKV containing timestamp. Returns (filename, offset_seconds)."""
    lo, hi = 0, len(mkv_index) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        start, end, mkv = mkv_index[mid]
        if ts < start:
            hi = mid - 1
        elif ts > end:
            lo = mid + 1
        else:
            return mkv, ts - start
    return None


def extract_frame_ffmpeg(mkv_path: str, offset_s: float) -> np.ndarray | None:
    """Extract a single frame from MKV at given offset using ffmpeg.

    Handles resolution mismatch: MKVs may be 720p while coordinates are 1080p.
    Upscales to 1920x1080 to match the passthrough gold coordinate space.
    """
    # Use ffmpeg to extract as JPEG (more robust than raw pipe for variable sizes)
    cmd = [
        "ffmpeg",
        "-ss", f"{offset_s:.3f}",
        "-i", mkv_path,
        "-frames:v", "1",
        "-f", "image2",
        "-vcodec", "mjpeg",
        "-q:v", "2",
        "-v", "quiet",
        "pipe:1",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=10)
        if result.returncode != 0 or len(result.stdout) == 0:
            return None
        # Decode JPEG
        jpg_array = np.frombuffer(result.stdout, dtype=np.uint8)
        frame = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
        if frame is None:
            return None
        h, w = frame.shape[:2]
        # Upscale to 1080p if needed (to match coordinate space)
        if w != FRAME_W or h != FRAME_H:
            frame = cv2.resize(frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_LANCZOS4)
        return frame
    except (subprocess.TimeoutExpired, Exception):
        return None


def extract_frames_batch_ffmpeg(
    mkv_path: str, offsets: list[tuple[float, int]]
) -> dict[int, np.ndarray]:
    """Extract multiple frames from one MKV using sequential single-frame seeks.

    Each frame is extracted individually via ffmpeg JPEG pipe (~100KB per frame,
    ~200ms per seek). Safe for any batch size — no risk of OOM.

    Args:
        mkv_path: Path to MKV file
        offsets: List of (offset_seconds, entry_index) pairs

    Returns:
        Dict mapping entry_index -> frame (numpy array)
    """
    if not offsets:
        return {}

    results = {}
    for offset_s, idx in sorted(offsets, key=lambda x: x[0]):
        frame = extract_frame_ffmpeg(mkv_path, offset_s)
        if frame is not None:
            results[idx] = frame
    return results


def process_passthrough_gold(
    mkv_index: list, max_per_mkv: int, dry_run: bool
) -> list[dict]:
    """Extract frames for passthrough gold entries. Returns label records."""
    # Load gold entries
    gold_entries = []
    with open(CURSOR_SAMPLES) as f:
        for line in f:
            e = json.loads(line)
            if e.get("source") == "passthrough_gold" and "x" in e and "y" in e:
                ts = e.get("timestamp")
                if isinstance(ts, (int, float)):
                    gold_entries.append(e)

    print(f"  Passthrough gold entries with coords: {len(gold_entries)}")

    # Group by MKV file
    mkv_groups: dict[str, list[tuple[float, int, dict]]] = defaultdict(list)
    for i, e in enumerate(gold_entries):
        result = find_mkv_for_timestamp(e["timestamp"], mkv_index)
        if result:
            mkv_name, offset = result
            mkv_groups[mkv_name].append((offset, i, e))

    print(f"  Grouped into {len(mkv_groups)} MKV files")

    if dry_run:
        total = sum(min(len(v), max_per_mkv) for v in mkv_groups.values())
        print(f"  Would extract: {total} frames (dry run)")
        return []

    frames_dir = OUTPUT_DIR / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    total_extracted = 0

    for mkv_name, entries in sorted(mkv_groups.items()):
        # Apply per-MKV limit (sample evenly if needed)
        if len(entries) > max_per_mkv:
            step = len(entries) / max_per_mkv
            entries = [entries[int(i * step)] for i in range(max_per_mkv)]

        mkv_path = str(RECORDINGS_DIR / mkv_name)
        if not os.path.exists(mkv_path):
            print(f"    [skip] {mkv_name}: file not found")
            continue

        offsets = [(offset, i) for offset, i, _ in entries]
        entry_map = {i: e for _, i, e in entries}

        print(f"    Extracting {len(entries)} frames from {mkv_name}...", end="", flush=True)
        frames = extract_frames_batch_ffmpeg(mkv_path, offsets)

        extracted = 0
        for idx, frame in frames.items():
            e = entry_map[idx]
            frame_id = f"gold_{total_extracted + extracted:06d}"
            frame_path = frames_dir / f"{frame_id}.jpg"
            cv2.imwrite(
                str(frame_path),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            labels.append({
                "frame_id": frame_id,
                "cursor_x": int(e["x"]),
                "cursor_y": int(e["y"]),
                "timestamp": float(e["timestamp"]),
                "source": "passthrough_gold",
                "confidence": float(e.get("confidence", 1.0)),
            })
            extracted += 1

        total_extracted += extracted
        print(f" {extracted}/{len(entries)}")

    print(f"  Total passthrough gold frames extracted: {total_extracted}")
    return labels


def process_yolo_v4(dry_run: bool) -> list[dict]:
    """Convert YOLO v4 positive images into Phase 1 dataset entries."""
    meta_path = YOLO_V4_DIR / "metadata.jsonl"
    labels_dir_train = YOLO_V4_DIR / "labels" / "train"
    labels_dir_val = YOLO_V4_DIR / "labels" / "val"
    images_dir_train = YOLO_V4_DIR / "images" / "train"
    images_dir_val = YOLO_V4_DIR / "images" / "val"

    # Load metadata for confidence and timestamp info
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                if not line.strip():
                    continue
                m = json.loads(line)
                # Key by source + position for matching
                key = f"{m.get('source', '')}_{m.get('cx', 0)}_{m.get('cy', 0)}"
                metadata[key] = m

    # Collect positive images (those with non-empty label files)
    positives = []
    for split, img_dir, lbl_dir in [
        ("train", images_dir_train, labels_dir_train),
        ("val", images_dir_val, labels_dir_val),
    ]:
        if not img_dir.exists():
            continue
        for img_file in sorted(img_dir.iterdir()):
            if not img_file.suffix in (".jpg", ".jpeg", ".png"):
                continue
            lbl_file = lbl_dir / (img_file.stem + ".txt")
            if not lbl_file.exists():
                continue
            content = lbl_file.read_text().strip()
            if not content:
                continue  # Negative sample (empty label)
            # Parse YOLO label: class cx cy w h (normalized)
            parts = content.split()
            if len(parts) >= 3:
                cx_norm = float(parts[1])
                cy_norm = float(parts[2])
                cx_px = int(cx_norm * FRAME_W)
                cy_px = int(cy_norm * FRAME_H)
                positives.append({
                    "img_path": img_file,
                    "cx": cx_px,
                    "cy": cy_px,
                    "split": split,
                })

    print(f"  YOLO v4 positive images found: {len(positives)}")

    if dry_run:
        print(f"  Would copy: {len(positives)} frames (dry run)")
        return []

    frames_dir = OUTPUT_DIR / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    labels = []
    for i, pos in enumerate(positives):
        frame_id = f"yolo_{i:06d}"
        dst = frames_dir / f"{frame_id}.jpg"

        # Symlink instead of copy to save disk space
        src = pos["img_path"]
        if dst.exists():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            # Fallback to copy if symlinks not supported
            shutil.copy2(src, dst)

        # Try to get metadata-based confidence
        meta_key = f"*_{pos['cx']}_{pos['cy']}"
        conf = 0.8  # Default for YOLO v4 (retro-CNN filtered)

        labels.append({
            "frame_id": frame_id,
            "cursor_x": pos["cx"],
            "cursor_y": pos["cy"],
            "timestamp": 0.0,  # Not available for YOLO images
            "source": f"yolo_v4_{pos['split']}",
            "confidence": conf,
        })

    print(f"  Total YOLO v4 frames added: {len(labels)}")
    return labels


def save_labels(labels: list[dict]):
    """Save labels as both Parquet and JSONL."""
    output_parquet = OUTPUT_DIR / "labels.parquet"
    output_jsonl = OUTPUT_DIR / "labels.jsonl"

    # Save JSONL (always works, no extra deps)
    with open(output_jsonl, "w") as f:
        for lbl in labels:
            f.write(json.dumps(lbl) + "\n")
    print(f"  Labels saved to {output_jsonl} ({len(labels)} entries)")

    # Try Parquet (preferred for training)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.table({
            "frame_id": [l["frame_id"] for l in labels],
            "cursor_x": [l["cursor_x"] for l in labels],
            "cursor_y": [l["cursor_y"] for l in labels],
            "timestamp": [l["timestamp"] for l in labels],
            "source": [l["source"] for l in labels],
            "confidence": [l["confidence"] for l in labels],
        })
        pq.write_table(table, output_parquet)
        print(f"  Labels saved to {output_parquet}")
    except ImportError:
        print("  [skip] pyarrow not installed — using JSONL only")
        print("  Install with: uv add pyarrow")


def print_summary(labels: list[dict]):
    """Print dataset summary statistics."""
    print(f"\n{'='*50}")
    print(f"  PHASE 1 DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"  Total labeled frames: {len(labels)}")

    from collections import Counter
    sources = Counter(l["source"] for l in labels)
    print(f"  By source:")
    for src, count in sources.most_common():
        print(f"    {src}: {count}")

    if labels:
        xs = [l["cursor_x"] for l in labels]
        ys = [l["cursor_y"] for l in labels]
        print(f"  Cursor X: mean={np.mean(xs):.0f}, std={np.std(xs):.0f}")
        print(f"  Cursor Y: mean={np.mean(ys):.0f}, std={np.std(ys):.0f}")

        # Quadrant balance
        xs_arr, ys_arr = np.array(xs), np.array(ys)
        tl = ((xs_arr < 960) & (ys_arr < 540)).sum()
        tr = ((xs_arr >= 960) & (ys_arr < 540)).sum()
        bl = ((xs_arr < 960) & (ys_arr >= 540)).sum()
        br = ((xs_arr >= 960) & (ys_arr >= 540)).sum()
        total = len(labels)
        print(f"  Quadrant balance:")
        print(f"    TL: {tl:>5} ({tl/total*100:5.1f}%) | TR: {tr:>5} ({tr/total*100:5.1f}%)")
        print(f"    BL: {bl:>5} ({bl/total*100:5.1f}%) | BR: {br:>5} ({br/total*100:5.1f}%)")

    frames_dir = OUTPUT_DIR / "frames"
    if frames_dir.exists():
        # Count real files vs symlinks
        n_files = len(list(frames_dir.iterdir()))
        n_symlinks = sum(1 for f in frames_dir.iterdir() if f.is_symlink())
        print(f"  Frames directory: {n_files} files ({n_symlinks} symlinks)")
        # Estimate disk usage (only real files)
        real_size = sum(
            f.stat().st_size
            for f in frames_dir.iterdir()
            if not f.is_symlink()
        )
        print(f"  Disk usage (extracted frames): {real_size / 1024**2:.0f} MB")

    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Build Phase 1 synchronized training dataset"
    )
    parser.add_argument(
        "--gold-only", action="store_true",
        help="Only extract passthrough gold frames"
    )
    parser.add_argument(
        "--yolo-only", action="store_true",
        help="Only include YOLO v4 frames"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show counts without extracting frames"
    )
    parser.add_argument(
        "--max-per-mkv", type=int, default=500,
        help="Max samples to extract per MKV file (default: 500)"
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove existing dataset before building"
    )
    args = parser.parse_args()

    if args.clean and OUTPUT_DIR.exists():
        print(f"Cleaning {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_labels = []

    # Source 1: Passthrough gold + MKV extraction
    if not args.yolo_only:
        print("\n[1/2] Processing passthrough gold → MKV frame extraction")
        mkv_index = build_mkv_index()
        gold_labels = process_passthrough_gold(
            mkv_index, args.max_per_mkv, args.dry_run
        )
        all_labels.extend(gold_labels)

    # Source 2: YOLO v4 positive images
    if not args.gold_only:
        print("\n[2/2] Processing YOLO v4 positive images")
        yolo_labels = process_yolo_v4(args.dry_run)
        all_labels.extend(yolo_labels)

    if not args.dry_run and all_labels:
        print("\nSaving labels...")
        save_labels(all_labels)
        print_summary(all_labels)
    elif args.dry_run:
        print(f"\n[dry-run] Total frames that would be created: {len(all_labels) if all_labels else 'see counts above'}")


if __name__ == "__main__":
    main()
