"""Generate synthetic YOLO training data from KVM recordings.

Extracts diverse background frames from MKV recordings and composites
cursor patches onto them to create labeled training data.

This augments the dataset from convert_to_yolo.py with much greater
background diversity (421 recordings vs 9 loss events).

Sources:
  - data/recordings/*.mkv — 1280x720 KVM capture recordings
  - data/cursor_samples/  — 64x64 cursor patches with confidence scores

Output: adds synth_*.jpg + synth_*.txt into data/yolo_training/{images,labels}/{train,val}/

Usage:
    uv run python scripts/synth_from_recordings.py
    uv run python scripts/synth_from_recordings.py --n-backgrounds 2000 --composites-per-bg 3
    uv run python scripts/synth_from_recordings.py --sample-interval 15
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RECORDINGS_DIR = PROJECT_ROOT / "data" / "recordings"
CURSOR_SAMPLES_DIR = PROJECT_ROOT / "data" / "cursor_samples"
OUTPUT_DIR = PROJECT_ROOT / "data" / "yolo_training"

DEFAULT_BBOX_W = 32
DEFAULT_BBOX_H = 32


def extract_backgrounds(
    recordings_dir: Path,
    sample_interval_s: float,
    max_backgrounds: int,
    seed: int = 42,
) -> list[dict]:
    """Extract frames from MKV recordings at regular intervals.

    Returns list of dicts with 'frame' (numpy array), 'source', 'frame_idx'.
    """
    rng = random.Random(seed)
    recordings = sorted(recordings_dir.glob("*.mkv"))
    if not recordings:
        print("No MKV recordings found!")
        return []

    print(f"  Found {len(recordings)} recordings")

    # Distribute budget across recordings
    frames_per_recording = max(1, max_backgrounds // len(recordings))

    backgrounds = []

    for i, rec_path in enumerate(recordings):
        cap = cv2.VideoCapture(str(rec_path))
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Frame positions at the sample interval
        interval_frames = int(sample_interval_s * fps)
        if interval_frames < 1:
            interval_frames = 1

        frame_positions = list(range(0, total_frames, interval_frames))

        # Randomly sample if too many
        if len(frame_positions) > frames_per_recording:
            frame_positions = rng.sample(frame_positions, frames_per_recording)

        for frame_idx in sorted(frame_positions):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Skip very dark/bright frames (transitions, black screens)
            mean_brightness = frame.mean()
            if mean_brightness < 15 or mean_brightness > 245:
                continue

            backgrounds.append({
                "frame": frame,
                "source": rec_path.name,
                "frame_idx": frame_idx,
            })

            if len(backgrounds) >= max_backgrounds:
                cap.release()
                print(f"  Reached {max_backgrounds} backgrounds at recording {i+1}/{len(recordings)}")
                return backgrounds

        cap.release()

        # Progress
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(recordings)} recordings, {len(backgrounds)} backgrounds so far")

    return backgrounds


def load_cursor_patches(min_confidence: float = 0.7) -> list[np.ndarray]:
    """Load positive cursor patches from cursor_samples directory."""
    index_path = CURSOR_SAMPLES_DIR / "index.jsonl"
    if not index_path.exists():
        return []

    patches = []
    for line in index_path.read_text().strip().split("\n"):
        entry = json.loads(line)
        if entry["label"] == "pos" and entry.get("confidence", 0) >= min_confidence:
            patch_path = CURSOR_SAMPLES_DIR / entry["filename"]
            if patch_path.exists():
                img = cv2.imread(str(patch_path))
                if img is not None:
                    patches.append(img)

    return patches


def augment_cursor_patch(patch: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply random augmentations to a cursor patch."""
    h, w = patch.shape[:2]

    # Random scale (0.7x - 1.5x)
    scale = rng.uniform(0.7, 1.5)
    new_w = max(4, int(w * scale))
    new_h = max(4, int(h * scale))
    patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Random brightness adjustment (±20%)
    brightness = rng.uniform(0.8, 1.2)
    patch = np.clip(patch * brightness, 0, 255).astype(np.uint8)

    return patch


def composite_cursor(
    bg_frame: np.ndarray,
    cursor_patch: np.ndarray,
    cx: int,
    cy: int,
    alpha: float = 0.85,
) -> np.ndarray:
    """Composite cursor patch onto background at (cx, cy)."""
    result = bg_frame.copy()
    h, w = result.shape[:2]
    ph, pw = cursor_patch.shape[:2]

    # Position patch centered at (cx, cy)
    x1 = cx - pw // 2
    y1 = cy - ph // 2
    x2 = x1 + pw
    y2 = y1 + ph

    # Source crop offsets (handle edge clipping)
    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    src_x2 = src_x1 + (x2 - x1)
    src_y2 = src_y1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return result

    roi = result[y1:y2, x1:x2]
    patch_crop = cursor_patch[src_y1:src_y2, src_x1:src_x2]

    if roi.shape != patch_crop.shape:
        return result

    blended = cv2.addWeighted(roi, 1.0 - alpha, patch_crop, alpha, 0)
    result[y1:y2, x1:x2] = blended
    return result


def make_yolo_label(
    cx: int, cy: int,
    frame_w: int, frame_h: int,
    bbox_w: int, bbox_h: int,
) -> str:
    """Create YOLO label: class_id cx cy w h (normalized 0-1)."""
    cx_norm = cx / frame_w
    cy_norm = cy / frame_h
    w_norm = bbox_w / frame_w
    h_norm = bbox_h / frame_h

    cx_norm = max(0.0, min(1.0, cx_norm))
    cy_norm = max(0.0, min(1.0, cy_norm))
    w_norm = min(w_norm, min(cx_norm, 1.0 - cx_norm) * 2)
    h_norm = min(h_norm, min(cy_norm, 1.0 - cy_norm) * 2)

    return f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"


def clean_old_synth(output_dir: Path):
    """Remove previously generated synthetic files."""
    count = 0
    for split in ("train", "val"):
        for f in (output_dir / "images" / split).glob("synth_*.jpg"):
            f.unlink()
            count += 1
        for f in (output_dir / "labels" / split).glob("synth_*.txt"):
            f.unlink()
            count += 1
    if count:
        print(f"  Cleaned {count} old synthetic files")


def generate(args):
    rng = random.Random(args.seed)

    # Clean old synthetic data
    print("Cleaning old synthetic data...")
    clean_old_synth(OUTPUT_DIR)

    # Step 1: Extract backgrounds
    print("\nStep 1: Extracting background frames from recordings...")
    backgrounds = extract_backgrounds(
        RECORDINGS_DIR, args.sample_interval, args.n_backgrounds, seed=args.seed,
    )
    print(f"  Total: {len(backgrounds)} backgrounds")

    if not backgrounds:
        print("No backgrounds extracted! Check data/recordings/ directory.")
        sys.exit(1)

    # Step 2: Load cursor patches
    print("\nStep 2: Loading cursor patches...")
    cursor_patches = load_cursor_patches(min_confidence=args.min_confidence)
    print(f"  Loaded {len(cursor_patches)} patches (conf >= {args.min_confidence})")

    if not cursor_patches:
        print("No cursor patches found!")
        sys.exit(1)

    # Step 3: Generate synthetic images
    print(f"\nStep 3: Generating synthetic YOLO data...")
    print(f"  {args.composites_per_bg} composites per background + {args.negative_ratio:.0%} negatives")

    # Ensure output dirs exist
    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Split backgrounds into train/val
    rng.shuffle(backgrounds)
    val_count = max(1, int(len(backgrounds) * args.val_ratio))
    val_backgrounds = backgrounds[:val_count]
    train_backgrounds = backgrounds[val_count:]

    stats = {"train_pos": 0, "train_neg": 0, "val_pos": 0, "val_neg": 0}
    synth_idx = 0

    for split, bgs in [("train", train_backgrounds), ("val", val_backgrounds)]:
        img_dir = OUTPUT_DIR / "images" / split
        lbl_dir = OUTPUT_DIR / "labels" / split

        for bg_entry in bgs:
            bg = bg_entry["frame"]
            h, w = bg.shape[:2]

            # Generate cursor composites
            for _ in range(args.composites_per_bg):
                patch = rng.choice(cursor_patches).copy()
                patch = augment_cursor_patch(patch, rng)

                ph, pw = patch.shape[:2]
                margin = max(ph, pw) + 10
                if margin >= w - margin or margin >= h - margin:
                    continue

                cx = rng.randint(margin, w - margin)
                cy = rng.randint(margin, h - margin)

                alpha = rng.uniform(0.7, 0.95)
                synth_frame = composite_cursor(bg, patch, cx, cy, alpha)

                img_name = f"synth_{synth_idx:05d}.jpg"
                lbl_name = f"synth_{synth_idx:05d}.txt"

                cv2.imwrite(
                    str(img_dir / img_name), synth_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )

                label = make_yolo_label(cx, cy, w, h, args.bbox_w, args.bbox_h)
                (lbl_dir / lbl_name).write_text(label + "\n")

                stats[f"{split}_pos"] += 1
                synth_idx += 1

            # Add background as negative (no cursor)
            if rng.random() < args.negative_ratio:
                img_name = f"synth_{synth_idx:05d}.jpg"
                lbl_name = f"synth_{synth_idx:05d}.txt"

                cv2.imwrite(
                    str(img_dir / img_name), bg,
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
                (lbl_dir / lbl_name).write_text("")

                stats[f"{split}_neg"] += 1
                synth_idx += 1

        print(f"  {split}: {stats[f'{split}_pos']} positive + {stats[f'{split}_neg']} negative")

    total_new = sum(stats.values())
    print(f"\n  Generated {total_new} synthetic images")

    # Report total dataset size
    all_train = list((OUTPUT_DIR / "images" / "train").glob("*.jpg"))
    all_val = list((OUTPUT_DIR / "images" / "val").glob("*.jpg"))
    print(f"\n  Full dataset (existing + synthetic):")
    print(f"    Train: {len(all_train)} images")
    print(f"    Val:   {len(all_val)} images")
    print(f"    Total: {len(all_train) + len(all_val)} images")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic YOLO training data from KVM recordings",
    )
    parser.add_argument("--n-backgrounds", type=int, default=1000,
                        help="Max background frames to extract (default: 1000)")
    parser.add_argument("--sample-interval", type=float, default=10.0,
                        help="Extract 1 frame per N seconds from each recording (default: 10)")
    parser.add_argument("--composites-per-bg", type=int, default=2,
                        help="Cursor composites per background frame (default: 2)")
    parser.add_argument("--negative-ratio", type=float, default=0.3,
                        help="Fraction of backgrounds also kept as negatives (default: 0.3)")
    parser.add_argument("--bbox-w", type=int, default=DEFAULT_BBOX_W,
                        help="YOLO bbox width in px (default: 32)")
    parser.add_argument("--bbox-h", type=int, default=DEFAULT_BBOX_H,
                        help="YOLO bbox height in px (default: 32)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Validation split ratio (default: 0.15)")
    parser.add_argument("--min-confidence", type=float, default=0.7,
                        help="Min cursor patch confidence to use (default: 0.7)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
