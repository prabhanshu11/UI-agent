"""Convert cursor detection training data to YOLO format.

Sources:
  1. Loss event frames (data/loss_events/*/frames/) — full 1920x1080 frames
     with tracked cursor positions from timeline.jsonl
  2. Cursor samples (data/cursor_samples/) — 64x64 patches with positions,
     composited onto loss event background frames to generate more training data

Output structure:
  data/yolo_training/
    images/train/  images/val/
    labels/train/  labels/val/
    data.yaml

YOLO label format: class_id cx cy w h (normalized 0-1)
Single class: 0 = cursor

Usage:
    uv run python scripts/convert_to_yolo.py
    uv run python scripts/convert_to_yolo.py --bbox-size 40  # larger bbox
    uv run python scripts/convert_to_yolo.py --val-ratio 0.2
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
LOSS_EVENTS_DIR = PROJECT_ROOT / "data" / "loss_events"
CURSOR_SAMPLES_DIR = PROJECT_ROOT / "data" / "cursor_samples"
OUTPUT_DIR = PROJECT_ROOT / "data" / "yolo_training"

# Standard cursor sizes at 1080p (px)
# Arrow: ~12x19, I-beam: ~4x20, Hand: ~18x22
# Use a generous bbox that covers all cursor types
DEFAULT_BBOX_W = 32
DEFAULT_BBOX_H = 32


def load_loss_event_frames(bbox_w: int, bbox_h: int) -> tuple[list[dict], list[dict]]:
    """Load frames from loss events with cursor positions.

    Returns (positive_frames, negative_frames) where each entry has:
      - image_path: Path to the JPEG frame
      - cursor_x, cursor_y: cursor position (for positives only)
      - event_id: source loss event
    """
    positives = []
    negatives = []

    for event_dir in sorted(LOSS_EVENTS_DIR.iterdir()):
        if not event_dir.is_dir():
            continue

        timeline_path = event_dir / "timeline.jsonl"
        if not timeline_path.exists():
            continue

        event_id = event_dir.name

        for line in timeline_path.read_text().strip().split("\n"):
            entry = json.loads(line)
            frame_rel = entry.get("frame")
            if not frame_rel:
                continue

            frame_path = event_dir / frame_rel
            if not frame_path.exists():
                continue

            cursor = entry.get("cursor", {})
            cx = cursor.get("x", 0)
            cy = cursor.get("y", 0)
            phase = entry.get("phase", "lost")

            if phase in ("had_it", "recovered") and cx > 0 and cy > 0:
                # Verify cursor position is within frame bounds
                if 0 < cx < 1920 and 0 < cy < 1080:
                    positives.append({
                        "image_path": frame_path,
                        "cursor_x": cx,
                        "cursor_y": cy,
                        "event_id": event_id,
                        "phase": phase,
                    })
            else:
                # Lost/recovering frames — cursor is on screen but position unknown
                # Use as hard negatives (empty label = no cursor annotation)
                negatives.append({
                    "image_path": frame_path,
                    "event_id": event_id,
                    "phase": phase,
                })

    return positives, negatives


def generate_composite_positives(
    negatives: list[dict],
    n_composites: int,
    bbox_w: int,
    bbox_h: int,
) -> list[dict]:
    """Generate synthetic positive frames by compositing cursor patches onto
    background frames from loss events.

    Takes cursor patches (64x64 with cursor) and pastes them onto full frames
    where the cursor position is unknown (lost phase). This gives us more
    positive training data with diverse backgrounds.
    """
    # Load positive cursor patches
    index_path = CURSOR_SAMPLES_DIR / "index.jsonl"
    if not index_path.exists():
        return []

    pos_patches = []
    for line in index_path.read_text().strip().split("\n"):
        entry = json.loads(line)
        if entry["label"] == "pos":
            patch_path = CURSOR_SAMPLES_DIR / entry["filename"]
            if patch_path.exists():
                pos_patches.append(patch_path)

    if not pos_patches or not negatives:
        return []

    composites = []
    composite_dir = OUTPUT_DIR / "composites"
    composite_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(n_composites, len(negatives))):
        bg_entry = negatives[i % len(negatives)]
        bg_frame = cv2.imread(str(bg_entry["image_path"]))
        if bg_frame is None:
            continue

        h, w = bg_frame.shape[:2]

        # Pick a random cursor patch
        patch_path = random.choice(pos_patches)
        patch = cv2.imread(str(patch_path))
        if patch is None:
            continue

        ph, pw = patch.shape[:2]

        # Random position (avoid edges)
        margin = max(pw, ph)
        cx = random.randint(margin, w - margin)
        cy = random.randint(margin, h - margin)

        # Paste patch onto background (centered at cx, cy)
        x1 = cx - pw // 2
        y1 = cy - ph // 2
        x2 = x1 + pw
        y2 = y1 + ph

        # Clamp to frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # Alpha blend (50% to make it look natural)
        roi = bg_frame[y1:y2, x1:x2]
        patch_crop = patch[:y2 - y1, :x2 - x1]
        if roi.shape != patch_crop.shape:
            continue

        blended = cv2.addWeighted(roi, 0.3, patch_crop, 0.7, 0)
        bg_frame[y1:y2, x1:x2] = blended

        # Save composite
        comp_name = f"composite_{i:04d}.jpg"
        comp_path = composite_dir / comp_name
        cv2.imwrite(str(comp_path), bg_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        composites.append({
            "image_path": comp_path,
            "cursor_x": cx,
            "cursor_y": cy,
            "event_id": f"composite_{i}",
            "phase": "synthetic",
        })

    return composites


def make_yolo_label(
    cursor_x: int,
    cursor_y: int,
    frame_w: int,
    frame_h: int,
    bbox_w: int,
    bbox_h: int,
) -> str:
    """Create YOLO format label line: class_id cx cy w h (normalized 0-1)."""
    # Normalize to 0-1
    cx_norm = cursor_x / frame_w
    cy_norm = cursor_y / frame_h
    w_norm = bbox_w / frame_w
    h_norm = bbox_h / frame_h

    # Clamp to valid range
    cx_norm = max(0.0, min(1.0, cx_norm))
    cy_norm = max(0.0, min(1.0, cy_norm))
    w_norm = min(w_norm, min(cx_norm, 1.0 - cx_norm) * 2)
    h_norm = min(h_norm, min(cy_norm, 1.0 - cy_norm) * 2)

    return f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"


def write_data_yaml(output_dir: Path):
    """Write YOLO data.yaml configuration."""
    yaml_content = f"""# Cursor detection dataset
# Auto-generated by convert_to_yolo.py

path: {output_dir.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: cursor
"""
    (output_dir / "data.yaml").write_text(yaml_content)


def convert(args):
    random.seed(42)

    print("Loading loss event frames...")
    positives, negatives = load_loss_event_frames(args.bbox_w, args.bbox_h)
    print(f"  Positive frames (reliable position): {len(positives)}")
    print(f"  Negative frames (unknown position): {len(negatives)}")

    # Generate composite positives from cursor patches
    if args.composites > 0:
        print(f"\nGenerating {args.composites} composite positives...")
        composites = generate_composite_positives(
            negatives, args.composites, args.bbox_w, args.bbox_h)
        print(f"  Generated: {len(composites)}")
        positives.extend(composites)

    if not positives:
        print("No positive training data! Run the dashboard to capture loss events first.")
        sys.exit(1)

    # Create output directories
    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Split into train/val
    random.shuffle(positives)
    random.shuffle(negatives)

    val_pos_count = max(1, int(len(positives) * args.val_ratio))
    val_neg_count = max(1, int(len(negatives) * args.val_ratio))

    val_positives = positives[:val_pos_count]
    train_positives = positives[val_pos_count:]
    val_negatives = negatives[:val_neg_count]
    train_negatives = negatives[val_neg_count:]

    # Limit negatives to avoid class imbalance (max 2x positives)
    max_neg = len(train_positives) * 2
    if len(train_negatives) > max_neg:
        train_negatives = train_negatives[:max_neg]
    max_neg_val = len(val_positives) * 2
    if len(val_negatives) > max_neg_val:
        val_negatives = val_negatives[:max_neg_val]

    print(f"\nDataset split:")
    print(f"  Train: {len(train_positives)} positive + {len(train_negatives)} negative")
    print(f"  Val: {len(val_positives)} positive + {len(val_negatives)} negative")

    # Frame dimensions (assumed 1920x1080)
    frame_w, frame_h = 1920, 1080

    idx = 0

    def write_sample(entry: dict, split: str, has_cursor: bool):
        nonlocal idx
        img_name = f"img_{idx:05d}.jpg"
        label_name = f"img_{idx:05d}.txt"

        # Copy/symlink image
        src = entry["image_path"]
        dst = OUTPUT_DIR / "images" / split / img_name
        shutil.copy2(src, dst)

        # Write label
        label_path = OUTPUT_DIR / "labels" / split / label_name
        if has_cursor:
            label_line = make_yolo_label(
                entry["cursor_x"], entry["cursor_y"],
                frame_w, frame_h,
                args.bbox_w, args.bbox_h,
            )
            label_path.write_text(label_line + "\n")
        else:
            # Empty label file = no objects in image
            label_path.write_text("")

        idx += 1

    # Write positive samples
    for entry in train_positives:
        write_sample(entry, "train", has_cursor=True)
    for entry in val_positives:
        write_sample(entry, "val", has_cursor=True)

    # Write negative samples (background-only)
    for entry in train_negatives:
        write_sample(entry, "train", has_cursor=False)
    for entry in val_negatives:
        write_sample(entry, "val", has_cursor=False)

    # Write data.yaml
    write_data_yaml(OUTPUT_DIR)

    total = len(train_positives) + len(val_positives) + len(train_negatives) + len(val_negatives)
    print(f"\nDataset written to {OUTPUT_DIR}")
    print(f"  Total images: {total}")
    print(f"  data.yaml: {OUTPUT_DIR / 'data.yaml'}")

    # Print sample label for verification
    if train_positives:
        entry = train_positives[0]
        label = make_yolo_label(
            entry["cursor_x"], entry["cursor_y"],
            frame_w, frame_h, args.bbox_w, args.bbox_h)
        print(f"\n  Sample label: {label}")
        print(f"    cursor at ({entry['cursor_x']}, {entry['cursor_y']}) "
              f"→ bbox {args.bbox_w}x{args.bbox_h}px")


def main():
    parser = argparse.ArgumentParser(description="Convert cursor data to YOLO format")
    parser.add_argument("--bbox-w", type=int, default=DEFAULT_BBOX_W,
                        help="Bounding box width in pixels (default: 32)")
    parser.add_argument("--bbox-h", type=int, default=DEFAULT_BBOX_H,
                        help="Bounding box height in pixels (default: 32)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation split ratio (default: 0.2)")
    parser.add_argument("--composites", type=int, default=300,
                        help="Number of synthetic composite images to generate (default: 300)")
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
