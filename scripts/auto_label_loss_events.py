"""Auto-label cursor patches from loss event data.

Mines loss_events/ directories to extract training data for both CNN and YOLO:

CNN patches (64x64 grayscale):
  - Positives: "had_it" phase with live_cnn confidence > threshold
  - Negatives: "lost" phase with retrospective_cnn confidence < threshold
  - Hard negatives: blob_patches from busy screens

YOLO annotations (full-frame):
  - Full frames from "had_it" phase with confirmed cursor positions
  - Creates bounding box annotations in YOLO format

Usage:
    uv run python scripts/auto_label_loss_events.py
    uv run python scripts/auto_label_loss_events.py --cnn-only
    uv run python scripts/auto_label_loss_events.py --yolo-only
    uv run python scripts/auto_label_loss_events.py --pos-threshold 0.8 --neg-threshold 0.15
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
LOSS_EVENTS_DIR = PROJECT_ROOT / "data" / "loss_events"
CURSOR_SAMPLES_DIR = PROJECT_ROOT / "data" / "cursor_samples"
YOLO_OUTPUT_DIR = PROJECT_ROOT / "data" / "yolo_training_v3"

# Cursor bbox size for YOLO annotations (pixels in 1280x720 frame)
CURSOR_BBOX_W = 32
CURSOR_BBOX_H = 32


def patch_hash(patch_path: Path) -> str:
    """Perceptual hash: downsample to 8x8, threshold at mean."""
    img = cv2.imread(str(patch_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return hashlib.md5(patch_path.read_bytes()).hexdigest()
    small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    mean = small.mean()
    bits = (small > mean).flatten()
    return "".join(str(int(b)) for b in bits)


def load_existing_hashes(samples_dir: Path) -> set:
    """Load perceptual hashes of existing samples to avoid duplicates."""
    hashes = set()
    for png in samples_dir.glob("*.png"):
        h = patch_hash(png)
        hashes.add(h)
    return hashes


def load_existing_index(samples_dir: Path) -> tuple[int, int]:
    """Get current pos/neg count from index.jsonl."""
    index_path = samples_dir / "index.jsonl"
    pos_count = 0
    neg_count = 0
    if index_path.exists():
        with open(index_path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("label") == "pos":
                    pos_count += 1
                elif entry.get("label") == "neg":
                    neg_count += 1
    return pos_count, neg_count


def mine_cnn_patches(
    pos_threshold: float = 0.7,
    neg_threshold: float = 0.2,
    max_blob_negatives_per_event: int = 10,
    skip_duplicates: bool = True,
):
    """Mine CNN training patches from loss events.

    Returns (positives, negatives, stats) where each entry is a dict
    with patch_path and metadata.
    """
    events = sorted(LOSS_EVENTS_DIR.iterdir())
    events = [e for e in events if e.is_dir() and (e / "timeline.jsonl").exists()]
    print(f"Found {len(events)} loss events with timelines")

    # Load existing hashes for dedup
    existing_hashes = set()
    if skip_duplicates:
        print("Computing hashes of existing samples...")
        existing_hashes = load_existing_hashes(CURSOR_SAMPLES_DIR)
        print(f"  {len(existing_hashes)} existing sample hashes loaded")

    positives = []
    negatives = []
    blob_negatives = []
    stats = {
        "events_processed": 0,
        "had_it_total": 0,
        "had_it_high_cnn": 0,
        "lost_total": 0,
        "lost_low_retro": 0,
        "blobs_total": 0,
        "duplicates_skipped": 0,
        "missing_patches": 0,
    }

    for event_dir in events:
        event_id = event_dir.name
        timeline_path = event_dir / "timeline.jsonl"
        patches_dir = event_dir / "patches"
        blob_patches_dir = event_dir / "blob_patches"

        if not patches_dir.exists():
            continue

        stats["events_processed"] += 1

        # Read timeline
        timeline = []
        with open(timeline_path) as f:
            for line in f:
                if line.strip():
                    timeline.append(json.loads(line))

        # Process each frame
        blob_neg_count = 0
        for entry in timeline:
            phase = entry.get("phase", "")
            idx = entry.get("index", 0)
            cursor = entry.get("cursor", {})
            live_cnn = entry.get("live_cnn", {})
            retro_cnn = entry.get("retrospective_cnn", {})
            velocity = entry.get("velocity_px_s", 0)

            # Construct patch filename
            patch_file = entry.get("patch", "")
            if patch_file:
                patch_path = event_dir / patch_file
            else:
                # Try pattern matching
                candidates = list(patches_dir.glob(f"p{idx:04d}_*.png"))
                patch_path = candidates[0] if candidates else None

            if not patch_path or not patch_path.exists():
                stats["missing_patches"] += 1
                continue

            # Dedup check
            if skip_duplicates:
                h = patch_hash(patch_path)
                if h in existing_hashes:
                    stats["duplicates_skipped"] += 1
                    continue
                existing_hashes.add(h)

            # --- POSITIVE: had_it + high live CNN ---
            if phase == "had_it":
                stats["had_it_total"] += 1
                live_conf = live_cnn.get("confidence", 0)
                if live_conf > pos_threshold:
                    stats["had_it_high_cnn"] += 1
                    positives.append({
                        "patch_path": patch_path,
                        "event_id": event_id,
                        "x": cursor.get("x", 0),
                        "y": cursor.get("y", 0),
                        "live_cnn": live_conf,
                        "velocity": velocity,
                        "source": f"loss_event_had_it",
                    })

            # --- NEGATIVE: lost + low retrospective CNN ---
            elif phase == "lost":
                stats["lost_total"] += 1
                retro_conf = retro_cnn.get("confidence", 0)
                if retro_conf < neg_threshold:
                    stats["lost_low_retro"] += 1
                    negatives.append({
                        "patch_path": patch_path,
                        "event_id": event_id,
                        "x": cursor.get("x", 0),
                        "y": cursor.get("y", 0),
                        "retro_cnn": retro_conf,
                        "velocity": velocity,
                        "source": f"loss_event_lost",
                    })

        # --- BLOB NEGATIVES: patches at blob positions ---
        if blob_patches_dir.exists() and blob_neg_count < max_blob_negatives_per_event:
            for bp in sorted(blob_patches_dir.glob("b*.png")):
                if blob_neg_count >= max_blob_negatives_per_event:
                    break
                stats["blobs_total"] += 1

                if skip_duplicates:
                    h = patch_hash(bp)
                    if h in existing_hashes:
                        stats["duplicates_skipped"] += 1
                        continue
                    existing_hashes.add(h)

                blob_negatives.append({
                    "patch_path": bp,
                    "event_id": event_id,
                    "source": "loss_event_blob",
                })
                blob_neg_count += 1

    return positives, negatives, blob_negatives, stats


def mine_yolo_annotations(
    pos_threshold: float = 0.7,
    frame_w: int = 1280,
    frame_h: int = 720,
):
    """Mine YOLO training data from loss event full frames.

    Returns list of (frame_path, bbox_annotation) for confirmed cursor positions.
    """
    events = sorted(LOSS_EVENTS_DIR.iterdir())
    events = [e for e in events if e.is_dir() and (e / "timeline.jsonl").exists()]

    annotations = []
    seen_positions = set()  # Dedup by approximate position

    for event_dir in events:
        timeline_path = event_dir / "timeline.jsonl"
        frames_dir = event_dir / "frames"

        if not frames_dir.exists():
            continue

        with open(timeline_path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                phase = entry.get("phase", "")
                live_cnn = entry.get("live_cnn", {})
                cursor = entry.get("cursor", {})

                if phase != "had_it":
                    continue
                if live_cnn.get("confidence", 0) < pos_threshold:
                    continue

                cx = cursor.get("x", 0)
                cy = cursor.get("y", 0)
                if cx <= 0 or cy <= 0:
                    continue

                # Dedup: skip if we already have a sample within 20px
                pos_key = (cx // 20, cy // 20)
                if pos_key in seen_positions:
                    continue
                seen_positions.add(pos_key)

                # Find the frame file
                frame_file = entry.get("frame", "")
                if frame_file:
                    frame_path = event_dir / frame_file
                else:
                    idx = entry.get("index", 0)
                    candidates = list(frames_dir.glob(f"f{idx:04d}_*.jpg"))
                    frame_path = candidates[0] if candidates else None

                if not frame_path or not frame_path.exists():
                    continue

                # YOLO format: class x_center y_center width height (normalized)
                x_center = cx / frame_w
                y_center = cy / frame_h
                w_norm = CURSOR_BBOX_W / frame_w
                h_norm = CURSOR_BBOX_H / frame_h

                # Clamp to valid range
                x_center = max(w_norm / 2, min(1 - w_norm / 2, x_center))
                y_center = max(h_norm / 2, min(1 - h_norm / 2, y_center))

                annotations.append({
                    "frame_path": frame_path,
                    "annotation": f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}",
                    "event_id": event_dir.name,
                    "cx": cx,
                    "cy": cy,
                    "live_cnn": live_cnn.get("confidence", 0),
                })

    return annotations


def save_cnn_patches(positives, negatives, blob_negatives):
    """Copy patches to cursor_samples/ and append to index.jsonl."""
    CURSOR_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Get current counters
    existing_pos, existing_neg = load_existing_index(CURSOR_SAMPLES_DIR)
    pos_counter = existing_pos
    neg_counter = existing_neg

    index_path = CURSOR_SAMPLES_DIR / "index.jsonl"
    new_entries = 0

    with open(index_path, "a") as f:
        # Save positives
        for item in positives:
            pos_counter += 1
            filename = f"pos_{pos_counter:06d}.png"
            dst = CURSOR_SAMPLES_DIR / filename
            shutil.copy2(item["patch_path"], dst)

            entry = {
                "filename": filename,
                "label": "pos",
                "x": item["x"],
                "y": item["y"],
                "source": item["source"],
                "confidence": item["live_cnn"],
                "velocity": item.get("velocity", 0),
                "event_id": item["event_id"],
                "timestamp": 0,
            }
            f.write(json.dumps(entry) + "\n")
            new_entries += 1

        # Save negatives (from lost phase)
        for item in negatives:
            neg_counter += 1
            filename = f"neg_{neg_counter:06d}.png"
            dst = CURSOR_SAMPLES_DIR / filename
            shutil.copy2(item["patch_path"], dst)

            entry = {
                "filename": filename,
                "label": "neg",
                "x": item["x"],
                "y": item["y"],
                "source": item["source"],
                "confidence": item.get("retro_cnn", 0),
                "velocity": item.get("velocity", 0),
                "event_id": item["event_id"],
                "timestamp": 0,
            }
            f.write(json.dumps(entry) + "\n")
            new_entries += 1

        # Save blob negatives
        for item in blob_negatives:
            neg_counter += 1
            filename = f"neg_{neg_counter:06d}.png"
            dst = CURSOR_SAMPLES_DIR / filename
            shutil.copy2(item["patch_path"], dst)

            entry = {
                "filename": filename,
                "label": "neg",
                "x": 0,
                "y": 0,
                "source": item["source"],
                "confidence": 0.0,
                "event_id": item["event_id"],
                "timestamp": 0,
            }
            f.write(json.dumps(entry) + "\n")
            new_entries += 1

    print(f"\nSaved {new_entries} new entries to {index_path}")
    print(f"  Positives: {existing_pos} → {pos_counter} (+{pos_counter - existing_pos})")
    print(f"  Negatives: {existing_neg} → {neg_counter} (+{neg_counter - existing_neg})")
    return pos_counter, neg_counter


def save_yolo_annotations(annotations, val_ratio: float = 0.15):
    """Save YOLO training data with train/val split."""
    import random
    random.shuffle(annotations)

    val_count = int(len(annotations) * val_ratio)
    val_set = annotations[:val_count]
    train_set = annotations[val_count:]

    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        img_dir = YOLO_OUTPUT_DIR / "images" / split_name
        lbl_dir = YOLO_OUTPUT_DIR / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for i, item in enumerate(split_data):
            stem = f"loss_{item['event_id']}_{i:04d}"
            dst_img = img_dir / f"{stem}.jpg"
            dst_lbl = lbl_dir / f"{stem}.txt"

            shutil.copy2(item["frame_path"], dst_img)
            dst_lbl.write_text(item["annotation"] + "\n")

    # Write dataset YAML
    yaml_path = YOLO_OUTPUT_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {YOLO_OUTPUT_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['cursor']\n"
    )

    print(f"\nYOLO data saved to {YOLO_OUTPUT_DIR}")
    print(f"  Train: {len(train_set)} images")
    print(f"  Val: {len(val_set)} images")
    print(f"  Dataset config: {yaml_path}")

    return len(train_set), len(val_set)


def extract_random_negatives_from_frames(
    n_per_event: int = 5,
    frame_w: int = 1280,
    frame_h: int = 720,
    patch_size: int = 64,
):
    """Extract random negative patches from full frames.

    Picks random locations far from the tracked cursor position.
    These are background patches — guaranteed not at cursor location.
    """
    import random
    events = sorted(LOSS_EVENTS_DIR.iterdir())
    events = [e for e in events if e.is_dir() and (e / "timeline.jsonl").exists()]

    negatives = []
    half = patch_size // 2

    for event_dir in events:
        timeline_path = event_dir / "timeline.jsonl"
        frames_dir = event_dir / "frames"
        if not frames_dir.exists():
            continue

        # Read timeline to get cursor positions (to avoid them)
        cursor_positions = []
        with open(timeline_path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                cx = entry.get("cursor", {}).get("x", 0)
                cy = entry.get("cursor", {}).get("y", 0)
                if cx > 0 and cy > 0:
                    cursor_positions.append((cx, cy))

        # Pick a few frames from this event
        all_frames = sorted(frames_dir.glob("f*.jpg"))
        if not all_frames:
            continue

        selected = random.sample(all_frames, min(n_per_event, len(all_frames)))

        for frame_path in selected:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue
            fh, fw = frame.shape[:2]

            # Pick random position far from any cursor position
            for _ in range(3):  # 3 attempts per frame
                rx = random.randint(half, fw - half - 1)
                ry = random.randint(half, fh - half - 1)

                # Check distance from all known cursor positions
                min_dist = min(
                    ((rx - cx) ** 2 + (ry - cy) ** 2) ** 0.5
                    for cx, cy in cursor_positions
                ) if cursor_positions else 999

                if min_dist < 100:  # Too close to cursor
                    continue

                # Extract patch
                patch = frame[ry - half:ry + half, rx - half:rx + half]
                if patch.shape[:2] != (patch_size, patch_size):
                    continue

                gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

                negatives.append({
                    "patch": gray,
                    "event_id": event_dir.name,
                    "x": rx,
                    "y": ry,
                    "source": "random_background",
                })

    return negatives


def save_random_negatives(random_negs):
    """Save random background negatives to cursor_samples/."""
    CURSOR_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    _, existing_neg = load_existing_index(CURSOR_SAMPLES_DIR)
    neg_counter = existing_neg

    index_path = CURSOR_SAMPLES_DIR / "index.jsonl"
    with open(index_path, "a") as f:
        for item in random_negs:
            neg_counter += 1
            filename = f"neg_{neg_counter:06d}.png"
            dst = CURSOR_SAMPLES_DIR / filename
            cv2.imwrite(str(dst), item["patch"])

            entry = {
                "filename": filename,
                "label": "neg",
                "x": item["x"],
                "y": item["y"],
                "source": item["source"],
                "confidence": 0.0,
                "event_id": item["event_id"],
                "timestamp": 0,
            }
            f.write(json.dumps(entry) + "\n")

    print(f"  Random negatives: {existing_neg} → {neg_counter} "
          f"(+{neg_counter - existing_neg})")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label cursor patches from loss events")
    parser.add_argument("--cnn-only", action="store_true",
                        help="Only extract CNN patches, skip YOLO")
    parser.add_argument("--yolo-only", action="store_true",
                        help="Only extract YOLO annotations, skip CNN")
    parser.add_argument("--pos-threshold", type=float, default=0.7,
                        help="Live CNN confidence threshold for positives")
    parser.add_argument("--neg-threshold", type=float, default=0.2,
                        help="Retrospective CNN confidence threshold for negatives")
    parser.add_argument("--max-blob-neg", type=int, default=10,
                        help="Max blob negatives per event")
    parser.add_argument("--random-neg-per-event", type=int, default=5,
                        help="Random background negatives to extract per event")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Skip duplicate detection (faster but may add dupes)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count what would be extracted without saving")
    args = parser.parse_args()

    if not LOSS_EVENTS_DIR.exists():
        print(f"No loss events directory at {LOSS_EVENTS_DIR}")
        sys.exit(1)

    # --- CNN patches ---
    if not args.yolo_only:
        print("=" * 60)
        print("Phase 1: Mining CNN patches from loss events")
        print("=" * 60)

        positives, negatives, blob_negatives, stats = mine_cnn_patches(
            pos_threshold=args.pos_threshold,
            neg_threshold=args.neg_threshold,
            max_blob_negatives_per_event=args.max_blob_neg,
            skip_duplicates=not args.no_dedup,
        )

        print(f"\nMining stats:")
        print(f"  Events processed: {stats['events_processed']}")
        print(f"  had_it total: {stats['had_it_total']}, "
              f"high CNN: {stats['had_it_high_cnn']}")
        print(f"  lost total: {stats['lost_total']}, "
              f"low retro: {stats['lost_low_retro']}")
        print(f"  Blob patches: {stats['blobs_total']}")
        print(f"  Duplicates skipped: {stats['duplicates_skipped']}")
        print(f"  Missing patches: {stats['missing_patches']}")
        print(f"\nExtracted:")
        print(f"  Positives: {len(positives)}")
        print(f"  Negatives (lost): {len(negatives)}")
        print(f"  Negatives (blobs): {len(blob_negatives)}")

        if not args.dry_run:
            save_cnn_patches(positives, negatives, blob_negatives)

            # Also extract random background negatives
            if args.random_neg_per_event > 0:
                print(f"\nExtracting random background negatives...")
                random_negs = extract_random_negatives_from_frames(
                    n_per_event=args.random_neg_per_event,
                )
                print(f"  Extracted {len(random_negs)} random negatives")
                save_random_negatives(random_negs)

    # --- YOLO annotations ---
    if not args.cnn_only:
        print("\n" + "=" * 60)
        print("Phase 2: Mining YOLO annotations from loss events")
        print("=" * 60)

        annotations = mine_yolo_annotations(
            pos_threshold=args.pos_threshold,
        )
        print(f"\nExtracted {len(annotations)} YOLO annotations")

        if not args.dry_run:
            save_yolo_annotations(annotations)

    print("\nDone!")


if __name__ == "__main__":
    main()
