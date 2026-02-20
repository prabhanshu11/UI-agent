"""Train YOLO v4 cursor detector with time-corrected, CNN-filtered labels.

v2/v3 problem: trained on loss event frames where ground truth came from
sil_motion_roi and motion_track — the same unreliable systems causing false
detections now. Most entries have validated=false, meaning labels may point
at the wrong position.

v4 strategy:
  1. Filter loss event positives by retrospective_cnn.confidence > 0.80
  2. Temporal sampling: max 1 frame per 2s to avoid near-duplicate overfitting
  3. Hard negatives: taskbar, system tray, notification area crops
  4. Velocity test frames with error < 20px (high-confidence ground truth)
  5. Composites for diversity
  6. Fine-tune from v3

Usage:
    uv run python scripts/train_yolo_v4.py --collect    # Collect training data
    uv run python scripts/train_yolo_v4.py --train      # Train YOLO v4
    uv run python scripts/train_yolo_v4.py --collect --train  # Both
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "yolo_training_v4"
LOSS_EVENTS_DIR = PROJECT_ROOT / "data" / "loss_events"
VTEST_DIR = PROJECT_ROOT / "data" / "velocity_test"
RECORDINGS_DIR = PROJECT_ROOT / "data" / "recordings"
CURSOR_SAMPLES_DIR = PROJECT_ROOT / "data" / "cursor_samples"

# Cursor bbox sizes (px) at 1080p
BBOX_W = 32
BBOX_H = 48  # cursor is taller than wide

# Minimum retrospective CNN confidence for positive labels
MIN_RETRO_CNN_CONF = 0.80

# Minimum temporal gap between sampled frames (seconds)
MIN_TEMPORAL_GAP_S = 2.0

# Hard negative regions at 1080p (taskbar, system tray, notification area)
HARD_NEGATIVE_REGIONS = [
    (0, 1040, 1920, 1080),       # Bottom taskbar
    (1750, 1040, 1920, 1080),    # System tray (bottom-right)
    (0, 0, 1920, 40),            # Top bar / title bars
    (1800, 0, 1920, 40),         # Top-right notification area
    (0, 0, 60, 1080),            # Left edge (dock/panel)
]


def make_label(cx: int, cy: int, fw: int = 1920, fh: int = 1080) -> str:
    """YOLO label: class cx cy w h (normalized 0-1)."""
    cx_n = max(0.001, min(0.999, cx / fw))
    cy_n = max(0.001, min(0.999, cy / fh))
    w_n = min(BBOX_W / fw, min(cx_n, 1.0 - cx_n) * 2)
    h_n = min(BBOX_H / fh, min(cy_n, 1.0 - cy_n) * 2)
    return f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}"


def collect_velocity_test_frames() -> list[dict]:
    """Collect frames from velocity test runs with commanded positions."""
    frames = []
    for run_dir in sorted(VTEST_DIR.glob("run_*")):
        results_path = run_dir / "results.jsonl"
        if not results_path.exists():
            continue
        for line in results_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            frame_name = r.get("frame")
            if not frame_name:
                continue
            frame_path = run_dir / "frames" / frame_name
            if not frame_path.exists():
                continue

            tracked = r["tracked"]
            error = r["error_px"]

            # Only use frames where position is accurate
            if r["profile"] == "V0" or error < 20:
                cx = tracked["x"]
                cy = tracked["y"]
                if 5 < cx < 1915 and 5 < cy < 1075:
                    frames.append({
                        "path": frame_path,
                        "cx": cx,
                        "cy": cy,
                        "source": f"vtest_{r['profile']}",
                    })

    return frames


def collect_loss_event_frames_v4() -> tuple[list[dict], list[dict]]:
    """Collect frames from loss events with CNN-filtered labels.

    Key v4 improvements:
    - Only use had_it frames where retrospective_cnn.confidence > 0.80
    - Temporal sampling: skip frames within 2s of each other per event
    - Lost phase frames are negatives (cursor NOT at tracked position)
    """
    positives = []
    negatives = []
    skipped_low_conf = 0
    skipped_temporal = 0

    for event_dir in sorted(LOSS_EVENTS_DIR.iterdir()):
        if not event_dir.is_dir():
            continue
        tl = event_dir / "timeline.jsonl"
        if not tl.exists():
            continue

        last_used_t = -999.0

        for line in tl.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            frame_rel = entry.get("frame")
            if not frame_rel:
                continue
            frame_path = event_dir / frame_rel
            if not frame_path.exists():
                continue

            cursor = entry.get("cursor", {})
            cx, cy = cursor.get("x", 0), cursor.get("y", 0)
            phase = entry.get("phase", "lost")
            t_offset = entry.get("t_offset_s", 0)
            retro_conf = entry.get("retrospective_cnn", {}).get("confidence", 0)

            if phase in ("had_it", "recovered") and 5 < cx < 1915 and 5 < cy < 1075:
                # v4: Only trust positions where retrospective CNN agrees
                if retro_conf < MIN_RETRO_CNN_CONF:
                    skipped_low_conf += 1
                    continue

                # v4: Temporal sampling — skip near-duplicate frames
                if t_offset - last_used_t < MIN_TEMPORAL_GAP_S:
                    skipped_temporal += 1
                    continue
                last_used_t = t_offset

                positives.append({
                    "path": frame_path,
                    "cx": cx,
                    "cy": cy,
                    "source": f"loss_{event_dir.name}",
                    "retro_conf": retro_conf,
                    "utc": entry.get("utc", ""),
                })
            elif phase == "lost":
                negatives.append({
                    "path": frame_path,
                    "source": f"loss_neg_{event_dir.name}",
                })

    print(f"  Skipped {skipped_low_conf} frames with retro_cnn < {MIN_RETRO_CNN_CONF}")
    print(f"  Skipped {skipped_temporal} frames within {MIN_TEMPORAL_GAP_S}s of each other")

    return positives, negatives


def extract_recording_frames(max_per_recording: int = 3, max_recordings: int = 100) -> list[dict]:
    """Extract random frames from .mkv recordings for background variety."""
    frames = []
    recordings = sorted(RECORDINGS_DIR.glob("*.mkv"))
    random.shuffle(recordings)

    for rec in recordings[:max_recordings]:
        cap = cv2.VideoCapture(str(rec))
        if not cap.isOpened():
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 10:
            cap.release()
            continue

        for _ in range(max_per_recording):
            frame_idx = random.randint(5, total - 5)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            out_name = f"rec_{rec.stem}_{frame_idx:05d}.jpg"
            out_path = OUTPUT_DIR / "recording_frames" / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            frames.append({
                "path": out_path,
                "source": "recording_neg",
            })

        cap.release()

    return frames


def extract_hard_negatives(max_total: int = 200) -> list[dict]:
    """Extract crops of taskbar, system tray, notification area.

    These are the exact regions motion_track falsely detects as cursor.
    Training YOLO to NOT detect these as cursor is critical.
    """
    hard_negs = []
    recordings = sorted(RECORDINGS_DIR.glob("*.mkv"))
    random.shuffle(recordings)

    per_recording = max(1, max_total // min(len(recordings), 50))

    for rec in recordings[:50]:
        cap = cv2.VideoCapture(str(rec))
        if not cap.isOpened():
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 10:
            cap.release()
            continue

        for i in range(per_recording):
            frame_idx = random.randint(5, total - 5)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            region = random.choice(HARD_NEGATIVE_REGIONS)
            x1, y1, x2, y2 = region

            # Scale for actual frame size (regions defined at 1080p)
            sx, sy = w / 1920, h / 1080
            rx1 = int(x1 * sx)
            ry1 = int(y1 * sy)
            rx2 = int(x2 * sx)
            ry2 = int(y2 * sy)

            crop = frame[ry1:ry2, rx1:rx2]
            if crop.size == 0:
                continue

            # Pad crop to full frame size (YOLO expects consistent input)
            padded = np.zeros((h, w, 3), dtype=np.uint8)
            padded[ry1:ry2, rx1:rx2] = crop

            out_name = f"hardneg_{rec.stem}_{frame_idx:05d}_{i}.jpg"
            out_path = OUTPUT_DIR / "hard_negatives" / out_name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), padded, [cv2.IMWRITE_JPEG_QUALITY, 90])

            hard_negs.append({
                "path": out_path,
                "source": "hard_negative",
            })

        cap.release()

        if len(hard_negs) >= max_total:
            break

    return hard_negs


def generate_composites(backgrounds: list[dict], n: int = 500) -> list[dict]:
    """Composite cursor patches onto background frames for diversity."""
    index_path = CURSOR_SAMPLES_DIR / "index.jsonl"
    if not index_path.exists():
        return []

    pos_patches = []
    for line in index_path.read_text().splitlines():
        entry = json.loads(line)
        if entry["label"] == "pos":
            p = CURSOR_SAMPLES_DIR / entry["filename"]
            if p.exists():
                pos_patches.append(p)

    if not pos_patches or not backgrounds:
        return []

    comp_dir = OUTPUT_DIR / "composites"
    comp_dir.mkdir(parents=True, exist_ok=True)

    composites = []
    for i in range(min(n, len(backgrounds) * 3)):
        bg_entry = random.choice(backgrounds)
        bg = cv2.imread(str(bg_entry["path"]))
        if bg is None:
            continue

        h, w = bg.shape[:2]
        patch = cv2.imread(str(random.choice(pos_patches)))
        if patch is None:
            continue

        ph, pw = patch.shape[:2]
        margin = max(pw, ph) + 10
        cx = random.randint(margin, w - margin)
        cy = random.randint(margin, h - margin)

        x1 = max(0, cx - pw // 2)
        y1 = max(0, cy - ph // 2)
        x2 = min(w, x1 + pw)
        y2 = min(h, y1 + ph)

        roi = bg[y1:y2, x1:x2]
        pc = patch[:y2 - y1, :x2 - x1]
        if roi.shape != pc.shape:
            continue

        bg[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.2, pc, 0.8, 0)

        comp_name = f"comp_{i:05d}.jpg"
        comp_path = comp_dir / comp_name
        cv2.imwrite(str(comp_path), bg, [cv2.IMWRITE_JPEG_QUALITY, 90])

        composites.append({
            "path": comp_path,
            "cx": cx,
            "cy": cy,
            "source": "composite",
        })

    return composites


def write_dataset(positives: list[dict], negatives: list[dict], val_ratio: float = 0.2):
    """Write YOLO format dataset with train/val split."""
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)

    random.shuffle(positives)
    random.shuffle(negatives)

    n_val_pos = max(1, int(len(positives) * val_ratio))
    n_val_neg = max(1, int(len(negatives) * val_ratio))

    pos_train = positives[n_val_pos:]
    pos_val = positives[:n_val_pos]
    neg_train = negatives[n_val_neg:]
    neg_val = negatives[:n_val_neg]

    written = {"train_pos": 0, "train_neg": 0, "val_pos": 0, "val_neg": 0}

    for split, pos_list, neg_list in [
        ("train", pos_train, neg_train),
        ("val", pos_val, neg_val),
    ]:
        for i, item in enumerate(pos_list):
            name = f"pos_{item['source']}_{i:05d}"
            img_dst = OUTPUT_DIR / f"images/{split}/{name}.jpg"
            lbl_dst = OUTPUT_DIR / f"labels/{split}/{name}.txt"

            src = item["path"]
            if src.suffix == ".jpg":
                shutil.copy2(str(src), str(img_dst))
            else:
                frame = cv2.imread(str(src))
                if frame is not None:
                    cv2.imwrite(str(img_dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            lbl_dst.write_text(make_label(item["cx"], item["cy"]) + "\n")
            written[f"{split}_pos"] += 1

        for i, item in enumerate(neg_list):
            name = f"neg_{item['source']}_{i:05d}"
            img_dst = OUTPUT_DIR / f"images/{split}/{name}.jpg"
            lbl_dst = OUTPUT_DIR / f"labels/{split}/{name}.txt"

            src = item["path"]
            if src.suffix == ".jpg":
                shutil.copy2(str(src), str(img_dst))
            else:
                frame = cv2.imread(str(src))
                if frame is not None:
                    cv2.imwrite(str(img_dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            lbl_dst.write_text("")  # Empty = no cursor
            written[f"{split}_neg"] += 1

    # Write data.yaml
    yaml = f"""# Cursor detection v4 dataset
# Filtered by retrospective CNN confidence > {MIN_RETRO_CNN_CONF}
# Temporal sampling: 1 frame per {MIN_TEMPORAL_GAP_S}s
path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: cursor
"""
    (OUTPUT_DIR / "data.yaml").write_text(yaml)

    # Write metadata sidecar with UTC timestamps for future temporal analysis
    metadata = []
    for item in positives:
        metadata.append({
            "source": item["source"],
            "cx": item["cx"],
            "cy": item["cy"],
            "retro_conf": item.get("retro_conf", 0),
            "utc": item.get("utc", ""),
        })
    meta_path = OUTPUT_DIR / "metadata.jsonl"
    with open(meta_path, "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    return written


def collect_all_data():
    """Collect training data from all sources."""
    print("=" * 60)
    print("YOLO v4 Data Collection (CNN-filtered + temporal sampling)")
    print("=" * 60)

    # Clean old data
    if OUTPUT_DIR.exists():
        for sub in ("images", "labels", "composites", "recording_frames", "hard_negatives"):
            p = OUTPUT_DIR / sub
            if p.exists():
                shutil.rmtree(str(p))
        print("  Cleaned old v4 training data")

    # Source 1: Velocity test frames (highest quality ground truth)
    print("\nStep 1: Velocity test frames...")
    vtest_frames = collect_velocity_test_frames()
    print(f"  Found {len(vtest_frames)} accurate velocity test frames")

    # Source 2: Loss event frames (CNN-filtered)
    print("\nStep 2: Loss event frames (retro_cnn > {:.0%})...".format(MIN_RETRO_CNN_CONF))
    loss_pos, loss_neg = collect_loss_event_frames_v4()
    print(f"  Positives: {len(loss_pos)}, Negatives: {len(loss_neg)}")

    # Source 3: Recording backgrounds (negatives)
    print("\nStep 3: Recording backgrounds (negatives)...")
    rec_frames = extract_recording_frames(max_per_recording=2, max_recordings=80)
    print(f"  Extracted {len(rec_frames)} recording frames")

    # Source 4: Hard negatives (taskbar, system tray crops)
    print("\nStep 4: Hard negatives (taskbar/system tray)...")
    hard_negs = extract_hard_negatives(max_total=200)
    print(f"  Extracted {len(hard_negs)} hard negative crops")

    # Source 5: Composites
    print("\nStep 5: Generating composites...")
    all_backgrounds = loss_neg + rec_frames
    composites = generate_composites(all_backgrounds, n=min(500, len(all_backgrounds) * 2))
    print(f"  Generated {len(composites)} composite images")

    # Merge
    all_positives = vtest_frames + loss_pos + composites
    all_negatives = loss_neg + rec_frames + hard_negs

    print(f"\nTotal positives: {len(all_positives)}")
    print(f"  Velocity test: {len(vtest_frames)}")
    print(f"  Loss events (CNN-filtered): {len(loss_pos)}")
    print(f"  Composites: {len(composites)}")
    print(f"Total negatives: {len(all_negatives)}")
    print(f"  Loss event negatives: {len(loss_neg)}")
    print(f"  Recording backgrounds: {len(rec_frames)}")
    print(f"  Hard negatives: {len(hard_negs)}")

    # Write dataset
    print("\nStep 6: Writing YOLO dataset...")
    written = write_dataset(all_positives, all_negatives)
    for k, v in written.items():
        print(f"  {k}: {v}")

    print(f"\n  Dataset: {OUTPUT_DIR}")
    print(f"  Config: {OUTPUT_DIR / 'data.yaml'}")
    print(f"  Metadata: {OUTPUT_DIR / 'metadata.jsonl'}")
    return written


def train_v4(epochs: int = 50, batch: int = 16):
    """Train YOLOv8n on v4 dataset, fine-tuning from v3."""
    from ultralytics import YOLO

    data_yaml = OUTPUT_DIR / "data.yaml"
    if not data_yaml.exists():
        print("ERROR: Run --collect first to create training data")
        return

    print("=" * 60)
    print("YOLO v4 Training")
    print("=" * 60)

    # Fine-tune from v3 (best available)
    v3_path = PROJECT_ROOT / "data" / "yolo_models" / "cursor_v3_best.pt"
    if v3_path.exists():
        print(f"  Fine-tuning from v3: {v3_path}")
        model = YOLO(str(v3_path))
    else:
        # Fallback chain: v2 → v1 → pretrained
        for fallback in ["cursor_v2_best.pt", "cursor_v1_best.pt"]:
            fb_path = PROJECT_ROOT / "data" / "yolo_models" / fallback
            if fb_path.exists():
                print(f"  Fine-tuning from fallback: {fb_path}")
                model = YOLO(str(fb_path))
                break
        else:
            print("  Training from pretrained yolov8n.pt")
            model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=batch,
        device=0,
        project=str(PROJECT_ROOT / "data" / "yolo_models" / "runs" / "detect"),
        name="cursor_v4",
        exist_ok=True,
        patience=10,
        save=True,
        plots=True,
    )

    # Copy best model
    best_src = PROJECT_ROOT / "data" / "yolo_models" / "runs" / "detect" / "cursor_v4" / "weights" / "best.pt"
    best_dst = PROJECT_ROOT / "data" / "yolo_models" / "cursor_v4_best.pt"
    if best_src.exists():
        shutil.copy2(str(best_src), str(best_dst))
        print(f"\n  Best model: {best_dst}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO v4 cursor detector")
    parser.add_argument("--collect", action="store_true", help="Collect training data")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    if not args.collect and not args.train:
        parser.print_help()
        return

    if args.collect:
        collect_all_data()

    if args.train:
        train_v4(epochs=args.epochs, batch=args.batch)


if __name__ == "__main__":
    main()
