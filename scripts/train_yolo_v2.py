"""Train YOLO v2 cursor detector with improved data.

v1 problem: trained only on loss event frames, doesn't detect cursor on live
frames (0.15 conf vs 0.71 for toolbar false positives).

v2 strategy:
  1. Live frame capture with daemon blob positions as annotations
  2. Velocity test frames with known commanded positions
  3. Recording snapshots with YOLO v1 + silhouette cross-validation
  4. Hard negatives: toolbar crops, static UI elements
  5. Diverse cursor positions and backgrounds

Usage:
    uv run python scripts/train_yolo_v2.py --collect    # Collect training data
    uv run python scripts/train_yolo_v2.py --train      # Train YOLO v2
    uv run python scripts/train_yolo_v2.py --collect --train  # Both
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
OUTPUT_DIR = PROJECT_ROOT / "data" / "yolo_training_v2"
LOSS_EVENTS_DIR = PROJECT_ROOT / "data" / "loss_events"
VTEST_DIR = PROJECT_ROOT / "data" / "velocity_test"
RECORDINGS_DIR = PROJECT_ROOT / "data" / "recordings"
CURSOR_SAMPLES_DIR = PROJECT_ROOT / "data" / "cursor_samples"

# Cursor bbox sizes (px) at 1080p
BBOX_W = 32
BBOX_H = 48  # cursor is taller than wide


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

            # Use commanded position as ground truth for V0 (stationary)
            # or tracked position for profiles where tracker was accurate
            cmd = r["commanded"]
            tracked = r["tracked"]
            error = r["error_px"]

            # Only use frames where we're confident about cursor position
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


def collect_loss_event_frames() -> tuple[list[dict], list[dict]]:
    """Collect frames from loss events. Returns (positives, negatives)."""
    positives = []
    negatives = []

    for event_dir in sorted(LOSS_EVENTS_DIR.iterdir()):
        if not event_dir.is_dir():
            continue
        tl = event_dir / "timeline.jsonl"
        if not tl.exists():
            continue

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

            if phase in ("had_it", "recovered") and 5 < cx < 1915 and 5 < cy < 1075:
                positives.append({
                    "path": frame_path,
                    "cx": cx,
                    "cy": cy,
                    "source": f"loss_{event_dir.name}",
                })
            else:
                negatives.append({
                    "path": frame_path,
                    "source": f"loss_neg_{event_dir.name}",
                })

    return positives, negatives


def capture_live_frames(n_frames: int = 50) -> list[dict]:
    """Capture live frames from the running dashboard with current cursor position.

    Requires dashboard running at localhost:8766.
    """
    import urllib.request

    frames = []
    print(f"  Capturing {n_frames} live frames from dashboard...")

    for i in range(n_frames):
        try:
            # Get state for cursor position
            with urllib.request.urlopen("http://localhost:8766/api/state", timeout=2) as resp:
                state = json.loads(resp.read())

            cursor = state.get("cursor", {})
            cx, cy = cursor.get("x", 0), cursor.get("y", 0)
            method = cursor.get("method", "")

            # Only use if position is reliable
            if method in ("sil_template", "yolo", "yolo_anchor", "probe_15") and 5 < cx < 1915 and 5 < cy < 1075:
                # Get frame
                with urllib.request.urlopen("http://localhost:8766/api/frame", timeout=2) as resp:
                    jpeg_data = resp.read()

                frame_path = OUTPUT_DIR / "live_captures" / f"live_{i:04d}.jpg"
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                frame_path.write_bytes(jpeg_data)

                frames.append({
                    "path": frame_path,
                    "cx": cx,
                    "cy": cy,
                    "source": f"live_{method}",
                })

        except Exception as e:
            pass  # Skip frame on error

        time.sleep(0.2)  # 5fps capture rate

    print(f"  Captured {len(frames)} reliable live frames")
    return frames


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

            # These are negatives (no known cursor position)
            frames.append({
                "path": out_path,
                "source": "recording_neg",
            })

        cap.release()

    return frames


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

    # Shuffle and split
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
        # Positives
        for i, item in enumerate(pos_list):
            name = f"pos_{item['source']}_{i:05d}"
            img_dst = OUTPUT_DIR / f"images/{split}/{name}.jpg"
            lbl_dst = OUTPUT_DIR / f"labels/{split}/{name}.txt"

            # Copy or symlink image
            src = item["path"]
            if src.suffix == ".jpg":
                shutil.copy2(str(src), str(img_dst))
            else:
                frame = cv2.imread(str(src))
                if frame is not None:
                    cv2.imwrite(str(img_dst), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

            lbl_dst.write_text(make_label(item["cx"], item["cy"]) + "\n")
            written[f"{split}_pos"] += 1

        # Negatives (empty label file)
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
    yaml = f"""# Cursor detection v2 dataset
path: {OUTPUT_DIR.resolve()}
train: images/train
val: images/val

nc: 1
names:
  0: cursor
"""
    (OUTPUT_DIR / "data.yaml").write_text(yaml)

    return written


def collect_all_data(skip_live: bool = False):
    """Collect training data from all sources."""
    print("=" * 60)
    print("YOLO v2 Data Collection")
    print("=" * 60)

    # Clean old data
    if OUTPUT_DIR.exists():
        for sub in ("images", "labels", "composites", "live_captures", "recording_frames"):
            p = OUTPUT_DIR / sub
            if p.exists():
                shutil.rmtree(str(p))
        print("  Cleaned old v2 training data")

    # Source 1: Velocity test frames
    print("\nStep 1: Velocity test frames...")
    vtest_frames = collect_velocity_test_frames()
    print(f"  Found {len(vtest_frames)} accurate velocity test frames")

    # Source 2: Loss event frames
    print("\nStep 2: Loss event frames...")
    loss_pos, loss_neg = collect_loss_event_frames()
    print(f"  Positives: {len(loss_pos)}, Negatives: {len(loss_neg)}")

    # Source 3: Live captures
    live_frames = []
    if not skip_live:
        print("\nStep 3: Live frame capture...")
        live_frames = capture_live_frames(n_frames=50)
    else:
        print("\nStep 3: Skipping live capture (--skip-live)")

    # Source 4: Recording backgrounds
    print("\nStep 4: Recording backgrounds (negatives)...")
    rec_frames = extract_recording_frames(max_per_recording=2, max_recordings=80)
    print(f"  Extracted {len(rec_frames)} recording frames")

    # Source 5: Composites
    print("\nStep 5: Generating composites...")
    all_backgrounds = loss_neg + rec_frames
    composites = generate_composites(all_backgrounds, n=min(500, len(all_backgrounds) * 2))
    print(f"  Generated {len(composites)} composite images")

    # Merge positives and negatives
    all_positives = vtest_frames + loss_pos + live_frames + composites
    all_negatives = loss_neg + rec_frames

    print(f"\nTotal positives: {len(all_positives)}")
    print(f"Total negatives: {len(all_negatives)}")

    # Write dataset
    print("\nStep 6: Writing YOLO dataset...")
    written = write_dataset(all_positives, all_negatives)
    for k, v in written.items():
        print(f"  {k}: {v}")

    print(f"\n  Dataset: {OUTPUT_DIR}")
    print(f"  Config: {OUTPUT_DIR / 'data.yaml'}")
    return written


def train_v2(epochs: int = 50, batch: int = 16):
    """Train YOLOv8n on v2 dataset."""
    from ultralytics import YOLO

    data_yaml = OUTPUT_DIR / "data.yaml"
    if not data_yaml.exists():
        print("ERROR: Run --collect first to create training data")
        return

    print("=" * 60)
    print("YOLO v2 Training")
    print("=" * 60)

    # Start from v1 weights (transfer learning)
    v1_path = PROJECT_ROOT / "data" / "yolo_models" / "cursor_v1_best.pt"
    if v1_path.exists():
        print(f"  Fine-tuning from v1: {v1_path}")
        model = YOLO(str(v1_path))
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
        name="cursor_v2",
        exist_ok=True,
        patience=10,  # Early stopping
        save=True,
        plots=True,
    )

    # Copy best model
    best_src = PROJECT_ROOT / "data" / "yolo_models" / "runs" / "detect" / "cursor_v2" / "weights" / "best.pt"
    best_dst = PROJECT_ROOT / "data" / "yolo_models" / "cursor_v2_best.pt"
    if best_src.exists():
        shutil.copy2(str(best_src), str(best_dst))
        print(f"\n  Best model: {best_dst}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO v2 cursor detector")
    parser.add_argument("--collect", action="store_true", help="Collect training data")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--skip-live", action="store_true", help="Skip live frame capture")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    if not args.collect and not args.train:
        parser.print_help()
        return

    if args.collect:
        collect_all_data(skip_live=args.skip_live)

    if args.train:
        train_v2(epochs=args.epochs, batch=args.batch)


if __name__ == "__main__":
    main()
