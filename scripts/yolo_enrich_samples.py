#!/usr/bin/env python3
"""Batch YOLO enrichment of cursor samples.

Pre-computes YOLO detection results for all cursor training samples and caches
them in data/cursor_samples/yolo_cache.jsonl.

Two enrichment passes:
  Pass 1 — Context crops (~72 samples with ctx_*.png): Run YOLO on 256x256 crop.
  Pass 2 — Loss event frame cross-reference (~3000+ samples): Match samples by
            timestamp to loss event frames, run YOLO on full 1920x1080 frame,
            check if any detection overlaps the sample's (x,y) position.

Cache invalidation: tracks YOLO model file hash. Re-runs only uncached or
invalidated entries.

Usage:
    uv run python scripts/yolo_enrich_samples.py [--force] [--model PATH]
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
SAMPLES_DIR = PROJECT_ROOT / "data" / "cursor_samples"
LOSS_EVENTS_DIR = PROJECT_ROOT / "data" / "loss_events"
YOLO_MODELS_DIR = PROJECT_ROOT / "data" / "yolo_models"
CACHE_PATH = SAMPLES_DIR / "yolo_cache.jsonl"
INDEX_PATH = SAMPLES_DIR / "index.jsonl"

# Position overlap tolerance in pixels
OVERLAP_TOLERANCE = 50


def file_hash(path: Path) -> str:
    """SHA256 hash of a file (first 16 hex chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def find_yolo_model(model_path: str | None = None) -> Path | None:
    """Find the best YOLO model to use."""
    if model_path:
        p = Path(model_path)
        if p.exists():
            return p

    # Look for fine-tuned cursor models
    pts = sorted(YOLO_MODELS_DIR.glob("cursor_*.pt"), reverse=True)
    if pts:
        return pts[0]

    # Fall back to base model
    base = YOLO_MODELS_DIR / "yolov8n.pt"
    if base.exists():
        return base

    return None


def load_index() -> list[dict]:
    """Load all sample entries from index.jsonl."""
    entries = []
    if not INDEX_PATH.exists():
        return entries
    with open(INDEX_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries


def load_cache() -> dict[str, dict]:
    """Load existing cache, keyed by filename."""
    cache = {}
    if not CACHE_PATH.exists():
        return cache
    with open(CACHE_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    cache[entry["filename"]] = entry
                except (json.JSONDecodeError, KeyError):
                    continue
    return cache


def save_cache(cache: dict[str, dict]):
    """Write cache to disk (full rewrite for consistency)."""
    with open(CACHE_PATH, "w") as f:
        for entry in sorted(cache.values(), key=lambda e: e["filename"]):
            f.write(json.dumps(entry) + "\n")


def build_loss_event_index() -> list[dict]:
    """Build a time-sorted index of all loss event frames.

    Returns list of {event_id, timestamp, frame_path} sorted by timestamp.
    """
    frames = []
    if not LOSS_EVENTS_DIR.exists():
        return frames

    for event_dir in LOSS_EVENTS_DIR.iterdir():
        if not event_dir.is_dir():
            continue
        manifest_path = event_dir / "manifest.json"
        timeline_path = event_dir / "timeline.jsonl"

        # Get event base timestamp from manifest
        event_ts = None
        event_id = event_dir.name
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                event_ts = manifest.get("trigger_utc")
            except (json.JSONDecodeError, KeyError):
                pass

        # Read individual frame timestamps from timeline
        if timeline_path.exists():
            try:
                for line in open(timeline_path):
                    line = line.strip()
                    if not line:
                        continue
                    tl = json.loads(line)
                    frame_rel = tl.get("frame")
                    utc = tl.get("utc")
                    if frame_rel and utc:
                        frame_path = event_dir / frame_rel
                        if frame_path.exists():
                            frames.append({
                                "event_id": event_id,
                                "timestamp": utc,
                                "frame_path": str(frame_path),
                            })
            except (json.JSONDecodeError, KeyError):
                pass

    frames.sort(key=lambda f: f["timestamp"])
    return frames


def find_closest_frame(sample_ts: float, frame_index: list[dict],
                       max_delta: float = 2.0) -> dict | None:
    """Binary search for closest frame within max_delta seconds."""
    if not frame_index:
        return None

    import bisect
    timestamps = [f["timestamp"] for f in frame_index]
    idx = bisect.bisect_left(timestamps, sample_ts)

    best = None
    best_delta = max_delta + 1

    for i in [idx - 1, idx]:
        if 0 <= i < len(frame_index):
            delta = abs(frame_index[i]["timestamp"] - sample_ts)
            if delta < best_delta:
                best_delta = delta
                best = frame_index[i]

    if best_delta <= max_delta:
        return best
    return None


def bbox_overlaps_point(det_bbox: tuple, x: int, y: int,
                        tolerance: int = OVERLAP_TOLERANCE) -> bool:
    """Check if a YOLO detection bbox overlaps a point within tolerance."""
    x1, y1, x2, y2 = det_bbox
    return (x1 - tolerance <= x <= x2 + tolerance and
            y1 - tolerance <= y <= y2 + tolerance)


def run_enrichment(model_path: str | None = None, force: bool = False):
    """Run the full enrichment pipeline."""
    # Find YOLO model
    yolo_path = find_yolo_model(model_path)
    if yolo_path is None:
        print("[YOLO Enrich] No YOLO model found. Aborting.")
        sys.exit(1)

    model_hash = file_hash(yolo_path)
    model_name = yolo_path.stem
    print(f"[YOLO Enrich] Model: {yolo_path.name} (hash: {model_hash})")

    # Load YOLO
    from ultralytics import YOLO
    yolo = YOLO(str(yolo_path))
    print("[YOLO Enrich] YOLO loaded.")

    # Load data
    entries = load_index()
    cache = load_cache() if not force else {}
    print(f"[YOLO Enrich] {len(entries)} samples, {len(cache)} cached")

    # Check which entries need enrichment
    needs_enrichment = []
    for entry in entries:
        fn = entry.get("filename", "")
        cached = cache.get(fn)
        if cached and cached.get("yolo_model_hash") == model_hash and not force:
            continue  # Already cached with this model version
        needs_enrichment.append(entry)

    if not needs_enrichment:
        print("[YOLO Enrich] All samples up to date. Nothing to do.")
        return

    print(f"[YOLO Enrich] {len(needs_enrichment)} samples need enrichment")

    # Build loss event frame index for Pass 2
    print("[YOLO Enrich] Building loss event frame index...")
    frame_index = build_loss_event_index()
    print(f"[YOLO Enrich] {len(frame_index)} loss event frames indexed")

    # Cache for YOLO results on full frames (avoid re-running same frame)
    frame_yolo_cache: dict[str, list] = {}

    t0 = time.time()
    enriched_count = 0
    pass1_count = 0
    pass2_count = 0
    no_source_count = 0

    for i, entry in enumerate(needs_enrichment):
        fn = entry.get("filename", "")
        sample_x = entry.get("x")
        sample_y = entry.get("y")
        sample_ts = entry.get("timestamp")

        result = {
            "filename": fn,
            "yolo_detected": False,
            "yolo_conf": None,
            "yolo_bbox": None,
            "yolo_model": model_name,
            "yolo_model_hash": model_hash,
            "enrichment_source": None,
            "loss_event_id": None,
            "enriched_at": time.time(),
        }

        # Pass 1: Context crops
        ctx_path = SAMPLES_DIR / f"ctx_{fn}"
        if ctx_path.exists():
            ctx_img = cv2.imread(str(ctx_path))
            if ctx_img is not None:
                # Run YOLO on context crop
                results = yolo.predict(ctx_img, conf=0.15, verbose=False)
                for r in results:
                    if r.boxes is not None and len(r.boxes) > 0:
                        # Take highest confidence detection
                        best_idx = r.boxes.conf.argmax()
                        box = r.boxes[best_idx]
                        coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        conf_val = float(box.conf[0].cpu())
                        result["yolo_detected"] = True
                        result["yolo_conf"] = round(conf_val, 4)
                        result["yolo_bbox"] = coords
                        result["enrichment_source"] = "context_crop"
                        pass1_count += 1
                        break

                if not result["yolo_detected"]:
                    result["enrichment_source"] = "context_crop"
                    pass1_count += 1

        # Pass 2: Loss event frame cross-reference (if no context crop result)
        elif sample_ts and sample_x is not None and sample_y is not None:
            closest = find_closest_frame(sample_ts, frame_index)
            if closest:
                frame_path = closest["frame_path"]

                # Check frame YOLO cache
                if frame_path not in frame_yolo_cache:
                    frame_img = cv2.imread(frame_path)
                    if frame_img is not None:
                        results = yolo.predict(frame_img, conf=0.15,
                                               verbose=False)
                        detections = []
                        for r in results:
                            if r.boxes is not None:
                                for box in r.boxes:
                                    coords = box.xyxy[0].cpu().numpy().astype(int).tolist()
                                    conf_val = float(box.conf[0].cpu())
                                    detections.append({
                                        "bbox": coords,
                                        "conf": conf_val,
                                    })
                        frame_yolo_cache[frame_path] = detections
                    else:
                        frame_yolo_cache[frame_path] = []

                # Check if any detection overlaps sample position
                detections = frame_yolo_cache[frame_path]
                for det in sorted(detections, key=lambda d: d["conf"],
                                  reverse=True):
                    if bbox_overlaps_point(tuple(det["bbox"]),
                                           sample_x, sample_y):
                        result["yolo_detected"] = True
                        result["yolo_conf"] = round(det["conf"], 4)
                        result["yolo_bbox"] = det["bbox"]
                        result["enrichment_source"] = "loss_event_frame"
                        result["loss_event_id"] = closest["event_id"]
                        break

                if not result["yolo_detected"]:
                    # Frame found but no overlapping detection
                    result["enrichment_source"] = "loss_event_frame"
                    result["loss_event_id"] = closest["event_id"]

                pass2_count += 1
            else:
                no_source_count += 1
        else:
            no_source_count += 1

        cache[fn] = result
        enriched_count += 1

        if (i + 1) % 100 == 0 or i == len(needs_enrichment) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"[YOLO Enrich] {i+1}/{len(needs_enrichment)} "
                  f"({rate:.1f}/s) — "
                  f"pass1={pass1_count} pass2={pass2_count} "
                  f"no_source={no_source_count}")

    # Save cache
    save_cache(cache)
    elapsed = time.time() - t0
    detected = sum(1 for v in cache.values() if v.get("yolo_detected"))
    print(f"\n[YOLO Enrich] Done in {elapsed:.1f}s")
    print(f"  Total cached: {len(cache)}")
    print(f"  YOLO detected: {detected}")
    print(f"  Pass 1 (context crops): {pass1_count}")
    print(f"  Pass 2 (loss events): {pass2_count}")
    print(f"  No enrichment source: {no_source_count}")
    print(f"  Cache file: {CACHE_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch YOLO enrichment")
    parser.add_argument("--force", action="store_true",
                        help="Re-enrich all samples (ignore cache)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to YOLO model (default: latest cursor_*.pt)")
    args = parser.parse_args()
    run_enrichment(model_path=args.model, force=args.force)
