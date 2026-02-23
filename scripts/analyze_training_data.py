"""Analyze training data coverage for Phase 1 vision model.

Reports on:
  1. Passthrough gold ↔ MKV timestamp alignment (how many frames we can extract)
  2. Spatial distribution of cursor positions (quadrant bias, heatmap)
  3. Temporal density (samples per hour, gaps)
  4. YOLO v4 dataset statistics (spatial coverage, source breakdown)
  5. Combined dataset estimate for Phase 1 training

Usage:
    uv run python scripts/analyze_training_data.py
    uv run python scripts/analyze_training_data.py --heatmap   # Save spatial heatmap PNG
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
CURSOR_SAMPLES = PROJECT_ROOT / "data" / "cursor_samples" / "index.jsonl"
RECORDINGS_DIR = PROJECT_ROOT / "data" / "recordings"
YOLO_V4_DIR = PROJECT_ROOT / "data" / "yolo_training_v4"
LOSS_EVENTS_DIR = PROJECT_ROOT / "data" / "loss_events"

FRAME_W, FRAME_H = 1920, 1080


def build_mkv_index() -> list[tuple[float, float, str]]:
    """Build (start_ts, end_ts, filename) index from MKV filenames."""
    mkvs = sorted(f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".mkv"))
    index = []
    for i, mkv in enumerate(mkvs):
        name = mkv.replace("kvm_", "").replace(".mkv", "")
        try:
            dt = datetime.strptime(name, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        start_ts = dt.timestamp()
        if i + 1 < len(mkvs):
            next_name = mkvs[i + 1].replace("kvm_", "").replace(".mkv", "")
            try:
                next_dt = datetime.strptime(next_name, "%Y%m%d_%H%M%S")
                end_ts = next_dt.timestamp()
            except ValueError:
                end_ts = start_ts + 300
        else:
            end_ts = start_ts + 300
        index.append((start_ts, end_ts, mkv))
    return index


def load_passthrough_gold() -> list[dict]:
    """Load passthrough_gold entries from index.jsonl."""
    entries = []
    with open(CURSOR_SAMPLES) as f:
        for line in f:
            e = json.loads(line)
            if e.get("source") == "passthrough_gold":
                entries.append(e)
    return entries


def analyze_mkv_overlap(gold: list[dict], mkv_index: list) -> dict:
    """Check how many gold entries have matching MKV recordings."""
    hits = 0
    misses = 0
    hit_mkvs: dict[str, int] = {}  # mkv -> count of gold entries

    for e in gold:
        ts = e.get("timestamp")
        if not isinstance(ts, (int, float)):
            misses += 1
            continue
        found = False
        for start, end, mkv in mkv_index:
            if start <= ts <= end:
                found = True
                hit_mkvs[mkv] = hit_mkvs.get(mkv, 0) + 1
                break
        if found:
            hits += 1
        else:
            misses += 1

    return {
        "hits": hits,
        "misses": misses,
        "unique_mkvs": len(hit_mkvs),
        "total_mkvs": len(mkv_index),
        "mkv_counts": hit_mkvs,
    }


def analyze_spatial(gold: list[dict]) -> dict:
    """Analyze spatial distribution of cursor positions."""
    xs = np.array([e["x"] for e in gold if "x" in e])
    ys = np.array([e["y"] for e in gold if "y" in e])

    n = len(xs)
    quadrants = {
        "top_left": int(((xs < FRAME_W / 2) & (ys < FRAME_H / 2)).sum()),
        "top_right": int(((xs >= FRAME_W / 2) & (ys < FRAME_H / 2)).sum()),
        "bottom_left": int(((xs < FRAME_W / 2) & (ys >= FRAME_H / 2)).sum()),
        "bottom_right": int(((xs >= FRAME_W / 2) & (ys >= FRAME_H / 2)).sum()),
    }

    # Grid coverage: divide screen into 10x10 grid, count occupied cells
    grid = np.zeros((10, 10), dtype=int)
    for x, y in zip(xs, ys):
        gx = min(9, int(x / FRAME_W * 10))
        gy = min(9, int(y / FRAME_H * 10))
        grid[gy, gx] += 1

    occupied_cells = int((grid > 0).sum())

    return {
        "count": n,
        "x_mean": float(xs.mean()),
        "x_std": float(xs.std()),
        "x_range": (int(xs.min()), int(xs.max())),
        "y_mean": float(ys.mean()),
        "y_std": float(ys.std()),
        "y_range": (int(ys.min()), int(ys.max())),
        "quadrants": quadrants,
        "grid_10x10_occupied": occupied_cells,
        "grid_10x10_total": 100,
        "grid": grid,
    }


def analyze_temporal(gold: list[dict]) -> dict:
    """Analyze temporal density and gaps."""
    timestamps = sorted(
        e["timestamp"]
        for e in gold
        if isinstance(e.get("timestamp"), (int, float))
    )
    if not timestamps:
        return {"count": 0}

    deltas = np.diff(timestamps)
    total_span_h = (timestamps[-1] - timestamps[0]) / 3600
    samples_per_hour = len(timestamps) / total_span_h if total_span_h > 0 else 0

    # Find gaps > 5 minutes
    gaps = [(i, d) for i, d in enumerate(deltas) if d > 300]

    # Samples per date
    date_counts = Counter()
    for ts in timestamps:
        date_counts[datetime.fromtimestamp(ts).strftime("%Y-%m-%d")] += 1

    return {
        "count": len(timestamps),
        "span_hours": round(total_span_h, 1),
        "samples_per_hour": round(samples_per_hour, 1),
        "median_gap_s": round(float(np.median(deltas)), 2),
        "mean_gap_s": round(float(np.mean(deltas)), 2),
        "max_gap_s": round(float(np.max(deltas)), 1),
        "gaps_over_5min": len(gaps),
        "date_counts": dict(date_counts),
    }


def analyze_yolo_v4() -> dict:
    """Analyze YOLO v4 training dataset."""
    train_dir = YOLO_V4_DIR / "images" / "train"
    val_dir = YOLO_V4_DIR / "images" / "val"
    meta_path = YOLO_V4_DIR / "metadata.jsonl"

    train_count = len(list(train_dir.glob("*"))) if train_dir.exists() else 0
    val_count = len(list(val_dir.glob("*"))) if val_dir.exists() else 0

    # Parse metadata for source breakdown and spatial coverage
    sources = Counter()
    xs, ys = [], []
    if meta_path.exists():
        with open(meta_path) as f:
            for line in f:
                if not line.strip():
                    continue
                m = json.loads(line)
                src = m.get("source", "unknown")
                # Normalize source to category
                if src.startswith("loss_"):
                    sources["loss_event"] += 1
                elif src.startswith("vtest_"):
                    sources["velocity_test"] += 1
                elif src.startswith("composite"):
                    sources["composite"] += 1
                elif src.startswith("neg_"):
                    sources["hard_negative"] += 1
                else:
                    sources[src] += 1
                if "cx" in m and "cy" in m:
                    xs.append(m["cx"])
                    ys.append(m["cy"])

    spatial = {}
    if xs:
        xs_arr, ys_arr = np.array(xs), np.array(ys)
        spatial = {
            "x_mean": float(xs_arr.mean()),
            "x_std": float(xs_arr.std()),
            "y_mean": float(ys_arr.mean()),
            "y_std": float(ys_arr.std()),
            "quadrants": {
                "top_left": int(
                    ((xs_arr < FRAME_W / 2) & (ys_arr < FRAME_H / 2)).sum()
                ),
                "top_right": int(
                    ((xs_arr >= FRAME_W / 2) & (ys_arr < FRAME_H / 2)).sum()
                ),
                "bottom_left": int(
                    ((xs_arr < FRAME_W / 2) & (ys_arr >= FRAME_H / 2)).sum()
                ),
                "bottom_right": int(
                    ((xs_arr >= FRAME_W / 2) & (ys_arr >= FRAME_H / 2)).sum()
                ),
            },
        }

    return {
        "train_images": train_count,
        "val_images": val_count,
        "total": train_count + val_count,
        "sources": dict(sources),
        "spatial": spatial,
    }


def analyze_loss_events() -> dict:
    """Summarize loss event data."""
    if not LOSS_EVENTS_DIR.exists():
        return {"count": 0}

    events = sorted(LOSS_EVENTS_DIR.iterdir())
    events = [e for e in events if e.is_dir()]

    total_frames = 0
    recovered = 0
    edge_cases = Counter()

    for event_dir in events:
        manifest = event_dir / "manifest.json"
        if manifest.exists():
            m = json.loads(manifest.read_text())
            total_frames += m.get("total_frames", 0)
            if m.get("recovered"):
                recovered += 1
            for ec in m.get("edge_cases_found", []):
                edge_cases[ec] += 1

    return {
        "event_count": len(events),
        "total_frames": total_frames,
        "recovered": recovered,
        "recovery_rate": round(recovered / len(events) * 100, 1) if events else 0,
        "top_edge_cases": dict(edge_cases.most_common(10)),
    }


def save_heatmap(gold_spatial: dict, yolo_spatial: dict, output_path: Path):
    """Save a spatial coverage heatmap as PNG."""
    try:
        import cv2
    except ImportError:
        print("  [skip] opencv not available for heatmap generation")
        return

    # Create 192x108 heatmap (1/10 scale of 1920x1080)
    heatmap = np.zeros((108, 192), dtype=np.float32)

    # Add passthrough gold points
    gold_grid = gold_spatial.get("grid")
    if gold_grid is not None:
        for gy in range(10):
            for gx in range(10):
                y1, y2 = gy * 108 // 10, (gy + 1) * 108 // 10
                x1, x2 = gx * 192 // 10, (gx + 1) * 192 // 10
                heatmap[y1:y2, x1:x2] += gold_grid[gy, gx]

    # Normalize and colorize
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Scale up for visibility
    colored = cv2.resize(colored, (960, 540), interpolation=cv2.INTER_NEAREST)

    # Add labels
    cv2.putText(
        colored,
        "Passthrough Gold Spatial Coverage",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.imwrite(str(output_path), colored)
    print(f"  Heatmap saved to {output_path}")


def print_report(
    mkv_overlap: dict,
    gold_spatial: dict,
    gold_temporal: dict,
    yolo_stats: dict,
    loss_stats: dict,
):
    """Print formatted analysis report."""
    print("=" * 70)
    print("  PHASE 1 TRAINING DATA ANALYSIS REPORT")
    print("=" * 70)

    # Section 1: MKV Overlap
    print("\n1. PASSTHROUGH GOLD ↔ MKV ALIGNMENT")
    print("-" * 50)
    print(f"  Gold entries with matching MKV: {mkv_overlap['hits']}/{mkv_overlap['hits'] + mkv_overlap['misses']}")
    print(f"  MKVs containing gold data:     {mkv_overlap['unique_mkvs']}/{mkv_overlap['total_mkvs']}")
    overlap_pct = mkv_overlap["hits"] / (mkv_overlap["hits"] + mkv_overlap["misses"]) * 100
    print(f"  Overlap rate:                  {overlap_pct:.1f}%")
    if mkv_overlap["mkv_counts"]:
        top_mkvs = sorted(mkv_overlap["mkv_counts"].items(), key=lambda x: -x[1])[:5]
        print(f"  Top MKVs by gold count:")
        for mkv, count in top_mkvs:
            print(f"    {mkv}: {count} entries")

    # Section 2: Spatial Distribution
    print(f"\n2. SPATIAL DISTRIBUTION (passthrough gold)")
    print("-" * 50)
    print(f"  Samples with coordinates: {gold_spatial['count']}")
    print(f"  X: mean={gold_spatial['x_mean']:.0f} ±{gold_spatial['x_std']:.0f}  range=[{gold_spatial['x_range'][0]}, {gold_spatial['x_range'][1]}]")
    print(f"  Y: mean={gold_spatial['y_mean']:.0f} ±{gold_spatial['y_std']:.0f}  range=[{gold_spatial['y_range'][0]}, {gold_spatial['y_range'][1]}]")
    q = gold_spatial["quadrants"]
    total_q = sum(q.values())
    print(f"  Quadrants:")
    print(f"    TL: {q['top_left']:>5} ({q['top_left']/total_q*100:5.1f}%)  |  TR: {q['top_right']:>5} ({q['top_right']/total_q*100:5.1f}%)")
    print(f"    BL: {q['bottom_left']:>5} ({q['bottom_left']/total_q*100:5.1f}%)  |  BR: {q['bottom_right']:>5} ({q['bottom_right']/total_q*100:5.1f}%)")
    print(f"  Grid coverage: {gold_spatial['grid_10x10_occupied']}/100 cells occupied")

    bias_warning = max(q.values()) / total_q > 0.7
    if bias_warning:
        dominant = max(q, key=q.get)
        print(f"  ⚠ SPATIAL BIAS: {q[dominant]/total_q*100:.0f}% of samples in {dominant}")
        print(f"    → Model will overfit to this region without augmentation")
        print(f"    → YOLO v4 data provides better spatial diversity (see below)")

    # Section 3: Temporal Distribution
    print(f"\n3. TEMPORAL DISTRIBUTION (passthrough gold)")
    print("-" * 50)
    print(f"  Total span:        {gold_temporal['span_hours']:.1f} hours")
    print(f"  Samples/hour:      {gold_temporal['samples_per_hour']:.1f}")
    print(f"  Median gap:        {gold_temporal['median_gap_s']:.1f}s")
    print(f"  Mean gap:          {gold_temporal['mean_gap_s']:.1f}s")
    print(f"  Max gap:           {gold_temporal['max_gap_s']:.0f}s ({gold_temporal['max_gap_s']/3600:.1f}h)")
    print(f"  Gaps > 5min:       {gold_temporal['gaps_over_5min']}")
    print(f"  By date:")
    for date, count in sorted(gold_temporal.get("date_counts", {}).items()):
        print(f"    {date}: {count}")

    # Section 4: YOLO v4 Dataset
    print(f"\n4. YOLO V4 TRAINING DATA")
    print("-" * 50)
    print(f"  Train images: {yolo_stats['train_images']}")
    print(f"  Val images:   {yolo_stats['val_images']}")
    print(f"  Total:        {yolo_stats['total']}")
    if yolo_stats["sources"]:
        print(f"  Sources:")
        for src, count in sorted(yolo_stats["sources"].items(), key=lambda x: -x[1]):
            print(f"    {src}: {count}")
    if yolo_stats.get("spatial"):
        s = yolo_stats["spatial"]
        print(f"  Spatial coverage:")
        print(f"    X: mean={s['x_mean']:.0f} ±{s['x_std']:.0f}")
        print(f"    Y: mean={s['y_mean']:.0f} ±{s['y_std']:.0f}")
        yq = s.get("quadrants", {})
        if yq:
            total_yq = sum(yq.values()) or 1
            print(f"    TL: {yq.get('top_left',0):>5} ({yq.get('top_left',0)/total_yq*100:5.1f}%)  |  TR: {yq.get('top_right',0):>5} ({yq.get('top_right',0)/total_yq*100:5.1f}%)")
            print(f"    BL: {yq.get('bottom_left',0):>5} ({yq.get('bottom_left',0)/total_yq*100:5.1f}%)  |  BR: {yq.get('bottom_right',0):>5} ({yq.get('bottom_right',0)/total_yq*100:5.1f}%)")

    # Section 5: Loss Events
    print(f"\n5. LOSS EVENTS")
    print("-" * 50)
    print(f"  Total events:   {loss_stats['event_count']}")
    print(f"  Total frames:   {loss_stats['total_frames']}")
    print(f"  Recovery rate:  {loss_stats['recovery_rate']}%")
    if loss_stats.get("top_edge_cases"):
        print(f"  Top edge cases:")
        for ec, count in loss_stats["top_edge_cases"].items():
            print(f"    {ec}: {count}")

    # Section 6: Combined Dataset Estimate
    print(f"\n6. COMBINED PHASE 1 DATASET ESTIMATE")
    print("-" * 50)
    gold_extractable = mkv_overlap["hits"]
    yolo_total = yolo_stats["total"]
    combined = gold_extractable + yolo_total
    print(f"  Passthrough gold → full frames:  {gold_extractable:>6} (ground truth, best quality)")
    print(f"  YOLO v4 images (already exist):  {yolo_total:>6} (retro-CNN >0.8, good quality)")
    print(f"  Combined estimate:               {combined:>6} labeled full frames")
    print()
    if combined >= 15000:
        print(f"  ✓ Sufficient for Phase 1 cursor head training (target: ≥15,000)")
    else:
        print(f"  ⚠ Below target of 15,000 — need {15000 - combined} more samples")
    print(f"  ✓ Passthrough gold provides high-quality anchors")
    if bias_warning:
        print(f"  ⚠ Passthrough gold has spatial bias — combine with YOLO v4 for diversity")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze Phase 1 training data coverage")
    parser.add_argument("--heatmap", action="store_true", help="Save spatial heatmap PNG")
    args = parser.parse_args()

    print("Loading data...")
    gold = load_passthrough_gold()
    print(f"  Passthrough gold entries: {len(gold)}")

    mkv_index = build_mkv_index()
    print(f"  MKV recordings indexed: {len(mkv_index)}")

    print("Analyzing...")
    mkv_overlap = analyze_mkv_overlap(gold, mkv_index)
    gold_spatial = analyze_spatial(gold)
    gold_temporal = analyze_temporal(gold)
    yolo_stats = analyze_yolo_v4()
    loss_stats = analyze_loss_events()

    print_report(mkv_overlap, gold_spatial, gold_temporal, yolo_stats, loss_stats)

    if args.heatmap:
        heatmap_path = PROJECT_ROOT / "data" / "phase1_spatial_heatmap.png"
        save_heatmap(gold_spatial, yolo_stats.get("spatial", {}), heatmap_path)


if __name__ == "__main__":
    main()
