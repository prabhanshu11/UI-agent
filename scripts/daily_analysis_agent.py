"""Daily analysis agent: Claude SDK subprocess that reviews 24h of system data.

Runs every 24 hours (via cron or systemd timer). Analyzes:
  1. Loss events — What triggered them? New failure patterns?
  2. Blob log statistics — Noise distribution, cursor blob characteristics
  3. Model performance — CNN confidence distribution, YOLO detection rate
  4. Passthrough gold coverage — What screen regions lack training data?
  5. Edge cases discovered — High-velocity events, partial cursor, new cursor types

Outputs:
  - data/daily_reports/YYYYMMDD.md — Full analysis report
  - data/daily_reports/YYYYMMDD_actions.json — Structured action items

Usage:
    uv run python scripts/daily_analysis_agent.py                   # Analyze last 24h
    uv run python scripts/daily_analysis_agent.py --hours 48        # Analyze last 48h
    uv run python scripts/daily_analysis_agent.py --dry-run         # Show what would be analyzed
"""

import argparse
import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
LOSS_EVENTS_DIR = PROJECT_ROOT / "data" / "loss_events"
BLOB_LOG_DIR = PROJECT_ROOT / "data" / "blob_log"
CURSOR_SAMPLES = PROJECT_ROOT / "data" / "cursor_samples" / "index.jsonl"
STATE_LOG = PROJECT_ROOT / "data" / "state_log.jsonl"
CURSOR_MODELS_DIR = PROJECT_ROOT / "data" / "cursor_models"
YOLO_MODELS_DIR = PROJECT_ROOT / "data" / "yolo_models"
REPORTS_DIR = PROJECT_ROOT / "data" / "daily_reports"

FRAME_W, FRAME_H = 1920, 1080


def collect_loss_events(since: datetime) -> list[dict]:
    """Collect loss event manifests from the last N hours."""
    events = []
    if not LOSS_EVENTS_DIR.exists():
        return events

    for event_dir in sorted(LOSS_EVENTS_DIR.iterdir()):
        if not event_dir.is_dir():
            continue
        manifest = event_dir / "manifest.json"
        if not manifest.exists():
            continue

        # Parse event timestamp from directory name (YYYYMMDD_HHMMSS)
        try:
            event_dt = datetime.strptime(event_dir.name, "%Y%m%d_%H%M%S")
        except ValueError:
            continue

        if event_dt >= since:
            m = json.loads(manifest.read_text())
            m["event_dir"] = str(event_dir.name)
            events.append(m)

    return events


def collect_blob_stats(since: datetime) -> dict:
    """Collect blob log statistics."""
    stats = {
        "files_analyzed": 0,
        "total_frames": 0,
        "total_blobs": 0,
        "max_blobs_per_frame": 0,
        "avg_blobs_per_frame": 0,
        "blob_area_stats": {"mean": 0, "std": 0, "max": 0},
    }

    if not BLOB_LOG_DIR.exists():
        return stats

    since_ts_ns = int(since.timestamp() * 1e9)
    areas = []

    for blob_file in sorted(BLOB_LOG_DIR.glob("blobs_*.jsonl")):
        # Parse file timestamp
        try:
            file_dt = datetime.strptime(
                blob_file.stem.replace("blobs_", ""), "%Y%m%d_%H%M%S"
            )
            if file_dt < since - timedelta(hours=1):
                continue  # Skip old files (with 1h buffer)
        except ValueError:
            continue

        stats["files_analyzed"] += 1
        with open(blob_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                ts = entry.get("ts", 0)
                if ts < since_ts_ns:
                    continue

                bc = entry.get("bc", 0)
                stats["total_frames"] += 1
                stats["total_blobs"] += bc
                stats["max_blobs_per_frame"] = max(stats["max_blobs_per_frame"], bc)

                for blob in entry.get("b", []):
                    areas.append(blob.get("a", 0))

    if stats["total_frames"] > 0:
        stats["avg_blobs_per_frame"] = round(
            stats["total_blobs"] / stats["total_frames"], 2
        )

    if areas:
        areas_arr = np.array(areas)
        stats["blob_area_stats"] = {
            "mean": round(float(areas_arr.mean()), 2),
            "std": round(float(areas_arr.std()), 2),
            "max": round(float(areas_arr.max()), 2),
        }

    return stats


def collect_passthrough_gold_stats(since: datetime) -> dict:
    """Analyze passthrough gold coverage in the time window."""
    stats = {
        "new_samples": 0,
        "spatial_coverage": {"quadrants": {}, "grid_occupied": 0},
        "sources": {},
    }

    since_ts = since.timestamp()
    xs, ys = [], []

    if not CURSOR_SAMPLES.exists():
        return stats

    with open(CURSOR_SAMPLES) as f:
        for line in f:
            e = json.loads(line)
            ts = e.get("timestamp")
            if not isinstance(ts, (int, float)):
                continue
            if ts < since_ts:
                continue
            if e.get("source") == "passthrough_gold":
                stats["new_samples"] += 1
                if "x" in e and "y" in e:
                    xs.append(e["x"])
                    ys.append(e["y"])

    if xs:
        xs_arr, ys_arr = np.array(xs), np.array(ys)
        stats["spatial_coverage"]["quadrants"] = {
            "top_left": int(((xs_arr < 960) & (ys_arr < 540)).sum()),
            "top_right": int(((xs_arr >= 960) & (ys_arr < 540)).sum()),
            "bottom_left": int(((xs_arr < 960) & (ys_arr >= 540)).sum()),
            "bottom_right": int(((xs_arr >= 960) & (ys_arr >= 540)).sum()),
        }
        # 10x10 grid coverage
        grid = np.zeros((10, 10), dtype=int)
        for x, y in zip(xs, ys):
            gx = min(9, int(x / FRAME_W * 10))
            gy = min(9, int(y / FRAME_H * 10))
            grid[gy, gx] += 1
        stats["spatial_coverage"]["grid_occupied"] = int((grid > 0).sum())

    return stats


def collect_model_status() -> dict:
    """Check current model versions and metrics."""
    status = {
        "cnn_models": 0,
        "cnn_latest": "",
        "yolo_models": 0,
        "yolo_latest": "",
    }

    if CURSOR_MODELS_DIR.exists():
        models = sorted(CURSOR_MODELS_DIR.glob("v*_*.pt"))
        status["cnn_models"] = len(models)
        if models:
            status["cnn_latest"] = models[-1].name

    if YOLO_MODELS_DIR.exists():
        models = sorted(YOLO_MODELS_DIR.glob("cursor_*.pt"))
        status["yolo_models"] = len(models)
        if models:
            status["yolo_latest"] = models[-1].name

    # Check training log for recent metrics
    cnn_log = CURSOR_MODELS_DIR / "training_log.jsonl"
    if cnn_log.exists():
        lines = cnn_log.read_text().strip().splitlines()
        if lines:
            latest = json.loads(lines[-1])
            status["cnn_latest_metrics"] = latest

    return status


def build_analysis_prompt(
    loss_events: list[dict],
    blob_stats: dict,
    gold_stats: dict,
    model_status: dict,
    hours: int,
) -> str:
    """Build the prompt for Claude analysis."""
    prompt = f"""You are analyzing {hours} hours of data from a cursor tracking system.
This system uses HDMI capture + ESP32 mouse passthrough to track a Windows cursor.

## Loss Events ({len(loss_events)} events)

"""
    for evt in loss_events[:20]:  # Cap at 20 for prompt length
        prompt += f"""### Event: {evt.get('event_id', '?')}
- Trigger: {evt.get('trigger', '?')}
- Duration: {evt.get('duration_s', '?')}s
- Recovered: {evt.get('recovered', False)}
- Total frames: {evt.get('total_frames', 0)}
- Phases: {json.dumps(evt.get('phases', {}))}
- Edge cases: {', '.join(evt.get('edge_cases_found', []))}
- Velocity: max={evt.get('velocity', {}).get('max_velocity_px_s', 0):.0f} px/s
"""

    prompt += f"""
## Blob Log Statistics
- Files analyzed: {blob_stats['files_analyzed']}
- Total frames: {blob_stats['total_frames']}
- Total blobs detected: {blob_stats['total_blobs']}
- Avg blobs/frame: {blob_stats['avg_blobs_per_frame']}
- Max blobs in single frame: {blob_stats['max_blobs_per_frame']}
- Blob area stats: mean={blob_stats['blob_area_stats']['mean']}, std={blob_stats['blob_area_stats']['std']}, max={blob_stats['blob_area_stats']['max']}

## Passthrough Gold (New Data)
- New samples in window: {gold_stats['new_samples']}
- Spatial coverage (quadrants): {json.dumps(gold_stats['spatial_coverage']['quadrants'])}
- Grid cells occupied (of 100): {gold_stats['spatial_coverage']['grid_occupied']}

## Model Status
- CNN models: {model_status['cnn_models']} (latest: {model_status['cnn_latest']})
- YOLO models: {model_status['yolo_models']} (latest: {model_status['yolo_latest']})

---

Generate a report with these sections:

1. **Edge Cases Discovered** — New failure modes, novel patterns in loss events
2. **Data Quality Assessment** — Is the training data representative? Gaps?
3. **Recommended Actions** — What needs human attention (prioritized)
4. **Training Data Suggestions** — What regions/scenarios need more data
5. **Model Improvement Priorities** — What should be retrained or improved next

Keep it concise and actionable. Focus on patterns, not individual events.
"""
    return prompt


def run_analysis(
    loss_events: list[dict],
    blob_stats: dict,
    gold_stats: dict,
    model_status: dict,
    hours: int,
) -> tuple[str, list[dict]]:
    """Run Claude analysis and return (report_md, action_items)."""
    from anthropic import Anthropic

    client = Anthropic()
    prompt = build_analysis_prompt(
        loss_events, blob_stats, gold_stats, model_status, hours
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    report_text = response.content[0].text

    # Extract action items (try to parse from the report)
    action_items = []
    for line in report_text.split("\n"):
        line = line.strip()
        if line.startswith("- [ ]") or line.startswith("- **Action"):
            action_items.append({
                "action": line.lstrip("- [ ]").lstrip("- **Action").strip("*: "),
                "source": "daily_analysis",
                "timestamp": datetime.now().isoformat(),
            })

    return report_text, action_items


def main():
    parser = argparse.ArgumentParser(description="Daily cursor system analysis agent")
    parser.add_argument("--hours", type=int, default=24, help="Analysis window in hours")
    parser.add_argument("--dry-run", action="store_true", help="Show data counts only")
    args = parser.parse_args()

    since = datetime.now() - timedelta(hours=args.hours)
    date_str = datetime.now().strftime("%Y%m%d")

    print(f"Analyzing last {args.hours} hours (since {since.strftime('%Y-%m-%d %H:%M')})")

    # Collect data
    print("\nCollecting data...")
    loss_events = collect_loss_events(since)
    print(f"  Loss events: {len(loss_events)}")

    blob_stats = collect_blob_stats(since)
    print(f"  Blob frames: {blob_stats['total_frames']}")

    gold_stats = collect_passthrough_gold_stats(since)
    print(f"  New passthrough gold: {gold_stats['new_samples']}")

    model_status = collect_model_status()
    print(f"  CNN models: {model_status['cnn_models']}")
    print(f"  YOLO models: {model_status['yolo_models']}")

    if args.dry_run:
        print("\n[dry-run] Would send to Claude for analysis. Data summary above.")
        return

    if loss_events == [] and blob_stats["total_frames"] == 0 and gold_stats["new_samples"] == 0:
        print("\nNo new data in the analysis window. Skipping report.")
        return

    # Run analysis
    print("\nRunning Claude analysis...")
    report_md, action_items = run_analysis(
        loss_events, blob_stats, gold_stats, model_status, args.hours
    )

    # Save report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / f"{date_str}.md"
    header = f"""# Daily Analysis Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}

**Window:** {args.hours} hours (since {since.strftime('%Y-%m-%d %H:%M')})
**Loss events:** {len(loss_events)}
**Blob frames:** {blob_stats['total_frames']}
**New gold samples:** {gold_stats['new_samples']}

---

"""
    report_path.write_text(header + report_md)
    print(f"\nReport saved: {report_path}")

    # Save action items
    actions_path = REPORTS_DIR / f"{date_str}_actions.json"
    actions_path.write_text(json.dumps(action_items, indent=2))
    print(f"Action items saved: {actions_path} ({len(action_items)} items)")

    # Print summary
    print(f"\n{'='*50}")
    print(f"  DAILY ANALYSIS SUMMARY")
    print(f"{'='*50}")
    print(report_md[:500])
    if len(report_md) > 500:
        print(f"\n  ... ({len(report_md)} chars total, see {report_path})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
