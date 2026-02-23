# /// script
# dependencies = ["orjson"]
# requires-python = ">=3.13"
# ///
"""Analyze a week of cursor tracking data.

Mines state_log.jsonl, velocity_test runs, and loss_events to find:
1. Where positions get "stuck" (frozen tracker)
2. Movement episodes and their characteristics
3. Cross-correlation between commanded mouse and observed displacement
4. Method effectiveness breakdown
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"

def parse_ts(ts_str: str) -> float:
    """Parse ISO timestamp to epoch seconds."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()


def analyze_state_log():
    """Analyze the main state_log.jsonl for stuck positions and movement patterns."""
    log_path = DATA / "state_log.jsonl"
    if not log_path.exists():
        print("state_log.jsonl not found")
        return

    print("=" * 70)
    print("  STATE LOG ANALYSIS")
    print("=" * 70)

    total = 0
    methods = Counter()
    ya_true = 0  # YOLO active count
    yolo_detections = 0  # yc > 0
    validated = 0  # v == true

    # Position tracking
    prev_pos = None
    prev_t = None
    stuck_runs = []  # (pos, start_t, end_t, count)
    current_stuck_start = None
    current_stuck_count = 0
    current_stuck_pos = None

    # Movement episodes
    moves = []  # (t, dx, dy, dt)

    # CNN confidence distribution
    cnn_buckets = Counter()  # 0.0-0.1, 0.1-0.2, ...
    sil_buckets = Counter()

    # Time range
    first_t = None
    last_t = None

    # Position change frequency
    position_changes = 0

    with open(log_path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            t = parse_ts(entry["t"])
            if first_t is None:
                first_t = t
            last_t = t

            cx, cy = entry["cx"], entry["cy"]
            method = entry.get("m", "unknown")
            cnn = entry.get("cnn", 0)
            sil = entry.get("sil", 0)

            methods[method] += 1
            if entry.get("ya"):
                ya_true += 1
            if entry.get("yc", 0) > 0:
                yolo_detections += 1
            if entry.get("v"):
                validated += 1

            # CNN confidence bucket
            bucket = min(int(cnn * 10), 9)
            cnn_buckets[bucket] += 1
            sil_bucket = min(int(sil * 10), 9)
            sil_buckets[sil_bucket] += 1

            pos = (cx, cy)
            if prev_pos is not None:
                if pos == prev_pos:
                    if current_stuck_pos == pos:
                        current_stuck_count += 1
                    else:
                        # New stuck run
                        if current_stuck_count > 1:
                            stuck_runs.append((current_stuck_pos, current_stuck_start, prev_t, current_stuck_count))
                        current_stuck_pos = pos
                        current_stuck_start = prev_t
                        current_stuck_count = 1
                else:
                    # Position changed
                    position_changes += 1
                    if current_stuck_count > 1:
                        stuck_runs.append((current_stuck_pos, current_stuck_start, prev_t, current_stuck_count))
                    current_stuck_pos = pos
                    current_stuck_start = t
                    current_stuck_count = 1

                    dx = cx - prev_pos[0]
                    dy = cy - prev_pos[1]
                    dt = t - prev_t
                    if dt > 0:
                        moves.append((t, dx, dy, dt))

            prev_pos = pos
            prev_t = t

    # Finalize last stuck run
    if current_stuck_count > 1:
        stuck_runs.append((current_stuck_pos, current_stuck_start, last_t, current_stuck_count))

    duration_h = (last_t - first_t) / 3600 if first_t and last_t else 0

    print(f"\n  Total entries: {total:,}")
    print(f"  Time range: {duration_h:.1f} hours")
    print(f"  Avg rate: {total / (last_t - first_t):.1f} Hz" if last_t != first_t else "")
    print(f"  Position changes: {position_changes:,} ({100*position_changes/total:.1f}%)")
    print(f"  YOLO active: {ya_true:,} ({100*ya_true/total:.1f}%)")
    print(f"  YOLO detections: {yolo_detections:,}")
    print(f"  Validated: {validated:,} ({100*validated/total:.1f}%)")

    print(f"\n  --- Methods ---")
    for m, c in methods.most_common():
        print(f"    {m}: {c:,} ({100*c/total:.1f}%)")

    print(f"\n  --- CNN Confidence Distribution ---")
    for b in range(10):
        count = cnn_buckets.get(b, 0)
        bar = "#" * max(1, int(40 * count / total))
        print(f"    {b*10:3d}-{(b+1)*10:3d}%: {count:6,} {bar}")

    print(f"\n  --- Silhouette Confidence Distribution ---")
    for b in range(10):
        count = sil_buckets.get(b, 0)
        bar = "#" * max(1, int(40 * count / total))
        print(f"    {b*10:3d}-{(b+1)*10:3d}%: {count:6,} {bar}")

    # Stuck analysis
    print(f"\n  --- STUCK POSITION ANALYSIS ---")
    print(f"  Total stuck runs (>1 consecutive same pos): {len(stuck_runs):,}")

    if stuck_runs:
        # Top 20 longest stuck periods
        stuck_runs.sort(key=lambda x: x[3], reverse=True)
        print(f"\n  Top 20 longest stuck periods:")
        print(f"  {'Position':>15s}  {'Entries':>8s}  {'Duration':>10s}  {'Start':>25s}")
        for pos, start, end, count in stuck_runs[:20]:
            dur = end - start
            start_str = datetime.fromtimestamp(start, tz=timezone.utc).strftime("%H:%M:%S")
            print(f"  ({pos[0]:4d},{pos[1]:4d})  {count:8,}  {dur:8.1f}s  {start_str}")

        # What % of all entries are in stuck runs > 100 entries?
        long_stuck = sum(s[3] for s in stuck_runs if s[3] > 100)
        print(f"\n  Entries in stuck runs >100: {long_stuck:,} ({100*long_stuck/total:.1f}%)")

        # What % of time is spent stuck?
        total_stuck_time = sum(s[2] - s[1] for s in stuck_runs if s[3] > 10)
        total_time = last_t - first_t
        print(f"  Time spent stuck (>10 entries): {total_stuck_time:.0f}s / {total_time:.0f}s ({100*total_stuck_time/total_time:.1f}%)")

    # Movement analysis
    print(f"\n  --- MOVEMENT EPISODES ---")
    print(f"  Total position changes: {position_changes:,}")

    if moves:
        speeds = [((dx**2 + dy**2)**0.5 / dt) for _, dx, dy, dt in moves if dt > 0]
        if speeds:
            speeds.sort()
            print(f"  Speed (px/s): median={speeds[len(speeds)//2]:.0f}, "
                  f"p95={speeds[int(len(speeds)*0.95)]:.0f}, "
                  f"max={speeds[-1]:.0f}")

        # Find continuous movement episodes (position changes within 2s of each other)
        episodes = []
        ep_start = moves[0][0]
        ep_moves = 1
        ep_last = moves[0][0]
        for t, dx, dy, dt in moves[1:]:
            if t - ep_last < 2.0:
                ep_moves += 1
                ep_last = t
            else:
                if ep_moves >= 3:
                    episodes.append((ep_start, ep_last, ep_moves))
                ep_start = t
                ep_moves = 1
                ep_last = t
        if ep_moves >= 3:
            episodes.append((ep_start, ep_last, ep_moves))

        print(f"  Movement episodes (>3 changes within 2s): {len(episodes)}")
        if episodes:
            print(f"\n  Top 10 longest movement episodes:")
            episodes.sort(key=lambda x: x[2], reverse=True)
            for start, end, count in episodes[:10]:
                dur = end - start
                start_str = datetime.fromtimestamp(start, tz=timezone.utc).strftime("%H:%M:%S")
                print(f"    {start_str}  {dur:.1f}s  {count} moves")


def analyze_velocity_tests():
    """Analyze velocity test runs — commanded speed vs observed accuracy."""
    vt_dir = DATA / "velocity_test"
    if not vt_dir.exists():
        print("\nNo velocity_test directory")
        return

    print("\n" + "=" * 70)
    print("  VELOCITY TEST ANALYSIS (Commanded vs Observed)")
    print("=" * 70)

    runs = sorted(vt_dir.iterdir())
    print(f"\n  Total runs: {len(runs)}")

    for run_dir in runs:
        summary_path = run_dir / "summary.json"
        results_path = run_dir / "results.jsonl"
        if not summary_path.exists():
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        print(f"\n  --- {run_dir.name} ---")

        # Parse each test scenario
        for scenario, data in summary.items():
            vel = data.get("velocity_px_s_mean", 0)
            err_mean = data.get("error_mean", 0)
            err_p95 = data.get("error_p95", 0)
            acc = data.get("accuracy_pct", 0)
            count = data.get("count", 0)
            labels = data.get("labels", {})
            print(f"    {scenario:>12s}: vel={vel:6.0f} px/s  err_mean={err_mean:6.0f}px  "
                  f"err_p95={err_p95:6.0f}px  acc={acc:5.1f}%  n={count}  "
                  f"[ok={labels.get('accurate',0)} drift={labels.get('drift',0)} "
                  f"wrong={labels.get('wrong',0)} lost={labels.get('lost',0)}]")

        # Also check per-frame results for motion curves
        if results_path.exists():
            frames = []
            with open(results_path) as f:
                for line in f:
                    try:
                        frames.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

            if frames:
                # Look for commanded vs observed position pairs
                has_commanded = any("commanded" in f or "expected" in f or "hid" in f for f in frames[:5])
                has_tracker = any("tracker" in f or "cursor" in f or "cx" in f for f in frames[:5])
                keys = list(frames[0].keys()) if frames else []
                print(f"    Per-frame fields: {keys[:8]}{'...' if len(keys) > 8 else ''}")
                print(f"    Per-frame entries: {len(frames)}")

                # Print first frame as sample
                if frames:
                    print(f"    Sample: {json.dumps(frames[0])[:200]}...")


def analyze_loss_events():
    """Analyze loss events for patterns."""
    le_dir = DATA / "loss_events"
    if not le_dir.exists():
        print("\nNo loss_events directory")
        return

    print("\n" + "=" * 70)
    print("  LOSS EVENT ANALYSIS")
    print("=" * 70)

    events = sorted(le_dir.iterdir())
    print(f"\n  Total events: {len(events)}")

    triggers = Counter()
    recovered = 0
    durations = []
    velocities = []
    edge_cases_all = Counter()

    for ev_dir in events:
        manifest = ev_dir / "manifest.json"
        if not manifest.exists():
            continue
        with open(manifest) as f:
            m = json.load(f)

        triggers[m.get("trigger", "unknown")] += 1
        if m.get("recovered"):
            recovered += 1
        durations.append(m.get("duration_s", 0))
        vel = m.get("velocity", {})
        if vel.get("max_velocity_px_s"):
            velocities.append(vel["max_velocity_px_s"])
        for ec in m.get("edge_cases_found", []):
            edge_cases_all[ec] += 1

    total = len(events)
    print(f"  Recovered: {recovered}/{total} ({100*recovered/total:.0f}%)")
    if durations:
        durations.sort()
        print(f"  Duration: median={durations[len(durations)//2]:.1f}s, "
              f"mean={sum(durations)/len(durations):.1f}s, "
              f"max={durations[-1]:.1f}s")
    if velocities:
        velocities.sort()
        print(f"  Max velocity at loss: median={velocities[len(velocities)//2]:.0f} px/s, "
              f"max={velocities[-1]:.0f} px/s")

    print(f"\n  Triggers:")
    for trig, c in triggers.most_common():
        print(f"    {trig}: {c}")

    print(f"\n  Edge cases:")
    for ec, c in edge_cases_all.most_common():
        print(f"    {ec}: {c}")

    # Check for stuck-at-loss pattern: events where post velocity is 0 but pre was high
    print(f"\n  --- Stuck-at-loss Pattern ---")
    stuck_at_loss = 0
    for ev_dir in events:
        manifest = ev_dir / "manifest.json"
        if not manifest.exists():
            continue
        with open(manifest) as f:
            m = json.load(f)
        vel = m.get("velocity", {})
        if vel.get("avg_velocity_pre", 0) > 20 and vel.get("avg_velocity_post", 0) == 0:
            stuck_at_loss += 1
            if stuck_at_loss <= 5:
                print(f"    {ev_dir.name}: pre_vel={vel['avg_velocity_pre']:.0f} → "
                      f"post_vel=0, duration={m.get('duration_s', 0):.1f}s, "
                      f"recovered={m.get('recovered', False)}")
    print(f"  Total stuck-at-loss events: {stuck_at_loss}/{total}")


def analyze_blob_log():
    """Check blob log for correlation with state log."""
    bl_dir = DATA / "blob_log"
    if not bl_dir.exists():
        print("\nNo blob_log directory")
        return

    print("\n" + "=" * 70)
    print("  BLOB LOG ANALYSIS")
    print("=" * 70)

    files = sorted(bl_dir.iterdir())
    print(f"  Blob log files: {len(files)}")

    total_entries = 0
    blob_counts = Counter()
    sample = None

    for bf in files[:3]:  # Just sample first 3
        with open(bf) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    total_entries += 1
                    bc = entry.get("blob_count", entry.get("count", 0))
                    blob_counts[bc] += 1
                    if sample is None:
                        sample = entry
                except json.JSONDecodeError:
                    continue

    print(f"  Sample entries (first 3 files): {total_entries}")
    if sample:
        print(f"  Sample fields: {list(sample.keys())[:10]}")
        print(f"  Sample: {json.dumps(sample)[:300]}...")

    if blob_counts:
        print(f"\n  Blob count distribution:")
        for bc, c in sorted(blob_counts.items()):
            print(f"    {bc} blobs: {c} entries")


def cross_reference_motion_mouse():
    """Look for data that links mouse HID commands to observed motion.

    The velocity tests have commanded velocities + observed positions.
    This IS the calibration data the user mentioned.
    """
    vt_dir = DATA / "velocity_test"
    if not vt_dir.exists():
        return

    print("\n" + "=" * 70)
    print("  MOTION-MOUSE CROSS REFERENCE (Calibration Data)")
    print("=" * 70)

    # Find runs with per-frame results
    all_pairs = []  # (commanded_vel, observed_displacement, direction)

    for run_dir in sorted(vt_dir.iterdir()):
        results_path = run_dir / "results.jsonl"
        summary_path = run_dir / "summary.json"
        if not results_path.exists() or not summary_path.exists():
            continue

        with open(summary_path) as f:
            summary = json.load(f)

        # Extract commanded velocity → observed error for each scenario
        for scenario, data in summary.items():
            vel = data.get("velocity_px_s_mean", 0)
            err_mean = data.get("error_mean", 0)
            acc = data.get("accuracy_pct", 0)
            count = data.get("count", 0)
            direction = "right" if "right" in scenario.lower() else "down"
            all_pairs.append({
                "run": run_dir.name,
                "scenario": scenario,
                "commanded_vel": vel,
                "error_mean": err_mean,
                "accuracy": acc,
                "count": count,
                "direction": direction,
            })

    if all_pairs:
        print(f"\n  Total scenario-level data points: {len(all_pairs)}")
        print(f"\n  Velocity → Error curve (calibration transfer function):")
        print(f"  {'Vel (px/s)':>12s}  {'Dir':>6s}  {'Error (px)':>10s}  {'Accuracy':>8s}  Run")

        # Group by velocity
        by_vel = defaultdict(list)
        for p in all_pairs:
            by_vel[p["commanded_vel"]].append(p)

        for vel in sorted(by_vel.keys()):
            points = by_vel[vel]
            for p in points:
                print(f"  {p['commanded_vel']:12.0f}  {p['direction']:>6s}  "
                      f"{p['error_mean']:10.1f}  {p['accuracy']:7.1f}%  {p['run']}")

        # Aggregate: velocity → mean error
        print(f"\n  Aggregate velocity → mean tracking error:")
        for vel in sorted(by_vel.keys()):
            points = by_vel[vel]
            avg_err = sum(p["error_mean"] for p in points) / len(points)
            avg_acc = sum(p["accuracy"] for p in points) / len(points)
            n = sum(p["count"] for p in points)
            print(f"    {vel:6.0f} px/s → {avg_err:6.1f} px error, {avg_acc:5.1f}% accurate (n={n})")

        print(f"\n  KEY FINDING: This IS calibration data!")
        print(f"  The velocity test runs commanded the mouse at known speeds")
        print(f"  and measured how far off the tracker was.")
        print(f"  This velocity→error curve is the transfer function.")


if __name__ == "__main__":
    print("UI-Agent Tracking Data Analysis")
    print(f"Data directory: {DATA}")
    print()

    analyze_state_log()
    analyze_velocity_tests()
    analyze_loss_events()
    analyze_blob_log()
    cross_reference_motion_mouse()

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
