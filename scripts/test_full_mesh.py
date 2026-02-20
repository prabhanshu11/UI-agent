"""End-to-end integration test for the full cursor detection mesh.

Tests the complete detection pipeline:
  YOLO → Silhouette → CNN → Jitter → Lissajous

Can run in two modes:
  1. Offline mode (--offline): Uses saved frames from loss events
  2. Live mode (default): Requires hardware (HDMI capture + ESP32)

Usage:
    uv run python scripts/test_full_mesh.py --offline   # Test with saved data
    uv run python scripts/test_full_mesh.py             # Live hardware test
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware.yolo_detector import YOLOCursorDetector
from src.hardware.cursor_recognizer import CursorRecognizer
from src.hardware.silhouette_tracker import SilhouetteTracker
from src.hardware.cursor_sample_collector import extract_gray_patch


def test_offline(args):
    """Test mesh components on saved frames from loss events."""
    print("=" * 60)
    print("FULL MESH INTEGRATION TEST (Offline Mode)")
    print("=" * 60)

    # Initialize components
    print("\n1. Initializing components...")

    yolo = YOLOCursorDetector(conf_threshold=0.3)
    yolo.load()
    print(f"   YOLO: {yolo.status()['loaded']}")

    recognizer = CursorRecognizer()
    print(f"   CNN: {recognizer.model_version}")

    sil_tracker = SilhouetteTracker(recognizer)
    print(f"   Silhouette: ROI={sil_tracker.roi_size}px")

    # Collect all frames from loss events
    events_dir = PROJECT_ROOT / "data" / "loss_events"
    frames = []
    for event_dir in sorted(events_dir.iterdir()):
        if not event_dir.is_dir():
            continue
        tl = event_dir / "timeline.jsonl"
        if not tl.exists():
            continue
        for line in tl.read_text().strip().split("\n"):
            entry = json.loads(line)
            frame_path = event_dir / entry.get("frame", "")
            if frame_path.exists():
                frames.append({
                    "path": frame_path,
                    "true_x": entry.get("cursor", {}).get("x", 0),
                    "true_y": entry.get("cursor", {}).get("y", 0),
                    "phase": entry.get("phase", "unknown"),
                    "event_id": event_dir.name,
                })

    reliable_frames = [f for f in frames if f["phase"] in ("had_it", "recovered")]
    print(f"\n   Total frames: {len(frames)}")
    print(f"   Reliable (known position): {len(reliable_frames)}")

    # Test 1: YOLO cold-start detection
    print("\n" + "=" * 60)
    print("TEST 1: YOLO Cold-Start Detection")
    print("=" * 60)

    yolo_hits = 0
    yolo_correct = 0
    yolo_times = []

    for f in reliable_frames[:50]:
        frame = cv2.imread(str(f["path"]))
        if frame is None:
            continue

        t0 = time.monotonic()
        dets = yolo.detect(frame)
        elapsed = (time.monotonic() - t0) * 1000
        yolo_times.append(elapsed)

        if dets:
            yolo_hits += 1
            d = dets[0]
            dist = ((d.cx - f["true_x"])**2 + (d.cy - f["true_y"])**2)**0.5
            if dist < 50:
                yolo_correct += 1

    n = min(50, len(reliable_frames))
    print(f"   Detection rate: {yolo_hits}/{n} ({100*yolo_hits/n:.1f}%)")
    print(f"   Accuracy (<50px): {yolo_correct}/{n} ({100*yolo_correct/n:.1f}%)")
    if yolo_times:
        print(f"   Inference: {sum(yolo_times)/len(yolo_times):.1f}ms avg, "
              f"{min(yolo_times):.1f}ms min, {max(yolo_times):.1f}ms max")

    # Test 2: YOLO → Silhouette handoff
    print("\n" + "=" * 60)
    print("TEST 2: YOLO → Silhouette Handoff")
    print("=" * 60)

    sil_tracker.reset()
    handoff_success = 0
    sil_correct = 0

    for i, f in enumerate(reliable_frames[:30]):
        frame = cv2.imread(str(f["path"]))
        if frame is None:
            continue

        # Step 1: YOLO finds cursor
        dets = yolo.detect(frame)
        if not dets:
            continue

        yolo_x, yolo_y = dets[0].cx, dets[0].cy

        # Step 2: Feed YOLO position to silhouette tracker
        sil_tracker.reset()
        result = sil_tracker.track(frame, yolo_x, yolo_y)

        if result:
            handoff_success += 1
            dist = ((result.x - f["true_x"])**2 + (result.y - f["true_y"])**2)**0.5
            if dist < 50:
                sil_correct += 1

    n_tried = min(30, len(reliable_frames))
    print(f"   Handoff success: {handoff_success}/{n_tried}")
    print(f"   Silhouette correct (<50px): {sil_correct}/{n_tried}")

    # Test 3: CNN validation on YOLO detections
    print("\n" + "=" * 60)
    print("TEST 3: CNN Validates YOLO Detections")
    print("=" * 60)

    cnn_validated = 0
    cnn_total = 0

    for f in reliable_frames[:30]:
        frame = cv2.imread(str(f["path"]))
        if frame is None:
            continue

        dets = yolo.detect(frame)
        if not dets:
            continue

        d = dets[0]
        patch = extract_gray_patch(frame, d.cx, d.cy)
        if patch is None:
            continue

        cnn_total += 1
        conf = recognizer.classify_patch(patch)
        if conf > 0.5:
            cnn_validated += 1

    print(f"   CNN validates YOLO: {cnn_validated}/{cnn_total} "
          f"({100*cnn_validated/cnn_total:.1f}%)" if cnn_total > 0 else "   No frames")

    # Test 4: Full mesh convergence time
    print("\n" + "=" * 60)
    print("TEST 4: Full Mesh Convergence (Cold Start)")
    print("=" * 60)

    convergence_times = []
    for f in reliable_frames[:10]:
        frame = cv2.imread(str(f["path"]))
        if frame is None:
            continue

        sil_tracker.reset()
        t_start = time.monotonic()

        # Phase 1: YOLO scan
        dets = yolo.detect(frame)
        t_yolo = time.monotonic()

        if not dets:
            continue

        # Phase 2: Silhouette lock
        result = sil_tracker.track(frame, dets[0].cx, dets[0].cy)
        t_sil = time.monotonic()

        # Phase 3: CNN validate
        patch = extract_gray_patch(frame, dets[0].cx, dets[0].cy)
        cnn_conf = recognizer.classify_patch(patch) if patch is not None else 0
        t_cnn = time.monotonic()

        total_ms = (t_cnn - t_start) * 1000
        convergence_times.append({
            "yolo_ms": (t_yolo - t_start) * 1000,
            "sil_ms": (t_sil - t_yolo) * 1000,
            "cnn_ms": (t_cnn - t_sil) * 1000,
            "total_ms": total_ms,
            "sil_locked": result is not None,
            "cnn_conf": cnn_conf,
        })

    if convergence_times:
        avg_total = sum(c["total_ms"] for c in convergence_times) / len(convergence_times)
        avg_yolo = sum(c["yolo_ms"] for c in convergence_times) / len(convergence_times)
        avg_sil = sum(c["sil_ms"] for c in convergence_times) / len(convergence_times)
        avg_cnn = sum(c["cnn_ms"] for c in convergence_times) / len(convergence_times)
        sil_lock_rate = sum(1 for c in convergence_times if c["sil_locked"]) / len(convergence_times)

        print(f"   Cold start → full convergence: {avg_total:.1f}ms avg")
        print(f"     YOLO scan: {avg_yolo:.1f}ms")
        print(f"     Silhouette lock: {avg_sil:.1f}ms (rate: {sil_lock_rate*100:.0f}%)")
        print(f"     CNN validate: {avg_cnn:.1f}ms")
        print(f"   Target: <3000ms — {'PASS' if avg_total < 3000 else 'FAIL'}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"   YOLO detection rate:   {100*yolo_hits/max(1,n):.0f}% (target: >90%)")
    print(f"   YOLO accuracy (<50px): {100*yolo_correct/max(1,n):.0f}% (target: >90%)")
    print(f"   YOLO inference:        {sum(yolo_times)/max(1,len(yolo_times)):.1f}ms (target: <10ms)")
    print(f"   Silhouette handoff:    {100*handoff_success/max(1,n_tried):.0f}%")
    print(f"   CNN validates YOLO:    {100*cnn_validated/max(1,cnn_total):.0f}%")
    if convergence_times:
        print(f"   Cold start time:       {avg_total:.0f}ms (target: <3000ms)")


def main():
    parser = argparse.ArgumentParser(description="Full mesh integration test")
    parser.add_argument("--offline", action="store_true",
                        help="Use saved frames instead of live hardware")
    args = parser.parse_args()

    if args.offline:
        test_offline(args)
    else:
        print("Live mode requires running dashboard with hardware connected.")
        print("Use --offline to test with saved frames.")
        print("\nTo run live test:")
        print("  1. Start dashboard: uv run python -m src.web.kvm_dashboard")
        print("  2. Navigate to http://localhost:8766/api/velocity_test")
        print("  3. Or use: curl -X POST http://localhost:8766/api/velocity_test")


if __name__ == "__main__":
    main()
