"""YOLOv8n-based full-screen cursor detector.

Fills the two missing mesh boxes from STAR_TREK_COMPUTER_RESEARCH.md:
  - Silhouette Scan (full-screen): find cursor anywhere on screen
  - CNN Full-Screen: confirm cursor is on this display

A single YOLOv8n inference scans the full 1920x1080 frame and returns
bounding box(es) + confidence. Runs at 10fps on RTX 2060 SUPER.

Usage:
    detector = YOLOCursorDetector()
    detector.load()  # loads fine-tuned or pretrained model
    detections = detector.detect(frame_rgb)
    if detections:
        best = detections[0]
        print(f"Cursor at ({best.cx}, {best.cy}) conf={best.confidence:.2f}")
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

MODELS_DIR = Path(__file__).parent.parent.parent / "data" / "yolo_models"


@dataclass
class YOLODetection:
    """Single YOLO detection result."""
    x1: int        # Bounding box left
    y1: int        # Bounding box top
    x2: int        # Bounding box right
    y2: int        # Bounding box bottom
    cx: int        # Center X (cursor position estimate)
    cy: int        # Center Y (cursor position estimate)
    confidence: float  # Detection confidence 0-1
    class_id: int  # Class index (0 = cursor)
    inference_ms: float = 0.0


class YOLOCursorDetector:
    """Full-screen cursor detection using YOLOv8n.

    Wraps ultralytics YOLO with cursor-specific defaults:
    - Single class (cursor)
    - Optimized for 1920x1080 HDMI capture frames
    - TensorRT export for low-latency inference
    - Confidence threshold tuned for cursor detection
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        conf_threshold: float = 0.25,
        device: str = "0",  # GPU 0
        imgsz: int = 640,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.conf_threshold = conf_threshold
        self.device = device
        self.imgsz = imgsz
        self._model = None
        self._loaded = False

        # Performance tracking
        self._inference_count = 0
        self._total_inference_ms = 0.0
        self._last_inference_ms = 0.0

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def avg_inference_ms(self) -> float:
        if self._inference_count == 0:
            return 0.0
        return self._total_inference_ms / self._inference_count

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    def load(self, model_path: Optional[str | Path] = None) -> bool:
        """Load YOLO model. Tries in order:
        1. Explicit model_path argument
        2. self.model_path from constructor
        3. Latest fine-tuned model in data/yolo_models/
        4. Pretrained YOLOv8n (for initial benchmarking)

        Returns True if loaded successfully.
        """
        from ultralytics import YOLO

        path = Path(model_path) if model_path else self.model_path

        if path is None:
            # Try latest fine-tuned model
            path = self._find_latest_model()

        if path is None:
            # Fall back to pretrained YOLOv8n
            print("[YOLO] No fine-tuned model found, loading pretrained yolov8n.pt")
            path = "yolov8n.pt"

        try:
            self._model = YOLO(str(path))
            self._loaded = True
            print(f"[YOLO] Loaded model: {path}")
            return True
        except Exception as e:
            print(f"[YOLO] Failed to load model {path}: {e}")
            self._loaded = False
            return False

    def _find_latest_model(self) -> Optional[Path]:
        """Find the most recent fine-tuned cursor model."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Check for TensorRT engine first (fastest)
        engines = sorted(MODELS_DIR.glob("*.engine"), reverse=True)
        if engines:
            return engines[0]

        # Then check for .pt files
        pts = sorted(MODELS_DIR.glob("cursor_*.pt"), reverse=True)
        if pts:
            return pts[0]

        # Check ultralytics training runs
        runs_dir = MODELS_DIR / "runs" / "detect"
        if runs_dir.exists():
            train_dirs = sorted(runs_dir.glob("cursor_*"), reverse=True)
            for d in train_dirs:
                best = d / "weights" / "best.pt"
                if best.exists():
                    return best

        return None

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> list[YOLODetection]:
        """Run YOLO inference on a full frame.

        Args:
            frame: RGB or BGR numpy array (H, W, 3).
            conf_threshold: Override default confidence threshold.

        Returns:
            List of YOLODetection sorted by confidence (highest first).
        """
        if not self._loaded or self._model is None:
            return []

        conf = conf_threshold or self.conf_threshold
        t0 = time.monotonic()

        results = self._model.predict(
            frame,
            conf=conf,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._last_inference_ms = elapsed_ms
        self._total_inference_ms += elapsed_ms
        self._inference_count += 1

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                conf_val = float(box.conf[0].cpu())
                cls_id = int(box.cls[0].cpu())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                detections.append(YOLODetection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    cx=cx, cy=cy,
                    confidence=conf_val,
                    class_id=cls_id,
                    inference_ms=elapsed_ms,
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect_crop(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        conf_threshold: Optional[float] = None,
    ) -> list[YOLODetection]:
        """Run YOLO on a crop region, returning detections in full-frame coords.

        Extracts the crop from (x1, y1, x2, y2), pads to square, runs inference,
        then maps detection coordinates back to the full frame.

        Args:
            frame: Full frame (H, W, 3).
            bbox: (x1, y1, x2, y2) crop region in frame coordinates.
            conf_threshold: Override default confidence threshold.

        Returns:
            List of YOLODetection with coordinates in full-frame space.
        """
        if not self._loaded or self._model is None:
            return []

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        # Pad to square for better YOLO performance
        ch, cw = crop.shape[:2]
        side = max(ch, cw, 64)  # minimum 64px
        square = np.zeros((side, side, 3), dtype=crop.dtype)
        oy = (side - ch) // 2
        ox = (side - cw) // 2
        square[oy:oy + ch, ox:ox + cw] = crop

        conf = conf_threshold or self.conf_threshold
        t0 = time.monotonic()

        results = self._model.predict(
            square,
            conf=conf,
            device=self.device,
            imgsz=max(64, min(self.imgsz, side)),
            verbose=False,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._last_inference_ms = elapsed_ms
        self._total_inference_ms += elapsed_ms
        self._inference_count += 1

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                coords = box.xyxy[0].cpu().numpy().astype(int)
                # Map from square coords → crop coords → full-frame coords
                dx1 = int(coords[0]) - ox + x1
                dy1 = int(coords[1]) - oy + y1
                dx2 = int(coords[2]) - ox + x1
                dy2 = int(coords[3]) - oy + y1
                conf_val = float(box.conf[0].cpu())
                cls_id = int(box.cls[0].cpu())
                cx = (dx1 + dx2) // 2
                cy = (dy1 + dy2) // 2
                detections.append(YOLODetection(
                    x1=dx1, y1=dy1, x2=dx2, y2=dy2,
                    cx=cx, cy=cy,
                    confidence=conf_val,
                    class_id=cls_id,
                    inference_ms=elapsed_ms,
                ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def benchmark(self, frame: np.ndarray, n_warmup: int = 5, n_runs: int = 20) -> dict:
        """Benchmark inference speed on a sample frame.

        Returns dict with timing statistics.
        """
        if not self._loaded:
            return {"error": "Model not loaded"}

        # Warmup
        for _ in range(n_warmup):
            self.detect(frame)

        # Timed runs
        times = []
        det_counts = []
        for _ in range(n_runs):
            t0 = time.monotonic()
            dets = self.detect(frame)
            times.append((time.monotonic() - t0) * 1000)
            det_counts.append(len(dets))

        return {
            "model": str(self.model_path or "yolov8n.pt"),
            "imgsz": self.imgsz,
            "device": self.device,
            "n_runs": n_runs,
            "mean_ms": round(sum(times) / len(times), 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
            "median_ms": round(sorted(times)[len(times) // 2], 2),
            "fps": round(1000 / (sum(times) / len(times)), 1),
            "avg_detections": round(sum(det_counts) / len(det_counts), 1),
        }

    def export_tensorrt(self, half: bool = True, int8: bool = False) -> Optional[Path]:
        """Export current model to TensorRT engine for faster inference.

        Args:
            half: Use FP16 (default, good balance of speed/accuracy).
            int8: Use INT8 (fastest, needs calibration data).

        Returns:
            Path to exported engine file.
        """
        if not self._loaded or self._model is None:
            print("[YOLO] Cannot export: model not loaded")
            return None

        try:
            engine_path = self._model.export(
                format="engine",
                half=half,
                int8=int8,
                imgsz=self.imgsz,
                device=self.device,
            )
            print(f"[YOLO] Exported TensorRT engine: {engine_path}")
            return Path(engine_path)
        except Exception as e:
            print(f"[YOLO] TensorRT export failed: {e}")
            return None

    def status(self) -> dict:
        """Return current detector status for dashboard display."""
        return {
            "loaded": self._loaded,
            "model": str(self.model_path or "none"),
            "device": self.device,
            "imgsz": self.imgsz,
            "conf_threshold": self.conf_threshold,
            "inference_count": self._inference_count,
            "avg_inference_ms": round(self.avg_inference_ms, 2),
            "last_inference_ms": round(self._last_inference_ms, 2),
        }
