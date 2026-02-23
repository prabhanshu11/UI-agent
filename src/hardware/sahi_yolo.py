"""SAHI-style tiled YOLO inference for reliable cursor detection.

Problem: At 1920x1080 downscaled to 640px for YOLO, the cursor (~20x33px)
becomes ~7x11px — near YOLO's detection floor. Many cursors are missed.

Solution: Split the frame into overlapping tiles, run YOLO on each tile
(where the cursor is relatively larger), then merge detections with NMS.

At 384x384 tiles on a 1920x1080 frame:
  - Tile grid: 5×3 = 15 tiles (with overlap)
  - Cursor in tile: ~20x33px in a 384x384 context → much easier for YOLO
  - Each tile scaled to 640px → cursor becomes ~34x55px (well within range)

Performance target: <100ms total on RTX 2060 SUPER (15 tiles × ~5ms each)

Usage:
    from src.hardware.sahi_yolo import SAHIYOLODetector

    sahi = SAHIYOLODetector(yolo_detector)
    detections = sahi.detect(frame)

    # Or with custom tile config:
    sahi = SAHIYOLODetector(yolo_detector, tile_size=512, overlap=0.3)
    detections = sahi.detect(frame)
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.hardware.yolo_detector import YOLOCursorDetector, YOLODetection

FRAME_W, FRAME_H = 1920, 1080


@dataclass
class TileConfig:
    """Configuration for tile-based inference."""

    tile_size: int = 384  # Tile width and height in pixels
    overlap_ratio: float = 0.25  # Fraction of tile size to overlap
    nms_iou_threshold: float = 0.3  # IoU threshold for NMS across tiles
    conf_boost: float = 0.0  # Bonus to add to tiled detection confidence
    min_conf: float = 0.15  # Minimum confidence to keep after NMS


@dataclass
class SAHIResult:
    """Result from SAHI tiled inference."""

    detections: list[YOLODetection]
    tiles_processed: int
    total_ms: float
    ms_per_tile: float
    raw_detections: int  # Before NMS


class SAHIYOLODetector:
    """Tiled YOLO inference for small object detection (SAHI pattern).

    Wraps an existing YOLOCursorDetector and runs it on overlapping tiles
    for better cursor detection at full resolution.
    """

    def __init__(
        self,
        yolo: YOLOCursorDetector,
        tile_size: int = 384,
        overlap_ratio: float = 0.25,
        nms_iou_threshold: float = 0.3,
    ):
        self._yolo = yolo
        self.config = TileConfig(
            tile_size=tile_size,
            overlap_ratio=overlap_ratio,
            nms_iou_threshold=nms_iou_threshold,
        )
        self._tiles: list[tuple[int, int, int, int]] = []
        self._tiles_computed_for: tuple[int, int] = (0, 0)

        # Performance tracking
        self._inference_count = 0
        self._total_ms = 0.0
        self._last_result: Optional[SAHIResult] = None

    def _compute_tiles(self, frame_w: int, frame_h: int) -> list[tuple[int, int, int, int]]:
        """Generate overlapping tile grid for given frame dimensions.

        Returns list of (x1, y1, x2, y2) tile coordinates.
        """
        if (frame_w, frame_h) == self._tiles_computed_for and self._tiles:
            return self._tiles

        tile_size = self.config.tile_size
        stride = int(tile_size * (1 - self.config.overlap_ratio))
        tiles = []

        # Generate grid positions
        y_positions = list(range(0, frame_h - tile_size + 1, stride))
        if not y_positions or y_positions[-1] + tile_size < frame_h:
            y_positions.append(frame_h - tile_size)

        x_positions = list(range(0, frame_w - tile_size + 1, stride))
        if not x_positions or x_positions[-1] + tile_size < frame_w:
            x_positions.append(frame_w - tile_size)

        for y in y_positions:
            for x in x_positions:
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(frame_w, x + tile_size)
                y2 = min(frame_h, y + tile_size)
                tiles.append((x1, y1, x2, y2))

        # Deduplicate (can happen at edges)
        seen = set()
        unique_tiles = []
        for t in tiles:
            if t not in seen:
                seen.add(t)
                unique_tiles.append(t)

        self._tiles = unique_tiles
        self._tiles_computed_for = (frame_w, frame_h)
        return unique_tiles

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
    ) -> SAHIResult:
        """Run tiled YOLO inference on full frame.

        Args:
            frame: Full frame (H, W, 3) as numpy array.
            conf_threshold: Override minimum confidence.

        Returns:
            SAHIResult with merged detections.
        """
        t0 = time.monotonic()
        h, w = frame.shape[:2]
        tiles = self._compute_tiles(w, h)

        all_detections: list[YOLODetection] = []

        for x1, y1, x2, y2 in tiles:
            # Use the existing detect_crop method which handles
            # coordinate mapping back to full frame
            dets = self._yolo.detect_crop(
                frame,
                bbox=(x1, y1, x2, y2),
                conf_threshold=conf_threshold or self.config.min_conf,
            )
            all_detections.extend(dets)

        raw_count = len(all_detections)

        # Apply NMS across all tiles
        merged = self._nms(all_detections)

        # Apply confidence boost for tiled detections
        if self.config.conf_boost > 0:
            for d in merged:
                d.confidence = min(1.0, d.confidence + self.config.conf_boost)

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._inference_count += 1
        self._total_ms += elapsed_ms

        result = SAHIResult(
            detections=merged,
            tiles_processed=len(tiles),
            total_ms=round(elapsed_ms, 2),
            ms_per_tile=round(elapsed_ms / len(tiles), 2) if tiles else 0,
            raw_detections=raw_count,
        )
        self._last_result = result
        return result

    def detect_focused(
        self,
        frame: np.ndarray,
        center_x: int,
        center_y: int,
        radius: int = 300,
        conf_threshold: Optional[float] = None,
    ) -> SAHIResult:
        """Run tiled inference on a focused region around a point.

        Useful when we have an approximate cursor position (e.g., from
        dead reckoning or motion trail) and want high-confidence confirmation.

        Args:
            frame: Full frame (H, W, 3).
            center_x, center_y: Center of focus region.
            radius: Half-size of focus region in pixels.
            conf_threshold: Override minimum confidence.

        Returns:
            SAHIResult with detections from the focused region.
        """
        t0 = time.monotonic()
        h, w = frame.shape[:2]

        # Compute focus region
        fx1 = max(0, center_x - radius)
        fy1 = max(0, center_y - radius)
        fx2 = min(w, center_x + radius)
        fy2 = min(h, center_y + radius)

        region_w = fx2 - fx1
        region_h = fy2 - fy1

        tile_size = self.config.tile_size
        stride = int(tile_size * (1 - self.config.overlap_ratio))

        tiles = []
        for y in range(fy1, fy2 - tile_size + 1, stride):
            for x in range(fx1, fx2 - tile_size + 1, stride):
                tiles.append((x, y, x + tile_size, y + tile_size))

        # Ensure at least one tile covers the center
        if not tiles:
            cx1 = max(0, center_x - tile_size // 2)
            cy1 = max(0, center_y - tile_size // 2)
            tiles.append((cx1, cy1, cx1 + tile_size, cy1 + tile_size))

        all_detections = []
        for x1, y1, x2, y2 in tiles:
            dets = self._yolo.detect_crop(
                frame,
                bbox=(x1, y1, min(w, x2), min(h, y2)),
                conf_threshold=conf_threshold or self.config.min_conf,
            )
            all_detections.extend(dets)

        raw_count = len(all_detections)
        merged = self._nms(all_detections)

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._inference_count += 1
        self._total_ms += elapsed_ms

        return SAHIResult(
            detections=merged,
            tiles_processed=len(tiles),
            total_ms=round(elapsed_ms, 2),
            ms_per_tile=round(elapsed_ms / len(tiles), 2) if tiles else 0,
            raw_detections=raw_count,
        )

    def _nms(self, detections: list[YOLODetection]) -> list[YOLODetection]:
        """Non-Maximum Suppression across tile detections.

        When the same cursor is detected in overlapping tiles, keep only
        the highest-confidence detection.
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence descending
        dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []
        suppressed = set()

        for i, det_i in enumerate(dets):
            if i in suppressed:
                continue
            keep.append(det_i)
            for j in range(i + 1, len(dets)):
                if j in suppressed:
                    continue
                if self._iou(det_i, dets[j]) > self.config.nms_iou_threshold:
                    suppressed.add(j)

        return keep

    @staticmethod
    def _iou(a: YOLODetection, b: YOLODetection) -> float:
        """Compute Intersection over Union between two detections."""
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0

        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0

    @property
    def tile_count(self) -> int:
        """Number of tiles for current configuration."""
        return len(self._tiles) if self._tiles else 0

    @property
    def avg_ms(self) -> float:
        """Average total inference time."""
        if self._inference_count == 0:
            return 0.0
        return self._total_ms / self._inference_count

    def status(self) -> dict:
        """Return current SAHI detector status for dashboard."""
        result = {
            "tile_size": self.config.tile_size,
            "overlap": self.config.overlap_ratio,
            "tile_count": self.tile_count,
            "nms_threshold": self.config.nms_iou_threshold,
            "inference_count": self._inference_count,
            "avg_total_ms": round(self.avg_ms, 2),
        }
        if self._last_result:
            result["last_tiles"] = self._last_result.tiles_processed
            result["last_total_ms"] = self._last_result.total_ms
            result["last_raw_dets"] = self._last_result.raw_detections
            result["last_merged_dets"] = len(self._last_result.detections)
        return result

    def benchmark(
        self, frame: np.ndarray, n_warmup: int = 3, n_runs: int = 10
    ) -> dict:
        """Benchmark SAHI tiled inference vs single-shot YOLO.

        Returns comparative timing statistics.
        """
        # Warmup
        for _ in range(n_warmup):
            self.detect(frame)
            self._yolo.detect(frame)

        # SAHI runs
        sahi_times = []
        sahi_det_counts = []
        for _ in range(n_runs):
            t0 = time.monotonic()
            result = self.detect(frame)
            sahi_times.append((time.monotonic() - t0) * 1000)
            sahi_det_counts.append(len(result.detections))

        # Single-shot YOLO runs
        yolo_times = []
        yolo_det_counts = []
        for _ in range(n_runs):
            t0 = time.monotonic()
            dets = self._yolo.detect(frame)
            yolo_times.append((time.monotonic() - t0) * 1000)
            yolo_det_counts.append(len(dets))

        return {
            "sahi": {
                "tile_count": self.tile_count,
                "mean_ms": round(sum(sahi_times) / len(sahi_times), 2),
                "min_ms": round(min(sahi_times), 2),
                "max_ms": round(max(sahi_times), 2),
                "avg_detections": round(sum(sahi_det_counts) / len(sahi_det_counts), 1),
            },
            "single_shot": {
                "mean_ms": round(sum(yolo_times) / len(yolo_times), 2),
                "min_ms": round(min(yolo_times), 2),
                "max_ms": round(max(yolo_times), 2),
                "avg_detections": round(sum(yolo_det_counts) / len(yolo_det_counts), 1),
            },
            "speedup": round(
                (sum(sahi_times) / len(sahi_times))
                / (sum(yolo_times) / len(yolo_times)),
                2,
            ),
        }
