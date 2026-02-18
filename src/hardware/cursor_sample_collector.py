"""Cursor sample collector for CNN training data.

Extracts 64x64 grayscale patches from HDMI capture frames at known cursor
positions and saves them as PNG files with metadata in index.jsonl.

Samples are collected from two sources:
1. High-confidence Lissajous locate results (positive + random negatives)
2. Motion tracking loop (periodic, lower confidence)

Data layout:
    data/cursor_samples/
        pos_000001.png      # 64x64 grayscale, cursor centered
        neg_000001.png      # 64x64 grayscale, random background
        index.jsonl         # One JSON object per line with metadata
"""

import json
import random
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PATCH_SIZE = 64
HALF = PATCH_SIZE // 2

# Default data directory (relative to project root)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "cursor_samples"


def extract_gray_patch(
    frame: np.ndarray, x: int, y: int, size: int = PATCH_SIZE,
) -> Optional[np.ndarray]:
    """Extract a grayscale patch centered at (x, y) from an RGB frame.

    Returns None if the patch would extend outside the frame.
    """
    half = size // 2
    h, w = frame.shape[:2]
    x0, y0 = x - half, y - half
    x1, y1 = x0 + size, y0 + size

    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None

    patch = frame[y0:y1, x0:x1]
    if len(patch.shape) == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    return patch


class CursorSampleCollector:
    """Collects 64x64 grayscale patches for cursor CNN training.

    Saves positive samples (cursor present) and negative samples
    (random background) as PNG files with metadata in index.jsonl.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.data_dir / "index.jsonl"

        # Count existing samples to continue numbering
        self._pos_count = len(list(self.data_dir.glob("pos_*.png")))
        self._neg_count = len(list(self.data_dir.glob("neg_*.png")))

    @property
    def total_samples(self) -> int:
        return self._pos_count + self._neg_count

    @property
    def positive_count(self) -> int:
        return self._pos_count

    @property
    def negative_count(self) -> int:
        return self._neg_count

    def _save_patch(self, patch: np.ndarray, label: str) -> Optional[str]:
        """Save a patch as PNG and return the filename."""
        if label == "pos":
            self._pos_count += 1
            filename = f"pos_{self._pos_count:06d}.png"
        else:
            self._neg_count += 1
            filename = f"neg_{self._neg_count:06d}.png"

        path = self.data_dir / filename
        cv2.imwrite(str(path), patch)
        return filename

    def _append_index(self, filename: str, label: str, x: int, y: int,
                      source: str, confidence: float):
        """Append a metadata entry to index.jsonl."""
        entry = {
            "filename": filename,
            "label": label,
            "x": x,
            "y": y,
            "source": source,
            "confidence": confidence,
            "timestamp": time.time(),
        }
        with open(self.index_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def collect_from_locate(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        correlation: float = 1.0,
        num_negatives: int = 3,
    ) -> int:
        """Collect samples from a Lissajous locate result.

        Args:
            frame: Full RGB frame from HDMI capture.
            x: Detected cursor X position.
            y: Detected cursor Y position.
            correlation: Match correlation from Lissajous detection.
            num_negatives: Number of random negative patches to extract.

        Returns:
            Number of samples saved.
        """
        saved = 0
        h, w = frame.shape[:2]

        # Positive: cursor patch
        patch = extract_gray_patch(frame, x, y)
        if patch is not None:
            filename = self._save_patch(patch, "pos")
            if filename:
                self._append_index(filename, "pos", x, y, "lissajous", correlation)
                saved += 1

        # Negatives: random positions far from cursor
        for _ in range(num_negatives):
            for _attempt in range(10):
                rx = random.randint(HALF, w - HALF - 1)
                ry = random.randint(HALF, h - HALF - 1)
                dist = ((rx - x) ** 2 + (ry - y) ** 2) ** 0.5
                if dist > 150:  # Far enough from cursor
                    break
            else:
                continue

            neg_patch = extract_gray_patch(frame, rx, ry)
            if neg_patch is not None:
                filename = self._save_patch(neg_patch, "neg")
                if filename:
                    self._append_index(filename, "neg", rx, ry, "lissajous_neg", 0.0)
                    saved += 1

        return saved

    def collect_from_tracking(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        confidence: float = 0.5,
    ) -> int:
        """Collect a sample from motion tracking (lower confidence).

        Called periodically from the motion detection loop when a blob
        is being tracked as the cursor.

        Args:
            frame: Full RGB frame.
            x: Blob centroid X.
            y: Blob centroid Y.
            confidence: Tracking confidence (0-1).

        Returns:
            Number of samples saved (0 or 1 positive + 1 negative).
        """
        saved = 0
        h, w = frame.shape[:2]

        patch = extract_gray_patch(frame, x, y)
        if patch is not None:
            filename = self._save_patch(patch, "pos")
            if filename:
                self._append_index(filename, "pos", x, y, "motion_track", confidence)
                saved += 1

        # One random negative
        for _attempt in range(10):
            rx = random.randint(HALF, w - HALF - 1)
            ry = random.randint(HALF, h - HALF - 1)
            dist = ((rx - x) ** 2 + (ry - y) ** 2) ** 0.5
            if dist > 150:
                break
        else:
            return saved

        neg_patch = extract_gray_patch(frame, rx, ry)
        if neg_patch is not None:
            filename = self._save_patch(neg_patch, "neg")
            if filename:
                self._append_index(filename, "neg", rx, ry, "motion_track_neg", 0.0)
                saved += 1

        return saved
