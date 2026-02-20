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
CONTEXT_SIZE = 256  # Larger crop saved alongside patch for high-value sources

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

    def _save_context_crop(
        self, frame: np.ndarray, x: int, y: int, filename: str,
    ) -> bool:
        """Save a 256x256 context crop centered at (x, y), clamped to frame edges."""
        ctx = extract_gray_patch(frame, x, y, size=CONTEXT_SIZE)
        if ctx is None:
            # Clamped extraction: pad with black if near edges
            half = CONTEXT_SIZE // 2
            h, w = frame.shape[:2]
            x0 = max(0, x - half)
            y0 = max(0, y - half)
            x1 = min(w, x + half)
            y1 = min(h, y + half)
            region = frame[y0:y1, x0:x1]
            if len(region.shape) == 3:
                region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
            ctx = np.zeros((CONTEXT_SIZE, CONTEXT_SIZE), dtype=np.uint8)
            ox = max(0, half - x)
            oy = max(0, half - y)
            ctx[oy:oy + region.shape[0], ox:ox + region.shape[1]] = region
        ctx_filename = "ctx_" + filename
        cv2.imwrite(str(self.data_dir / ctx_filename), ctx)
        return True

    def _append_index(self, filename: str, label: str, x: int, y: int,
                      source: str, confidence: float,
                      cursor_type: Optional[str] = None,
                      is_confusing: Optional[int] = None,
                      distance_from_estimated: Optional[str] = None):
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
        if cursor_type is not None:
            entry["cursor_type"] = cursor_type
        if is_confusing is not None:
            entry["is_confusing"] = is_confusing
        if distance_from_estimated is not None:
            entry["distance_from_estimated"] = distance_from_estimated
        with open(self.index_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def collect_from_locate(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        correlation: float = 1.0,
        num_negatives: int = 3,
        source: str = "lissajous",
        cursor_type: Optional[str] = None,
        is_confusing: Optional[int] = None,
        distance_from_estimated: Optional[str] = None,
    ) -> int:
        """Collect samples from a locate result (Lissajous, Claude vision, etc.).

        Args:
            frame: Full RGB frame from HDMI capture.
            x: Detected cursor X position.
            y: Detected cursor Y position.
            correlation: Match correlation / confidence score.
            num_negatives: Number of random negative patches to extract.
            source: Label source ("lissajous", "claude_vision", etc.).
            cursor_type: Optional cursor type tag (arrow, ibeam, hand, etc.).
            is_confusing: Optional confusion flag from Claude (0/1).
            distance_from_estimated: Optional distance category (close/medium/far).

        Returns:
            Number of samples saved.
        """
        saved = 0
        h, w = frame.shape[:2]

        # Positive: cursor patch
        _high_value = source in ("claude_vision", "ground_truth", "lissajous")
        patch = extract_gray_patch(frame, x, y)
        if patch is not None:
            filename = self._save_patch(patch, "pos")
            if filename:
                self._append_index(
                    filename, "pos", x, y, source, correlation,
                    cursor_type=cursor_type,
                    is_confusing=is_confusing,
                    distance_from_estimated=distance_from_estimated,
                )
                saved += 1
                if _high_value:
                    self._save_context_crop(frame, x, y, filename)

        # Negatives: random positions far from cursor
        neg_source = f"{source}_neg"
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
                    self._append_index(filename, "neg", rx, ry, neg_source, 0.0)
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
