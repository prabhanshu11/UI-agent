"""Frame-diff cursor position detector.

Detects cursor position by subtracting two HDMI capture frames taken
before and after a known mouse movement. The changed pixels correspond
to the cursor's old and new positions.

This is a deterministic alternative to LLM vision for cursor detection —
since we control the only moving element (the mouse), frame subtraction
reliably isolates the cursor.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class CursorDetection:
    """Result of cursor detection via frame subtraction."""
    x: int  # Centroid X of changed region
    y: int  # Centroid Y of changed region
    pixel_count: int  # Number of changed pixels
    bbox: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    confidence: float  # 0.0–1.0 based on cluster quality


def detect_cursor_movement(
    frame_before: np.ndarray,
    frame_after: np.ndarray,
    channel_threshold: int = 30,
    min_pixels: int = 10,
    max_pixels: int = 5000,
) -> Optional[CursorDetection]:
    """Detect cursor position from frame subtraction.

    Subtracts two frames and finds the centroid of changed pixels.
    The cursor is the primary source of change when we control movement.

    Args:
        frame_before: Frame captured before mouse movement (H, W, 3).
        frame_after: Frame captured after mouse movement (H, W, 3).
        channel_threshold: Sum of per-channel absolute differences needed
            to count a pixel as "changed". 30 works well for PNG captures.
        min_pixels: Minimum changed pixels to count as a valid detection.
            Filters out noise. Cursor is typically 15-30px wide.
        max_pixels: Maximum changed pixels. If exceeded, likely a screen
            transition rather than cursor movement.

    Returns:
        CursorDetection with position info, or None if no cursor found.
    """
    diff = np.abs(frame_after.astype(np.int16) - frame_before.astype(np.int16))
    diff_magnitude = diff.sum(axis=2)  # Sum across RGB channels
    mask = diff_magnitude > channel_threshold

    pixel_count = int(mask.sum())

    if pixel_count < min_pixels or pixel_count > max_pixels:
        return None

    ys, xs = np.where(mask)
    cx = int(xs.mean())
    cy = int(ys.mean())

    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    # Confidence based on cluster compactness — a cursor produces a tight
    # cluster, while noise is scattered across the frame
    bbox_area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
    density = pixel_count / max(bbox_area, 1)
    confidence = min(1.0, density * 2)  # Dense cluster = high confidence

    return CursorDetection(
        x=cx, y=cy,
        pixel_count=pixel_count,
        bbox=bbox,
        confidence=confidence,
    )


def detect_cursor_after_movement(
    frame_after: np.ndarray,
    frame_before: np.ndarray,
    move_dx: int,
    move_dy: int,
    channel_threshold: int = 30,
    min_pixels: int = 10,
    max_pixels: int = 5000,
) -> Optional[Tuple[int, int]]:
    """Detect cursor position after a known movement.

    Since we know the movement direction, we can pick the cluster
    that corresponds to the NEW cursor position (vs the old one).
    The frame diff will show two clusters: where the cursor was,
    and where it moved to.

    Args:
        frame_after: Frame after movement.
        frame_before: Frame before movement.
        move_dx: Known horizontal movement applied.
        move_dy: Known vertical movement applied.
        channel_threshold: Per-pixel change threshold.
        min_pixels: Minimum changed pixels for valid detection.
        max_pixels: Maximum changed pixels.

    Returns:
        (x, y) of cursor's new position, or None.
    """
    diff = np.abs(frame_after.astype(np.int16) - frame_before.astype(np.int16))
    diff_magnitude = diff.sum(axis=2)
    mask = diff_magnitude > channel_threshold

    pixel_count = int(mask.sum())
    if pixel_count < min_pixels or pixel_count > max_pixels:
        return None

    ys, xs = np.where(mask)

    # If movement is large enough, there will be two distinct clusters.
    # The "after" cluster is shifted in the direction of movement.
    # Simple approach: use the centroid — for small movements (50px),
    # the two cursor positions overlap heavily, so centroid approximates
    # the midpoint. Offset by half the movement to get the new position.
    cx = int(xs.mean()) + move_dx // 2
    cy = int(ys.mean()) + move_dy // 2

    return (cx, cy)
