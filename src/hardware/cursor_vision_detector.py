"""Claude Vision cursor detector — slow but reliable cursor finder.

Uses the Claude Code Agent SDK (subscription-based, no API key needed) to
analyze HDMI frames for cursor presence. Captures a frame, chops it into
4 quadrants, saves as temp JPEGs, and spawns a Claude agent that reads
the images and returns structured JSON output.

This is Phase 1 of the Star Trek Computer model: the SLOW but RELIABLE fallback
that works WITHOUT mouse/keyboard control. Claude's responses become gold-standard
training labels for TinyCursorNet (the fast local CNN).

Per-quadrant structured output columns (designed for CNN training):
    is_cursor_present (0/1)
    distance_from_estimated_point (close/medium/far/none)
    is_the_image_confusing (0/1)

Usage:
    detector = ClaudeVisionCursorDetector()
    result = await detector.detect(frame, estimated_x=500, estimated_y=300)
    if result.cursor_found:
        print(f"Cursor at ({result.cursor_x}, {result.cursor_y})")

    # Save training samples from high-confidence detections
    detector.save_training_samples(frame, result, collector)
"""

import asyncio
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Data classes ─────────────────────────────────────────────────

@dataclass
class QuadrantResult:
    """Detection result for a single quadrant."""
    quadrant: int  # 1-4
    is_cursor_present: int  # 0 or 1
    distance_from_estimated_point: str  # close/medium/far/none
    is_the_image_confusing: int  # 0 or 1


@dataclass
class CursorDetectionResult:
    """Full detection result from Claude vision."""
    quadrants: list[QuadrantResult]
    cursor_found: bool
    cursor_x: Optional[int] = None  # 0-1920 full frame coords
    cursor_y: Optional[int] = None  # 0-1080 full frame coords
    cursor_type: Optional[str] = None
    reasoning: str = ""
    latency_ms: float = 0.0
    has_confusing_quadrants: bool = False
    raw_response: str = ""

    @property
    def confidence(self) -> str:
        """Derive confidence from quadrant agreement and confusion flags."""
        if not self.cursor_found:
            return "none"
        cursor_q = [q for q in self.quadrants if q.is_cursor_present == 1]
        if len(cursor_q) != 1:
            return "low"  # Multiple or no quadrants claim cursor
        if cursor_q[0].is_the_image_confusing:
            return "low"
        if cursor_q[0].distance_from_estimated_point == "close":
            return "high"
        return "medium"


# ── Agent SDK system prompt ──────────────────────────────────────

SYSTEM_PROMPT = """You are a cursor detection system. You analyze Windows desktop screenshots captured via HDMI.

The screen is split into 4 quadrants. You will be given paths to 4 JPEG images (Q1-Q4).
A green dot labeled "EST" may mark where the system ESTIMATES the cursor is.

Quadrant layout (full frame is 1920x1080):
- Q1 = top-left (x: 0-960, y: 0-540)
- Q2 = top-right (x: 960-1920, y: 0-540)
- Q3 = bottom-left (x: 0-960, y: 540-1080)
- Q4 = bottom-right (x: 960-1920, y: 540-1080)

## Your Task

1. Read all 4 quadrant images using the Read tool
2. For EACH quadrant, determine:
   - is_cursor_present: 1 if a mouse cursor is visible, 0 if not
   - distance_from_estimated_point: how far the real cursor is from the green dot
     (close=<50px, medium=50-200px, far=>200px, none=no green dot in this quadrant)
   - is_the_image_confusing: 1 if the image has elements easily mistaken for a cursor
     (blinking text cursors, UI arrows, loading spinners, tooltip pointers)
3. If cursor found, estimate its full-frame (x, y) coordinates and type

Cursor types: arrow (white arrow, most common), ibeam (text fields), hand (links),
loading, resize, crosshair, move, forbidden, unknown.
The cursor is typically 12-20px tall with sharp edges on a 1080p screen.

## Output Format

After reading all 4 images, output ONLY a JSON block (no other text):

```json
{
  "quadrants": [
    {"quadrant": 1, "is_cursor_present": 0, "distance_from_estimated_point": "none", "is_the_image_confusing": 0},
    {"quadrant": 2, "is_cursor_present": 1, "distance_from_estimated_point": "medium", "is_the_image_confusing": 0},
    {"quadrant": 3, "is_cursor_present": 0, "distance_from_estimated_point": "none", "is_the_image_confusing": 1},
    {"quadrant": 4, "is_cursor_present": 0, "distance_from_estimated_point": "none", "is_the_image_confusing": 0}
  ],
  "cursor_x": 1150,
  "cursor_y": 320,
  "cursor_type": "arrow",
  "reasoning": "White arrow cursor visible in Q2 near toolbar."
}
```

If no cursor found in any quadrant, set cursor_x/cursor_y/cursor_type to null.
Output ONLY the JSON block, nothing else."""


# ── Quadrant helpers ─────────────────────────────────────────────

def _point_to_quadrant(x: int, y: int) -> int:
    """Return which quadrant (1-4) contains point (x, y)."""
    if x < 960:
        return 1 if y < 540 else 3
    else:
        return 2 if y < 540 else 4


def _chop_and_save_quadrants(
    frame: np.ndarray,
    output_dir: str,
    estimated_x: Optional[int] = None,
    estimated_y: Optional[int] = None,
) -> list[tuple[int, str]]:
    """Chop frame into 4 quadrants, draw green dot, save as JPEGs.

    Args:
        frame: RGB numpy array (1920x1080 or similar).
        output_dir: Directory to save quadrant JPEGs.
        estimated_x: Current estimated cursor X (for green dot).
        estimated_y: Current estimated cursor Y (for green dot).

    Returns:
        List of (quadrant_number, file_path) tuples.
    """
    h, w = frame.shape[:2]
    mid_x, mid_y = w // 2, h // 2

    # Determine which quadrant gets the green dot
    dot_quadrant = None
    if estimated_x is not None and estimated_y is not None:
        dot_quadrant = _point_to_quadrant(estimated_x, estimated_y)

    results = []
    slices = {
        1: (0, 0, mid_x, mid_y),
        2: (mid_x, 0, w, mid_y),
        3: (0, mid_y, mid_x, h),
        4: (mid_x, mid_y, w, h),
    }

    for q_num, (x0, y0, x1, y1) in slices.items():
        quad = frame[y0:y1, x0:x1].copy()

        # Draw green dot if this quadrant contains the estimated position
        if dot_quadrant == q_num and estimated_x is not None and estimated_y is not None:
            local_x = estimated_x - x0
            local_y = estimated_y - y0
            cv2.circle(quad, (local_x, local_y), 8, (0, 255, 0), 2)
            cv2.circle(quad, (local_x, local_y), 2, (0, 255, 0), -1)
            cv2.putText(
                quad, "EST", (local_x + 12, local_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
            )

        # Save as JPEG (convert RGB to BGR for cv2)
        quad_bgr = cv2.cvtColor(quad, cv2.COLOR_RGB2BGR)
        filepath = os.path.join(output_dir, f"q{q_num}.jpg")
        cv2.imwrite(filepath, quad_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        results.append((q_num, filepath))

    return results


def _parse_json_response(text: str) -> Optional[dict]:
    """Extract JSON from Claude's text response.

    Handles both raw JSON and JSON inside markdown code blocks.
    """
    # Try to find JSON in code blocks first
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try the whole text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON-like structure in the text
    brace_match = re.search(r'\{.*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _parse_detection_result(data: dict) -> CursorDetectionResult:
    """Parse the JSON response into CursorDetectionResult."""
    quadrants = []
    for q_data in data.get("quadrants", []):
        quadrants.append(QuadrantResult(
            quadrant=q_data.get("quadrant", 0),
            is_cursor_present=q_data.get("is_cursor_present", 0),
            distance_from_estimated_point=q_data.get("distance_from_estimated_point", "none"),
            is_the_image_confusing=q_data.get("is_the_image_confusing", 0),
        ))

    quadrants.sort(key=lambda q: q.quadrant)

    cursor_found = any(q.is_cursor_present == 1 for q in quadrants)
    has_confusing = any(q.is_the_image_confusing == 1 for q in quadrants)

    return CursorDetectionResult(
        quadrants=quadrants,
        cursor_found=cursor_found,
        cursor_x=data.get("cursor_x"),
        cursor_y=data.get("cursor_y"),
        cursor_type=data.get("cursor_type"),
        reasoning=data.get("reasoning", ""),
        has_confusing_quadrants=has_confusing,
    )


# ── Main detector class ─────────────────────────────────────────

class ClaudeVisionCursorDetector:
    """Detects cursor position using Claude Agent SDK vision.

    Chops HDMI frame into 4 quadrants, saves as temp JPEGs, spawns a
    Claude agent that reads the images and returns structured JSON with
    per-quadrant cursor analysis.

    Uses Claude subscription (no API key needed).
    """

    def __init__(self, model: str = "sonnet"):
        """
        Args:
            model: Model to use — "sonnet", "opus", or "haiku".
        """
        self.model = model
        self._detect_count = 0
        self._last_result: Optional[CursorDetectionResult] = None
        self._last_time: float = 0.0

    @property
    def detect_count(self) -> int:
        return self._detect_count

    @property
    def last_result(self) -> Optional[CursorDetectionResult]:
        return self._last_result

    @property
    def last_time(self) -> float:
        return self._last_time

    async def detect(
        self,
        frame: np.ndarray,
        estimated_x: Optional[int] = None,
        estimated_y: Optional[int] = None,
    ) -> CursorDetectionResult:
        """Detect cursor in a full HDMI frame using Claude Agent SDK.

        Saves quadrant images to temp files, spawns a Claude agent that
        reads them with the Read tool, and parses the structured JSON response.

        Args:
            frame: RGB numpy array (expected 1920x1080).
            estimated_x: Current estimated cursor X (for green dot).
            estimated_y: Current estimated cursor Y (for green dot).

        Returns:
            CursorDetectionResult with per-quadrant analysis and cursor position.
        """
        t0 = time.monotonic()

        # Agent SDK refuses to run inside another Claude Code session
        os.environ.pop("CLAUDECODE", None)

        # Lazy import — avoids import errors when Agent SDK isn't installed
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            AssistantMessage,
            ResultMessage,
            TextBlock,
            query,
        )

        # Save quadrant images to temp directory
        with tempfile.TemporaryDirectory(prefix="cursor_detect_") as tmpdir:
            quadrant_files = _chop_and_save_quadrants(
                frame, tmpdir, estimated_x, estimated_y,
            )

            # Build the prompt with file paths
            file_list = "\n".join(
                f"- Q{q_num}: {fpath}" for q_num, fpath in quadrant_files
            )

            est_info = ""
            if estimated_x is not None and estimated_y is not None:
                est_q = _point_to_quadrant(estimated_x, estimated_y)
                est_info = (
                    f"\nThe system estimates the cursor is at ({estimated_x}, {estimated_y}) "
                    f"in Q{est_q}. A green dot labeled 'EST' is drawn at that position."
                )
            else:
                est_info = "\nNo estimated cursor position available (green dot not drawn)."

            prompt = (
                f"Read all 4 quadrant images and find the mouse cursor.\n\n"
                f"Quadrant image files:\n{file_list}\n"
                f"{est_info}\n\n"
                f"Read each image file, analyze for cursor presence, "
                f"then output ONLY the JSON result."
            )

            options = ClaudeAgentOptions(
                system_prompt=SYSTEM_PROMPT,
                allowed_tools=["Read"],
                permission_mode="bypassPermissions",
                max_turns=6,  # Read 4 images + respond
                model=self.model,
            )

            # Collect the agent's text response
            response_text = ""
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            response_text += block.text
                elif isinstance(message, ResultMessage):
                    if message.result:
                        response_text += message.result

        # Parse the JSON response
        parsed = _parse_json_response(response_text)

        if parsed is not None:
            result = _parse_detection_result(parsed)
            result.raw_response = response_text
        else:
            result = CursorDetectionResult(
                quadrants=[],
                cursor_found=False,
                reasoning=f"Failed to parse JSON from response: {response_text[:200]}",
                raw_response=response_text,
            )

        result.latency_ms = (time.monotonic() - t0) * 1000
        self._detect_count += 1
        self._last_result = result
        self._last_time = time.monotonic()

        return result

    def detect_sync(
        self,
        frame: np.ndarray,
        estimated_x: Optional[int] = None,
        estimated_y: Optional[int] = None,
    ) -> CursorDetectionResult:
        """Synchronous wrapper for detect()."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context — run in a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.detect(frame, estimated_x, estimated_y),
                )
                return future.result(timeout=60)
        else:
            return asyncio.run(self.detect(frame, estimated_x, estimated_y))

    def save_training_samples(
        self,
        frame: np.ndarray,
        result: CursorDetectionResult,
        collector,
    ) -> int:
        """Save training samples from a high-confidence Claude detection.

        Only saves when cursor is found with high/medium confidence and
        the quadrant is not marked as confusing.

        Args:
            frame: RGB numpy array used for detection.
            result: CursorDetectionResult from detect().
            collector: CursorSampleCollector instance.

        Returns:
            Number of samples saved.
        """
        if not result.cursor_found:
            return 0
        if result.cursor_x is None or result.cursor_y is None:
            return 0

        # Don't auto-save confusing images — those need human review
        cursor_q = [q for q in result.quadrants if q.is_cursor_present == 1]
        if cursor_q and cursor_q[0].is_the_image_confusing:
            return 0

        # Use confidence as correlation score
        conf_score = {"high": 1.0, "medium": 0.8, "low": 0.5, "none": 0.0}
        correlation = conf_score.get(result.confidence, 0.5)

        if correlation < 0.8:
            return 0  # Only save high/medium confidence

        # Get distance and confusion info for the cursor quadrant
        distance = cursor_q[0].distance_from_estimated_point if cursor_q else None
        is_confusing = cursor_q[0].is_the_image_confusing if cursor_q else None

        return collector.collect_from_locate(
            frame=frame,
            x=result.cursor_x,
            y=result.cursor_y,
            correlation=correlation,
            num_negatives=3,
            source="claude_vision",
            cursor_type=result.cursor_type,
            is_confusing=is_confusing,
            distance_from_estimated=distance,
        )

    def to_dict(self) -> dict:
        """Serialize last result for API responses."""
        if self._last_result is None:
            return {"detect_count": self._detect_count, "last_result": None}

        r = self._last_result
        return {
            "detect_count": self._detect_count,
            "last_time": self._last_time,
            "last_result": {
                "cursor_found": r.cursor_found,
                "cursor_x": r.cursor_x,
                "cursor_y": r.cursor_y,
                "cursor_type": r.cursor_type,
                "confidence": r.confidence,
                "reasoning": r.reasoning,
                "latency_ms": round(r.latency_ms, 1),
                "has_confusing_quadrants": r.has_confusing_quadrants,
                "quadrants": [
                    {
                        "quadrant": q.quadrant,
                        "is_cursor_present": q.is_cursor_present,
                        "distance_from_estimated_point": q.distance_from_estimated_point,
                        "is_the_image_confusing": q.is_the_image_confusing,
                    }
                    for q in r.quadrants
                ],
            },
        }
