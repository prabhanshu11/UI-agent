"""Cursor recognizer — validates cursor presence and provides shape templates.

Two operating modes:
  Stage 0 (no CNN): Synthetic Windows arrow template + cv2.matchTemplate
  Stage 2+: TinyCursorNet inference (auto-loaded when model file exists)

Also maintains a "cursor template" — the last-known cursor appearance — used
by the motion detector for fast shape-informed blob scoring every frame.

Usage:
    recognizer = CursorRecognizer()
    confidence = recognizer.classify_patch(gray_64x64)  # 0.0-1.0
    recognizer.update_cursor_template(gray_64x64)       # after confirmed position
    score = recognizer.score_blob_shape(gray_64x64)     # fast, every frame
"""

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PATCH_SIZE = 64

# Default model directory
_PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = _PROJECT_ROOT / "data" / "cursor_models"
VIT_MODEL_DIR = _PROJECT_ROOT / "data" / "phase1_models"


def _make_arrow_template(size: int = 32) -> np.ndarray:
    """Generate a synthetic Windows-style arrow cursor template.

    White arrow with black outline on transparent (gray 128) background.
    Returns a grayscale uint8 image.
    """
    img = np.full((size, size), 128, dtype=np.uint8)

    # Arrow polygon: standard Windows cursor shape
    # Vertices scaled to the given size
    s = size / 32.0
    arrow_pts = np.array([
        [int(3 * s), int(1 * s)],
        [int(3 * s), int(23 * s)],
        [int(8 * s), int(18 * s)],
        [int(12 * s), int(26 * s)],
        [int(15 * s), int(24 * s)],
        [int(11 * s), int(16 * s)],
        [int(17 * s), int(16 * s)],
    ], dtype=np.int32)

    # Black outline (2px)
    cv2.polylines(img, [arrow_pts], isClosed=True, color=0, thickness=2)
    # White fill
    cv2.fillPoly(img, [arrow_pts], color=255)
    # Black border on top (1px) for crisp edge
    cv2.polylines(img, [arrow_pts], isClosed=True, color=0, thickness=1)

    return img


def _make_ibeam_template(size: int = 32) -> np.ndarray:
    """Generate a synthetic I-beam (text cursor) template."""
    img = np.full((size, size), 128, dtype=np.uint8)
    s = size / 32.0
    cx = int(16 * s)

    # Vertical bar
    cv2.line(img, (cx, int(4 * s)), (cx, int(28 * s)), 0, 2)
    # Top serif
    cv2.line(img, (int(10 * s), int(4 * s)), (int(22 * s), int(4 * s)), 0, 2)
    # Bottom serif
    cv2.line(img, (int(10 * s), int(28 * s)), (int(22 * s), int(28 * s)), 0, 2)

    return img


def _make_hand_template(size: int = 32) -> np.ndarray:
    """Generate a synthetic hand/pointer cursor template."""
    img = np.full((size, size), 128, dtype=np.uint8)
    s = size / 32.0
    # Simplified pointing hand: index finger + palm
    hand_pts = np.array([
        [int(10 * s), int(2 * s)],   # finger tip
        [int(13 * s), int(2 * s)],
        [int(13 * s), int(12 * s)],
        [int(20 * s), int(12 * s)],  # knuckles
        [int(22 * s), int(14 * s)],
        [int(22 * s), int(24 * s)],  # palm bottom
        [int(8 * s), int(24 * s)],
        [int(8 * s), int(14 * s)],
        [int(10 * s), int(12 * s)],
    ], dtype=np.int32)
    cv2.fillPoly(img, [hand_pts], color=255)
    cv2.polylines(img, [hand_pts], isClosed=True, color=0, thickness=1)
    return img


class CursorRecognizer:
    """Validates cursor presence and provides shape templates.

    Operates in two stages:
      Stage 0: Template matching against synthetic cursor shapes
      Stage 2+: TinyCursorNet CNN inference (when trained model exists)

    Also maintains a learned cursor template from confirmed positions,
    used for fast shape scoring in the motion detection loop.
    """

    def __init__(self, model_dir: Optional[Path] = None,
                 model_file: Optional[Path] = None):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR

        # Synthetic templates for Stage 0
        self._arrow_template = _make_arrow_template(32)
        self._ibeam_template = _make_ibeam_template(32)
        self._hand_template = _make_hand_template(32)

        # Learned cursor template (updated from confirmed positions)
        self._cursor_template: Optional[np.ndarray] = None

        # CNN model (loaded lazily)
        self._cnn_model = None
        self._cnn_version = "none"
        if model_file:
            self._load_specific_model(model_file)
        else:
            self._load_cnn_model()

    @property
    def cursor_template(self) -> Optional[np.ndarray]:
        """Last-known cursor appearance as 64x64 grayscale. None if unknown."""
        return self._cursor_template

    @property
    def model_version(self) -> str:
        """Current model version string, or 'template' for Stage 0."""
        return self._cnn_version if self._cnn_model else "template"

    def _load_cnn_model(self):
        """Try to load the latest CNN model from model_dir."""
        if not self.model_dir.exists():
            return

        model_files = sorted(self.model_dir.glob("v*_*.pt"), reverse=True)
        if not model_files:
            return

        try:
            import torch
            from .cursor_cnn import TinyCursorNet

            model = TinyCursorNet()
            model.load_state_dict(torch.load(model_files[0], map_location="cpu", weights_only=True))
            model.eval()
            self._cnn_model = model
            self._cnn_version = model_files[0].stem
        except (ImportError, Exception):
            pass  # torch not installed or model incompatible — stay on Stage 0

    def _load_specific_model(self, model_file: Path):
        """Load a specific model file by path."""
        try:
            import torch
            from .cursor_cnn import TinyCursorNet

            model = TinyCursorNet()
            model.load_state_dict(torch.load(model_file, map_location="cpu", weights_only=True))
            model.eval()
            self._cnn_model = model
            self._cnn_version = model_file.stem
        except (ImportError, Exception):
            pass

    def classify_patch(self, patch: np.ndarray) -> float:
        """Classify a 64x64 grayscale patch as cursor (1.0) or not (0.0).

        Uses CNN if available, falls back to template matching.

        Args:
            patch: 64x64 uint8 grayscale image.

        Returns:
            Confidence score 0.0-1.0.
        """
        if patch is None or patch.shape != (PATCH_SIZE, PATCH_SIZE):
            return 0.0

        # Stage 2+: CNN inference
        if self._cnn_model is not None:
            return self._classify_cnn(patch)

        # Stage 0: Template matching
        return self._classify_template(patch)

    def _classify_template(self, patch: np.ndarray) -> float:
        """Stage 0: Match against synthetic cursor templates."""
        best_score = 0.0

        for template in (self._arrow_template, self._ibeam_template, self._hand_template):
            result = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            best_score = max(best_score, max_val)

        # Also match against learned template if available
        if self._cursor_template is not None:
            # Extract center 32x32 from learned template for matching
            center = self._cursor_template[16:48, 16:48]
            result = cv2.matchTemplate(patch, center, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            best_score = max(best_score, max_val)

        # Normalize: matchTemplate returns -1 to 1, we want 0 to 1
        return max(0.0, min(1.0, (best_score + 1.0) / 2.0))

    def _classify_cnn(self, patch: np.ndarray) -> float:
        """Stage 2+: CNN inference on a 64x64 grayscale patch."""
        import torch

        tensor = torch.from_numpy(patch.astype(np.float32) / 255.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)

        with torch.no_grad():
            confidence = self._cnn_model(tensor).item()

        return confidence

    def grid_scan(
        self,
        frame: np.ndarray,
        stride: int = 64,
        threshold: float = 0.6,
    ) -> list[tuple[int, int, float]]:
        """Scan the full frame with CNN on a grid of patches.

        Phase 1 visual detection: find the cursor anywhere on screen
        without knowing its position. Uses batched inference for speed.

        Args:
            frame: Full color or grayscale frame (H x W x 3 or H x W).
            stride: Grid stride in pixels. 64 = non-overlapping, 32 = 50% overlap.
            threshold: Minimum confidence to include in results.

        Returns:
            List of (x, y, confidence) sorted by confidence descending.
            x, y are center coordinates of the patch.
        """
        if self._cnn_model is None:
            return []

        import torch

        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        h, w = gray.shape
        half = PATCH_SIZE // 2

        # Extract patches on grid
        patches = []
        positions = []
        for y in range(half, h - half, stride):
            for x in range(half, w - half, stride):
                patch = gray[y - half:y + half, x - half:x + half]
                if patch.shape == (PATCH_SIZE, PATCH_SIZE):
                    patches.append(patch.astype(np.float32) / 255.0)
                    positions.append((x, y))

        if not patches:
            return []

        # Batch inference
        batch = torch.from_numpy(np.array(patches)).unsqueeze(1)  # (N, 1, 64, 64)
        with torch.no_grad():
            confidences = self._cnn_model(batch).squeeze(-1).numpy()  # (N,)

        # Filter and sort
        results = []
        for i, conf in enumerate(confidences):
            if conf >= threshold:
                x, y = positions[i]
                results.append((x, y, float(conf)))

        results.sort(key=lambda r: r[2], reverse=True)

        # NMS: suppress nearby lower-confidence hits (cursor can't be in two
        # places). Keeps only the highest-confidence hit within suppress_radius.
        if results:
            suppress_radius = stride * 1.5
            kept = []
            for x, y, c in results:
                suppressed = False
                for kx, ky, kc in kept:
                    dist = ((x - kx) ** 2 + (y - ky) ** 2) ** 0.5
                    if dist < suppress_radius:
                        suppressed = True
                        break
                if not suppressed:
                    kept.append((x, y, c))
            results = kept

        return results

    def update_cursor_template(self, patch: np.ndarray):
        """Update the learned cursor template from a confirmed cursor position.

        Called after CNN validation confirms cursor is at a known position,
        or after successful jitter re-acquisition.

        Args:
            patch: 64x64 grayscale image centered on cursor.
        """
        if patch is not None and patch.shape == (PATCH_SIZE, PATCH_SIZE):
            self._cursor_template = patch.copy()

    def score_blob_shape(self, blob_patch: np.ndarray) -> float:
        """Fast shape match: compare a blob's patch against current cursor template.

        This runs every frame in the motion detection loop, so it must be fast.
        Uses cv2.matchTemplate against the learned cursor template.

        Args:
            blob_patch: 64x64 grayscale patch around a motion blob centroid.

        Returns:
            Shape match score 0.0-1.0. Higher = more cursor-like.
        """
        if blob_patch is None or blob_patch.shape != (PATCH_SIZE, PATCH_SIZE):
            return 0.0

        if self._cursor_template is None:
            # No learned template — fall back to synthetic arrow
            template = self._arrow_template
        else:
            # Use center 32x32 of learned template
            template = self._cursor_template[16:48, 16:48]

        result = cv2.matchTemplate(blob_patch, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        return max(0.0, min(1.0, (max_val + 1.0) / 2.0))

    def reload_model(self):
        """Re-check for new CNN model files and load the latest."""
        self._load_cnn_model()

    # ── ViT-S cursor regression model ──────────────────────────────

    def load_vit_model(self) -> bool:
        """Load the latest ViT-S cursor regression model.

        These models predict (x, y) from a full 384x384 frame.
        Returns True if a model was loaded.
        """
        if not VIT_MODEL_DIR.exists():
            return False

        model_files = sorted(VIT_MODEL_DIR.glob("cursor_vit_v*.pt"), reverse=True)
        if not model_files:
            return False

        try:
            import torch

            checkpoint = torch.load(
                model_files[0], map_location="cpu", weights_only=False)
            from scripts.train_phase1_model import CursorViT
            vit = CursorViT()
            vit.load_state_dict(checkpoint["model_state_dict"])
            # Set to inference mode (no dropout, frozen batchnorm)
            vit.train(False)
            self._vit_model = vit
            self._vit_version = model_files[0].stem
            self._vit_mode = checkpoint.get("mode", "full")
            return True
        except Exception:
            return False

    def vit_predict(self, frame: np.ndarray) -> Optional[tuple[int, int, float]]:
        """Predict cursor position using ViT-S regression model.

        Args:
            frame: Full color frame (H, W, 3).

        Returns:
            (x, y, confidence) or None if model not loaded or prediction failed.
        """
        if not hasattr(self, '_vit_model') or self._vit_model is None:
            return None

        try:
            import torch
            import torchvision.transforms as T

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if len(frame.shape) == 3 else frame

            resized = cv2.resize(rgb, (384, 384))

            transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
            tensor = transform(resized).unsqueeze(0)

            with torch.no_grad():
                pred = self._vit_model(tensor)

            nx, ny = pred[0, 0].item(), pred[0, 1].item()
            px = max(0, min(w - 1, int(nx * w)))
            py = max(0, min(h - 1, int(ny * h)))

            return (px, py, 0.5)
        except Exception:
            return None

    def visual_detect(
        self,
        frame: np.ndarray,
        sahi_detector=None,
    ) -> Optional[tuple[int, int, float, str]]:
        """Full visual detection cascade — find cursor without mouse.

        Tries methods in order of speed, returns first success:
          1. SAHI tiled YOLO (~50ms, best for visible cursors)
          2. ViT-S regression (~30ms, predicts from full frame)
          3. CNN grid scan (~200ms, exhaustive search)

        Args:
            frame: Full color frame (H, W, 3).
            sahi_detector: Optional SAHIYOLODetector instance.

        Returns:
            (x, y, confidence, method) or None if cursor not found.
        """
        # 1. SAHI tiled YOLO detection
        if sahi_detector is not None:
            try:
                result = sahi_detector.detect(frame)
                if result.detections:
                    det = result.detections[0]
                    return (int(det.cx), int(det.cy),
                            float(det.confidence), "sahi_yolo")
            except Exception:
                pass

        # 2. ViT-S regression + CNN validation
        vit_result = self.vit_predict(frame)
        if vit_result is not None:
            vx, vy, _ = vit_result
            patch = self._extract_vit_patch(frame, vx, vy)
            if patch is not None:
                conf = self.classify_patch(patch)
                if conf > 0.6:
                    return (vx, vy, conf, "vit_s")

        # 3. CNN grid scan (slowest but most thorough)
        hits = self.grid_scan(frame, stride=64, threshold=0.85)
        if hits:
            gx, gy, gc = hits[0]
            return (gx, gy, gc, "cnn_grid")

        return None

    def _extract_vit_patch(self, frame: np.ndarray,
                           x: int, y: int) -> Optional[np.ndarray]:
        """Extract a 64x64 grayscale patch for CNN validation."""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        h, w = gray.shape[:2]
        half = PATCH_SIZE // 2
        x0, y0 = max(0, x - half), max(0, y - half)
        x1, y1 = min(w, x + half), min(h, y + half)

        patch = gray[y0:y1, x0:x1]
        if patch.shape != (PATCH_SIZE, PATCH_SIZE):
            padded = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
            ph, pw = patch.shape[:2]
            padded[:ph, :pw] = patch
            return padded
        return patch
