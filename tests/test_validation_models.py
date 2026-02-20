"""Tests for validation page multi-model comparison endpoints.

Tests: model listing, comparison model loading, dual inference,
and context crop functionality.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.hardware.cursor_cnn import TinyCursorNet


# ── Fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def tmp_samples_dir(tmp_path):
    """Create a temp samples directory with a few test patches and index."""
    samples = tmp_path / "cursor_samples"
    samples.mkdir()

    import cv2
    for i in range(1, 4):
        patch = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        cv2.imwrite(str(samples / f"pos_{i:06d}.png"), patch)
        cv2.imwrite(str(samples / f"neg_{i:06d}.png"), patch)

    entries = []
    for i in range(1, 4):
        entries.append({"filename": f"pos_{i:06d}.png", "label": "pos",
                        "x": 100, "y": 100, "source": "motion_track",
                        "confidence": 0.8, "timestamp": 1000.0 + i})
        entries.append({"filename": f"neg_{i:06d}.png", "label": "neg",
                        "x": 500, "y": 500, "source": "motion_track_neg",
                        "confidence": 0.0, "timestamp": 1000.0 + i})
    with open(samples / "index.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    return samples


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Create a temp models directory with saved TinyCursorNet models."""
    models = tmp_path / "cursor_models"
    models.mkdir()

    for version in ("v001_20260219", "v006_20260220"):
        model = TinyCursorNet()
        torch.save(model.state_dict(), str(models / f"{version}.pt"))

    log = [
        {"model_file": "v001_20260219.pt", "version": 1, "timestamp": "2026-02-19",
         "train_samples": 1000, "val_samples": 250, "pos_samples": 500, "neg_samples": 500,
         "epochs_trained": 30, "best_val_loss": 0.519, "val_acc": 0.744,
         "major": "v1", "notes": "initial"},
        {"model_file": "v006_20260220.pt", "version": 6, "timestamp": "2026-02-20",
         "train_samples": 2580, "val_samples": 643, "pos_samples": 1514, "neg_samples": 1709,
         "epochs_trained": 30, "best_val_loss": 0.389, "val_acc": 0.835,
         "major": "v2", "notes": "with human corrections"},
    ]
    with open(models / "training_log.jsonl", "w") as f:
        for entry in log:
            f.write(json.dumps(entry) + "\n")

    return models


# ── Unit tests for TinyCursorNet ───────────────────────────────────

class TestTinyCursorNet:
    def test_model_output_shape(self):
        model = TinyCursorNet()
        model.eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1)

    def test_model_output_range(self):
        """Output should be in [0, 1] due to sigmoid."""
        model = TinyCursorNet()
        model.eval()
        for _ in range(10):
            x = torch.randn(1, 1, 64, 64)
            with torch.no_grad():
                out = model(x).item()
            assert 0.0 <= out <= 1.0

    def test_model_batch_inference(self):
        model = TinyCursorNet()
        model.eval()
        x = torch.randn(8, 1, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 1)

    def test_model_save_load_roundtrip(self, tmp_path):
        """Save and reload should produce identical outputs."""
        model1 = TinyCursorNet()
        model1.eval()
        path = tmp_path / "test_model.pt"
        torch.save(model1.state_dict(), str(path))

        model2 = TinyCursorNet()
        model2.load_state_dict(torch.load(str(path), weights_only=True))
        model2.eval()

        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            out1 = model1(x).item()
            out2 = model2(x).item()
        assert abs(out1 - out2) < 1e-6

    def test_two_models_produce_valid_outputs(self):
        """Two randomly initialized models should both produce valid outputs."""
        m1 = TinyCursorNet()
        m2 = TinyCursorNet()
        m1.eval()
        m2.eval()
        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            o1 = m1(x).item()
            o2 = m2(x).item()
        assert 0.0 <= o1 <= 1.0
        assert 0.0 <= o2 <= 1.0


# ── Tests for model listing logic ─────────────────────────────────

class TestModelListing:
    def test_parse_training_log(self, tmp_models_dir):
        """Verify training log can be parsed into model list."""
        log_path = tmp_models_dir / "training_log.jsonl"
        models = []
        for line in log_path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            models.append({
                "version": entry["model_file"].replace(".pt", ""),
                "val_acc": entry.get("val_acc"),
                "major": entry.get("major"),
            })
        assert len(models) == 2
        assert models[0]["version"] == "v001_20260219"
        assert models[0]["val_acc"] == 0.744
        assert models[0]["major"] == "v1"
        assert models[1]["version"] == "v006_20260220"
        assert models[1]["val_acc"] == 0.835
        assert models[1]["major"] == "v2"

    def test_model_files_exist(self, tmp_models_dir):
        """All models in log should have .pt files."""
        log_path = tmp_models_dir / "training_log.jsonl"
        for line in log_path.read_text().splitlines():
            entry = json.loads(line)
            pt_file = tmp_models_dir / entry["model_file"]
            assert pt_file.exists(), f"Missing model file: {entry['model_file']}"


# ── Tests for comparison model loading ─────────────────────────────

class TestComparisonModel:
    def test_load_comparison_model(self, tmp_models_dir):
        """Load a model file and verify it produces inference."""
        model = TinyCursorNet()
        model_path = tmp_models_dir / "v001_20260219.pt"
        model.load_state_dict(torch.load(str(model_path), weights_only=True))
        model.eval()

        x = torch.randn(1, 1, 64, 64)
        with torch.no_grad():
            conf = model(x).item()
        assert 0.0 <= conf <= 1.0

    def test_load_nonexistent_model_fails(self, tmp_models_dir):
        """Loading a missing model should raise an error."""
        with pytest.raises(FileNotFoundError):
            torch.load(str(tmp_models_dir / "v999_nonexistent.pt"), weights_only=True)

    def test_comparison_state_management(self):
        """Test the comparison model state dict pattern."""
        state = {"version": None, "model": None}

        # Set comparison
        model = TinyCursorNet()
        model.eval()
        state = {"version": "v001_20260219", "model": model}
        assert state["version"] == "v001_20260219"
        assert state["model"] is not None

        # Clear comparison
        state = {"version": None, "model": None}
        assert state["version"] is None
        assert state["model"] is None


# ── Tests for dual inference ───────────────────────────────────────

class TestDualInference:
    def test_dual_inference_same_patch(self, tmp_samples_dir):
        """Two models should both produce valid confidence for the same patch."""
        import cv2
        patch_path = tmp_samples_dir / "pos_000001.png"
        patch = cv2.imread(str(patch_path), cv2.IMREAD_GRAYSCALE)
        assert patch is not None
        assert patch.shape == (64, 64)

        tensor = torch.from_numpy(
            patch.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

        m1 = TinyCursorNet()
        m2 = TinyCursorNet()
        m1.eval()
        m2.eval()

        with torch.no_grad():
            conf1 = m1(tensor).item()
            conf2 = m2(tensor).item()

        assert 0.0 <= conf1 <= 1.0
        assert 0.0 <= conf2 <= 1.0

    def test_models_disagree_flag(self):
        """Test models_disagree logic."""
        assert not _models_disagree("pos", "pos")
        assert not _models_disagree("neg", "neg")
        assert _models_disagree("pos", "neg")
        assert _models_disagree("neg", "pos")
        assert not _models_disagree(None, "pos")
        assert not _models_disagree("pos", None)
        assert not _models_disagree(None, None)

    def test_cnn_predicted_from_confidence(self):
        """Test confidence to predicted label mapping."""
        assert _conf_to_label(0.8) == "pos"
        assert _conf_to_label(0.51) == "pos"
        assert _conf_to_label(0.5) == "neg"
        assert _conf_to_label(0.3) == "neg"
        assert _conf_to_label(0.0) == "neg"
        assert _conf_to_label(1.0) == "pos"
        assert _conf_to_label(None) is None

    def test_sample_response_fields(self):
        """Verify the expected fields in a dual-inference sample dict."""
        sample = {
            "filename": "pos_000001.png",
            "label": "pos",
            "cnn_confidence": 0.821,
            "cnn_predicted": "pos",
            "comparison_confidence": 0.312,
            "comp_predicted": "neg",
            "models_disagree": True,
            "origin_type": "sensor",
        }
        assert sample["models_disagree"] is True
        assert sample["cnn_predicted"] != sample["comp_predicted"]
        assert sample["comparison_confidence"] is not None


# ── Tests for context crop ─────────────────────────────────────────

class TestContextCrop:
    def test_context_crop_detection(self, tmp_samples_dir):
        """has_context should be True when ctx_ file exists."""
        import cv2
        ctx = np.zeros((256, 256), dtype=np.uint8)
        cv2.imwrite(str(tmp_samples_dir / "ctx_pos_000001.png"), ctx)

        assert (tmp_samples_dir / "ctx_pos_000001.png").exists()
        assert not (tmp_samples_dir / "ctx_pos_000002.png").exists()

    def test_extract_gray_patch_context_size(self):
        """extract_gray_patch should work with CONTEXT_SIZE."""
        from src.hardware.cursor_sample_collector import extract_gray_patch, CONTEXT_SIZE
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        patch = extract_gray_patch(frame, 960, 540, size=CONTEXT_SIZE)
        assert patch is not None
        assert patch.shape == (256, 256)

    def test_extract_gray_patch_near_edge_returns_none(self):
        """extract_gray_patch with CONTEXT_SIZE near edge returns None."""
        from src.hardware.cursor_sample_collector import extract_gray_patch, CONTEXT_SIZE
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        patch = extract_gray_patch(frame, 10, 10, size=CONTEXT_SIZE)
        assert patch is None

    def test_save_context_crop(self, tmp_path):
        """Test _save_context_crop writes a 256x256 PNG."""
        from src.hardware.cursor_sample_collector import CursorSampleCollector
        collector = CursorSampleCollector(data_dir=tmp_path / "samples")
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        result = collector._save_context_crop(frame, 960, 540, "pos_000001.png")
        assert result is True
        ctx_path = tmp_path / "samples" / "ctx_pos_000001.png"
        assert ctx_path.exists()
        import cv2
        img = cv2.imread(str(ctx_path), cv2.IMREAD_GRAYSCALE)
        assert img.shape == (256, 256)

    def test_save_context_crop_near_edge(self, tmp_path):
        """Context crop near edge should still save (with black padding)."""
        from src.hardware.cursor_sample_collector import CursorSampleCollector
        collector = CursorSampleCollector(data_dir=tmp_path / "samples")
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        result = collector._save_context_crop(frame, 10, 10, "pos_000099.png")
        assert result is True
        ctx_path = tmp_path / "samples" / "ctx_pos_000099.png"
        assert ctx_path.exists()
        import cv2
        img = cv2.imread(str(ctx_path), cv2.IMREAD_GRAYSCALE)
        assert img.shape == (256, 256)

    def test_collect_from_locate_saves_context_for_claude_vision(self, tmp_path):
        """High-value sources should get context crops."""
        from src.hardware.cursor_sample_collector import CursorSampleCollector
        collector = CursorSampleCollector(data_dir=tmp_path / "samples")
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        count = collector.collect_from_locate(
            frame, 960, 540, correlation=1.0, num_negatives=0,
            source="claude_vision")
        assert count >= 1
        # Context crop should exist for the positive sample
        ctx_files = list((tmp_path / "samples").glob("ctx_pos_*.png"))
        assert len(ctx_files) == 1

    def test_collect_from_tracking_no_context(self, tmp_path):
        """Motion tracking samples should NOT get context crops."""
        from src.hardware.cursor_sample_collector import CursorSampleCollector
        collector = CursorSampleCollector(data_dir=tmp_path / "samples")
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        collector.collect_from_tracking(frame, 960, 540, confidence=0.8)
        ctx_files = list((tmp_path / "samples").glob("ctx_*.png"))
        assert len(ctx_files) == 0


# ── Helper functions matching backend logic ────────────────────────

def _models_disagree(pred1, pred2):
    return (pred1 is not None and pred2 is not None and pred1 != pred2)


def _conf_to_label(conf):
    if conf is None:
        return None
    return "pos" if conf > 0.5 else "neg"
