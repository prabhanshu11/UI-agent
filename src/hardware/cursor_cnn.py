"""TinyCursorNet — lightweight CNN for cursor presence classification.

Binary classifier: "Is there a mouse cursor in this 64x64 grayscale patch?"

Architecture:
    Input: 1x64x64 grayscale
    Conv1: 1->16 channels, 5x5, stride 2, padding 2 -> 16x32x32
    Conv2: 16->32 channels, 3x3, stride 2, padding 1 -> 32x16x16
    Conv3: 32->32 channels, 3x3, stride 2, padding 1 -> 32x8x8
    Global Average Pool -> 32
    FC: 32->1 -> sigmoid

Total: ~14K parameters. Inference: <2ms on CPU (Ryzen 5 3600).

Usage:
    model = TinyCursorNet()
    # Train with BCE loss...
    save_model(model, model_dir, metrics={"val_acc": 0.95})

    # Later:
    model = load_latest_model(model_dir)
    confidence = model(tensor_1x1x64x64).item()
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


class TinyCursorNet(nn.Module):
    """Lightweight binary classifier for cursor detection in 64x64 patches."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),  # -> 16x32x32
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # -> 32x16x16
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  # -> 32x8x8
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)  # -> 32x1x1
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, 1, 64, 64) float tensor in [0, 1].

        Returns:
            (batch, 1) confidence scores in [0, 1].
        """
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def save_model(
    model: TinyCursorNet,
    model_dir: Path,
    metrics: Optional[dict] = None,
) -> Path:
    """Save model with versioned naming and log training metrics.

    Naming: vNNN_YYYYMMDD.pt where NNN increments from existing files.

    Args:
        model: Trained TinyCursorNet.
        model_dir: Directory for model files.
        metrics: Optional training metrics to log.

    Returns:
        Path to saved model file.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Find next version number
    existing = sorted(model_dir.glob("v*_*.pt"))
    if existing:
        last_name = existing[-1].stem  # e.g., "v003_20260218"
        last_num = int(last_name.split("_")[0][1:])
        next_num = last_num + 1
    else:
        next_num = 1

    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"v{next_num:03d}_{date_str}.pt"
    model_path = model_dir / filename

    torch.save(model.state_dict(), model_path)

    # Log training metrics
    if metrics:
        log_path = model_dir / "training_log.jsonl"
        entry = {
            "model_file": filename,
            "version": next_num,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    return model_path


def load_latest_model(model_dir: Path) -> Optional[TinyCursorNet]:
    """Load the most recent model from model_dir.

    Returns None if no models found or loading fails.
    """
    model_dir = Path(model_dir)
    model_files = sorted(model_dir.glob("v*_*.pt"), reverse=True)
    if not model_files:
        return None

    model = TinyCursorNet()
    model.load_state_dict(torch.load(model_files[0], map_location="cpu", weights_only=True))
    model.eval()
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
