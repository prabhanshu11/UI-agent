"""Train Phase 1 vision model: ViT-S encoder + cursor regression head.

Predicts cursor (x, y) position from a full frame using a pretrained ViT-S/16
encoder (frozen) with a trainable 2-layer regression head.

Two training modes:
  1. full-frame: Downscale 1920x1080 -> 384x384, predict normalized (x,y)
     Simpler but cursor is ~3x5px in the input.

  2. crop-based: Sample 384x384 crops containing cursor at native resolution.
     Each crop has cursor at known offset. The head predicts (x,y) within the
     crop. At inference, run multiple crops and pick highest-confidence one.

Input:  data/phase1_dataset/ (from build_phase1_dataset.py)
Output: data/phase1_models/cursor_vit_v{NNN}.pt

Usage:
    uv run python scripts/train_phase1_model.py                      # Full-frame mode
    uv run python scripts/train_phase1_model.py --mode crop          # Crop-based mode
    uv run python scripts/train_phase1_model.py --epochs 50 --lr 1e-4
    uv run python scripts/train_phase1_model.py --eval-only --model data/phase1_models/cursor_vit_v001.pt
"""

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "phase1_dataset"
MODELS_DIR = PROJECT_ROOT / "data" / "phase1_models"

FRAME_W, FRAME_H = 1920, 1080
VIT_INPUT_SIZE = 384  # ViT-S/16 default input resolution


# --- Dataset ---


class CursorRegressionDataset(Dataset):
    """Load (frame, cursor_x, cursor_y) from Phase 1 dataset."""

    def __init__(
        self,
        dataset_dir: Path,
        mode: str = "full",
        crop_size: int = VIT_INPUT_SIZE,
        augment: bool = False,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.frames_dir = self.dataset_dir / "frames"
        self.mode = mode
        self.crop_size = crop_size
        self.augment = augment

        # Load labels
        labels_path = self.dataset_dir / "labels.jsonl"
        self.labels = []
        with open(labels_path) as f:
            for line in f:
                entry = json.loads(line)
                frame_path = self.frames_dir / f"{entry['frame_id']}.jpg"
                if frame_path.exists() or frame_path.is_symlink():
                    self.labels.append(entry)

        print(f"  Dataset: {len(self.labels)} frames, mode={mode}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        entry = self.labels[idx]
        frame_path = self.frames_dir / f"{entry['frame_id']}.jpg"

        # Resolve symlinks
        if frame_path.is_symlink():
            frame_path = frame_path.resolve()

        frame = cv2.imread(str(frame_path))
        if frame is None:
            # Return a black frame with center cursor as fallback
            frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
            entry = {**entry, "cursor_x": FRAME_W // 2, "cursor_y": FRAME_H // 2}

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cx, cy = entry["cursor_x"], entry["cursor_y"]

        if self.mode == "crop":
            return self._get_crop(frame, cx, cy)
        else:
            return self._get_fullframe(frame, cx, cy)

    def _get_fullframe(self, frame, cx, cy):
        """Downscale full frame to VIT_INPUT_SIZE, normalize cursor coords."""
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (VIT_INPUT_SIZE, VIT_INPUT_SIZE))

        # Normalize cursor to [0, 1]
        cx_norm = cx / w
        cy_norm = cy / h

        if self.augment:
            resized = self._augment_image(resized)

        tensor = self._to_tensor(resized)
        target = torch.tensor([cx_norm, cy_norm], dtype=torch.float32)
        return tensor, target

    def _get_crop(self, frame, cx, cy):
        """Extract a crop containing the cursor at native resolution."""
        h, w = frame.shape[:2]
        half = self.crop_size // 2

        # Random offset so cursor isn't always centered (during training)
        if self.augment:
            offset_x = random.randint(-half // 2, half // 2)
            offset_y = random.randint(-half // 2, half // 2)
        else:
            offset_x, offset_y = 0, 0

        # Crop center = cursor position + random offset
        crop_cx = cx + offset_x
        crop_cy = cy + offset_y

        x1 = max(0, crop_cx - half)
        y1 = max(0, crop_cy - half)
        x2 = min(w, x1 + self.crop_size)
        y2 = min(h, y1 + self.crop_size)
        x1 = max(0, x2 - self.crop_size)
        y1 = max(0, y2 - self.crop_size)

        crop = frame[y1:y2, x1:x2]
        if crop.shape[0] != self.crop_size or crop.shape[1] != self.crop_size:
            crop = cv2.resize(crop, (self.crop_size, self.crop_size))

        # Cursor position within crop, normalized to [0, 1]
        cx_in_crop = (cx - x1) / self.crop_size
        cy_in_crop = (cy - y1) / self.crop_size
        cx_in_crop = max(0.0, min(1.0, cx_in_crop))
        cy_in_crop = max(0.0, min(1.0, cy_in_crop))

        if self.augment:
            crop = self._augment_image(crop)

        tensor = self._to_tensor(crop)
        target = torch.tensor([cx_in_crop, cy_in_crop], dtype=torch.float32)
        return tensor, target

    def _augment_image(self, img):
        """Apply training augmentations."""
        # Brightness/contrast jitter
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.randint(-20, 20)  # brightness
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(
                np.uint8
            )

        # Gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    @staticmethod
    def _to_tensor(img):
        """Convert HWC uint8 image to CHW float32 tensor, ImageNet-normalized."""
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        # ImageNet normalization (required for pretrained ViT)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor


# --- Model ---


class CursorRegressionHead(nn.Module):
    """2-layer MLP regression head for cursor (x, y) prediction.

    Takes ViT encoder features (cls_token or pooled) and outputs
    normalized (x, y) in [0, 1].
    """

    def __init__(self, in_features: int, hidden_dim: int = 256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, x):
        return self.head(x)


class CursorViT(nn.Module):
    """Frozen ViT-S/16 encoder + trainable cursor regression head."""

    def __init__(self, encoder_name: str = "vit_small_patch16_384"):
        super().__init__()

        import timm

        # Load pretrained encoder
        self.encoder = timm.create_model(
            encoder_name, pretrained=True, num_classes=0  # Remove classifier
        )

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Get encoder output dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, VIT_INPUT_SIZE, VIT_INPUT_SIZE)
            feat_dim = self.encoder(dummy).shape[-1]

        self.head = CursorRegressionHead(feat_dim)

        print(f"  Encoder: {encoder_name} ({feat_dim}d features, frozen)")
        print(
            f"  Head: {sum(p.numel() for p in self.head.parameters())} trainable params"
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.head(features)

    def unfreeze_encoder(self, n_layers: int = 2):
        """Unfreeze the last N transformer blocks for fine-tuning."""
        blocks = list(self.encoder.blocks)
        for block in blocks[-n_layers:]:
            for param in block.parameters():
                param.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Unfroze last {n_layers} encoder blocks ({trainable} trainable params)")


# --- Training ---


def train_one_epoch(
    model: CursorViT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict:
    """Train for one epoch. Returns loss and pixel error metrics."""
    model.train()
    total_loss = 0.0
    pixel_errors = []
    n_batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)
        loss = nn.functional.mse_loss(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        # Compute pixel error (denormalize to 1080p)
        with torch.no_grad():
            pred_px = predictions.cpu().numpy()
            true_px = targets.cpu().numpy()
            pred_px[:, 0] *= FRAME_W
            pred_px[:, 1] *= FRAME_H
            true_px[:, 0] *= FRAME_W
            true_px[:, 1] *= FRAME_H
            errors = np.sqrt(((pred_px - true_px) ** 2).sum(axis=1))
            pixel_errors.extend(errors.tolist())

    pixel_errors = np.array(pixel_errors)
    return {
        "loss": total_loss / n_batches,
        "pixel_error_median": float(np.median(pixel_errors)),
        "pixel_error_mean": float(np.mean(pixel_errors)),
        "pixel_error_90th": float(np.percentile(pixel_errors, 90)),
        "pixel_error_max": float(np.max(pixel_errors)),
    }


@torch.no_grad()
def validate(
    model: CursorViT,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate model. Returns loss and pixel error metrics."""
    model.eval()
    total_loss = 0.0
    pixel_errors = []
    n_batches = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)
        loss = nn.functional.mse_loss(predictions, targets)

        total_loss += loss.item()
        n_batches += 1

        pred_px = predictions.cpu().numpy()
        true_px = targets.cpu().numpy()
        pred_px[:, 0] *= FRAME_W
        pred_px[:, 1] *= FRAME_H
        true_px[:, 0] *= FRAME_W
        true_px[:, 1] *= FRAME_H
        errors = np.sqrt(((pred_px - true_px) ** 2).sum(axis=1))
        pixel_errors.extend(errors.tolist())

    pixel_errors = np.array(pixel_errors)
    return {
        "loss": total_loss / max(1, n_batches),
        "pixel_error_median": float(np.median(pixel_errors)),
        "pixel_error_mean": float(np.mean(pixel_errors)),
        "pixel_error_90th": float(np.percentile(pixel_errors, 90)),
        "pixel_error_max": float(np.max(pixel_errors)),
    }


def save_checkpoint(
    model: CursorViT,
    metrics: dict,
    models_dir: Path,
    mode: str,
) -> Path:
    """Save model checkpoint with versioned naming."""
    models_dir.mkdir(parents=True, exist_ok=True)

    # Find next version — extract NNN from cursor_vit_vNNN_...
    import re as _re
    existing = sorted(models_dir.glob("cursor_vit_v*.pt"))
    if existing:
        nums = []
        for f in existing:
            m = _re.search(r"cursor_vit_v(\d+)", f.stem)
            if m:
                nums.append(int(m.group(1)))
        next_num = max(nums) + 1 if nums else 1
    else:
        next_num = 1

    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"cursor_vit_v{next_num:03d}_{date_str}_{mode}.pt"
    path = models_dir / filename

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "head_state_dict": model.head.state_dict(),
            "metrics": metrics,
            "mode": mode,
            "timestamp": datetime.now().isoformat(),
        },
        path,
    )

    # Log training metrics
    log_path = models_dir / "training_log.jsonl"
    entry = {
        "model_file": filename,
        "version": next_num,
        "mode": mode,
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    return path


def main():
    parser = argparse.ArgumentParser(description="Train Phase 1 cursor regression model")
    parser.add_argument("--mode", choices=["full", "crop"], default="full",
                       help="Training mode: full-frame or crop-based")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--encoder", default="vit_small_patch16_384",
                       help="timm encoder model name")
    parser.add_argument("--unfreeze-after", type=int, default=0,
                       help="Unfreeze last N encoder blocks after this epoch (0=never)")
    parser.add_argument("--unfreeze-layers", type=int, default=2,
                       help="Number of encoder blocks to unfreeze")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--model", type=str, help="Path to model for eval-only")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Mode: {args.mode}")
    print(f"Encoder: {args.encoder}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = CursorRegressionDataset(
        DATASET_DIR,
        mode=args.mode,
        augment=True,
    )

    if len(dataset) < 50:
        print(f"ERROR: Only {len(dataset)} samples. Need at least 50.")
        print("Run scripts/build_phase1_dataset.py first.")
        return

    # Train/val split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # Disable augmentation for val set by creating a wrapper
    val_dataset_no_aug = CursorRegressionDataset(
        DATASET_DIR, mode=args.mode, augment=False
    )
    val_indices = val_set.indices
    val_set_clean = torch.utils.data.Subset(val_dataset_no_aug, val_indices)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set_clean, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True,
    )

    print(f"  Train: {train_size}, Val: {val_size}")

    # Build model
    print("\nBuilding model...")
    device = torch.device(args.device)
    model = CursorViT(encoder_name=args.encoder).to(device)

    if args.eval_only:
        if not args.model:
            print("ERROR: --model required for --eval-only")
            return
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics = validate(model, val_loader, device)
        print(f"\nValidation results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return

    # Training
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_error = float("inf")
    best_epoch = 0

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"{'Epoch':>5} {'TrLoss':>8} {'TrPxMed':>8} {'VlLoss':>8} {'VlPxMed':>8} {'VlPx90':>8} {'LR':>10}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # Optional encoder unfreezing
        if args.unfreeze_after > 0 and epoch == args.unfreeze_after:
            model.unfreeze_encoder(args.unfreeze_layers)
            # Add encoder params to optimizer
            encoder_params = [
                p for p in model.encoder.parameters() if p.requires_grad
            ]
            optimizer.add_param_group({"params": encoder_params, "lr": args.lr * 0.1})

        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch:5d} "
            f"{train_metrics['loss']:8.6f} "
            f"{train_metrics['pixel_error_median']:8.1f} "
            f"{val_metrics['loss']:8.6f} "
            f"{val_metrics['pixel_error_median']:8.1f} "
            f"{val_metrics['pixel_error_90th']:8.1f} "
            f"{lr:10.2e} "
            f"({elapsed:.1f}s)"
        )

        # Save best model
        if val_metrics["pixel_error_median"] < best_val_error:
            best_val_error = val_metrics["pixel_error_median"]
            best_epoch = epoch
            path = save_checkpoint(
                model,
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                },
                MODELS_DIR,
                args.mode,
            )
            print(f"       -> Saved best model: {path.name} (median px error: {best_val_error:.1f})")

        # Early stopping: no improvement in 20 epochs
        if epoch - best_epoch >= 20:
            print(f"\nEarly stopping at epoch {epoch} (best was epoch {best_epoch})")
            break

    print(f"\nTraining complete.")
    print(f"Best validation pixel error (median): {best_val_error:.1f}px at epoch {best_epoch}")
    print(f"Models saved to: {MODELS_DIR}")

    # Final report
    print(f"\nFinal validation on best model:")
    best_models = sorted(MODELS_DIR.glob("cursor_vit_v*.pt"), reverse=True)
    if best_models:
        checkpoint = torch.load(best_models[0], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        final_metrics = validate(model, val_loader, device)
        for k, v in final_metrics.items():
            if "loss" in k:
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v:.1f}px")

        # Performance targets
        print(f"\n  Target checks:")
        med = final_metrics["pixel_error_median"]
        p90 = final_metrics["pixel_error_90th"]
        print(f"    Median pixel error < 10px: {'PASS' if med < 10 else 'FAIL'} ({med:.1f}px)")
        print(f"    90th pctile error < 25px:  {'PASS' if p90 < 25 else 'FAIL'} ({p90:.1f}px)")


if __name__ == "__main__":
    main()
