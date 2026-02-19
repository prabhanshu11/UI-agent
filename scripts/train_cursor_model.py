"""Offline training script for TinyCursorNet.

Loads cursor samples from data/cursor_samples/index.jsonl, trains the model,
and saves a versioned .pt file to data/cursor_models/.

Usage:
    uv run python scripts/train_cursor_model.py
    uv run python scripts/train_cursor_model.py --from-model data/cursor_models/v001_20260218.pt
    uv run python scripts/train_cursor_model.py --epochs 50 --lr 0.0005

Requirements:
    Minimum 100 positive + 100 negative samples in data/cursor_samples/.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.hardware.cursor_cnn import TinyCursorNet, save_model, count_parameters

SAMPLES_DIR = PROJECT_ROOT / "data" / "cursor_samples"
MODELS_DIR = PROJECT_ROOT / "data" / "cursor_models"


class CursorPatchDataset(Dataset):
    """Dataset of 64x64 grayscale cursor patches with binary labels."""

    def __init__(self, entries: list[dict], data_dir: Path, augment: bool = False):
        self.entries = entries
        self.data_dir = data_dir
        self.augment = augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = self.data_dir / entry["filename"]
        patch = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        if patch is None:
            # Missing file -- return zero patch with negative label
            patch = np.zeros((64, 64), dtype=np.uint8)
            label = 0.0
        else:
            label = 1.0 if entry["label"] == "pos" else 0.0

        if self.augment:
            patch = self._augment(patch)

        tensor = torch.from_numpy(patch.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, torch.tensor([label], dtype=torch.float32)

    def _augment(self, patch: np.ndarray) -> np.ndarray:
        """Apply random augmentations: flip, rotation, brightness jitter."""
        # Random horizontal flip
        if random.random() > 0.5:
            patch = np.fliplr(patch).copy()

        # Random rotation (0, 90, 180, 270 degrees)
        k = random.randint(0, 3)
        if k > 0:
            patch = np.rot90(patch, k).copy()

        # Brightness jitter (+-30)
        jitter = random.randint(-30, 30)
        patch = np.clip(patch.astype(np.int16) + jitter, 0, 255).astype(np.uint8)

        return patch


def load_reviews(samples_dir: Path) -> dict[str, dict]:
    """Load human reviews from reviews.jsonl, keyed by filename."""
    reviews_path = samples_dir / "reviews.jsonl"
    if not reviews_path.exists():
        return {}

    reviews = {}
    with open(reviews_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    review = json.loads(line)
                    # Last review for each filename wins
                    reviews[review["filename"]] = review
                except (json.JSONDecodeError, KeyError):
                    continue
    return reviews


def load_index(samples_dir: Path) -> list[dict]:
    """Load all entries from index.jsonl, applying review overrides.

    Reviews can flip labels (wrong) or remove samples (delete).
    """
    index_path = samples_dir / "index.jsonl"
    if not index_path.exists():
        return []

    reviews = load_reviews(samples_dir)
    deleted = 0
    flipped = 0

    entries = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            filename = entry.get("filename", "")

            # Apply review overrides
            if filename in reviews:
                review = reviews[filename]
                action = review.get("action")
                if action == "delete":
                    deleted += 1
                    continue  # Skip deleted samples
                elif action == "wrong":
                    # Flip the label
                    entry["label"] = review.get("corrected_label",
                                                "neg" if entry["label"] == "pos" else "pos")
                    flipped += 1

            entries.append(entry)

    if deleted or flipped:
        print(f"Reviews applied: {flipped} label flips, {deleted} deletions")

    # Log source distribution
    sources = {}
    for e in entries:
        src = e.get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    if sources:
        dist = ", ".join(f"{k}: {v}" for k, v in sorted(sources.items()))
        print(f"Source distribution: {dist}")

    return entries


def stratified_split(entries: list[dict], val_ratio: float = 0.2):
    """Split entries into train/val sets, stratified by label."""
    pos = [e for e in entries if e["label"] == "pos"]
    neg = [e for e in entries if e["label"] == "neg"]

    random.shuffle(pos)
    random.shuffle(neg)

    val_pos = int(len(pos) * val_ratio)
    val_neg = int(len(neg) * val_ratio)

    val_entries = pos[:val_pos] + neg[:val_neg]
    train_entries = pos[val_pos:] + neg[val_neg:]

    random.shuffle(val_entries)
    random.shuffle(train_entries)

    return train_entries, val_entries


def train(args):
    entries = load_index(SAMPLES_DIR)
    if not entries:
        print("No samples found in", SAMPLES_DIR / "index.jsonl")
        sys.exit(1)

    pos_count = sum(1 for e in entries if e["label"] == "pos")
    neg_count = sum(1 for e in entries if e["label"] == "neg")
    print(f"Loaded {len(entries)} samples: {pos_count} positive, {neg_count} negative")

    if pos_count < args.min_samples or neg_count < args.min_samples:
        print(f"Need at least {args.min_samples} positive AND {args.min_samples} negative samples.")
        print(f"Currently have: {pos_count} pos, {neg_count} neg. Collect more data first.")
        sys.exit(1)

    # Filter out entries with missing files
    valid_entries = []
    for e in entries:
        if (SAMPLES_DIR / e["filename"]).exists():
            valid_entries.append(e)
    if len(valid_entries) < len(entries):
        print(f"Warning: {len(entries) - len(valid_entries)} samples have missing files, skipped")
    entries = valid_entries

    train_entries, val_entries = stratified_split(entries, val_ratio=0.2)
    print(f"Split: {len(train_entries)} train, {len(val_entries)} val")

    train_ds = CursorPatchDataset(train_entries, SAMPLES_DIR, augment=True)
    val_ds = CursorPatchDataset(val_entries, SAMPLES_DIR, augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    # Model
    model = TinyCursorNet()
    if args.from_model:
        print(f"Fine-tuning from {args.from_model}")
        model.load_state_dict(torch.load(args.from_model, map_location="cpu", weights_only=True))
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for X, y in train_dl:
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            train_correct += ((pred > 0.5).float() == y).sum().item()
            train_total += X.size(0)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X, y in val_dl:
                pred = model(X)
                loss = criterion(pred, y)
                val_loss += loss.item() * X.size(0)
                val_correct += ((pred > 0.5).float() == y).sum().item()
                val_total += X.size(0)

        train_loss /= train_total
        val_loss /= val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # Restore best model
    if best_state:
        model.load_state_dict(best_state)

    # Save
    metrics = {
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "pos_samples": pos_count,
        "neg_samples": neg_count,
        "epochs_trained": epoch,
        "best_val_loss": round(best_val_loss, 4),
        "val_acc": round(val_acc, 4),
        "from_model": str(args.from_model) if args.from_model else None,
    }
    model_path = save_model(model, MODELS_DIR, metrics=metrics)
    print(f"\nModel saved to {model_path}")
    print(f"Best val_loss={best_val_loss:.4f}, val_acc={val_acc:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train TinyCursorNet on cursor samples")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--min-samples", type=int, default=100,
                        help="Minimum samples per class required to train")
    parser.add_argument("--from-model", type=Path, default=None,
                        help="Path to existing .pt model for fine-tuning")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
