#!/usr/bin/env python3
"""
train_stage_b.py — EfficientNet fish species classifier training script (Stage B)

Stage B classifies fish detected by Stage A into 15 species/family classes:

  ORIGINAL (6):
  0: pike           — щука           (Esox lucius)
  1: taimen         — таймень        (Hucho taimen)
  2: grayling       — хариус         (Thymallus spp.)
  3: whitefish      — сиг            (Coregonus spp.)
  4: perch          — окунь          (Perca fluviatilis)

  NEW — Salmonidae (3):
  5: brown_trout    — форель/кумжа   (Salmo trutta)
  6: rainbow_trout  — радужная форель(Oncorhynchus mykiss)
  7: atlantic_salmon— сёмга         (Salmo salar)

  NEW — Cyprinidae (5):
  8: common_carp    — карп/сазан     (Cyprinus carpio)
  9: crucian_carp   — карась         (Carassius carassius)
  10: bream         — лещ            (Abramis brama)
  11: roach         — плотва         (Rutilus rutilus)
  12: ide           — язь            (Leuciscus idus)

  NEW — Siluriformes (1):
  13: wels_catfish  — сом            (Silurus glanis)

  FALLBACK (1):
  14: unknown_fish  — неизвестная рыба

Model selection rationale:
  - EfficientNet-B0 (5.3M params, ~6ms CPU): default. Good for datasets < 500 img/class.
    Use when total dataset is < 5000 images. Regularization via label smoothing (ε=0.1)
    compensates for small dataset size.
  - EfficientNet-B2 (9.1M params, ~12ms CPU): use with --model b2 when dataset grows
    to > 500 images per class. Better accuracy on fine-grained discrimination.
    B2 adds ~6ms latency per inference on CPU — acceptable for a Telegram bot.
  - Decision: ship B0 as default, B2 as opt-in flag. Never use B3+ on CPU.

Training strategy — two phases:
  Phase 1 (default 10 epochs): Freeze backbone, train only classifier head.
    LR=1e-3, weight_decay=1e-4  — fast convergence on new head weights
  Phase 2 (default 20 epochs): Unfreeze all layers, end-to-end fine-tuning.
    LR=1e-4, cosine annealing  — gentle fine-tuning to preserve ImageNet features

Regularization additions (v2):
  - Label smoothing (ε=0.1) — reduces overconfidence on small dataset
  - Mixup augmentation (α=0.2) — reduces overfitting by interpolating samples
  - Class-weighted cross-entropy — handles species imbalance

Dataset split: 70% train / 15% val / 15% test, stratified, seed=42.

Usage:
    python3 scripts/train_stage_b.py [--epochs_phase1 10] [--epochs_phase2 20] \\
                                     [--batch 32] [--device cpu] [--model b0|b2]
"""

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
STAGE_B_DIR = DATA_ROOT / "fish_dataset" / "stage_b"
MODELS_DIR = DATA_ROOT / "fish_models"

# Module-level model variant (set by CLI --model flag; also used by update_metadata)
_MODEL_VARIANT: str = "b0"

CLASS_NAMES_B = {
    "0":  "pike",
    "1":  "taimen",
    "2":  "grayling",
    "3":  "whitefish",
    "4":  "perch",
    "5":  "brown_trout",
    "6":  "rainbow_trout",
    "7":  "atlantic_salmon",
    "8":  "common_carp",
    "9":  "crucian_carp",
    "10": "bream",
    "11": "roach",
    "12": "ide",
    "13": "wels_catfish",
    "14": "unknown_fish",
}
# Reverse lookup: folder name -> index
NAME_TO_IDX = {v: int(k) for k, v in CLASS_NAMES_B.items()}

MIN_IMAGES_PER_CLASS = 15
SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 0.15 (implicit)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def _ok(msg: str) -> None:
    print(f"[OK]  {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------

def check_torch() -> None:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        _ok(f"torch {torch.__version__}, torchvision {torchvision.__version__}")
    except ImportError as exc:
        _fail(f"PyTorch/torchvision not installed: {exc}")
        _fail("Install with:  pip install torch torchvision")
        sys.exit(1)


def check_stage_b_data() -> dict[str, list[Path]]:
    """
    Verify species folders exist and have enough images.
    Returns dict mapping species_name -> list of image paths.
    """
    if not STAGE_B_DIR.exists():
        _fail(f"Stage B directory not found: {STAGE_B_DIR}")
        sys.exit(1)

    species_images: dict[str, list[Path]] = {}
    for name in CLASS_NAMES_B.values():
        folder = STAGE_B_DIR / name
        if not folder.exists():
            _info(f"Species folder missing (will be skipped): {folder}")
            continue
        imgs = _images_in(folder)
        species_images[name] = imgs

    # Count classes with enough images
    ok_classes = [n for n, imgs in species_images.items() if len(imgs) >= MIN_IMAGES_PER_CLASS]
    if len(ok_classes) < 2:
        _fail(
            f"Need at least 2 species with >= {MIN_IMAGES_PER_CLASS} images each. "
            f"Found: {len(ok_classes)}."
        )
        for name, imgs in species_images.items():
            status = "OK" if len(imgs) >= MIN_IMAGES_PER_CLASS else "TOO FEW"
            print(f"  [{status}] {name}: {len(imgs)} images", file=sys.stderr)
        sys.exit(1)

    for name, imgs in sorted(species_images.items()):
        status = "OK" if len(imgs) >= MIN_IMAGES_PER_CLASS else "WARN(too few)"
        _info(f"  {name:<15}: {len(imgs):>4} images  [{status}]")

    return species_images


def check_models_dir() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _ok(f"Models directory ready: {MODELS_DIR}")


# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------

def split_dataset(
    species_images: dict[str, list[Path]],
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    """
    Stratified split per species. Returns (train, val, test) as lists of (path, class_idx).
    Only includes species that appear in CLASS_NAMES_B and have >= MIN_IMAGES_PER_CLASS images.
    """
    rng = random.Random(SPLIT_SEED)

    train_items: list[tuple[Path, int]] = []
    val_items: list[tuple[Path, int]] = []
    test_items: list[tuple[Path, int]] = []

    _banner("Dataset Split (seed=42, stratified 70/15/15)")
    print(f"  {'Species':<15} {'Total':>6} {'Train':>6} {'Val':>5} {'Test':>5}")
    print(f"  {'-' * 40}")

    for name, imgs in sorted(species_images.items()):
        if name not in NAME_TO_IDX:
            continue
        if len(imgs) < MIN_IMAGES_PER_CLASS:
            _info(f"  Skipping {name} (only {len(imgs)} images < {MIN_IMAGES_PER_CLASS})")
            continue

        idx = NAME_TO_IDX[name]
        shuffled = imgs.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        # remainder goes to test
        n_test = n - n_train - n_val
        if n_test < 1:
            # edge case: ensure at least 1 in test by reducing val
            n_val = max(0, n_val - 1)
            n_test = n - n_train - n_val

        train_split = shuffled[:n_train]
        val_split = shuffled[n_train : n_train + n_val]
        test_split = shuffled[n_train + n_val :]

        print(f"  {name:<15} {n:>6} {len(train_split):>6} {len(val_split):>5} {len(test_split):>5}")

        train_items.extend((p, idx) for p in train_split)
        val_items.extend((p, idx) for p in val_split)
        test_items.extend((p, idx) for p in test_split)

    print()
    _ok(f"Total: train={len(train_items)}, val={len(val_items)}, test={len(test_items)}")
    return train_items, val_items, test_items


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------

class FishDataset:
    """Minimal PyTorch Dataset wrapping a list of (path, class_idx) pairs."""

    def __init__(self, items: list[tuple[Path, int]], transform):
        from PIL import Image  # type: ignore

        self.items = items
        self.transform = transform
        self._Image = Image

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        import torch

        path, label = self.items[idx]
        try:
            img = self._Image.open(path).convert("RGB")
        except Exception as exc:
            # Return a black image on read error rather than crashing the worker
            print(f"[WARN] Could not read {path}: {exc}", file=sys.stderr)
            import torch
            img = self._Image.new("RGB", (224, 224))
        img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)


def build_transforms():
    from torchvision import transforms

    train_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # hue=0 avoids numpy uint8 overflow bug in torchvision 0.17
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            # ImageNet normalisation — required by EfficientNet pretrained weights
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_transforms, val_transforms


def compute_class_weights(
    train_items: list[tuple[Path, int]], num_classes: int
) -> "torch.Tensor":
    """Inverse-frequency class weights to handle imbalanced species counts."""
    import torch

    counts = [0] * num_classes
    for _, idx in train_items:
        counts[idx] += 1

    total = sum(counts)
    weights = []
    for c in counts:
        weights.append(total / (num_classes * c) if c > 0 else 1.0)

    _info("Class weights (inverse frequency):")
    for i, w in enumerate(weights):
        name = CLASS_NAMES_B.get(str(i), str(i))
        print(f"    {name:<15}: {w:.4f}  (count={counts[i]})")

    return torch.tensor(weights, dtype=torch.float32)


def build_loaders(
    train_items, val_items, test_items, batch_size: int, num_workers: int = 0
):
    import torch
    from torch.utils.data import DataLoader

    train_tf, val_tf = build_transforms()

    train_ds = FishDataset(train_items, train_tf)
    val_ds = FishDataset(val_items, val_tf)
    test_ds = FishDataset(test_items, val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int, model_variant: str = "b0") -> "torch.nn.Module":
    """
    Build EfficientNet model for fish species classification.

    model_variant: "b0" (default, 5.3M params, ~6ms CPU) or "b2" (9.1M, ~12ms CPU).

    Selection guidance:
      - b0: Use when dataset < 500 images per class. Faster inference, sufficient for
            fine-grained discrimination with label smoothing + mixup regularization.
      - b2: Use when dataset > 500 images per class. Better accuracy on 15-class problem.
            Adds ~6ms latency on CPU — acceptable for Telegram bot async pipeline.
    """
    import torch.nn as nn

    if model_variant == "b2":
        from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2
        model = efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[1].in_features  # 1408 for B2
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        _ok(
            f"EfficientNet-B2 loaded (ImageNet weights). "
            f"Head replaced: {num_ftrs} -> {num_classes} classes. "
            f"Use this when dataset has > 500 images per class."
        )
    else:
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        # Original head: model.classifier = Sequential(Dropout, Linear(1280, 1000))
        num_ftrs = model.classifier[1].in_features  # 1280 for B0
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        _ok(
            f"EfficientNet-B0 loaded (ImageNet weights). "
            f"Head replaced: {num_ftrs} -> {num_classes} classes."
        )

    return model


def freeze_backbone(model: "torch.nn.Module") -> None:
    """Freeze all layers except the classifier head (Phase 1)."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _info(f"Backbone frozen. Trainable params: {trainable:,}")


def unfreeze_all(model: "torch.nn.Module") -> None:
    """Unfreeze all layers for end-to-end fine-tuning (Phase 2)."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _info(f"All layers unfrozen. Trainable params: {trainable:,}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run_epoch(
    model,
    loader,
    criterion,
    optimizer: Optional[object],
    device: "torch.device",
    train: bool,
) -> tuple[float, float]:
    """Run one epoch. Returns (avg_loss, accuracy)."""
    import torch

    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if train and optimizer is not None:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train and optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total if total > 0 else float("inf")
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def mixup_batch(
    images: "torch.Tensor",
    labels: "torch.Tensor",
    alpha: float = 0.2,
    num_classes: int = 15,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    """
    Mixup augmentation: interpolate pairs of training samples.
    Returns mixed_images, mixed_labels (as soft label distributions).

    Why mixup for fish classification?
    - Fish species share morphological features (fins, scale patterns, body shape)
    - Mixup forces the model to learn smooth decision boundaries
    - Reduces overconfident predictions on the small dataset
    - Works well for fine-grained species discrimination

    Reference: Zhang et al. (2018) "mixup: Beyond Empirical Risk Minimization"
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    if alpha <= 0:
        # Convert labels to one-hot for consistency
        soft = F.one_hot(labels, num_classes=num_classes).float()
        return images, soft

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_x = lam * images + (1 - lam) * images[index]

    # Soft labels: convex combination of one-hot encodings
    y_a = F.one_hot(labels, num_classes=num_classes).float()
    y_b = F.one_hot(labels[index], num_classes=num_classes).float()
    mixed_y = lam * y_a + (1 - lam) * y_b

    return mixed_x, mixed_y


def _run_epoch_mixup(
    model,
    loader,
    soft_criterion,
    optimizer,
    device: "torch.device",
    mixup_alpha: float = 0.2,
    num_classes: int = 15,
) -> float:
    """Training-only epoch with mixup augmentation. Returns avg_loss."""
    import torch

    model.train(True)
    total_loss = 0.0
    total = 0

    with torch.enable_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Apply mixup
            mixed_images, soft_labels = mixup_batch(images, labels, alpha=mixup_alpha, num_classes=num_classes)
            mixed_images = mixed_images.to(device)
            soft_labels = soft_labels.to(device)

            outputs = model(mixed_images)
            loss = soft_criterion(outputs, soft_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total += images.size(0)

    return total_loss / total if total > 0 else float("inf")


def train_phase(
    phase: int,
    model,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: "torch.device",
    class_weights: "torch.Tensor",
    patience: int = 10,
    best_val_acc: float = 0.0,
    best_state: Optional[dict] = None,
    label_smoothing: float = 0.1,
    mixup_alpha: float = 0.2,
    num_classes: int = 15,
) -> tuple[float, Optional[dict]]:
    """
    Train one phase. Returns (best_val_acc, best_state_dict).
    best_state is carried across phases so Phase 2 can beat Phase 1's checkpoint.

    Regularization in v2:
    - label_smoothing=0.1: prevents overconfident predictions on small dataset
    - mixup_alpha=0.2: interpolates samples for smoother decision boundaries
    Both are disabled (set to 0) for val/test to get clean accuracy numbers.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    # Label smoothing cross-entropy: built into PyTorch >= 1.10
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=label_smoothing,
    )
    # Soft-label loss (used with mixup) — manual KL divergence
    def soft_criterion(logits, soft_labels):
        log_probs = F.log_softmax(logits, dim=1)
        return -(soft_labels * log_probs).sum(dim=1).mean()

    if phase == 1:
        # Phase 1: train head only with a higher LR
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        scheduler = None
    else:
        # Phase 2: fine-tune everything with cosine annealing
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    patience_counter = 0

    print(f"\n  {'Epoch':<6} {'TrainLoss':>10} {'ValLoss':>9} {'ValAcc':>8} {'Best':>7}")
    print(f"  {'-' * 45}")

    for epoch in range(1, epochs + 1):
        # Training pass: use mixup if alpha > 0
        if mixup_alpha > 0:
            train_loss = _run_epoch_mixup(
                model, train_loader, soft_criterion, optimizer, device,
                mixup_alpha=mixup_alpha, num_classes=num_classes,
            )
        else:
            train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)

        # Validation pass: no mixup, clean labels
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device, train=False)

        if scheduler is not None:
            scheduler.step()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "*"
        else:
            patience_counter += 1
            marker = ""

        print(
            f"  {epoch:<6} {train_loss:>10.4f} {val_loss:>9.4f} "
            f"{val_acc:>7.2%} {best_val_acc:>6.2%} {marker}"
        )

        if patience_counter >= patience:
            _info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    return best_val_acc, best_state


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_test_set(
    model,
    test_loader,
    device: "torch.device",
    num_classes: int,
    present_classes: list[int],
) -> dict:
    """Run inference on test set, print confusion matrix, return metrics dict.

    present_classes is the sorted list of class indices actually present in the
    training data — may be fewer than len(CLASS_NAMES_B) if some species had
    too few images to train on.
    """
    import torch

    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    # Build confusion matrix sized to the actual number of classes trained
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for true, pred in zip(all_labels, all_preds):
        if 0 <= true < num_classes and 0 <= pred < num_classes:
            matrix[true][pred] += 1

    correct = sum(matrix[i][i] for i in range(num_classes))
    total = len(all_labels)
    overall_acc = correct / total if total > 0 else 0.0

    # Build name list aligned to the model's class indices.
    # The model output index IS the original CLASS_NAMES_B index (labels are assigned
    # directly via NAME_TO_IDX which maps to the canonical class index).
    # Do NOT use present_classes as a positional list — that shifts names when any
    # intermediate class index (e.g. taimen=1) has no training data.
    def _class_name(model_idx: int) -> str:
        return CLASS_NAMES_B.get(str(model_idx), str(model_idx))

    _banner("Test Set Evaluation")
    print(f"  Overall accuracy: {overall_acc:.2%}  ({correct}/{total})")

    # Per-class accuracy — only report classes that actually had test samples.
    print(f"\n  {'Class':<15} {'Correct':>8} {'Total':>7} {'Accuracy':>10}")
    print(f"  {'-' * 43}")
    per_class: dict[str, float] = {}
    for i in range(num_classes):
        name = _class_name(i)
        row_total = sum(matrix[i])
        row_correct = matrix[i][i]
        acc = row_correct / row_total if row_total > 0 else 0.0
        if row_total == 0:
            # Ghost class (index present due to gap in class numbering but no data).
            # Exclude from per_class_accuracy to avoid misleading 0.0 entries.
            print(f"  {name:<15} {'—':>8} {'0':>7} {'(no samples)':>10}")
            continue
        per_class[name] = round(acc, 4)
        print(f"  {name:<15} {row_correct:>8} {row_total:>7} {acc:>9.2%}")

    # Confusion matrix table
    print("\n  Confusion matrix (rows=true, cols=predicted):")
    names_short = [_class_name(i)[:8] for i in range(num_classes)]
    header = "  " + " " * 15 + "".join(f"{n:>9}" for n in names_short)
    print(header)
    for i in range(num_classes):
        row_label = _class_name(i)[:14]
        row = "  " + f"{row_label:<15}" + "".join(f"{matrix[i][j]:>9}" for j in range(num_classes))
        print(row)

    return {
        "overall_accuracy": round(overall_acc, 4),
        "per_class_accuracy": per_class,
    }


# ---------------------------------------------------------------------------
# Output persistence
# ---------------------------------------------------------------------------

def save_model(state_dict: dict) -> None:
    import torch

    dest = MODELS_DIR / "classifier_v1.pt"
    torch.save(state_dict, dest)
    _ok(f"Saved model state dict -> {dest}")


def write_class_names() -> None:
    dest = MODELS_DIR / "class_names_b.json"
    dest.write_text(json.dumps(CLASS_NAMES_B, indent=2, ensure_ascii=False), encoding="utf-8")
    _ok(f"Wrote {dest}")


def update_metadata(
    metrics: dict,
    train_size: int,
    val_size: int,
    test_size: int,
    epochs_phase1: int,
    epochs_phase2: int,
) -> None:
    meta_path = MODELS_DIR / "metadata.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    else:
        meta = {}

    meta.setdefault("classifier", {})
    meta["classifier"].update(
        {
            "backend": f"efficientnet_{_MODEL_VARIANT}",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "num_classes": len(CLASS_NAMES_B),
            "class_names": CLASS_NAMES_B,
            "dataset_size": {
                "train": train_size,
                "val": val_size,
                "test": test_size,
            },
            "training": {
                "epochs_phase1": epochs_phase1,
                "epochs_phase2": epochs_phase2,
                "split_seed": SPLIT_SEED,
                "label_smoothing": 0.1,
                "mixup_alpha": 0.2,
            },
            "metrics": metrics,
        }
    )

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    _ok(f"Updated {meta_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNet fish species classifier (Stage B) — 15 classes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs_phase1", type=int, default=10,
                        help="Epochs for Phase 1 (frozen backbone, head only)")
    parser.add_argument("--epochs_phase2", type=int, default=20,
                        help="Epochs for Phase 2 (full fine-tuning)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device: 'cpu', 'cuda', 'mps' (Apple Silicon)"
    )
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader num_workers (0 = main process, safe on macOS)")
    parser.add_argument(
        "--model", type=str, default="b0", choices=["b0", "b2"],
        help=(
            "EfficientNet variant. b0=5.3M params/~6ms CPU (default, use with <500 img/class). "
            "b2=9.1M params/~12ms CPU (use when dataset grows past 500 img/class)."
        ),
    )
    parser.add_argument(
        "--label-smoothing", type=float, default=0.1,
        help="Label smoothing epsilon (0=disabled). Reduces overconfidence on small datasets.",
    )
    parser.add_argument(
        "--mixup-alpha", type=float, default=0.2,
        help="Mixup alpha (0=disabled). Interpolates samples for smoother decision boundaries.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    global _MODEL_VARIANT
    args = parse_args()
    _MODEL_VARIANT = args.model

    _banner(f"Stage B — EfficientNet-{args.model.upper()} Fish Species Classifier Training")
    _info(f"Repo root      : {REPO_ROOT}")
    _info(f"Classes        : {len(CLASS_NAMES_B)} ({', '.join(CLASS_NAMES_B.values())})")
    _info(f"Phase 1 epochs : {args.epochs_phase1}  (frozen backbone, head only, LR=1e-3)")
    _info(f"Phase 2 epochs : {args.epochs_phase2}  (full fine-tune, LR=1e-4, cosine LR)")
    _info(f"Batch size     : {args.batch}")
    _info(f"Device         : {args.device}")
    _info(f"Model variant  : EfficientNet-{args.model.upper()}")
    _info(f"Label smoothing: {args.label_smoothing}")
    _info(f"Mixup alpha    : {args.mixup_alpha}")

    # ---- Prerequisites ----
    _banner("Prerequisite Checks")
    check_torch()
    species_images = check_stage_b_data()
    check_models_dir()

    # ---- Split ----
    train_items, val_items, test_items = split_dataset(species_images)

    # Determine actual number of classes present in training data
    present_classes = sorted(set(idx for _, idx in train_items))
    num_classes = max(present_classes) + 1
    _info(f"Classes in training data: {num_classes} (indices {present_classes})")
    trained_names = [CLASS_NAMES_B.get(str(i), str(i)) for i in present_classes]
    _info(f"  Species: {', '.join(trained_names)}")

    # Safety check: detect gaps in present_classes (ghost neurons).
    # A gap means one or more class indices in [0..max] have no training data,
    # creating "ghost" output neurons that waste softmax capacity.
    # The evaluation reporter handles this correctly (skips zero-sample rows),
    # but flag it clearly so the operator knows why num_classes > len(present_classes).
    expected_contiguous = list(range(len(present_classes)))
    remapped = list(range(num_classes))
    gaps = sorted(set(remapped) - set(present_classes))
    if gaps:
        gap_names = [CLASS_NAMES_B.get(str(g), str(g)) for g in gaps]
        _warn(
            f"Class index gaps detected: indices {gaps} ({', '.join(gap_names)}) have no "
            f"training data. Model will have {len(gaps)} ghost output neuron(s). "
            f"This is expected when some taxonomy classes lack images. "
            f"Evaluation correctly skips zero-sample rows."
        )

    # ---- DataLoaders ----
    train_loader, val_loader, test_loader = build_loaders(
        train_items, val_items, test_items, args.batch, args.workers
    )

    # ---- Device ----
    import torch

    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        if args.device not in ("cpu",):
            _info(f"Device '{args.device}' not available — falling back to CPU")
        device = torch.device("cpu")
    _ok(f"Using device: {device}")

    # ---- Model ----
    _banner("Building Model")
    model = build_model(num_classes, model_variant=args.model)
    model = model.to(device)

    # ---- Class weights ----
    class_weights = compute_class_weights(train_items, num_classes)

    # ---- Phase 1: train head only ----
    _banner(f"Phase 1 — Train Classifier Head ({args.epochs_phase1} epochs, LR=1e-3)")
    freeze_backbone(model)

    best_val_acc = 0.0
    best_state: Optional[dict] = None

    try:
        best_val_acc, best_state = train_phase(
            phase=1,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs_phase1,
            lr=1e-3,
            weight_decay=1e-4,
            device=device,
            class_weights=class_weights,
            patience=10,
            best_val_acc=best_val_acc,
            best_state=best_state,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            num_classes=num_classes,
        )
        _ok(f"Phase 1 complete. Best val accuracy: {best_val_acc:.2%}")
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Phase 1 interrupted.", file=sys.stderr)
        if best_state is not None:
            _info("Saving best checkpoint so far...")
            save_model(best_state)
        sys.exit(1)

    # ---- Phase 2: fine-tune all layers ----
    _banner(f"Phase 2 — Fine-tune All Layers ({args.epochs_phase2} epochs, LR=1e-4, cosine)")
    unfreeze_all(model)

    try:
        best_val_acc, best_state = train_phase(
            phase=2,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs_phase2,
            lr=1e-4,
            weight_decay=1e-4,
            device=device,
            class_weights=class_weights,
            patience=10,
            best_val_acc=best_val_acc,
            best_state=best_state,
            label_smoothing=args.label_smoothing,
            mixup_alpha=args.mixup_alpha,
            num_classes=num_classes,
        )
        _ok(f"Phase 2 complete. Best val accuracy: {best_val_acc:.2%}")
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Phase 2 interrupted.", file=sys.stderr)
        if best_state is not None:
            _info("Saving best checkpoint so far...")
            save_model(best_state)
        sys.exit(1)

    # ---- Load best weights for evaluation ----
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    # ---- Test set evaluation ----
    test_metrics = evaluate_test_set(model, test_loader, device, num_classes, present_classes)
    test_metrics["best_val_accuracy"] = round(best_val_acc, 4)

    # ---- Save outputs ----
    _banner("Saving Outputs")
    if best_state is None:
        _fail("No model checkpoint was saved (training produced no improvement). Exiting.")
        sys.exit(1)

    save_model(best_state)
    write_class_names()
    update_metadata(
        metrics=test_metrics,
        train_size=len(train_items),
        val_size=len(val_items),
        test_size=len(test_items),
        epochs_phase1=args.epochs_phase1,
        epochs_phase2=args.epochs_phase2,
    )

    _banner("Done")
    _ok(f"Model   -> {MODELS_DIR / 'classifier_v1.pt'}")
    _ok(f"Classes -> {MODELS_DIR / 'class_names_b.json'}")
    _ok(f"Meta    -> {MODELS_DIR / 'metadata.json'}")
    _ok(f"Final test accuracy: {test_metrics['overall_accuracy']:.2%}")
    print()


if __name__ == "__main__":
    main()
