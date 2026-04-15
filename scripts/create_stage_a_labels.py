#!/usr/bin/env python3
"""
create_stage_a_labels.py — Generate synthetic YOLO labels for Stage A detector
                            from raw (unannotated) class-organized images.

WHY SYNTHETIC FULL-IMAGE LABELS?
---------------------------------
We have class-organized raw images (whole_fish/, lure/, no_fish/, etc.)
but no bounding box annotations yet.

Bootstrap approach:
  - Assign each image a "full-image" bounding box: cx=0.5, cy=0.5, w=1.0, h=1.0
  - This tells YOLO: "the dominant object occupying this photo is class X"
  - YOLOv8 trained this way learns to recognize the dominant class even with
    imprecise boxes, especially with mosaic augmentation
  - This is a standard approach for image-level classification bootstrapping

Limitations (honest):
  - The model will learn "this whole image contains X" rather than "object X
    is at THIS location"
  - For our use case (single dominant object in each fishing photo),
    this is acceptable for Stage A classification
  - True bounding boxes (from Roboflow annotation or CVAT) will give a
    stronger model when available
  - These synthetic labels are CLEARLY MARKED as bootstrap so they can be
    replaced when real annotations exist

Output structure:
    data/fish_dataset/stage_a/labeled/train/images/
    data/fish_dataset/stage_a/labeled/train/labels/
    data/fish_dataset/stage_a/labeled/val/images/
    data/fish_dataset/stage_a/labeled/val/labels/
    data/fish_dataset/stage_a/labeled/test/images/
    data/fish_dataset/stage_a/labeled/test/labels/

Usage:
    python3 scripts/create_stage_a_labels.py [--split 0.7 0.15 0.15] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
RAW_DIR = DATA_ROOT / "fish_dataset" / "stage_a" / "raw"
LABELED_DIR = DATA_ROOT / "fish_dataset" / "stage_a" / "labeled"
MODELS_DIR = DATA_ROOT / "fish_models"

# YOLO class ID → class name (must match train_stage_a.py)
CLASS_NAMES = {
    0: "whole_fish",
    1: "lure",
    2: "fish_part",
    3: "fry",
    4: "no_fish",
}
NAME_TO_ID = {v: k for k, v in CLASS_NAMES.items()}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Synthetic bounding box: full image (cx, cy, w, h in YOLO normalized format)
FULL_IMAGE_BBOX = "0.5 0.5 1.0 1.0"

MIN_PER_CLASS = 5  # Warn if fewer than this (don't abort)
SPLIT_SEED = 42


def _banner(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def _ok(msg: str) -> None:
    print(f"[OK]  {msg}")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def _images_in(folder: Path) -> list[Path]:
    """Return sorted list of image files in folder (non-recursive)."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _count_images(folder: Path) -> int:
    return len(_images_in(folder))


# ─── Collect and split ─────────────────────────────────────────────────────

def collect_raw_images() -> dict[str, list[Path]]:
    """
    Scan raw/ for each class directory and collect image paths.
    Returns dict: class_name → [image paths]
    """
    result: dict[str, list[Path]] = {}

    for class_name in CLASS_NAMES.values():
        class_dir = RAW_DIR / class_name
        imgs = _images_in(class_dir)
        result[class_name] = imgs
        status = "OK" if len(imgs) >= MIN_PER_CLASS else f"WARN ({len(imgs)} < {MIN_PER_CLASS} recommended)"
        _info(f"  {class_name:<12}: {len(imgs):>4} raw images  [{status}]")

    return result


def split_images(
    class_images: dict[str, list[Path]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, dict[str, list[tuple[Path, str]]]]:
    """
    Stratified split per class.
    Returns: split_name → {class_name → [(image_path, class_name)]}
    Actually returns: split_name → list[(path, class_name)]
    """
    rng = random.Random(seed)
    splits: dict[str, list[tuple[Path, str]]] = {"train": [], "val": [], "test": []}

    _banner("Dataset Split")
    print(f"  {'Class':<12} {'Total':>6} {'Train':>6} {'Val':>5} {'Test':>5}")
    print(f"  {'-' * 37}")

    for class_name, imgs in sorted(class_images.items()):
        if not imgs:
            _info(f"  {class_name}: no images, skipping")
            continue

        shuffled = imgs.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)

        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = n - n_train - n_val

        if n_test < 0:
            n_val = max(0, n - n_train)
            n_test = 0

        train_imgs = shuffled[:n_train]
        val_imgs = shuffled[n_train:n_train + n_val]
        test_imgs = shuffled[n_train + n_val:]

        splits["train"].extend((p, class_name) for p in train_imgs)
        splits["val"].extend((p, class_name) for p in val_imgs)
        splits["test"].extend((p, class_name) for p in test_imgs)

        print(f"  {class_name:<12} {n:>6} {len(train_imgs):>6} {len(val_imgs):>5} {len(test_imgs):>5}")

    print()
    for split_name, items in splits.items():
        _ok(f"{split_name}: {len(items)} images")

    return splits


# ─── Write labeled dataset ─────────────────────────────────────────────────

def write_labeled_split(
    split_name: str,
    items: list[tuple[Path, str]],
    labeled_dir: Path,
    overwrite: bool = False,
    label_type: str = "bootstrap",
) -> int:
    """
    Copy images and write YOLO label files to labeled/split_name/images|labels/.

    label_type: "bootstrap" = full-image box; "skip" = write empty label (negative)
    Returns count of images written.
    """
    images_dir = labeled_dir / split_name / "images"
    labels_dir = labeled_dir / split_name / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for src_path, class_name in items:
        cls_id = NAME_TO_ID.get(class_name)
        if cls_id is None:
            _warn(f"  Unknown class name '{class_name}' for {src_path.name}, skipping")
            continue

        dest_img = images_dir / src_path.name
        dest_lbl = labels_dir / (src_path.stem + ".txt")

        # Skip if already present (avoid re-copying on re-runs)
        if dest_img.exists() and dest_lbl.exists() and not overwrite:
            continue

        try:
            shutil.copy2(src_path, dest_img)
        except Exception as exc:
            _warn(f"  Could not copy {src_path}: {exc}")
            continue

        # Write YOLO label
        if class_name == "no_fish":
            # no_fish = background — write empty label file (standard YOLO practice)
            label_content = ""
        else:
            # Synthetic full-image bounding box
            # Format: class_id cx cy w h (all normalized 0-1)
            # No comment lines — YOLO validator rejects them
            label_content = f"{cls_id} {FULL_IMAGE_BBOX}\n"

        dest_lbl.write_text(label_content, encoding="utf-8")
        written += 1

    return written


# ─── Metadata ──────────────────────────────────────────────────────────────

def write_bootstrap_metadata(
    class_images: dict[str, list[Path]],
    splits: dict[str, list[tuple[Path, str]]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    """Write metadata about the bootstrap labeling to fish_models/metadata.json."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = MODELS_DIR / "metadata.json"

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    else:
        meta = {}

    from datetime import datetime, timezone

    meta.setdefault("detector", {})
    meta["detector"]["stage_a_labeling"] = {
        "type": "bootstrap_full_image_boxes",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split_seed": seed,
        "split_ratios": {"train": train_ratio, "val": val_ratio, "test": round(1 - train_ratio - val_ratio, 2)},
        "note": (
            "Synthetic full-image YOLO labels. Each image assigned class_id with "
            "bbox=whole_image. Suitable for bootstrap training of image-level "
            "classifier. Replace with real bounding boxes from Roboflow/CVAT for "
            "proper object detection performance."
        ),
        "class_counts_raw": {
            name: len(imgs) for name, imgs in class_images.items()
        },
        "split_counts": {
            split: len(items) for split, items in splits.items()
        },
    }

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    _ok(f"Wrote metadata to {meta_path}")


# ─── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate synthetic YOLO labels for Stage A detector from raw class-organized images.\n"
            "Uses full-image bounding boxes as a bootstrap approach."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--train-ratio", type=float, default=0.70, help="Train split ratio")
    p.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    p.add_argument("--seed", type=int, default=SPLIT_SEED, help="Random seed for reproducibility")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing labeled files")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    test_ratio = round(1.0 - args.train_ratio - args.val_ratio, 4)
    if test_ratio < 0:
        _fail("train_ratio + val_ratio must be <= 1.0")
        sys.exit(1)

    _banner("Stage A — Bootstrap YOLO Label Generator")
    _info(f"Raw directory : {RAW_DIR}")
    _info(f"Labeled output: {LABELED_DIR}")
    _info(f"Split ratios  : train={args.train_ratio}, val={args.val_ratio}, test={test_ratio}")
    _info(f"Seed          : {args.seed}")
    _info("")
    _info("NOTE: This creates SYNTHETIC full-image bounding boxes.")
    _info("These are a bootstrap approach — replace with real annotations")
    _info("from Roboflow/CVAT for better bounding box precision.")

    # ── Collect raw images ──
    _banner("Scanning Raw Images")
    class_images = collect_raw_images()

    total_raw = sum(len(imgs) for imgs in class_images.values())
    if total_raw == 0:
        _fail("No raw images found in any class directory.")
        _fail(f"Run fetch_inaturalist.py and fetch_wikimedia_lures.py first.")
        sys.exit(1)

    nonempty = {k: v for k, v in class_images.items() if v}
    _ok(f"Found {total_raw} total raw images across {len(nonempty)} non-empty classes")

    # ── Split ──
    splits = split_images(class_images, args.train_ratio, args.val_ratio, args.seed)

    # ── Write labeled splits ──
    _banner("Writing Labeled Dataset (YOLO format)")
    total_written = 0
    for split_name, items in splits.items():
        if not items:
            _info(f"  {split_name}: no items, skipping")
            continue
        n = write_labeled_split(split_name, items, LABELED_DIR, overwrite=args.overwrite)
        total_written += n
        _ok(f"  {split_name}: {n} image+label pairs written")

    # ── Metadata ──
    write_bootstrap_metadata(class_images, splits, args.train_ratio, args.val_ratio, args.seed)

    # ── Summary ──
    _banner("Complete")
    _ok(f"Total image+label pairs written: {total_written}")

    for split_name in ["train", "val", "test"]:
        n_imgs = _count_images(LABELED_DIR / split_name / "images")
        print(f"  {split_name:<6}: {n_imgs} images in {LABELED_DIR / split_name / 'images'}")

    print()
    _info("IMPORTANT: These are bootstrap labels with full-image bounding boxes.")
    _info("The Stage A model will be trained as an image-level classifier.")
    _info("For a production-quality object detector, annotate real bounding boxes in Roboflow.")
    print()
    _info("Next steps:")
    _info("  python3 scripts/validate_dataset.py")
    _info("  python3 scripts/train_stage_a.py --epochs 50 --device cpu")
    print()


if __name__ == "__main__":
    main()
