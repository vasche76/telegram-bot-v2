#!/usr/bin/env python3
"""
augment_dataset.py — Augment small Stage A classes to reach training thresholds.

Problem: Some Stage A classes (lure, fish_part, fry) have very few images
because automated downloads are limited. This script synthetically augments
them using PIL transforms until each class reaches a target count.

Augmentations applied (PIL-only, no extra dependencies):
  1. Horizontal flip
  2. Vertical flip (low probability)
  3. Rotation: +15°, +30°, -15°, -30°
  4. Brightness jitter (darker, lighter)
  5. Contrast jitter
  6. Combined: flip + rotate + brightness

Each source image generates up to 8 augmented variants.
Augmented files are written alongside originals with an _aug_N suffix.
Provenance is tracked in AUGMENTATION.json.

Usage:
    python3 scripts/augment_dataset.py [--target 60] [--classes lure,fish_part,fry]

    # Augment only lure class to 80 images minimum:
    python3 scripts/augment_dataset.py --target 80 --classes lure

    # Dry run to see what would be created:
    python3 scripts/augment_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
STAGE_A_RAW = DATA_ROOT / "fish_dataset" / "stage_a" / "raw"

# Default classes to augment (only small Stage A classes — Stage B has enough from iNat)
DEFAULT_CLASSES = ["lure", "fish_part", "fry"]

# Default target count per class
DEFAULT_TARGET = 60

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


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


def _count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def _source_images(directory: Path) -> list[Path]:
    """Return only NON-augmented source images (exclude _aug_ files)."""
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.suffix.lower() in IMAGE_EXTS and "_aug_" not in p.stem
    )


# ─── Augmentation transform definitions ──────────────────────────────────────

def _augmentation_transforms() -> list[tuple[str, Callable]]:
    """
    Return list of (name, transform_fn) pairs.
    Each transform takes a PIL Image and returns a PIL Image.
    """
    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except ImportError:
        print("[FAIL] Pillow is not installed. Run: pip install Pillow", file=sys.stderr)
        sys.exit(1)

    transforms = [
        ("hflip", lambda img: img.transpose(Image.FLIP_LEFT_RIGHT)),
        ("rot15", lambda img: img.rotate(15, expand=True, fillcolor=(128, 128, 128))),
        ("rot_15", lambda img: img.rotate(-15, expand=True, fillcolor=(128, 128, 128))),
        ("rot30", lambda img: img.rotate(30, expand=True, fillcolor=(128, 128, 128))),
        ("bright_up", lambda img: ImageEnhance.Brightness(img).enhance(1.4)),
        ("bright_dn", lambda img: ImageEnhance.Brightness(img).enhance(0.65)),
        ("contrast_up", lambda img: ImageEnhance.Contrast(img).enhance(1.5)),
        (
            "hflip_rot15_bright",
            lambda img: ImageEnhance.Brightness(
                img.transpose(Image.FLIP_LEFT_RIGHT).rotate(15, expand=True, fillcolor=(128, 128, 128))
            ).enhance(1.2),
        ),
    ]
    return transforms


# ─── Main augmentation logic ──────────────────────────────────────────────────

def augment_class(
    class_dir: Path,
    target: int,
    dry_run: bool = False,
) -> int:
    """
    Augment images in class_dir until total count reaches target.

    Returns number of new augmented images created.
    """
    class_name = class_dir.name
    source_imgs = _source_images(class_dir)
    current_total = _count_images(class_dir)

    if current_total >= target:
        _info(f"  {class_name}: already {current_total} images (>= target {target}), skipping")
        return 0

    if not source_imgs:
        _warn(f"  {class_name}: no source images — cannot augment empty directory")
        return 0

    needed = target - current_total
    _info(f"  {class_name}: {current_total} images, need {needed} more to reach {target}")

    try:
        from PIL import Image
    except ImportError:
        print("[FAIL] Pillow is not installed. Run: pip install Pillow", file=sys.stderr)
        sys.exit(1)

    transforms = _augmentation_transforms()
    created = 0

    # Track what we've already augmented
    aug_log: dict[str, list[str]] = {}

    prov_path = class_dir / "AUGMENTATION.json"
    if prov_path.exists():
        try:
            existing = json.loads(prov_path.read_text(encoding="utf-8"))
            aug_log = existing.get("augmented", {})
        except Exception:
            aug_log = {}

    # Cycle through source images with augmentation transforms
    # until we reach the target
    cycle_idx = 0
    max_attempts = len(source_imgs) * len(transforms) * 3  # safety cap

    while created < needed and cycle_idx < max_attempts:
        src_img = source_imgs[cycle_idx % len(source_imgs)]
        transform_idx = (cycle_idx // len(source_imgs)) % len(transforms)
        cycle_count = (cycle_idx // (len(source_imgs) * len(transforms)))

        transform_name, transform_fn = transforms[transform_idx]
        aug_suffix = f"_aug_{transform_name}"
        if cycle_count > 0:
            aug_suffix += f"_v{cycle_count + 1}"

        aug_filename = src_img.stem + aug_suffix + src_img.suffix
        aug_path = class_dir / aug_filename

        cycle_idx += 1

        if aug_path.exists():
            continue  # already created

        if dry_run:
            _info(f"    [DRY-RUN] Would create: {aug_filename}")
            created += 1
            continue

        try:
            with Image.open(src_img) as img:
                img_rgb = img.convert("RGB")
                aug_img = transform_fn(img_rgb)
                aug_img.save(aug_path, quality=92)

            aug_log.setdefault(src_img.name, []).append(aug_filename)
            created += 1

            if created % 10 == 0:
                _info(f"    Created {created}/{needed} augmented images...")

        except Exception as exc:
            _warn(f"    Augmentation failed for {src_img.name}: {exc}")

    if not dry_run:
        prov_data = {
            "source": "augment_dataset.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target": target,
            "augmented": aug_log,
        }
        prov_path.write_text(json.dumps(prov_data, indent=2, ensure_ascii=False), encoding="utf-8")

    new_total = _count_images(class_dir)
    _ok(f"  {class_name}: created {created} augmented images, total now {new_total}")
    return created


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Augment small Stage A classes to reach training thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated Stage A class names to augment",
    )
    p.add_argument(
        "--target",
        type=int,
        default=DEFAULT_TARGET,
        help="Target image count per class (augmentation stops when reached)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be created without writing files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if not classes:
        print("[FAIL] --classes is empty", file=sys.stderr)
        sys.exit(1)

    _banner(f"Dataset Augmentation (target={args.target} per class)")
    _info(f"  Classes  : {classes}")
    _info(f"  Target   : {args.target} images per class")
    if args.dry_run:
        _info("  DRY-RUN mode — no files will be written")

    total_created = 0

    for class_name in classes:
        class_dir = STAGE_A_RAW / class_name
        if not class_dir.exists():
            _warn(f"  Class directory not found: {class_dir} — skipping")
            continue

        _banner(f"Augmenting: {class_name}")
        n = augment_class(class_dir, args.target, dry_run=args.dry_run)
        total_created += n

    _banner("Summary")
    _ok(f"Total augmented images created: {total_created}")

    print()
    print("  Current counts (after augmentation):")
    for class_name in classes:
        class_dir = STAGE_A_RAW / class_name
        n = _count_images(class_dir)
        status = "OK" if n >= 20 else f"WARN ({n} < 20 minimum)"
        print(f"    {class_name:<12}: {n:>4} images  [{status}]")

    print()
    _info("Next: run scripts/create_stage_a_labels.py --overwrite")
    _info("      then  scripts/train_stage_a.py")


if __name__ == "__main__":
    main()
