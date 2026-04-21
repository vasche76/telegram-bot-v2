#!/usr/bin/env python3
"""
validate_class_coverage.py — Dataset coverage validator for Stage B classifier.

Reads class_names_b.json and checks image counts per class folder in stage_b/.
Classifies each class as OK / WEAK / EMPTY relative to MIN_TRAIN_IMAGES.

Exit code: 0 if all classes OK, 1 if any WEAK or EMPTY classes detected.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_B_DIR = REPO_ROOT / "data" / "fish_dataset" / "stage_b"
CLASS_NAMES_B_PATH = REPO_ROOT / "data" / "fish_models" / "class_names_b.json"

MIN_TRAIN_IMAGES = 15
SOFT_WARN_IMAGES = 50    # classes below this threshold get a soft underfit warning
SPECIAL_CLASSES = {"unknown_fish"}   # excluded from EMPTY/WEAK/SOFT checks
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def validate_coverage(
    class_names_path: Path = CLASS_NAMES_B_PATH,
    dataset_dir: Path = STAGE_B_DIR,
    min_images: int = MIN_TRAIN_IMAGES,
    verbose: bool = True,
) -> dict:
    """
    Validate dataset coverage per class.

    Returns dict with keys:
        ok:        list[str]        — class names with >= min_images
        soft_warn: list[tuple]      — (class_name, count) with min_images <= count < SOFT_WARN_IMAGES
        weak:      list[tuple]      — (class_name, count) with 0 < count < min_images
        empty:     list[str]        — class names with 0 images (folder missing or empty)
        counts:    dict[str, int]   — image count per class name

    Note: classes in SPECIAL_CLASSES (e.g. unknown_fish) are excluded from
    EMPTY/WEAK/SOFT checks — they are always treated as OK.
    """
    if not class_names_path.exists():
        raise FileNotFoundError(f"class_names_b.json not found: {class_names_path}")

    with open(class_names_path, encoding="utf-8") as f:
        mapping: dict[str, str] = json.load(f)

    class_names = [mapping[k] for k in sorted(mapping, key=lambda x: int(x))]

    ok_classes: list[str] = []
    soft_warn_classes: list[tuple[str, int]] = []
    weak_classes: list[tuple[str, int]] = []
    empty_classes: list[str] = []
    counts: dict[str, int] = {}

    for name in class_names:
        n = _count_images(dataset_dir / name)
        counts[name] = n
        # Special classes (e.g. unknown_fish) are excluded from all coverage checks
        if name in SPECIAL_CLASSES:
            ok_classes.append(name)
            continue
        if n == 0:
            empty_classes.append(name)
        elif n < min_images:
            weak_classes.append((name, n))
        elif n < SOFT_WARN_IMAGES:
            soft_warn_classes.append((name, n))
        else:
            ok_classes.append(name)

    if verbose:
        print(f"\nDataset coverage check  (threshold: {min_images} images/class, soft warn: {SOFT_WARN_IMAGES})")
        print(f"{'Class':<20} {'Images':>7}  {'Status'}")
        print("-" * 40)
        for name in class_names:
            n = counts[name]
            if name in SPECIAL_CLASSES:
                status = "OK (special)"
            elif n == 0:
                status = "EMPTY"
            elif n < min_images:
                status = f"WEAK  ({n}/{min_images})"
            elif n < SOFT_WARN_IMAGES:
                status = f"WARN  ({n}/{SOFT_WARN_IMAGES} soft min)"
            else:
                status = "OK"
            print(f"{name:<20} {n:>7}  {status}")

        print()
        print(f"OK     : {len(ok_classes)} classes")
        if soft_warn_classes:
            print(f"WARN   : {len(soft_warn_classes)} classes (underfit risk) — " +
                  ", ".join(f"{n}({c} imgs)" for n, c in soft_warn_classes))
        else:
            print(f"WARN   : 0 classes")
        if weak_classes:
            print(f"WEAK   : {len(weak_classes)} classes — " +
                  ", ".join(f"{n}({c} imgs)" for n, c in weak_classes))
        else:
            print(f"WEAK   : 0 classes")
        if empty_classes:
            print(f"EMPTY  : {len(empty_classes)} classes — {', '.join(empty_classes)}")
        else:
            print(f"EMPTY  : 0 classes")

    return {
        "ok": ok_classes,
        "soft_warn": soft_warn_classes,
        "weak": weak_classes,
        "empty": empty_classes,
        "counts": counts,
    }


def get_inactive_classes(
    class_names_path: Path = CLASS_NAMES_B_PATH,
    dataset_dir: Path = STAGE_B_DIR,
    min_images: int = MIN_TRAIN_IMAGES,
) -> set[str]:
    """Return set of class names with < min_images — unsafe for training or inference."""
    result = validate_coverage(class_names_path, dataset_dir, min_images, verbose=False)
    inactive: set[str] = set(result["empty"])
    inactive.update(name for name, _ in result["weak"])
    return inactive


if __name__ == "__main__":
    try:
        result = validate_coverage()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    problems = len(result["empty"]) + len(result["weak"])
    soft_problems = len(result["soft_warn"])
    if problems > 0:
        print(f"\n{problems} class(es) need attention before training.", file=sys.stderr)
        sys.exit(1)

    if soft_problems > 0:
        print(f"\n{soft_problems} class(es) have underfit risk (< {SOFT_WARN_IMAGES} images).")

    print("\nAll required classes OK — dataset ready for training.")
    sys.exit(0)
