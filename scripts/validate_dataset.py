#!/usr/bin/env python3
"""
validate_dataset.py — Dataset validation gate for the two-stage fish recognition system.

Usage:
    python3 scripts/validate_dataset.py [--help]

Exit codes:
    0 — Dataset passes all minimums and labeled data is valid (ready for training)
    1 — One or more errors found (minimums not met or YOLO format invalid)
"""

import sys
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_A_RAW = REPO_ROOT / "data" / "fish_dataset" / "stage_a" / "raw"
STAGE_A_LABELED = REPO_ROOT / "data" / "fish_dataset" / "stage_a" / "labeled"
STAGE_B = REPO_ROOT / "data" / "fish_dataset" / "stage_b"

STAGE_A_CLASSES = ["whole_fish", "lure", "fish_part", "fry", "no_fish"]
STAGE_A_CLASS_IDS = {i: name for i, name in enumerate(STAGE_A_CLASSES)}
VALID_CLASS_IDS = set(range(len(STAGE_A_CLASSES)))

STAGE_B_SPECIES = ["pike", "taimen", "grayling", "whitefish", "perch", "brown_trout", "rainbow_trout", "atlantic_salmon", "common_carp", "crucian_carp", "bream", "roach", "ide", "wels_catfish", "unknown_fish"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

STAGE_A_WARN_THRESHOLD = 50
STAGE_A_ERROR_THRESHOLD = 20
STAGE_B_WARN_THRESHOLD = 30
STAGE_B_ERROR_THRESHOLD = 15

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_images(directory: Path) -> int:
    """Return count of image files in a directory (non-recursive)."""
    if not directory.exists():
        return 0
    return sum(
        1 for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def count_images_recursive(directory: Path) -> int:
    """Return count of image files in a directory tree (recursive)."""
    if not directory.exists():
        return 0
    return sum(
        1 for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def separator(char: str = "-", width: int = 70) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Stage A — raw directory check
# ---------------------------------------------------------------------------


def check_stage_a_raw() -> tuple[dict[str, int], list[str], list[str]]:
    """
    Returns (counts_dict, warnings, errors).
    counts_dict: class_name -> image count
    """
    counts: dict[str, int] = {}
    warnings: list[str] = []
    errors: list[str] = []

    for cls in STAGE_A_CLASSES:
        cls_dir = STAGE_A_RAW / cls
        n = count_images(cls_dir)
        counts[cls] = n

        if n < STAGE_A_ERROR_THRESHOLD:
            errors.append(
                f"  Stage A raw / {cls}: {n} images — MINIMUM {STAGE_A_ERROR_THRESHOLD} required (ERROR)"
            )
        elif n < STAGE_A_WARN_THRESHOLD:
            warnings.append(
                f"  Stage A raw / {cls}: {n} images — recommended {STAGE_A_WARN_THRESHOLD}+ (WARNING)"
            )

    return counts, warnings, errors


# ---------------------------------------------------------------------------
# Stage A — labeled YOLO directory check
# ---------------------------------------------------------------------------


def validate_yolo_label_file(label_path: Path) -> list[str]:
    """
    Validate a single YOLO .txt label file.
    Returns a list of error strings (empty means valid).
    """
    errs: list[str] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        return [f"    Cannot read {label_path.name}: {exc}"]

    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue  # blank lines are tolerated
        parts = line.split()
        if len(parts) != 5:
            errs.append(
                f"    {label_path.name}:{lineno} — expected 5 values, got {len(parts)}: '{line}'"
            )
            continue
        try:
            cls_id = int(parts[0])
            values = [float(p) for p in parts[1:]]
        except ValueError:
            errs.append(
                f"    {label_path.name}:{lineno} — non-numeric value: '{line}'"
            )
            continue

        if cls_id not in VALID_CLASS_IDS:
            errs.append(
                f"    {label_path.name}:{lineno} — invalid class_id {cls_id} "
                f"(valid: 0-{len(STAGE_A_CLASSES)-1})"
            )
        for idx, val in enumerate(values):
            coord_names = ["cx", "cy", "w", "h"]
            if not (0.0 <= val <= 1.0):
                errs.append(
                    f"    {label_path.name}:{lineno} — {coord_names[idx]}={val} "
                    f"out of range [0, 1]"
                )
    return errs


def check_stage_a_labeled() -> tuple[dict[str, int], list[str], list[str]]:
    """
    Validate labeled YOLO data for train/val/test splits.
    Returns (split_counts, warnings, errors).
    split_counts: split_name -> image count
    """
    split_counts: dict[str, int] = {}
    warnings: list[str] = []
    errors: list[str] = []

    if not STAGE_A_LABELED.exists():
        warnings.append("  Stage A labeled/ directory does not exist — skipping YOLO validation")
        return split_counts, warnings, errors

    splits = ["train", "val", "test"]
    for split in splits:
        images_dir = STAGE_A_LABELED / split / "images"
        labels_dir = STAGE_A_LABELED / split / "labels"

        n_images = count_images(images_dir)
        split_counts[split] = n_images

        if n_images == 0:
            # Empty split — not necessarily an error (user might not have val/test yet)
            warnings.append(f"  Stage A labeled/{split}/images/ is empty")
            continue

        if not labels_dir.exists():
            errors.append(f"  Stage A labeled/{split}/labels/ directory missing")
            continue

        # Check every image has a matching label file
        image_files = [
            f for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
        missing_labels: list[str] = []
        label_errors: list[str] = []

        for img_file in image_files:
            label_file = labels_dir / (img_file.stem + ".txt")
            if not label_file.exists():
                missing_labels.append(img_file.name)
            else:
                errs = validate_yolo_label_file(label_file)
                label_errors.extend(errs)

        if missing_labels:
            errors.append(
                f"  Stage A labeled/{split}: {len(missing_labels)} image(s) have no label file:"
            )
            for name in missing_labels[:10]:
                errors.append(f"    - {name}")
            if len(missing_labels) > 10:
                errors.append(f"    ... and {len(missing_labels) - 10} more")

        if label_errors:
            errors.append(f"  Stage A labeled/{split}: YOLO format errors found:")
            errors.extend(label_errors[:20])
            if len(label_errors) > 20:
                errors.append(f"    ... and {len(label_errors) - 20} more errors")

    return split_counts, warnings, errors


# ---------------------------------------------------------------------------
# Stage A — class coverage check (bootstrap label audit)
# ---------------------------------------------------------------------------


def check_stage_a_class_coverage() -> tuple[dict[int, int], list[str], list[str]]:
    """
    Reads all .txt label files across train/val/test labeled splits and counts
    annotations per class_id (0-4).

    class_id 4 (no_fish) having 0 annotations is expected — treated as WARNING.
    Any other class_id (0-3) with 0 annotations is an ERROR.

    Returns (class_counts, warnings, errors).
    class_counts: class_id -> total annotation count across all splits
    """
    class_counts: dict[int, int] = {i: 0 for i in range(len(STAGE_A_CLASSES))}
    warnings: list[str] = []
    errors: list[str] = []

    splits = ["train", "val", "test"]
    for split in splits:
        labels_dir = STAGE_A_LABELED / split / "labels"
        if not labels_dir.exists():
            continue
        for label_file in labels_dir.glob("*.txt"):
            try:
                lines = label_file.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                try:
                    cls_id = int(parts[0])
                except ValueError:
                    continue
                if cls_id in class_counts:
                    class_counts[cls_id] += 1

    for cls_id, count in class_counts.items():
        cls_name = STAGE_A_CLASS_IDS.get(cls_id, f"class_{cls_id}")
        if count == 0:
            if cls_id == 4:  # no_fish — background images have empty label files
                warnings.append(
                    f"Stage A train: class_id {cls_id} ({cls_name}) has 0 annotations in labels"
                )
            else:
                errors.append(
                    f"Stage A train: class_id {cls_id} ({cls_name}) has 0 annotations in labels"
                )

    return class_counts, warnings, errors


# ---------------------------------------------------------------------------
# Stage B — species directory check
# ---------------------------------------------------------------------------


def check_stage_b() -> tuple[dict[str, int], list[str], list[str]]:
    """
    Returns (counts_dict, warnings, errors).
    """
    counts: dict[str, int] = {}
    warnings: list[str] = []
    errors: list[str] = []

    for species in STAGE_B_SPECIES:
        species_dir = STAGE_B / species
        n = count_images(species_dir)
        counts[species] = n

        # unknown_fish is a catch-all — no minimum enforced
        if species == "unknown_fish":
            continue

        if n < STAGE_B_ERROR_THRESHOLD:
            errors.append(
                f"  Stage B / {species}: {n} images — MINIMUM {STAGE_B_ERROR_THRESHOLD} required (ERROR)"
            )
        elif n < STAGE_B_WARN_THRESHOLD:
            warnings.append(
                f"  Stage B / {species}: {n} images — recommended {STAGE_B_WARN_THRESHOLD}+ (WARNING)"
            )

    return counts, warnings, errors


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_table_row(label: str, count: int, warn: int, error: int, col_width: int = 20) -> None:
    if count == 0:
        status = "EMPTY"
    elif count < error:
        status = f"ERROR (<{error})"
    elif count < warn:
        status = f"WARN (<{warn})"
    else:
        status = "OK"
    print(f"  {label:<{col_width}} {count:>6}    {status}")


def print_report(
    raw_counts: dict[str, int],
    labeled_counts: dict[str, int],
    b_counts: dict[str, int],
) -> None:
    w = 70
    print()
    print(separator("=", w))
    print(" DATASET VALIDATION REPORT".center(w))
    print(separator("=", w))

    # Stage A raw
    print()
    print(f"  STAGE A — Raw images  ({STAGE_A_RAW})")
    print(separator("-", w))
    print(f"  {'Class':<20} {'Count':>6}    {'Status'}")
    print(separator("-", w))
    for cls in STAGE_A_CLASSES:
        print_table_row(cls, raw_counts.get(cls, 0), STAGE_A_WARN_THRESHOLD, STAGE_A_ERROR_THRESHOLD)
    total_raw = sum(raw_counts.values())
    print(separator("-", w))
    print(f"  {'TOTAL':<20} {total_raw:>6}")

    # Stage A labeled
    print()
    print(f"  STAGE A — Labeled (YOLO)  ({STAGE_A_LABELED})")
    print(separator("-", w))
    print(f"  {'Split':<20} {'Images':>6}")
    print(separator("-", w))
    for split in ["train", "val", "test"]:
        n = labeled_counts.get(split, 0)
        print(f"  {split:<20} {n:>6}")
    total_labeled = sum(labeled_counts.values())
    print(separator("-", w))
    print(f"  {'TOTAL':<20} {total_labeled:>6}")

    # Stage B
    print()
    print(f"  STAGE B — Species classification  ({STAGE_B})")
    print(separator("-", w))
    print(f"  {'Species':<20} {'Count':>6}    {'Status'}")
    print(separator("-", w))
    for species in STAGE_B_SPECIES:
        if species == "unknown_fish":
            n = b_counts.get(species, 0)
            print(f"  {species:<20} {n:>6}    (no minimum)")
        else:
            print_table_row(species, b_counts.get(species, 0), STAGE_B_WARN_THRESHOLD, STAGE_B_ERROR_THRESHOLD)
    total_b = sum(b_counts.values())
    print(separator("-", w))
    print(f"  {'TOTAL':<20} {total_b:>6}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the fish dataset for the two-stage recognition pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    args = parser.parse_args()
    _ = args  # no extra flags yet

    all_warnings: list[str] = []
    all_errors: list[str] = []

    # Run checks
    raw_counts, raw_warnings, raw_errors = check_stage_a_raw()
    all_warnings.extend(raw_warnings)
    all_errors.extend(raw_errors)

    labeled_counts, labeled_warnings, labeled_errors = check_stage_a_labeled()
    all_warnings.extend(labeled_warnings)
    all_errors.extend(labeled_errors)

    _class_counts, coverage_warnings, coverage_errors = check_stage_a_class_coverage()
    all_warnings.extend(coverage_warnings)
    all_errors.extend(coverage_errors)

    b_counts, b_warnings, b_errors = check_stage_b()
    all_warnings.extend(b_warnings)
    all_errors.extend(b_errors)

    # Print structured report
    print_report(raw_counts, labeled_counts, b_counts)

    w = 70
    # Print warnings
    if all_warnings:
        print(separator("-", w))
        print(f"  WARNINGS ({len(all_warnings)}):")
        for msg in all_warnings:
            print(f"  [WARN]  {msg.strip()}")
        print()

    # Print errors
    if all_errors:
        print(separator("=", w))
        print(f"  ERRORS ({len(all_errors)}) — training NOT ready:")
        print(separator("=", w))
        for msg in all_errors:
            print(f"  [ERROR] {msg.strip()}")
        print()
        print("  Result: FAIL — fix the errors above before training.")
        print(separator("=", w))
        return 1
    else:
        print(separator("=", w))
        if all_warnings:
            print("  Result: PASS (with warnings) — dataset meets minimums.")
        else:
            print("  Result: PASS — dataset is ready for training.")
        print(separator("=", w))
        return 0


if __name__ == "__main__":
    sys.exit(main())
