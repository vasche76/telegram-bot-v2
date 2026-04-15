#!/usr/bin/env python3
"""
dataset_stats.py — Dataset statistics for the two-stage fish recognition system.

Prints:
  - Image counts per class (stage_a raw + labeled, stage_b)
  - Class balance ratio (max/min)
  - Total disk usage per stage
  - Estimated training time at 100ms/image (rough guide)
  - ASCII bar chart of class sizes
  - Recommendations: which classes need more data

Usage:
    python3 scripts/dataset_stats.py [--help]
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
STAGE_B_SPECIES = ["pike", "taimen", "grayling", "whitefish", "perch", "brown_trout", "rainbow_trout", "atlantic_salmon", "common_carp", "crucian_carp", "bream", "roach", "ide", "wels_catfish", "unknown_fish"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

STAGE_A_WARN = 50
STAGE_A_MIN = 20
STAGE_B_WARN = 30
STAGE_B_MIN = 15

MS_PER_IMAGE = 100  # rough training time estimate

BAR_MAX_WIDTH = 40   # max bar length in chars

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_images(directory: Path) -> int:
    """Count image files (non-recursive) in a directory."""
    if not directory.exists():
        return 0
    return sum(
        1 for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def dir_size_bytes(directory: Path) -> int:
    """Return total size in bytes of all files under directory (recursive)."""
    if not directory.exists():
        return 0
    return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())


def human_size(n_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024 or unit == "TB":
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def human_time(milliseconds: int) -> str:
    """Format milliseconds as human-readable time string."""
    seconds = milliseconds // 1000
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    seconds %= 60
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours = minutes // 60
    minutes %= 60
    return f"{hours}h {minutes}m"


def separator(char: str = "-", width: int = 70) -> str:
    return char * width


def bar_chart_line(label: str, count: int, max_count: int, col_width: int = 20) -> str:
    """Render a single ASCII bar chart row."""
    bar_len = int((count / max_count) * BAR_MAX_WIDTH) if max_count > 0 else 0
    bar = "#" * bar_len
    return f"  {label:<{col_width}} {count:>5}  |{bar}"


# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------


def gather_stage_a_raw() -> dict[str, int]:
    return {cls: count_images(STAGE_A_RAW / cls) for cls in STAGE_A_CLASSES}


def gather_stage_a_labeled() -> dict[str, int]:
    """Count labeled images per split."""
    result: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        result[split] = count_images(STAGE_A_LABELED / split / "images")
    return result


def gather_stage_b() -> dict[str, int]:
    return {sp: count_images(STAGE_B / sp) for sp in STAGE_B_SPECIES}


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def print_stage_a_raw_section(counts: dict[str, int]) -> None:
    w = 70
    print(f"\n  STAGE A — Raw images per class  ({STAGE_A_RAW})")
    print(separator("-", w))
    print(f"  {'Class':<20} {'Count':>6}    {'Disk Usage':>12}    {'Status'}")
    print(separator("-", w))
    for cls in STAGE_A_CLASSES:
        n = counts[cls]
        size = dir_size_bytes(STAGE_A_RAW / cls)
        if n == 0:
            status = "EMPTY"
        elif n < STAGE_A_MIN:
            status = f"ERROR (<{STAGE_A_MIN})"
        elif n < STAGE_A_WARN:
            status = f"WARN (<{STAGE_A_WARN})"
        else:
            status = "OK"
        print(f"  {cls:<20} {n:>6}    {human_size(size):>12}    {status}")
    total_n = sum(counts.values())
    total_size = dir_size_bytes(STAGE_A_RAW)
    print(separator("-", w))
    print(f"  {'TOTAL':<20} {total_n:>6}    {human_size(total_size):>12}")


def print_stage_a_labeled_section(counts: dict[str, int]) -> None:
    w = 70
    total_labeled = sum(counts.values())
    print(f"\n  STAGE A — Labeled (YOLO) splits  ({STAGE_A_LABELED})")
    print(separator("-", w))
    print(f"  {'Split':<20} {'Images':>6}    {'Disk Usage':>12}")
    print(separator("-", w))
    for split in ["train", "val", "test"]:
        n = counts[split]
        size = dir_size_bytes(STAGE_A_LABELED / split)
        print(f"  {split:<20} {n:>6}    {human_size(size):>12}")
    total_size = dir_size_bytes(STAGE_A_LABELED)
    print(separator("-", w))
    print(f"  {'TOTAL':<20} {total_labeled:>6}    {human_size(total_size):>12}")


def print_stage_b_section(counts: dict[str, int]) -> None:
    w = 70
    print(f"\n  STAGE B — Species images  ({STAGE_B})")
    print(separator("-", w))
    print(f"  {'Species':<20} {'Count':>6}    {'Disk Usage':>12}    {'Status'}")
    print(separator("-", w))
    for sp in STAGE_B_SPECIES:
        n = counts[sp]
        size = dir_size_bytes(STAGE_B / sp)
        if sp == "unknown_fish":
            status = "(no minimum)"
        elif n == 0:
            status = "EMPTY"
        elif n < STAGE_B_MIN:
            status = f"ERROR (<{STAGE_B_MIN})"
        elif n < STAGE_B_WARN:
            status = f"WARN (<{STAGE_B_WARN})"
        else:
            status = "OK"
        print(f"  {sp:<20} {n:>6}    {human_size(size):>12}    {status}")
    total_n = sum(counts.values())
    total_size = dir_size_bytes(STAGE_B)
    print(separator("-", w))
    print(f"  {'TOTAL':<20} {total_n:>6}    {human_size(total_size):>12}")


def print_balance_section(
    raw_counts: dict[str, int],
    b_counts: dict[str, int],
) -> None:
    w = 70
    print(f"\n  CLASS BALANCE ANALYSIS")
    print(separator("-", w))

    # Stage A
    raw_vals = [v for v in raw_counts.values() if v > 0]
    if len(raw_vals) >= 2:
        ratio_a = max(raw_vals) / min(raw_vals)
        best_a = max(raw_counts, key=lambda k: raw_counts[k])
        worst_a = min(raw_counts, key=lambda k: raw_counts[k])
        print(f"  Stage A balance ratio (max/min): {ratio_a:.2f}x")
        print(f"    Most populated : {best_a} ({raw_counts[best_a]} images)")
        print(f"    Least populated: {worst_a} ({raw_counts[worst_a]} images)")
        if ratio_a > 3.0:
            print(f"    [!] Ratio > 3.0 — class imbalance may hurt training accuracy")
    elif len(raw_vals) == 1:
        print("  Stage A: only one class has data — add images to other classes")
    else:
        print("  Stage A: no images found")

    print()

    # Stage B (exclude unknown_fish from balance calc)
    b_vals = {k: v for k, v in b_counts.items() if k != "unknown_fish" and v > 0}
    if len(b_vals) >= 2:
        ratio_b = max(b_vals.values()) / min(b_vals.values())
        best_b = max(b_vals, key=lambda k: b_vals[k])
        worst_b = min(b_vals, key=lambda k: b_vals[k])
        print(f"  Stage B balance ratio (max/min): {ratio_b:.2f}x")
        print(f"    Most populated : {best_b} ({b_vals[best_b]} images)")
        print(f"    Least populated: {worst_b} ({b_vals[worst_b]} images)")
        if ratio_b > 3.0:
            print(f"    [!] Ratio > 3.0 — class imbalance may hurt training accuracy")
    elif len(b_vals) == 1:
        print("  Stage B: only one species has data — add images to other species")
    else:
        print("  Stage B: no species images found")


def print_training_time_section(
    raw_counts: dict[str, int],
    labeled_counts: dict[str, int],
    b_counts: dict[str, int],
) -> None:
    w = 70
    print(f"\n  ESTIMATED TRAINING TIME  (@ {MS_PER_IMAGE}ms/image, rough guide)")
    print(separator("-", w))

    total_a_labeled = sum(labeled_counts.values())
    total_b = sum(v for k, v in b_counts.items() if k != "unknown_fish")

    # Typical training runs over labeled data for stage A
    epochs_a = 100
    epochs_b = 50
    est_a = total_a_labeled * MS_PER_IMAGE * epochs_a
    est_b = total_b * MS_PER_IMAGE * epochs_b

    print(f"  Stage A (YOLO, {total_a_labeled} labeled images x {epochs_a} epochs): "
          f"{human_time(est_a)}")
    print(f"  Stage B (classifier, {total_b} images x {epochs_b} epochs):          "
          f"{human_time(est_b)}")
    print(f"  Note: actual GPU/CPU time varies widely; use this as a rough order-of-magnitude.")


def print_bar_chart(
    raw_counts: dict[str, int],
    b_counts: dict[str, int],
) -> None:
    w = 70
    all_counts = list(raw_counts.values()) + list(b_counts.values())
    max_count = max(all_counts) if all_counts else 1
    if max_count == 0:
        max_count = 1

    print(f"\n  CLASS SIZE — ASCII BAR CHART  (each '#' ≈ {max(1, max_count // BAR_MAX_WIDTH)} images)")
    print(separator("-", w))
    print("  --- Stage A (raw) ---")
    for cls in STAGE_A_CLASSES:
        print(bar_chart_line(cls, raw_counts[cls], max_count))
    print("  --- Stage B ---")
    for sp in STAGE_B_SPECIES:
        print(bar_chart_line(sp, b_counts[sp], max_count))


def print_recommendations(
    raw_counts: dict[str, int],
    b_counts: dict[str, int],
) -> None:
    w = 70
    print(f"\n  RECOMMENDATIONS")
    print(separator("-", w))

    recs: list[str] = []

    for cls in STAGE_A_CLASSES:
        n = raw_counts[cls]
        if n < STAGE_A_MIN:
            recs.append(
                f"  [URGENT] Stage A / {cls}: {n} images — need at least "
                f"{STAGE_A_MIN - n} more (minimum {STAGE_A_MIN})"
            )
        elif n < STAGE_A_WARN:
            recs.append(
                f"  [RECOMMENDED] Stage A / {cls}: {n} images — collect "
                f"{STAGE_A_WARN - n} more for best results (target {STAGE_A_WARN})"
            )

    for sp in STAGE_B_SPECIES:
        if sp == "unknown_fish":
            continue
        n = b_counts[sp]
        if n < STAGE_B_MIN:
            recs.append(
                f"  [URGENT] Stage B / {sp}: {n} images — need at least "
                f"{STAGE_B_MIN - n} more (minimum {STAGE_B_MIN})"
            )
        elif n < STAGE_B_WARN:
            recs.append(
                f"  [RECOMMENDED] Stage B / {sp}: {n} images — collect "
                f"{STAGE_B_WARN - n} more for best results (target {STAGE_B_WARN})"
            )

    if recs:
        for rec in recs:
            print(rec)
    else:
        print("  All classes meet recommended minimums. Dataset looks healthy!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print dataset statistics for the fish recognition pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.parse_args()

    raw_counts = gather_stage_a_raw()
    labeled_counts = gather_stage_a_labeled()
    b_counts = gather_stage_b()

    w = 70
    print()
    print(separator("=", w))
    print(" FISH DATASET STATISTICS".center(w))
    print(separator("=", w))

    print_stage_a_raw_section(raw_counts)
    print_stage_a_labeled_section(labeled_counts)
    print_stage_b_section(b_counts)
    print_balance_section(raw_counts, b_counts)
    print_training_time_section(raw_counts, labeled_counts, b_counts)
    print_bar_chart(raw_counts, b_counts)
    print_recommendations(raw_counts, b_counts)

    print()
    print(separator("=", w))
    print("  Run  python3 scripts/validate_dataset.py  for full validation.")
    print(separator("=", w))
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
