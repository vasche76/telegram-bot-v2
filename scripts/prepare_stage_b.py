#!/usr/bin/env python3
"""
prepare_stage_b.py — Stage B dataset validation and split preview

Run this BEFORE train_stage_b.py to inspect your species data without
committing to a training run.

Shows:
  - Image counts per species folder
  - Estimated train/val/test split sizes (70/15/15, seed=42)
  - Inverse-frequency class weights (for imbalance awareness)
  - Species with too few images flagged clearly

Usage:
    python3 scripts/prepare_stage_b.py
"""

import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths / constants (must match train_stage_b.py)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
STAGE_B_DIR = DATA_ROOT / "fish_dataset" / "stage_b"

CLASS_NAMES_B = {
    "0":  "pike",
    "1":  "taimen",
    "2":  "grayling",
    "3":  "whitefish",
    "4":  "perch",
    # New Salmonidae
    "5":  "brown_trout",
    "6":  "rainbow_trout",
    "7":  "atlantic_salmon",
    # New Cyprinidae
    "8":  "common_carp",
    "9":  "crucian_carp",
    "10": "bream",
    "11": "roach",
    "12": "ide",
    # New Siluriformes
    "13": "wels_catfish",
    # Fallback
    "14": "unknown_fish",
}

MIN_IMAGES_PER_CLASS = 15
SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ANSI colours (disabled if stdout is not a terminal)
_USE_COLOUR = sys.stdout.isatty()

RED = "\033[31m" if _USE_COLOUR else ""
YELLOW = "\033[33m" if _USE_COLOUR else ""
GREEN = "\033[32m" if _USE_COLOUR else ""
RESET = "\033[0m" if _USE_COLOUR else ""
BOLD = "\033[1m" if _USE_COLOUR else ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _colour(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"\n{BOLD}Stage B Dataset Preparation Preview{RESET}")
    print(f"Dataset directory: {STAGE_B_DIR}\n")

    if not STAGE_B_DIR.exists():
        print(_colour(f"[ERROR] Stage B directory not found: {STAGE_B_DIR}", RED))
        print("  Create it and populate with species subfolders.")
        sys.exit(1)

    # ---- Collect image counts ----
    species_data: list[dict] = []
    for idx_str, name in CLASS_NAMES_B.items():
        folder = STAGE_B_DIR / name
        if not folder.exists():
            species_data.append({
                "idx": int(idx_str),
                "name": name,
                "images": [],
                "exists": False,
            })
            continue
        imgs = _images_in(folder)
        species_data.append({
            "idx": int(idx_str),
            "name": name,
            "images": imgs,
            "exists": True,
        })

    # ---- Header ----
    header = (
        f"  {'#':>3}  {'Species':<15} {'Images':>7}  "
        f"{'Train':>6}  {'Val':>5}  {'Test':>5}  {'Weight':>7}  Status"
    )
    print(header)
    print("  " + "-" * 70)

    rng = random.Random(SPLIT_SEED)
    total_images = 0
    usable_counts: list[int] = []
    rows: list[str] = []

    for sd in species_data:
        name = sd["name"]
        imgs = sd["images"]
        n = len(imgs)
        total_images += n

        if not sd["exists"]:
            status = _colour("MISSING FOLDER", RED)
            row = (
                f"  {sd['idx']:>3}  {name:<15} {n:>7}  "
                f"{'—':>6}  {'—':>5}  {'—':>5}  {'—':>7}  {status}"
            )
            rows.append(row)
            continue

        if n < MIN_IMAGES_PER_CLASS:
            status = _colour(f"TOO FEW (need >= {MIN_IMAGES_PER_CLASS})", RED)
            row = (
                f"  {sd['idx']:>3}  {name:<15} {n:>7}  "
                f"{'—':>6}  {'—':>5}  {'—':>5}  {'—':>7}  {status}"
            )
            rows.append(row)
            continue

        usable_counts.append(n)

        # Compute split sizes exactly as train_stage_b.py does
        shuffled = imgs.copy()
        rng.shuffle(shuffled)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_val = max(0, n_val - 1)
            n_test = n - n_train - n_val

        sd["n_train"] = n_train
        sd["n_val"] = n_val
        sd["n_test"] = n_test

        if n < 30:
            status = _colour("LOW (consider adding more)", YELLOW)
        else:
            status = _colour("OK", GREEN)

        # Weight placeholder — filled below once totals are known
        sd["_row_fmt"] = (
            f"  {sd['idx']:>3}  {name:<15} {n:>7}  "
            f"{n_train:>6}  {n_val:>5}  {n_test:>5}  {{weight:>7}}  {status}"
        )

    # ---- Compute inverse-frequency weights for usable classes ----
    total_usable = sum(usable_counts)
    num_usable = len(usable_counts)

    usable_idx = 0
    for sd in species_data:
        if "_row_fmt" in sd:
            n = len(sd["images"])
            weight = total_usable / (num_usable * n) if n > 0 else 0.0
            rows.append(sd["_row_fmt"].format(weight=f"{weight:.3f}"))
        # else the row was already appended above (missing/too few)

    # Re-collect rows in order (they were appended differently for too-few vs OK)
    # Reset and rebuild rows in species order
    rows = []
    rng2 = random.Random(SPLIT_SEED)
    for sd in species_data:
        name = sd["name"]
        imgs = sd["images"]
        n = len(imgs)

        if not sd["exists"]:
            status = _colour("MISSING FOLDER", RED)
            rows.append(
                f"  {sd['idx']:>3}  {name:<15} {n:>7}  "
                f"{'—':>6}  {'—':>5}  {'—':>5}  {'—':>7}  {status}"
            )
            continue

        if n < MIN_IMAGES_PER_CLASS:
            status = _colour(f"TOO FEW (need >= {MIN_IMAGES_PER_CLASS})", RED)
            rows.append(
                f"  {sd['idx']:>3}  {name:<15} {n:>7}  "
                f"{'—':>6}  {'—':>5}  {'—':>5}  {'—':>7}  {status}"
            )
            continue

        # Recompute split with fresh rng in species order (consistent with train script)
        shuffled = imgs.copy()
        rng2.shuffle(shuffled)
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO))
        n_test = n - n_train - n_val
        if n_test < 1:
            n_val = max(0, n_val - 1)
            n_test = n - n_train - n_val

        weight = total_usable / (num_usable * n) if (num_usable > 0 and n > 0) else 0.0

        if n < 30:
            status = _colour("LOW (consider adding more)", YELLOW)
        else:
            status = _colour("OK", GREEN)

        rows.append(
            f"  {sd['idx']:>3}  {name:<15} {n:>7}  "
            f"{n_train:>6}  {n_val:>5}  {n_test:>5}  {weight:>7.3f}  {status}"
        )

    for row in rows:
        print(row)

    # ---- Summary ----
    ok_classes = [sd for sd in species_data if sd["exists"] and len(sd["images"]) >= MIN_IMAGES_PER_CLASS]
    problem_classes = [sd for sd in species_data if not sd["exists"] or len(sd["images"]) < MIN_IMAGES_PER_CLASS]

    print()
    print(f"  Total images across all species : {total_images}")
    print(f"  Usable species (>= {MIN_IMAGES_PER_CLASS} images)   : {len(ok_classes)}")
    if problem_classes:
        print()
        print(_colour("  Issues requiring attention:", YELLOW))
        for sd in problem_classes:
            if not sd["exists"]:
                print(_colour(f"    - {sd['name']}: folder does not exist at {STAGE_B_DIR / sd['name']}", RED))
            else:
                print(_colour(f"    - {sd['name']}: only {len(sd['images'])} images (need >= {MIN_IMAGES_PER_CLASS})", RED))

    # ---- Readiness verdict ----
    print()
    if len(ok_classes) >= 2:
        print(_colour("  Dataset is READY for train_stage_b.py", GREEN))
        print(f"  Run:  python3 scripts/train_stage_b.py")
    else:
        print(_colour("  Dataset is NOT ready for training.", RED))
        print(f"  Need at least 2 species with >= {MIN_IMAGES_PER_CLASS} images each.")
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
