#!/usr/bin/env python3
"""
fetch_deepfish.py — Ingest DeepFish dataset for Stage A whole_fish / no_fish classes.

Dataset:
  Name: DeepFish (James Cook University, Australia)
  URL:  https://alzayats.github.io/DeepFish/
  Data: https://data.qld.edu.au/article/c0d00059-dfcb-49f5-98ba-d0b6e27ac7ce
  License: CC-BY 4.0 (data) + MIT (code)
  Species: Tropical Australian freshwater fish (20 habitats, ~40K images)

Why useful for this pipeline?
  - Stage A is species-agnostic: it detects whole_fish vs lure vs no_fish
  - DeepFish provides diverse fish-present and fish-absent frames
  - Australian tropical fish look different from Russian freshwater species
    → adds visual diversity to the whole_fish and no_fish classes
  - NOT useful for Stage B (wrong species entirely)

Download instructions:
  The dataset is split into subdirectories per habitat.
  Files available at: https://data.qld.edu.au/article/c0d00059-dfcb-49f5-98ba-d0b6e27ac7ce

  This script downloads the "Presence" (fish present) and "Absence" (no fish) subsets
  which are the most useful for Stage A detection training.

  MANUAL STEP REQUIRED:
    Due to QLD data repository using a JavaScript-rendered download page,
    direct wget/curl of individual files works but the index page requires
    a browser visit to get the exact file URLs.

    Recommended approach:
    1. Visit: https://data.qld.edu.au/article/c0d00059-dfcb-49f5-98ba-d0b6e27ac7ce
    2. Download the dataset ZIP file (approx 9 GB)
    3. Extract to ~/Downloads/DeepFish/
    4. Run: python3 scripts/fetch_deepfish.py --source ~/Downloads/DeepFish/

    Alternatively, if the DOI redirects to a direct download:
    wget -O deepfish.zip "https://data.qld.edu.au/dataset/c0d00059-dfcb-49f5-98ba-d0b6e27ac7ce"

Usage:
    python3 scripts/fetch_deepfish.py --source ~/Downloads/DeepFish/ [--max 200] [--dry-run]

Output:
    data/fish_dataset/stage_a/raw/whole_fish/  <- fish-present frames
    data/fish_dataset/stage_a/raw/no_fish/     <- fish-absent frames (background)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
STAGE_A_RAW = DATA_ROOT / "fish_dataset" / "stage_a" / "raw"
WHOLE_FISH_DIR = STAGE_A_RAW / "whole_fish"
NO_FISH_DIR = STAGE_A_RAW / "no_fish"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# DeepFish directory structure:
# DeepFish/
#   Presence/   <- fish present in frame (whole_fish class)
#     <habitat_id>/
#       *.jpg
#   Absence/    <- fish absent (no_fish class)
#     <habitat_id>/
#       *.jpg
#   Segmentation/  <- annotated frames (skip for our use)
#   Count/         <- count annotations (skip)

DEEPFISH_PRESENCE_DIR = "Presence"
DEEPFISH_ABSENCE_DIR = "Absence"


def _banner(msg: str) -> None:
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


def _ok(msg: str) -> None:
    print(f"[OK]  {msg}")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _existing_hashes(directory: Path) -> set[str]:
    hashes: set[str] = set()
    if not directory.exists():
        return hashes
    for p in directory.iterdir():
        if p.suffix.lower() in IMAGE_EXTS:
            try:
                hashes.add(_md5(p))
            except OSError:
                pass
    return hashes


def collect_images(source_dir: Path, subdir_name: str) -> list[Path]:
    """Recursively collect all images under source_dir/subdir_name/."""
    top = source_dir / subdir_name
    if not top.exists():
        # Try case-insensitive search
        for child in source_dir.iterdir():
            if child.name.lower() == subdir_name.lower() and child.is_dir():
                top = child
                break
        else:
            _warn(f"  Directory not found: {top}")
            return []
    return sorted(
        p for p in top.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def ingest_class(
    images: list[Path],
    dest_dir: Path,
    class_label: str,
    max_images: int,
    dry_run: bool,
    existing_hashes: set[str],
    provenance: list[dict],
) -> int:
    """Copy up to max_images into dest_dir, deduplicating by MD5."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing_count = sum(1 for p in dest_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    need = max_images - existing_count
    if need <= 0:
        _info(f"  {class_label}: already has {existing_count} images — skipping")
        return 0

    copied = 0
    for src in images:
        if copied >= need:
            break
        if dry_run:
            _info(f"  [DRY-RUN] Would copy: {src.name} -> {class_label}")
            copied += 1
            continue
        try:
            h = _md5(src)
        except OSError:
            continue
        if h in existing_hashes:
            continue
        existing_hashes.add(h)
        dest_name = f"deepfish_{class_label}_{src.name}"
        # Normalize filename
        dest_name = dest_name.replace(" ", "_").replace("(", "").replace(")", "")
        dst = dest_dir / dest_name
        try:
            shutil.copy2(src, dst)
        except OSError as exc:
            _warn(f"  Copy failed: {src} -> {dst}: {exc}")
            continue
        copied += 1
        provenance.append({
            "file": dest_name,
            "source": "deepfish",
            "class": class_label,
            "original_path": str(src),
            "license": "CC-BY 4.0",
            "attribution": "DeepFish dataset, James Cook University, CC-BY 4.0. "
                           "Alzayats et al. (2020). DeepFish: A benchmark dataset for "
                           "fish segmentation, counting, and localization in underwater video.",
        })
    return copied


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ingest DeepFish dataset into Stage A training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to extracted DeepFish directory (contains Presence/, Absence/ subdirs)",
    )
    p.add_argument(
        "--max-whole-fish",
        type=int,
        default=200,
        help="Max images to add to whole_fish class (Stage A)",
    )
    p.add_argument(
        "--max-no-fish",
        type=int,
        default=200,
        help="Max images to add to no_fish class (Stage A)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be copied without doing it.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _banner("DeepFish Dataset Ingester — Stage A")
    _info(f"  Source: {args.source}")
    _info(f"  Max whole_fish: {args.max_whole_fish}")
    _info(f"  Max no_fish:    {args.max_no_fish}")

    if not args.source.exists():
        _fail(f"Source directory does not exist: {args.source}")
        _fail("")
        _fail("Manual download instructions:")
        _fail("  1. Visit: https://data.qld.edu.au/article/c0d00059-dfcb-49f5-98ba-d0b6e27ac7ce")
        _fail("  2. Download the dataset ZIP (~9 GB)")
        _fail("  3. Extract to ~/Downloads/DeepFish/")
        _fail("  4. Re-run this script with: --source ~/Downloads/DeepFish/")
        sys.exit(1)

    provenance: list[dict] = []

    # ── Presence (whole_fish) ─────────────────────────────────────────────────
    _banner("Step 1 — Presence frames -> whole_fish")
    presence_images = collect_images(args.source, DEEPFISH_PRESENCE_DIR)
    _info(f"  Found {len(presence_images)} presence images")

    existing_wf = _existing_hashes(WHOLE_FISH_DIR)
    n_wf = ingest_class(
        images=presence_images,
        dest_dir=WHOLE_FISH_DIR,
        class_label="whole_fish",
        max_images=args.max_whole_fish,
        dry_run=args.dry_run,
        existing_hashes=existing_wf,
        provenance=provenance,
    )
    _ok(f"whole_fish: {n_wf} images added")

    # ── Absence (no_fish) ─────────────────────────────────────────────────────
    _banner("Step 2 — Absence frames -> no_fish")
    absence_images = collect_images(args.source, DEEPFISH_ABSENCE_DIR)
    _info(f"  Found {len(absence_images)} absence images")

    existing_nf = _existing_hashes(NO_FISH_DIR)
    n_nf = ingest_class(
        images=absence_images,
        dest_dir=NO_FISH_DIR,
        class_label="no_fish",
        max_images=args.max_no_fish,
        dry_run=args.dry_run,
        existing_hashes=existing_nf,
        provenance=provenance,
    )
    _ok(f"no_fish: {n_nf} images added")

    # ── Provenance ────────────────────────────────────────────────────────────
    if not args.dry_run and provenance:
        for dest_dir, label in [(WHOLE_FISH_DIR, "whole_fish"), (NO_FISH_DIR, "no_fish")]:
            records = [r for r in provenance if r["class"] == label]
            if records:
                prov_path = dest_dir / "PROVENANCE_deepfish.json"
                existing_records: list[dict] = []
                if prov_path.exists():
                    try:
                        existing_records = json.loads(prov_path.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        pass
                existing_records.extend(records)
                prov_path.write_text(
                    json.dumps(existing_records, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

    _banner("Done")
    _ok(f"Total images ingested: {n_wf + n_nf}")
    print()
    print("  Stage A raw counts after ingestion:")
    for cls in ["whole_fish", "lure", "fish_part", "fry", "no_fish"]:
        d = STAGE_A_RAW / cls
        n = sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS) if d.exists() else 0
        print(f"    {cls:<12}: {n:>4}")
    print()
    print("  Attribution: DeepFish — James Cook University, CC-BY 4.0")
    print("  https://alzayats.github.io/DeepFish/")


if __name__ == "__main__":
    main()
