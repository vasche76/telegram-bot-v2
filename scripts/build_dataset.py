#!/usr/bin/env python3
"""
build_dataset.py — Unified dataset build orchestrator.

Runs the full pipeline from raw images to training-ready labeled datasets:

  Step 1: Download fish species images from iNaturalist  (fetch_inaturalist.py)
  Step 2: Download lure images from Wikimedia Commons   (fetch_wikimedia_lures.py)
  Step 3: Download fry + fish_part images               (fetch_fish_parts_fry.py)
  Step 4: Augment small classes                         (augment_dataset.py)
  Step 5: Generate YOLO labels for Stage A              (create_stage_a_labels.py)
  Step 6: Validate dataset                              (validate_dataset.py)
  Step 7: (Optional) Train Stage A                      (train_stage_a.py)
  Step 8: (Optional) Train Stage B                      (train_stage_b.py)

Each step can be run individually or skipped with --skip-* flags.

Usage:
    # Full pipeline (download + label + validate, skip training):
    python3 scripts/build_dataset.py

    # Full pipeline including training:
    python3 scripts/build_dataset.py --train --device mps

    # Skip downloads, just rebuild labels:
    python3 scripts/build_dataset.py --skip-download

    # Only download and augment, then stop:
    python3 scripts/build_dataset.py --skip-labels --skip-validate

    # Dry run (see what would happen, no network/file writes):
    python3 scripts/build_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
SCRIPTS_DIR = REPO_ROOT / "scripts"


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


def _run(script: str, extra_args: list[str] = (), dry_run: bool = False) -> int:
    """Run a scripts/ script as a subprocess. Returns exit code."""
    cmd = [sys.executable, str(SCRIPTS_DIR / script)] + list(extra_args)
    _info(f"  Running: {' '.join(cmd)}")
    if dry_run:
        _info(f"  [DRY-RUN] Would run: {script} {' '.join(extra_args)}")
        return 0
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode


def _count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for f in directory.iterdir() if f.suffix.lower() in exts)


def _print_status() -> None:
    """Print current dataset status."""
    print()
    print("  Stage A raw image counts:")
    for cls in ["whole_fish", "lure", "fish_part", "fry", "no_fish"]:
        n = _count_images(DATA_ROOT / "fish_dataset" / "stage_a" / "raw" / cls)
        status = "OK" if n >= 20 else f"WARN (<20)"
        print(f"    {cls:<12}: {n:>4}  [{status}]")

    print()
    print("  Stage B species image counts:")
    all_species = [
        # Original
        "pike", "taimen", "grayling", "whitefish", "perch",
        # New Salmonidae
        "brown_trout", "rainbow_trout", "atlantic_salmon",
        # New Cyprinidae
        "common_carp", "crucian_carp", "bream", "roach", "ide",
        # New Siluriformes
        "wels_catfish",
        # Fallback
        "unknown_fish",
    ]
    for sp in all_species:
        n = _count_images(DATA_ROOT / "fish_dataset" / "stage_b" / sp)
        if sp == "unknown_fish":
            status = "—"
        else:
            status = "OK" if n >= 15 else f"NEED {max(0, 15-n)} MORE"
        print(f"    {sp:<20}: {n:>4}  [{status}]")

    print()
    print("  Stage A labeled splits:")
    for split in ["train", "val", "test"]:
        n = _count_images(DATA_ROOT / "fish_dataset" / "stage_a" / "labeled" / split / "images")
        print(f"    {split:<6}: {n:>4}")

    print()
    models_dir = DATA_ROOT / "fish_models"
    detector = (models_dir / "detector_v1.pt").exists()
    classifier = (models_dir / "classifier_v1.pt").exists()
    print(f"  Models:")
    print(f"    detector_v1.pt  : {'EXISTS' if detector else 'MISSING'}")
    print(f"    classifier_v1.pt: {'EXISTS' if classifier else 'MISSING'}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified fish dataset build pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--skip-download", action="store_true",
                   help="Skip all download steps (use existing raw images)")
    p.add_argument("--skip-augment", action="store_true",
                   help="Skip augmentation of small classes")
    p.add_argument("--skip-labels", action="store_true",
                   help="Skip Stage A YOLO label generation")
    p.add_argument("--skip-validate", action="store_true",
                   help="Skip dataset validation")
    p.add_argument("--train", action="store_true",
                   help="Run training after dataset is ready (Stage A + Stage B)")
    p.add_argument("--train-a-only", action="store_true",
                   help="Train Stage A (YOLO) only")
    p.add_argument("--train-b-only", action="store_true",
                   help="Train Stage B (EfficientNet) only")
    p.add_argument("--device", default="cpu",
                   help="Training device: cpu, mps (Apple Silicon), cuda")
    p.add_argument("--epochs-a", type=int, default=100,
                   help="Epochs for Stage A YOLO training")
    p.add_argument("--epochs-b-phase1", type=int, default=15,
                   help="Epochs for Stage B Phase 1 (frozen backbone)")
    p.add_argument("--epochs-b-phase2", type=int, default=25,
                   help="Epochs for Stage B Phase 2 (full fine-tune)")
    p.add_argument("--augment-target", type=int, default=60,
                   help="Target count per class for augmentation")
    p.add_argument("--inat-max", type=int, default=80,
                   help="Max images per species to download from iNaturalist")
    p.add_argument("--gbif-max", type=int, default=80,
                   help="Max images per species to download from GBIF (supplements iNaturalist)")
    p.add_argument("--skip-gbif", action="store_true",
                   help="Skip GBIF download step")
    p.add_argument("--lure-max", type=int, default=80,
                   help="Max lure images to download from Wikimedia")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would run without executing")
    p.add_argument("--status", action="store_true",
                   help="Print current dataset status and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start_time = datetime.now(timezone.utc)

    _banner("Fish Dataset Build Pipeline")

    if args.status:
        _print_status()
        return

    _info(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M UTC')}")
    _info(f"  Repo   : {REPO_ROOT}")

    errors: list[str] = []

    # ── Step 1: Download fish species (iNaturalist) ──────────────────────────
    if not args.skip_download:
        _banner("Step 1 — Download Fish Species (iNaturalist)")
        rc = _run(
            "fetch_inaturalist.py",
            ["--species", "pike,taimen,grayling,whitefish,perch",
             "--max", str(args.inat_max),
             "--no-fish-max", "60"],
            dry_run=args.dry_run,
        )
        if rc != 0:
            _fail(f"fetch_inaturalist.py exited with code {rc}")
            errors.append("iNaturalist download failed")

        # ── Step 1b: Download new taxonomy species (GBIF) ────────────────────
        if not args.skip_gbif:
            _banner("Step 1b — Download New Taxonomy Species (GBIF)")
            _info("  Fetching: brown_trout, rainbow_trout, atlantic_salmon,")
            _info("            common_carp, crucian_carp, bream, roach, ide, wels_catfish")
            _info("  Also supplementing existing species where GBIF has more data.")
            rc = _run(
                "fetch_gbif.py",
                ["--all", "--max", str(args.gbif_max)],
                dry_run=args.dry_run,
            )
            if rc != 0:
                _fail(f"fetch_gbif.py exited with code {rc}")
                errors.append("GBIF download failed")
        else:
            _info("Skipping GBIF download (--skip-gbif)")

        # ── Step 2: Download lures (Wikimedia) ──────────────────────────────
        _banner("Step 2 — Download Lures (Wikimedia Commons)")
        rc = _run(
            "fetch_wikimedia_lures.py",
            ["--max", str(args.lure_max)],
            dry_run=args.dry_run,
        )
        if rc != 0:
            _fail(f"fetch_wikimedia_lures.py exited with code {rc}")
            errors.append("Wikimedia lure download failed")

        # ── Step 3: Download fry + fish_part ────────────────────────────────
        _banner("Step 3 — Download Fry + Fish Part")
        rc = _run(
            "fetch_fish_parts_fry.py",
            ["--fry-max", "60", "--fish-part-max", "60"],
            dry_run=args.dry_run,
        )
        if rc != 0:
            _fail(f"fetch_fish_parts_fry.py exited with code {rc}")
            errors.append("Fry/fish_part download failed")
    else:
        _info("Skipping download steps (--skip-download)")

    # ── Step 4: Augment small classes ────────────────────────────────────────
    if not args.skip_augment:
        _banner("Step 4 — Augment Small Classes")
        rc = _run(
            "augment_dataset.py",
            ["--classes", "lure,fish_part,fry",
             "--target", str(args.augment_target)],
            dry_run=args.dry_run,
        )
        if rc != 0:
            _fail(f"augment_dataset.py exited with code {rc}")
            errors.append("Augmentation failed")
    else:
        _info("Skipping augmentation (--skip-augment)")

    # ── Step 5: Generate YOLO labels ─────────────────────────────────────────
    if not args.skip_labels:
        _banner("Step 5 — Generate Stage A YOLO Labels")
        rc = _run(
            "create_stage_a_labels.py",
            ["--overwrite"],
            dry_run=args.dry_run,
        )
        if rc != 0:
            _fail(f"create_stage_a_labels.py exited with code {rc}")
            errors.append("YOLO label generation failed")
    else:
        _info("Skipping label generation (--skip-labels)")

    # ── Step 6: Validate ─────────────────────────────────────────────────────
    if not args.skip_validate:
        _banner("Step 6 — Validate Dataset")
        rc = _run("validate_dataset.py", dry_run=args.dry_run)
        if rc != 0:
            _fail("Dataset validation failed — review errors above before training")
            errors.append("Dataset validation failed")
    else:
        _info("Skipping validation (--skip-validate)")

    # ── Training ─────────────────────────────────────────────────────────────
    train_a = args.train or args.train_a_only
    train_b = args.train or args.train_b_only

    if train_b:
        _banner("Step 7a — Train Stage B (EfficientNet Species Classifier)")
        if errors:
            _fail("Skipping training due to earlier errors")
        else:
            rc = _run(
                "train_stage_b.py",
                [
                    "--epochs_phase1", str(args.epochs_b_phase1),
                    "--epochs_phase2", str(args.epochs_b_phase2),
                    "--device", args.device,
                ],
                dry_run=args.dry_run,
            )
            if rc != 0:
                errors.append("Stage B training failed")

    if train_a:
        _banner("Step 7b — Train Stage A (YOLO Detector)")
        if "Dataset validation failed" in errors:
            _fail("Skipping YOLO training due to dataset validation failure")
        else:
            rc = _run(
                "train_stage_a.py",
                ["--epochs", str(args.epochs_a), "--device", args.device],
                dry_run=args.dry_run,
            )
            if rc != 0:
                errors.append("Stage A training failed")

    # ── Final status ─────────────────────────────────────────────────────────
    _banner("Final Dataset Status")
    _print_status()

    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    _banner("Build Complete")

    if errors:
        print(f"\n  Completed with {len(errors)} error(s):")
        for e in errors:
            print(f"    - {e}")
        print(f"\n  Elapsed: {elapsed:.0f}s")
        sys.exit(1)
    else:
        _ok(f"All steps completed successfully in {elapsed:.0f}s")
        if not (args.train or args.train_a_only or args.train_b_only):
            print()
            _info("To train the models, run:")
            _info("  python3 scripts/build_dataset.py --train --device mps")
            _info("  or")
            _info("  python3 scripts/train_stage_b.py --device mps")
            _info("  python3 scripts/train_stage_a.py --device mps")


if __name__ == "__main__":
    main()
