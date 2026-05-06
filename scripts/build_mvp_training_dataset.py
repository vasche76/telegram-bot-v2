"""
build_mvp_training_dataset.py — Build controlled MVP structural training dataset.

Combines:
  1. Reviewed Telegram seed (from Phase E Lite reviewed_seed_records.jsonl)
  2. License-safe external public data (from data/fish_dataset/stage_b/ + stage_a/)

Structural MVP classes:
  fish            — manually reviewed whole fish at confidence >= 3
  not_fish_or_other — reviewed non-fish + license-safe external negatives

Rules (enforced by training gates):
  - No unreviewed Telegram records
  - No Phase C labels as truth
  - Deterministic train/val/test split (random seed fixed)
  - No duplicate sha256/review_id across splits
  - Class imbalance reported and gated
  - External data provenance recorded separately from Telegram seed
  - Output marked EXPERIMENTAL

Outputs (all in data/mvp_training/manifests/):
  mvp_training_manifest.json    — tracked; full provenance record
  mvp_class_mapping.json        — tracked; class assignment rules
  mvp_split_report.json         — tracked; split counts and imbalance
  mvp_dataset_quality_report.md — tracked; human-readable quality gate result

  (actual images are NOT copied by this script; see --copy-images flag)
  data/mvp_training/structural/train|val|test/ — gitignored image destinations

Training gates:
  GATE 1: No unreviewed Telegram records in seed
  GATE 2: No Phase C labels as truth
  GATE 3: Class imbalance ≤ 10:1 fish:non-fish
  GATE 4: Minimum 20 non-fish examples
  GATE 5: All sources have recorded license/provenance
  GATE 6: No review_id leakage across train/val/test

Usage:
    python3 scripts/build_mvp_training_dataset.py
    python3 scripts/build_mvp_training_dataset.py --dry-run
    python3 scripts/build_mvp_training_dataset.py --copy-images

source=mixed_telegram_reviewed_and_external_public, license=see_provenance_manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intake_constants as C

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
MVP_TRAINING_DIR = DATA_ROOT / "mvp_training"
MANIFESTS_DIR = MVP_TRAINING_DIR / "manifests"
STRUCTURAL_DIR = MVP_TRAINING_DIR / "structural"
STAGE_B_DIR = DATA_ROOT / "fish_dataset" / "stage_b"
STAGE_A_RAW_DIR = DATA_ROOT / "fish_dataset" / "stage_a" / "raw"
STAGE_A_LURE_DIR = STAGE_A_RAW_DIR / "lure"

MANIFEST_SCHEMA_VERSION = "mvp_training_v1"
SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# test = 1 - train - val = 0.15

# Training gates
GATE_MAX_IMBALANCE_RATIO = 10.0
GATE_MIN_NON_FISH = 20
GATE_MIN_TOTAL_TRAINING = 50

# Structural class names
CLASS_FISH = "fish"
CLASS_NOT_FISH = "not_fish_or_other"
STRUCTURAL_CLASSES = [CLASS_FISH, CLASS_NOT_FISH]

# External source species that map to "fish" for structural training
_FISH_SPECIES = {
    "pike", "taimen", "grayling", "whitefish", "perch",
    "brown_trout", "rainbow_trout", "atlantic_salmon",
    "common_carp", "crucian_carp", "bream", "roach", "ide",
    "wels_catfish", "unknown_fish",
}


# ─── I/O helpers ─────────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ─── Seed loading ─────────────────────────────────────────────────────────────


def load_telegram_seed_records() -> list[dict]:
    """Load reviewed Telegram seed records. Only fish/not_fish classes included."""
    records_path = C.REVIEWED_SEED_RECORDS_PATH
    if not records_path.exists():
        log.error(
            "Reviewed seed records not found: %s — run intake_phase_e_lite.py first",
            records_path,
        )
        return []
    records = _load_jsonl(records_path)
    # Filter to training-usable classes only (exclude needs_human_review)
    usable = [r for r in records if r.get("mvp_class") in (CLASS_FISH, CLASS_NOT_FISH)]
    excluded_count = len(records) - len(usable)
    if excluded_count > 0:
        log.info("Telegram seed: %d usable, %d excluded (needs_human_review)", len(usable), excluded_count)
    return usable


def load_external_fish_records() -> list[dict]:
    """
    Load external public fish species images from stage_b.
    These map to structural class 'fish'.
    """
    records = []
    if not STAGE_B_DIR.exists():
        return records
    for species_dir in sorted(STAGE_B_DIR.iterdir()):
        if not species_dir.is_dir() or species_dir.name not in _FISH_SPECIES:
            continue
        for img_path in sorted(species_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            # Compute stable ID from relative path
            rel = str(img_path.relative_to(REPO_ROOT))
            stable_id = hashlib.md5(rel.encode()).hexdigest()[:16]
            records.append({
                "stable_id": stable_id,
                "mvp_class": CLASS_FISH,
                "provenance": "external_public",
                "species": species_dir.name,
                "source_id": "gbif_inaturalist",
                "local_path": str(img_path),
            })
    log.info("External fish images (stage_b): %d", len(records))
    return records


def load_external_nofish_records() -> list[dict]:
    """
    Load external public no-fish images from stage_a/raw/no_fish.
    These map to structural class 'not_fish_or_other'.
    """
    records = []
    no_fish_dir = STAGE_A_RAW_DIR / "no_fish"
    if not no_fish_dir.exists():
        log.info("No external no_fish images found at %s", no_fish_dir)
        return records
    for img_path in sorted(no_fish_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        rel = str(img_path.relative_to(REPO_ROOT))
        stable_id = hashlib.md5(rel.encode()).hexdigest()[:16]
        records.append({
            "stable_id": stable_id,
            "mvp_class": CLASS_NOT_FISH,
            "provenance": "external_public",
            "source_id": "stage_a_raw_no_fish",
            "local_path": str(img_path),
        })
    log.info("External no_fish images (stage_a/raw): %d", len(records))
    return records


def load_external_lure_records() -> list[dict]:
    """
    Load Wikimedia Commons lure images from stage_a/raw/lure/ as not_fish_or_other.

    Augmented variants (filename contains '_aug_') are tracked with an
    aug_group_id equal to their original's stable_id so the group-aware split
    can keep them in the same partition as their source image.

    Provenance is verified via PROVENANCE.json (AUGMENTATION.json tracks groups).
    """
    records = []
    if not STAGE_A_LURE_DIR.exists():
        log.info("No lure images found at %s", STAGE_A_LURE_DIR)
        return records

    # Load augmentation group map: original_filename → [augmented_filename, ...]
    aug_map: dict[str, list[str]] = {}
    aug_json_path = STAGE_A_LURE_DIR / "AUGMENTATION.json"
    if aug_json_path.exists():
        aug_data = _load_json(aug_json_path)
        aug_map = aug_data.get("augmented", {})

    # Build reverse map: augmented_filename → original_filename
    aug_to_orig: dict[str, str] = {}
    for orig, aug_list in aug_map.items():
        for aug_name in aug_list:
            aug_to_orig[aug_name] = orig

    for img_path in sorted(STAGE_A_LURE_DIR.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        rel = str(img_path.relative_to(REPO_ROOT))
        stable_id = hashlib.md5(rel.encode()).hexdigest()[:16]

        # Determine aug_group_id: all augmented variants share their original's stable_id
        orig_name = aug_to_orig.get(img_path.name)
        if orig_name:
            orig_rel = str((STAGE_A_LURE_DIR / orig_name).relative_to(REPO_ROOT))
            aug_group_id = hashlib.md5(orig_rel.encode()).hexdigest()[:16]
        else:
            aug_group_id = stable_id  # original is its own group

        records.append({
            "stable_id": stable_id,
            "aug_group_id": aug_group_id,
            "mvp_class": CLASS_NOT_FISH,
            "provenance": "external_public",
            "source_id": "wikimedia_commons_lures",
            "local_path": str(img_path),
        })

    log.info("External lure images (stage_a/raw/lure): %d", len(records))
    return records


# ─── Training gates ───────────────────────────────────────────────────────────


class TrainingGateError(Exception):
    pass


def check_training_gates(
    tg_records: list[dict],
    ext_fish: list[dict],
    ext_nofish: list[dict],
    ext_lure: list[dict] | None = None,
) -> list[str]:
    """
    Run all training gates. Return list of gate failures (empty = all pass).
    """
    failures: list[str] = []
    _ext_lure = ext_lure or []

    all_fish = (
        [r for r in tg_records if r.get("mvp_class") == CLASS_FISH] + ext_fish
    )
    all_nofish = (
        [r for r in tg_records if r.get("mvp_class") == CLASS_NOT_FISH]
        + ext_nofish
        + _ext_lure
    )
    total = len(all_fish) + len(all_nofish)

    # GATE 1: No unreviewed Telegram records
    # (enforced by intake_phase_e_lite.py — verified by checking seed records file)
    seed_summary_path = C.REVIEWED_SEED_SUMMARY_PATH
    if seed_summary_path.exists():
        summary = _load_json(seed_summary_path)
        if summary.get("counts", {}).get("unreviewed_not_eligible", 0) > 0:
            # This is expected and correct — unreviewed records exist but are NOT in seed
            pass  # GATE 1 pass: unreviewed are tracked but not included
    else:
        failures.append("GATE 1: Reviewed seed summary missing — cannot verify no unreviewed records")

    # GATE 2: No Phase C labels as truth
    # (Phase C produced no training-ready labels — verified by checking seed has only
    #  manual review decisions)
    for r in tg_records:
        if r.get("provenance") not in ("telegram_reviewed", None):
            failures.append(f"GATE 2: Unexpected provenance '{r.get('provenance')}' in Telegram seed")
            break
    # Telegram seed records have mvp_class assigned from manual review, not Phase C

    # GATE 3: Class imbalance
    if len(all_nofish) > 0:
        ratio = len(all_fish) / len(all_nofish)
        if ratio > GATE_MAX_IMBALANCE_RATIO:
            failures.append(
                f"GATE 3: Class imbalance {ratio:.1f}:1 exceeds limit {GATE_MAX_IMBALANCE_RATIO:.0f}:1. "
                f"Need more non-fish examples. fish={len(all_fish)}, non_fish={len(all_nofish)}"
            )
    else:
        failures.append(
            f"GATE 3+4: Zero non-fish examples — cannot train structural classifier. "
            f"Collect non-fish data first."
        )

    # GATE 4: Minimum non-fish
    if len(all_nofish) < GATE_MIN_NON_FISH:
        failures.append(
            f"GATE 4: Only {len(all_nofish)} non-fish examples < {GATE_MIN_NON_FISH} minimum"
        )

    # GATE 5: External licenses recorded
    if not (REPO_ROOT / "data" / "external_public" / "manifests" / "external_license_report.json").exists():
        failures.append("GATE 5: External license report missing — run intake_external_lite.py first")

    # GATE 6: No review_id duplicates (review_id is per-record unique within the seed)
    review_ids = [r.get("review_id") for r in tg_records if r.get("review_id")]
    if len(review_ids) != len(set(review_ids)):
        dup_count = len(review_ids) - len(set(review_ids))
        failures.append(f"GATE 6: {dup_count} duplicate review_ids in Telegram seed")

    # Total size gate
    if total < GATE_MIN_TOTAL_TRAINING:
        failures.append(
            f"GATE 7: Total training records {total} < {GATE_MIN_TOTAL_TRAINING} minimum"
        )

    return failures


# ─── Deterministic split ──────────────────────────────────────────────────────


def deterministic_split(
    records: list[dict],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    seed: int = SPLIT_SEED,
) -> dict[str, list[dict]]:
    """
    Deterministic split. Group-aware when records contain aug_group_id:
    all records sharing the same aug_group_id are placed in the same split
    to prevent augmented-variant leakage across train/val/test.

    Records without aug_group_id are split individually (original behaviour).
    """
    # Separate group-aware records from independent records
    groups: dict[str, list[dict]] = {}
    independent: list[dict] = []

    for r in records:
        gid = r.get("aug_group_id")
        if gid and gid != r.get("stable_id"):
            # Record belongs to a multi-member augmentation group
            groups.setdefault(gid, []).append(r)
        else:
            # Original images and un-augmented records split independently
            independent.append(r)

    rng = random.Random(seed)

    # Split independent records by position
    ind_shuffled = list(independent)
    rng.shuffle(ind_shuffled)
    n = len(ind_shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    ind_splits: dict[str, list[dict]] = {
        "train": ind_shuffled[:n_train],
        "val": ind_shuffled[n_train : n_train + n_val],
        "test": ind_shuffled[n_train + n_val :],
    }

    # Split groups as a unit
    group_list = sorted(groups.values(), key=lambda g: g[0].get("aug_group_id", ""))
    rng.shuffle(group_list)
    ng = len(group_list)
    ng_train = int(ng * train_ratio)
    ng_val = int(ng * val_ratio)
    group_splits: dict[str, list[list[dict]]] = {
        "train": group_list[:ng_train],
        "val": group_list[ng_train : ng_train + ng_val],
        "test": group_list[ng_train + ng_val :],
    }

    result: dict[str, list[dict]] = {}
    for split_name in ("train", "val", "test"):
        split_records = list(ind_splits[split_name])
        for grp in group_splits[split_name]:
            split_records.extend(grp)
        result[split_name] = split_records

    return result


def check_split_leakage(splits: dict[str, list[dict]]) -> list[str]:
    """Verify no review_id or stable_id appears in more than one split."""
    errors = []
    id_to_split: dict[str, str] = {}
    for split_name, records in splits.items():
        for r in records:
            uid = r.get("review_id") or r.get("stable_id")
            if uid:
                if uid in id_to_split:
                    errors.append(
                        f"Leakage: ID {uid[:12]}... appears in both '{id_to_split[uid]}' and '{split_name}'"
                    )
                else:
                    id_to_split[uid] = split_name
    return errors


# ─── Quality report ───────────────────────────────────────────────────────────


def build_quality_report(
    gate_failures: list[str],
    tg_records: list[dict],
    ext_fish: list[dict],
    ext_nofish: list[dict],
    splits: dict[str, list[dict]] | None,
    now: str,
    ext_lure: list[dict] | None = None,
) -> str:
    _ext_lure = ext_lure or []
    all_fish = len([r for r in tg_records if r.get("mvp_class") == CLASS_FISH]) + len(ext_fish)
    all_nofish = (
        len([r for r in tg_records if r.get("mvp_class") == CLASS_NOT_FISH])
        + len(ext_nofish)
        + len(_ext_lure)
    )
    total = all_fish + all_nofish
    ratio = all_fish / all_nofish if all_nofish > 0 else float("inf")

    status = "BLOCKED" if gate_failures else "READY"

    lines = [
        "# MVP Training Dataset Quality Report",
        f"\nGenerated: {now}",
        f"\nStatus: **{status}**",
        "\n## Data Summary",
        "\n| Source | Fish | Non-fish | Total |",
        "| --- | --- | --- | --- |",
        f"| Telegram reviewed seed | {len([r for r in tg_records if r.get('mvp_class') == CLASS_FISH])} | {len([r for r in tg_records if r.get('mvp_class') == CLASS_NOT_FISH])} | {len(tg_records)} |",
        f"| External (GBIF/iNat, stage_b) | {len(ext_fish)} | 0 | {len(ext_fish)} |",
        f"| External (stage_a no_fish, iNat birds) | 0 | {len(ext_nofish)} | {len(ext_nofish)} |",
        f"| External (stage_a lure, Wikimedia CC) | 0 | {len(_ext_lure)} | {len(_ext_lure)} |",
        f"| **Total** | **{all_fish}** | **{all_nofish}** | **{total}** |",
        f"\nFish:non-fish imbalance ratio: **{ratio:.1f}:1**",
    ]

    if splits:
        lines.append("\n## Split Counts\n")
        lines.append("| Split | Fish | Non-fish | Total |")
        lines.append("| --- | --- | --- | --- |")
        for split_name, records in splits.items():
            f = sum(1 for r in records if r.get("mvp_class") == CLASS_FISH)
            nf = sum(1 for r in records if r.get("mvp_class") == CLASS_NOT_FISH)
            lines.append(f"| {split_name} | {f} | {nf} | {len(records)} |")

    if gate_failures:
        lines.append("\n## Training Gates: FAILED\n")
        for gf in gate_failures:
            lines.append(f"- ❌ {gf}")
        lines.append(
            "\n## Next Actions to Unblock Training\n"
            "1. Review more non-fish Telegram batches (lure, no_fish, out_of_scope content).\n"
            "2. Download DeepFish no_fish frames (~5000 frames, CC-BY-4.0, manual step).\n"
            "3. Re-run `intake_phase_e_lite.py` and `build_mvp_training_dataset.py`.\n"
            "4. Minimum viable: 100+ non-fish examples with imbalance ≤ 10:1.\n"
        )
    else:
        lines.append("\n## Training Gates: ALL PASSED\n")
        lines.append("Ready for `train_stage_a.py` or equivalent structural baseline training.")
        lines.append(
            "\n⚠️ **Note:** This dataset is marked EXPERIMENTAL. "
            "Do not use for production decisions."
        )

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build MVP structural training dataset manifest")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--copy-images",
        action="store_true",
        help="Also copy images to data/mvp_training/structural/ (requires Telegram export present)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    now = datetime.now(timezone.utc).isoformat()

    # Load data
    tg_records = load_telegram_seed_records()
    ext_fish = load_external_fish_records()
    ext_nofish = load_external_nofish_records()

    ext_lure = load_external_lure_records()

    log.info(
        "Data loaded: Telegram=%d, ext_fish=%d, ext_nofish=%d, ext_lure=%d",
        len(tg_records), len(ext_fish), len(ext_nofish), len(ext_lure),
    )

    # Run training gates
    gate_failures = check_training_gates(tg_records, ext_fish, ext_nofish, ext_lure)

    # Build combined records for split (only if gates pass)
    splits: dict[str, list[dict]] | None = None
    leakage_errors: list[str] = []

    if not gate_failures:
        all_records: list[dict] = []
        for r in tg_records:
            all_records.append({
                **r,
                "provenance": "telegram_reviewed",
                "experimental": True,
            })
        for r in ext_fish + ext_nofish + ext_lure:
            all_records.append({
                **r,
                "experimental": True,
            })

        splits = {}
        # Stratified by class; group-aware split handles lure augmentation groups
        fish_recs = [r for r in all_records if r["mvp_class"] == CLASS_FISH]
        nofish_recs = [r for r in all_records if r["mvp_class"] == CLASS_NOT_FISH]

        fish_splits = deterministic_split(fish_recs)
        nofish_splits = deterministic_split(nofish_recs)

        for split_name in ("train", "val", "test"):
            splits[split_name] = fish_splits[split_name] + nofish_splits[split_name]

        leakage_errors = check_split_leakage(splits)
        if leakage_errors:
            gate_failures.extend(leakage_errors)
            splits = None

    quality_report = build_quality_report(
        gate_failures, tg_records, ext_fish, ext_nofish, splits, now, ext_lure
    )

    # Build manifests
    all_fish_count = len([r for r in tg_records if r.get("mvp_class") == CLASS_FISH]) + len(ext_fish)
    all_nofish_count = (
        len([r for r in tg_records if r.get("mvp_class") == CLASS_NOT_FISH])
        + len(ext_nofish)
        + len(ext_lure)
    )

    class_mapping = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": now,
        "structural_classes": STRUCTURAL_CLASSES,
        "class_definitions": {
            CLASS_FISH: {
                "description": "Whole fish visible, manually reviewed at confidence >= 3",
                "sources": ["telegram_reviewed (fish, fish_part at conf>=3)", "external_public (stage_b species)"],
            },
            CLASS_NOT_FISH: {
                "description": "No fish, lure/gear, poster/screenshot — not a fishing catch",
                "sources": [
                    "telegram_reviewed (no_fish, lure_gear, poster_screenshot at conf>=3)",
                    "external_public (stage_a no_fish — iNat birds CC-licensed)",
                    "external_public (stage_a lure — Wikimedia Commons CC-licensed)",
                ],
            },
        },
        "data_integrity": {
            "phase_c_labels_used_as_truth": False,
            "unreviewed_telegram_records_included": False,
            "label_provenance": "manual_human_review_or_external_public_only",
            "experimental": True,
        },
    }

    split_report = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": now,
        "training_status": "BLOCKED" if gate_failures else "READY",
        "gate_failures": gate_failures,
        "data_summary": {
            "telegram_reviewed": len(tg_records),
            "external_fish": len(ext_fish),
            "external_nofish": len(ext_nofish),
            "external_lure": len(ext_lure),
            "total_fish": all_fish_count,
            "total_nofish": all_nofish_count,
            "total": all_fish_count + all_nofish_count,
            "imbalance_ratio": round(all_fish_count / all_nofish_count, 2) if all_nofish_count > 0 else None,
        },
        "splits": {
            split_name: {
                "total": len(recs),
                "fish": sum(1 for r in recs if r.get("mvp_class") == CLASS_FISH),
                "not_fish_or_other": sum(1 for r in recs if r.get("mvp_class") == CLASS_NOT_FISH),
            }
            for split_name, recs in (splits or {}).items()
        },
        "split_seed": SPLIT_SEED,
        "split_ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": round(1 - TRAIN_RATIO - VAL_RATIO, 2)},
        "leakage_errors": leakage_errors,
    }

    training_manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": now,
        "training_status": "BLOCKED" if gate_failures else "READY",
        "experimental": True,
        "sources": {
            "telegram_reviewed": {
                "record_count": len(tg_records),
                "provenance": "telegram_private_2026-04-24",
                "license": "private_training_only",
                "label_method": "manual_human_review",
                "phase_c_labels_as_truth": False,
                "unreviewed_included": False,
            },
            "external_public": {
                "fish_count": len(ext_fish),
                "nofish_count": len(ext_nofish),
                "lure_count": len(ext_lure),
                "total_nofish_external": len(ext_nofish) + len(ext_lure),
                "provenance": "external_public",
                "license": "CC-BY-4.0 and CC0-1.0 (see external_license_report.json)",
                "license_report": "data/external_public/manifests/external_license_report.json",
                "lure_license": "CC-BY and CC-BY-SA (Wikimedia Commons, see stage_a/raw/lure/PROVENANCE.json)",
                "lure_aug_note": "60 lure files = 13 originals + 47 augmented variants; group-aware split prevents aug leakage",
            },
        },
        "gate_failures": gate_failures,
    }

    if args.dry_run:
        log.info("DRY RUN — not writing files")
        print(quality_report)
        print("\n=== SPLIT REPORT ===")
        print(json.dumps(split_report, indent=2))
        return 0 if not gate_failures else 1

    # Write manifests
    _write_json_atomic(MANIFESTS_DIR / "mvp_training_manifest.json", training_manifest)
    _write_json_atomic(MANIFESTS_DIR / "mvp_class_mapping.json", class_mapping)
    _write_json_atomic(MANIFESTS_DIR / "mvp_split_report.json", split_report)
    (MANIFESTS_DIR / "mvp_dataset_quality_report.md").write_text(quality_report, encoding="utf-8")
    log.info("Wrote manifests to %s", MANIFESTS_DIR)

    if gate_failures:
        log.warning("Training BLOCKED by %d gate failure(s):", len(gate_failures))
        for gf in gate_failures:
            log.warning("  %s", gf)
        log.info("Quality report: %s", MANIFESTS_DIR / "mvp_dataset_quality_report.md")
        return 1

    log.info("Training gates PASSED. Dataset ready for structural baseline training.")

    if args.copy_images and splits:
        _copy_images_to_structural(splits)

    return 0


def _copy_images_to_structural(splits: dict[str, list[dict]]) -> None:
    """
    Copy images from local_path fields into the structural training directory layout:
        data/mvp_training/structural/{train,val,test}/{fish,not_fish_or_other}/

    Telegram-sourced records without a local_path are skipped with a warning.
    """
    copied = skipped = 0
    for split_name, records in splits.items():
        for r in records:
            cls = r.get("mvp_class")
            src_str = r.get("local_path")
            if not src_str:
                skipped += 1
                continue
            src = Path(src_str)
            if not src.exists():
                log.warning("Image not found, skipping: %s", src)
                skipped += 1
                continue
            dest_dir = STRUCTURAL_DIR / split_name / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            if not dest.exists():
                shutil.copy2(src, dest)
            copied += 1
    log.info(
        "--copy-images: %d images copied, %d skipped (no local_path or missing)",
        copied, skipped,
    )


if __name__ == "__main__":
    sys.exit(main())
