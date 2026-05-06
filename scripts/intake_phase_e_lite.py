"""
intake_phase_e_lite.py — U4 Phase E Lite: Reviewed-only seed materialization.

Reads validated manual-review decision batches and produces the privacy-safe
Telegram gold seed. ONLY manually reviewed batches are included. Unreviewed
records are explicitly tracked as unreviewed_not_eligible and never enter the seed.

MVP class mapping:
  fish                   → fish
  fish_part              → fish          (positive, structural)
  fry_juvenile           → needs_human_review (too few, ambiguous for structural)
  no_fish                → not_fish_or_other
  lure_gear              → not_fish_or_other
  poster_screenshot      → not_fish_or_other
  bad_quality            → needs_human_review (excluded from training)
  out_of_scope           → needs_human_review (excluded from training)
  duplicate_suspect      → needs_human_review (excluded from training)
  unsure                 → needs_human_review

Low-confidence override:
  confidence < 3         → needs_human_review (regardless of category)

Outputs:
  reviewed_seed/reviewed_seed_manifest.json  — tracked; provenance, counts
  reviewed_seed/reviewed_seed_summary.json   — tracked; class-breakdown counts only
  reviewed_seed/reviewed_seed_mvp_readiness.md — tracked; readiness assessment
  reviewed_seed/reviewed_seed_records.jsonl  — gitignored; review_id + class
  reviewed_seed/reviewed_seed_excluded.jsonl — gitignored; excluded records + reasons

Usage:
    python3 scripts/intake_phase_e_lite.py
    python3 scripts/intake_phase_e_lite.py --run-id rvrun_20260427T184629Z
    python3 scripts/intake_phase_e_lite.py --dry-run

source=telegram_private_2026-04-24, license=private_training_only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intake_constants as C

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr)
log = logging.getLogger(__name__)

# ─── Schema version ──────────────────────────────────────────────────────────

SEED_SCHEMA_VERSION = "u4_phase_e_seed_v1"
SEED_SUMMARY_SCHEMA_VERSION = "u4_phase_e_seed_summary_v1"

# ─── MVP class mapping ────────────────────────────────────────────────────────

# Threshold: confidence < MVP_CONFIDENCE_MIN → needs_human_review
MVP_CONFIDENCE_MIN: int = 3

# Map final_category → mvp_class
_CATEGORY_TO_MVP: dict[str, str] = {
    C.FINAL_CATEGORY_FISH:              "fish",
    C.FINAL_CATEGORY_FISH_PART:         "fish",
    C.FINAL_CATEGORY_FRY_JUVENILE:      "needs_human_review",
    C.FINAL_CATEGORY_NO_FISH:           "not_fish_or_other",
    C.FINAL_CATEGORY_LURE_GEAR:         "not_fish_or_other",
    C.FINAL_CATEGORY_POSTER_SCREENSHOT: "not_fish_or_other",
    C.FINAL_CATEGORY_BAD_QUALITY:       "needs_human_review",
    C.FINAL_CATEGORY_OUT_OF_SCOPE:      "needs_human_review",
    C.FINAL_CATEGORY_DUPLICATE_SUSPECT: "needs_human_review",
    C.FINAL_CATEGORY_UNSURE:            "needs_human_review",
}

MVP_CLASSES: list[str] = ["fish", "not_fish_or_other", "needs_human_review"]

# Training-usable MVP classes (not "needs_human_review")
TRAINING_MVP_CLASSES: frozenset[str] = frozenset({"fish", "not_fish_or_other"})

# Private fields that must never appear in tracked summaries
_DISALLOWED_TRACKED_FIELDS = frozenset({
    "review_id", "filename", "sha256", "caption", "sender",
    "source_path", "exif", "notes", "refinement",
})


# ─── I/O helpers ─────────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── Decision file discovery ─────────────────────────────────────────────────


def _is_blank_template(records: list[dict]) -> bool:
    return all(r.get("decision_type") is None for r in records)


def find_reviewed_batches(run_id: str, review_dir: Path) -> list[tuple[str, Path]]:
    """Return (batch_id, path) pairs for all non-blank decision files in run_id order."""
    prefix = f"filter_decisions_{run_id}_"
    results = []
    for path in sorted(review_dir.glob(f"{prefix}????.json")):
        try:
            data = _load_json(path)
        except Exception as exc:
            log.warning("Cannot load %s: %s", path.name, exc)
            continue
        records = data.get("records", [])
        if not records or _is_blank_template(records):
            continue
        batch_id = data.get("batch_id", path.stem.split("_")[-1])
        results.append((str(batch_id), path))
    return results


# ─── MVP class assignment ─────────────────────────────────────────────────────


def assign_mvp_class(final_category: str, human_confidence: int | None) -> str:
    """Assign an MVP structural class, downgrading low-confidence records."""
    conf = human_confidence if isinstance(human_confidence, int) else 0
    if conf < MVP_CONFIDENCE_MIN:
        return "needs_human_review"
    return _CATEGORY_TO_MVP.get(final_category, "needs_human_review")


# ─── Record processing ────────────────────────────────────────────────────────


def process_batch(batch_path: Path) -> tuple[list[dict], list[dict]]:
    """
    Read a decision batch and return (seed_records, excluded_records).

    seed_records: review_id, final_category, human_confidence, mvp_class, batch_id
    excluded_records: review_id, final_category, human_confidence, mvp_class=needs_human_review,
                      batch_id, exclude_reason
    """
    data = _load_json(batch_path)
    batch_id = data.get("batch_id", "unknown")
    seed: list[dict] = []
    excluded: list[dict] = []

    for rec in data.get("records", []):
        review_id = rec.get("review_id", "")
        dt = rec.get("decision_type")
        fc = rec.get("final_category", "")
        conf = rec.get("human_confidence")

        if not review_id:
            log.warning("Record missing review_id in batch %s — skipping", batch_id)
            continue

        if dt is None:
            log.warning("Blank record %s in batch %s — skipping", review_id[:8], batch_id)
            continue

        mvp_class = assign_mvp_class(fc, conf)

        entry = {
            "review_id": review_id,
            "final_category": fc,
            "human_confidence": conf,
            "mvp_class": mvp_class,
            "batch_id": str(batch_id),
        }

        if mvp_class == "needs_human_review":
            reasons = []
            if isinstance(conf, int) and conf < MVP_CONFIDENCE_MIN:
                reasons.append(f"low_confidence_{conf}")
            if fc in C.TRAINING_INELIGIBLE_CATEGORIES or fc == C.FINAL_CATEGORY_FRY_JUVENILE:
                reasons.append(f"ineligible_category_{fc}")
            if dt == C.DECISION_TYPE_UNSURE:
                reasons.append("decision_unsure")
            entry["exclude_reason"] = "|".join(reasons) if reasons else "mapped_to_needs_review"
            excluded.append(entry)
        else:
            seed.append(entry)

    return seed, excluded


# ─── Summary and manifest building ───────────────────────────────────────────


def _assert_summary_privacy(summary: dict) -> None:
    """Raise if any disallowed private field appears anywhere in summary."""
    summary_str = json.dumps(summary)
    for field in _DISALLOWED_TRACKED_FIELDS:
        if f'"{field}"' in summary_str:
            raise ValueError(f"Privacy violation: disallowed field '{field}' in tracked summary")


def build_summary(
    seed: list[dict],
    excluded: list[dict],
    unreviewed_not_eligible: int,
    run_id: str,
    batch_ids: list[str],
    now: str,
) -> dict:
    mvp_counts: dict[str, int] = Counter(r["mvp_class"] for r in seed)  # type: ignore[assignment]
    for cls in MVP_CLASSES:
        mvp_counts.setdefault(cls, 0)

    excluded_reasons: dict[str, int] = {}
    for r in excluded:
        for reason in r.get("exclude_reason", "").split("|"):
            if reason:
                excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1

    fish_count = mvp_counts.get("fish", 0)
    not_fish_count = mvp_counts.get("not_fish_or_other", 0)
    total_training_usable = fish_count + not_fish_count

    imbalance_ratio: float | None = None
    if not_fish_count > 0:
        imbalance_ratio = round(fish_count / not_fish_count, 2)

    summary = {
        "schema_version": SEED_SUMMARY_SCHEMA_VERSION,
        "source": C.SOURCE_TAG,
        "run_id": run_id,
        "phase": "U4_PHASE_E_LITE",
        "generated_at": now,
        "batches_included": sorted(batch_ids),
        "counts": {
            "total_reviewed_in_seed": len(seed),
            "total_excluded": len(excluded),
            "unreviewed_not_eligible": unreviewed_not_eligible,
        },
        "mvp_class_counts": {k: mvp_counts.get(k, 0) for k in MVP_CLASSES},
        "training_usable": {
            "total": total_training_usable,
            "fish": fish_count,
            "not_fish_or_other": not_fish_count,
            "imbalance_ratio_fish_to_not_fish": imbalance_ratio,
        },
        "excluded_reason_counts": excluded_reasons,
        "privacy_status": {
            "contains_review_ids": False,
            "contains_filenames": False,
            "contains_captions": False,
            "counts_only": True,
        },
    }
    _assert_summary_privacy(summary)
    return summary


def build_manifest(
    seed: list[dict],
    excluded: list[dict],
    unreviewed_not_eligible: int,
    run_id: str,
    batch_ids: list[str],
    now: str,
) -> dict:
    fish_count = sum(1 for r in seed if r["mvp_class"] == "fish")
    not_fish_count = sum(1 for r in seed if r["mvp_class"] == "not_fish_or_other")

    manifest = {
        "schema_version": SEED_SCHEMA_VERSION,
        "source": C.SOURCE_TAG,
        "license": C.LICENSE_TAG,
        "run_id": run_id,
        "phase": "U4_PHASE_E_LITE",
        "generated_at": now,
        "batches_included": sorted(batch_ids),
        "record_counts": {
            "seed_total": len(seed),
            "excluded_total": len(excluded),
            "unreviewed_not_eligible": unreviewed_not_eligible,
        },
        "mvp_class_mapping": _CATEGORY_TO_MVP,
        "mvp_confidence_min": MVP_CONFIDENCE_MIN,
        "mvp_class_counts": {
            cls: sum(1 for r in seed if r["mvp_class"] == cls) for cls in MVP_CLASSES
        },
        "training_usable_counts": {
            "fish": fish_count,
            "not_fish_or_other": not_fish_count,
            "total": fish_count + not_fish_count,
        },
        "data_integrity": {
            "source_phase_c_labels_used_as_truth": False,
            "unreviewed_records_included": False,
            "label_provenance": "manual_human_review_only",
        },
    }
    return manifest


# ─── Readiness report ─────────────────────────────────────────────────────────


_IMBALANCE_WARNING_RATIO = 5.0   # fish:non-fish > 5:1 → warn
_IMBALANCE_BLOCK_RATIO = 10.0    # fish:non-fish > 10:1 → block
_MIN_NON_FISH_FOR_TRAINING = 20  # absolute minimum non-fish examples


def build_readiness_report(
    seed: list[dict],
    excluded: list[dict],
    unreviewed_not_eligible: int,
    now: str,
) -> str:
    fish_count = sum(1 for r in seed if r["mvp_class"] == "fish")
    not_fish_count = sum(1 for r in seed if r["mvp_class"] == "not_fish_or_other")
    needs_review_count = len(excluded)
    total_training_usable = fish_count + not_fish_count

    ratio = fish_count / not_fish_count if not_fish_count > 0 else float("inf")

    enough_for_structural = (
        total_training_usable >= 50
        and not_fish_count >= _MIN_NON_FISH_FOR_TRAINING
        and ratio <= _IMBALANCE_BLOCK_RATIO
    )

    status = "BLOCKED" if not enough_for_structural else (
        "CAUTION" if ratio > _IMBALANCE_WARNING_RATIO else "READY"
    )

    lines = [
        "# Reviewed Seed MVP Readiness Report",
        f"\nGenerated: {now}",
        f"\nSource: {C.SOURCE_TAG}",
        "\n## Summary",
        f"\n| Metric | Value |",
        "| --- | --- |",
        f"| Reviewed records in seed | {len(seed)} |",
        f"| Excluded (needs_human_review) | {needs_review_count} |",
        f"| Unreviewed (not eligible) | {unreviewed_not_eligible} |",
        f"| Training-usable total | {total_training_usable} |",
        f"| MVP class: fish | {fish_count} |",
        f"| MVP class: not_fish_or_other | {not_fish_count} |",
        f"| Fish:non-fish ratio | {ratio:.1f}:1 |",
        f"\n## Structural MVP Status: **{status}**",
    ]

    if status == "BLOCKED":
        lines.append("\n### Why blocked")
        if not_fish_count < _MIN_NON_FISH_FOR_TRAINING:
            lines.append(
                f"- Not enough non-fish examples: {not_fish_count} < {_MIN_NON_FISH_FOR_TRAINING} minimum"
            )
        if ratio > _IMBALANCE_BLOCK_RATIO:
            lines.append(
                f"- Class imbalance too severe: {ratio:.1f}:1 fish-to-non-fish (limit {_IMBALANCE_BLOCK_RATIO:.0f}:1)"
            )
        if total_training_usable < 50:
            lines.append(f"- Total training-usable too small: {total_training_usable} < 50")
        lines.append(
            "\n### Next action to unblock\n"
            "1. Review more batches with non-fish content (batches 0003–0130 contain ~31,920 unreviewed records).\n"
            "2. OR ingest external public negatives (DeepFish no_fish class, GBIF background images).\n"
            "3. Minimum target: 100+ non-fish at confidence >= 3 before re-running this script."
        )
    elif status == "CAUTION":
        lines.append(
            f"\nClass imbalance is moderate ({ratio:.1f}:1). Training may bias toward fish class. "
            "Consider reviewing more non-fish batches or adding external negatives before training."
        )
    else:
        lines.append("\nData is sufficient for a conservative structural MVP baseline.")

    lines.append("\n## Class Mapping Used\n")
    lines.append("| final_category | mvp_class | min_confidence |")
    lines.append("| --- | --- | --- |")
    for cat, mvc in sorted(_CATEGORY_TO_MVP.items()):
        lines.append(f"| {cat} | {mvc} | {MVP_CONFIDENCE_MIN} |")

    lines.append(
        "\n## Recommended Actions\n"
        "1. Review 10+ more batches to increase non-fish coverage.\n"
        "2. Run `intake_external_lite.py` to record license-safe external sources.\n"
        "3. Re-run `intake_phase_e_lite.py` after each batch review session.\n"
        "4. Only run `build_mvp_training_dataset.py` when status is READY or CAUTION.\n"
    )
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="U4 Phase E Lite: materialize reviewed-only seed"
    )
    parser.add_argument(
        "--run-id",
        default="rvrun_20260427T184629Z",
        help="Review run ID (default: rvrun_20260427T184629Z)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary without writing any files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run_id = args.run_id
    dry_run = args.dry_run

    if dry_run:
        log.info("DRY RUN — no files will be written")

    review_dir = C.FILTER_REVIEW_DIR
    if not review_dir.exists():
        log.error("Review directory not found: %s", review_dir)
        return 1

    # Load partial summary for unreviewed count
    partial_summary_path = C.FILTER_REVIEW_PARTIAL_SUMMARY_PATH
    if not partial_summary_path.exists():
        log.error(
            "Partial summary not found: %s  — run intake_review_aggregate.py --partial first",
            partial_summary_path,
        )
        return 1

    partial = _load_json(partial_summary_path)
    unreviewed_not_eligible: int = partial.get("progress", {}).get("unreviewed_records", 0)

    # Discover reviewed batches
    batch_pairs = find_reviewed_batches(run_id, review_dir)
    if not batch_pairs:
        log.error("No reviewed (non-blank) decision files found for run_id=%s in %s", run_id, review_dir)
        return 1

    log.info("Found %d reviewed batch(es): %s", len(batch_pairs), [b for b, _ in batch_pairs])

    # Process all reviewed batches
    all_seed: list[dict] = []
    all_excluded: list[dict] = []
    batch_ids: list[str] = []

    for batch_id, batch_path in batch_pairs:
        seed, excluded = process_batch(batch_path)
        all_seed.extend(seed)
        all_excluded.extend(excluded)
        batch_ids.append(batch_id)
        log.info(
            "  Batch %s: %d seed records, %d excluded", batch_id, len(seed), len(excluded)
        )

    now = datetime.now(timezone.utc).isoformat()

    summary = build_summary(all_seed, all_excluded, unreviewed_not_eligible, run_id, batch_ids, now)
    manifest = build_manifest(all_seed, all_excluded, unreviewed_not_eligible, run_id, batch_ids, now)
    readiness = build_readiness_report(all_seed, all_excluded, unreviewed_not_eligible, now)

    # Log summary
    log.info(
        "Seed: %d records (%d fish, %d not_fish_or_other, %d needs_human_review), %d excluded",
        len(all_seed),
        summary["mvp_class_counts"]["fish"],
        summary["mvp_class_counts"]["not_fish_or_other"],
        summary["mvp_class_counts"]["needs_human_review"],
        len(all_excluded),
    )

    if dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    # Write outputs
    seed_dir = C.REVIEWED_SEED_DIR
    _write_json_atomic(C.REVIEWED_SEED_MANIFEST_PATH, manifest)
    _write_json_atomic(C.REVIEWED_SEED_SUMMARY_PATH, summary)
    _write_jsonl_atomic(C.REVIEWED_SEED_RECORDS_PATH, all_seed)
    _write_jsonl_atomic(C.REVIEWED_SEED_EXCLUDED_PATH, all_excluded)
    (seed_dir / "reviewed_seed_mvp_readiness.md").write_text(readiness, encoding="utf-8")

    log.info("Wrote: %s", C.REVIEWED_SEED_MANIFEST_PATH)
    log.info("Wrote: %s", C.REVIEWED_SEED_SUMMARY_PATH)
    log.info("Wrote: %s (gitignored)", C.REVIEWED_SEED_RECORDS_PATH)
    log.info("Wrote: %s (gitignored)", C.REVIEWED_SEED_EXCLUDED_PATH)
    log.info("Wrote: %s", C.REVIEWED_SEED_READINESS_PATH)
    log.info("Phase E Lite DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
