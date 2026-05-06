"""
intake_review_aggregate.py — U4 Phase D: Aggregate completed decisions.

Requires a successful validation pass (--require-validation-pass).

Reads the local manifest + all decision files and produces:
  - Local aggregate: review/filter_decisions_aggregate_<run_id>.json
  - Tracked summary: data/intake_meta/tg_2026-04-24/filter_review_summary.json
    (counts only — no review IDs, filenames, paths, captions, sender metadata)

SAFETY CHECKS:
  - Reviewed count must equal expected count (no missing decisions for PASS).
  - Invalid decisions must be 0 for PASS.
  - Conflict records must be explicitly accounted for.
  - Tracked summary must contain counts only.

source=telegram_private_2026-04-24, license=private_training_only
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intake_constants as C
from intake_review_validate import (
    load_manifest,
    load_decision_files,
    validate,
    validate_partial,
    BATCH_STATUS_REVIEWED_VALID,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr)
log = logging.getLogger(__name__)


# ─── I/O helpers ─────────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ─── Aggregation ─────────────────────────────────────────────────────────────


def aggregate_decisions(
    manifest: dict,
    decision_files: list[tuple[Path, dict]],
    validation_report: dict,
    require_validation_pass: bool,
) -> tuple[dict, dict]:
    """
    Aggregate all validated decisions.
    Returns (local_aggregate, tracked_summary).
    """
    run_id = manifest["run_id"]
    source = manifest.get("source", C.SOURCE_TAG)
    now = datetime.now(timezone.utc).isoformat()

    if require_validation_pass and not validation_report["passed"]:
        errors = validation_report.get("errors", [])
        raise ValueError(
            f"Validation FAILED with {len(errors)} error(s). "
            f"Run validate first and fix all errors before aggregating. "
            f"First error: {errors[0] if errors else '(none)'}"
        )

    # Collect all records
    all_records: list[dict] = []
    for _path, data in decision_files:
        all_records.extend(data.get("records", []))

    # Count aggregates
    decision_type_counts: dict[str, int] = {dt: 0 for dt in C.DECISION_TYPES}
    final_category_counts: dict[str, int] = {fc: 0 for fc in C.FINAL_CATEGORIES}
    confidence_counts: dict[str, int] = {str(i): 0 for i in range(C.CONFIDENCE_MIN, C.CONFIDENCE_MAX + 1)}
    training_eligible_count = 0  # confidence >= 4, not UNSURE
    conflict_resolved_count = 0
    conflict_unresolved_count = 0  # UNSURE on a conflict record
    unsure_count = 0

    for rec in all_records:
        dt = rec.get("decision_type", "")
        fc = rec.get("final_category", "")
        conf = rec.get("human_confidence")

        decision_type_counts[dt] = decision_type_counts.get(dt, 0) + 1
        final_category_counts[fc] = final_category_counts.get(fc, 0) + 1

        if isinstance(conf, int):
            key = str(conf)
            confidence_counts[key] = confidence_counts.get(key, 0) + 1
            if conf >= 4 and dt != C.DECISION_TYPE_UNSURE:
                training_eligible_count += 1

        if dt == C.DECISION_TYPE_UNSURE:
            unsure_count += 1

    # Local aggregate (per-record, includes review_ids — local only)
    local_aggregate: dict = {
        "schema_version": C.REVIEW_SCHEMA_VERSION,
        "run_id": run_id,
        "source": source,
        "phase": C.REVIEW_PHASE,
        "aggregated_at": now,
        "validation": {
            "passed": validation_report["passed"],
            "expected_count": validation_report["expected_count"],
            "reviewed_count": validation_report["reviewed_count"],
            "missing_count": validation_report["missing_count"],
            "invalid_count": validation_report["invalid_decisions"],
        },
        "summary": {
            "total_reviewed": len(all_records),
            "decision_type_counts": decision_type_counts,
            "final_category_counts": final_category_counts,
            "confidence_counts": confidence_counts,
            "training_eligible_count": training_eligible_count,
            "unsure_count": unsure_count,
        },
        "records": all_records,  # full per-record data (local-only)
    }

    # Tracked summary (counts only — no review IDs, filenames, captions, sender data)
    tracked_summary: dict = {
        "schema_version": C.REVIEW_SUMMARY_SCHEMA_VERSION,
        "source": source,
        "phase": C.REVIEW_PHASE,
        "run_id": run_id,
        "aggregated_at": now,
        "input_counts": {
            "expected_records": validation_report["expected_count"],
        },
        "reviewed_counts": {
            "total_reviewed": len(all_records),
            "missing_decision_count": validation_report["missing_count"],
            "duplicate_decision_count": 0,  # duplicates caught in validation
            "invalid_decision_count": validation_report["invalid_decisions"],
            "unknown_id_count": validation_report["unknown_count"],
        },
        "decision_type_counts": decision_type_counts,
        "final_category_counts": final_category_counts,
        "confidence_counts": confidence_counts,
        "conflict_resolution_counts": {
            "training_eligible_count": training_eligible_count,
            "unsure_count": unsure_count,
        },
        "privacy_status": {
            "contains_review_ids": False,
            "contains_filenames": False,
            "contains_paths": False,
            "contains_captions": False,
            "contains_sender_metadata": False,
            "contains_telegram_metadata": False,
        },
        "validation_status": {
            "passed": validation_report["passed"],
            "error_count": validation_report["error_count"],
            "errors": validation_report["errors"] if not validation_report["passed"] else [],
        },
    }

    return local_aggregate, tracked_summary


def _compute_training_buckets(all_records: list[dict]) -> dict:
    """
    Compute training eligibility buckets for reviewed records.

    Buckets:
      - training_eligible_count: conf>=4, not UNSURE, not in TRAINING_INELIGIBLE_CATEGORIES
      - reviewed_high_confidence_fish: eligible + final_category in TRAINING_FISH_CATEGORIES
      - reviewed_high_confidence_non_fish_or_excluded: conf>=4 but NOT eligible for training
          (either UNSURE, ineligible category like out_of_scope/bad_quality, OR eligible
           non-fish like lure_gear/no_fish). The name "_or_excluded" signals that this
           bucket is NOT the complement of high_confidence_fish within training_eligible_count.
      - reviewed_low_confidence: conf < 4 (regardless of category)

    IMPORTANT non-additivity invariant:
      high_conf_fish + high_conf_non_fish_or_excluded != training_eligible_count
      when any ineligible categories (e.g. out_of_scope, bad_quality) are present at conf>=4,
      because those records appear in non_fish_or_excluded but NOT in training_eligible_count.
      Callers must NOT sum the two high-confidence buckets to derive eligibility.
    """
    eligible = 0
    high_conf_fish = 0
    high_conf_non_fish = 0
    low_conf = 0

    for rec in all_records:
        dt = rec.get("decision_type", "")
        fc = rec.get("final_category", "")
        conf = rec.get("human_confidence")
        if not isinstance(conf, int):
            conf = 0

        if conf < C.CONFIDENCE_MIN + 3:  # conf < 4
            low_conf += 1
            continue

        # conf >= 4
        if dt == C.DECISION_TYPE_UNSURE or fc in C.TRAINING_INELIGIBLE_CATEGORIES:
            # Ineligible at high confidence: excluded category or UNSURE decision
            high_conf_non_fish += 1
            continue

        eligible += 1
        if fc in C.TRAINING_FISH_CATEGORIES:
            high_conf_fish += 1
        else:
            # Eligible non-fish (e.g. lure_gear, no_fish, poster_screenshot)
            high_conf_non_fish += 1

    return {
        "training_eligible_count": eligible,
        "reviewed_high_confidence_fish": high_conf_fish,
        "reviewed_high_confidence_non_fish_or_excluded": high_conf_non_fish,
        "reviewed_low_confidence": low_conf,
    }


def aggregate_partial(
    manifest: dict,
    review_dir: Path,
    run_id: str,
) -> tuple[dict, dict]:
    """
    Partial aggregation: aggregate only reviewed_valid batches.
    Unreviewed records stay explicitly unreviewed — never training-ready.
    Returns (partial_aggregate_local, partial_summary_tracked).
    The partial_aggregate_local contains per-record data (local-only).
    The partial_summary_tracked contains counts only (privacy-safe, committable).
    """
    partial_report = validate_partial(manifest, review_dir, run_id)

    if partial_report["reviewed_invalid"] > 0:
        invalid = partial_report["invalid_batches"]
        raise ValueError(
            f"Partial aggregation refused: {len(invalid)} reviewed batch(es) failed validation: "
            f"{invalid}. Fix validation errors before aggregating."
        )

    # Load only reviewed_valid batch files
    source = manifest.get("source", C.SOURCE_TAG)
    now = datetime.now(timezone.utc).isoformat()
    all_records: list[dict] = []

    for batch in manifest.get("batches", []):
        batch_id = str(batch.get("batch_id", ""))
        if batch_id not in partial_report["reviewed_valid_batches"]:
            continue
        decision_file = batch.get("decision_file", f"filter_decisions_{run_id}_{batch_id}.json")
        path = review_dir / decision_file
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        all_records.extend(data.get("records", []))

    # Aggregate counts
    decision_type_counts: dict[str, int] = {dt: 0 for dt in C.DECISION_TYPES}
    final_category_counts: dict[str, int] = {fc: 0 for fc in C.FINAL_CATEGORIES}
    confidence_counts: dict[str, int] = {str(i): 0 for i in range(C.CONFIDENCE_MIN, C.CONFIDENCE_MAX + 1)}

    for rec in all_records:
        dt = rec.get("decision_type", "")
        fc = rec.get("final_category", "")
        conf = rec.get("human_confidence")
        decision_type_counts[dt] = decision_type_counts.get(dt, 0) + 1
        final_category_counts[fc] = final_category_counts.get(fc, 0) + 1
        if isinstance(conf, int):
            key = str(conf)
            confidence_counts[key] = confidence_counts.get(key, 0) + 1

    training_buckets = _compute_training_buckets(all_records)
    unreviewed_count = partial_report["unreviewed_records"]

    # Local partial aggregate (per-record, local-only)
    local_aggregate: dict = {
        "schema_version": C.REVIEW_SCHEMA_VERSION,
        "run_id": run_id,
        "source": source,
        "phase": C.REVIEW_PHASE,
        "aggregated_at": now,
        "mode": "partial",
        "progress": {
            "total_batches": partial_report["total_batches"],
            "reviewed_valid": partial_report["reviewed_valid"],
            "blank_unreviewed": partial_report["blank_unreviewed"],
            "reviewed_records": partial_report["reviewed_records"],
            "unreviewed_records": unreviewed_count,
        },
        "summary": {
            "total_reviewed": len(all_records),
            "decision_type_counts": decision_type_counts,
            "final_category_counts": final_category_counts,
            "confidence_counts": confidence_counts,
            **training_buckets,
            "unreviewed_not_eligible": unreviewed_count,
        },
        "records": all_records,  # full per-record data (local-only)
    }

    # Tracked partial summary (counts only — no review IDs, filenames, etc.)
    tracked_summary: dict = {
        "schema_version": C.REVIEW_SUMMARY_SCHEMA_VERSION,
        "source": source,
        "phase": C.REVIEW_PHASE,
        "run_id": run_id,
        "mode": "partial",
        "aggregated_at": now,
        "progress": {
            "total_batches": partial_report["total_batches"],
            "reviewed_valid_batches": partial_report["reviewed_valid"],
            "blank_unreviewed_batches": partial_report["blank_unreviewed"],
            "reviewed_records": partial_report["reviewed_records"],
            "unreviewed_records": unreviewed_count,
            "next_unreviewed_batches": partial_report["next_unreviewed_batches"],
        },
        "decision_type_counts": decision_type_counts,
        "final_category_counts": final_category_counts,
        "confidence_counts": confidence_counts,
        "training_eligibility": {
            **training_buckets,
            "unreviewed_not_eligible": unreviewed_count,
        },
        "privacy_status": {
            "contains_review_ids": False,
            "contains_filenames": False,
            "contains_paths": False,
            "contains_captions": False,
            "contains_sender_metadata": False,
            "contains_telegram_metadata": False,
        },
    }

    return local_aggregate, tracked_summary


def _assert_summary_privacy(summary: dict) -> None:
    """Verify tracked summary contains no private data."""
    import re
    forbidden_patterns = [
        re.compile(r'\brv_[0-9a-f]{16}\b'),       # review IDs (16 hex chars after rv_)
        re.compile(r"photos/photo_", re.IGNORECASE),
        re.compile(r"\b[0-9a-f]{64}\b"),          # sha256
        re.compile(r"ChatExport_", re.IGNORECASE),
    ]
    content = json.dumps(summary)
    for pattern in forbidden_patterns:
        if pattern.search(content):
            raise ValueError(
                f"PRIVACY VIOLATION: tracked summary contains forbidden pattern '{pattern.pattern}'"
            )
    log.info("Privacy assertion passed: tracked summary contains no forbidden data")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="U4 Phase D: Aggregate completed decisions into summary.",
    )
    p.add_argument(
        "--review-dir",
        type=Path,
        default=C.FILTER_REVIEW_DIR,
        help="Review artifacts directory (default: %(default)s)",
    )
    p.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Review run ID (e.g. rvrun_20260427T200000)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override path to manifest JSON",
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=C.FILTER_REVIEW_SUMMARY_PATH,
        help="Output path for tracked privacy-safe summary (default: %(default)s)",
    )
    p.add_argument(
        "--no-require-validation-pass",
        action="store_false",
        dest="require_validation_pass",
        help="Allow aggregation even if validation has errors (not recommended)",
    )
    p.set_defaults(require_validation_pass=True)
    p.add_argument(
        "--partial",
        action="store_true",
        dest="partial",
        help=(
            "Partial mode: aggregate only reviewed_valid batches; unreviewed records stay "
            "unreviewed and not training-ready. Writes filter_review_partial_summary.json."
        ),
    )
    p.add_argument(
        "--partial-summary-out",
        type=Path,
        default=C.FILTER_REVIEW_PARTIAL_SUMMARY_PATH,
        help="Output path for partial privacy-safe summary (default: %(default)s)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    review_dir: Path = args.review_dir
    run_id: str = args.run_id

    manifest_path = args.manifest or review_dir / f"filter_review_manifest_{run_id}.json"

    try:
        manifest = load_manifest(manifest_path)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    if args.partial:
        try:
            local_aggregate, tracked_summary = aggregate_partial(manifest, review_dir, run_id)
        except ValueError as exc:
            log.error("%s", exc)
            sys.exit(1)

        try:
            _assert_summary_privacy(tracked_summary)
        except ValueError as exc:
            log.error("%s", exc)
            sys.exit(1)

        # Write local partial aggregate (local-only, in review dir)
        agg_path = review_dir / f"filter_decisions_partial_aggregate_{run_id}.json"
        _write_json_atomic(agg_path, local_aggregate)
        log.info("Local partial aggregate written: %s", agg_path)

        _write_json_atomic(args.partial_summary_out, tracked_summary)
        log.info("Partial tracked summary written: %s", args.partial_summary_out)

        prog = tracked_summary["progress"]
        tb = tracked_summary["training_eligibility"]
        log.info(
            "Partial aggregation complete: %d/%d batches reviewed, "
            "%d reviewed records, %d unreviewed, %d training-eligible",
            prog["reviewed_valid_batches"],
            prog["total_batches"],
            prog["reviewed_records"],
            prog["unreviewed_records"],
            tb["training_eligible_count"],
        )
        return

    try:
        decision_files = load_decision_files(review_dir, run_id)
    except (OSError, ValueError) as exc:
        log.error("Failed to load decision files: %s", exc)
        sys.exit(1)

    if not decision_files:
        log.error("No decision files found in %s for run_id=%s", review_dir, run_id)
        sys.exit(1)

    # Run validation inline
    log.info("Running validation pass before aggregation...")
    validation_report = validate(manifest, decision_files)
    v_status = "PASS" if validation_report["passed"] else "FAIL"
    log.info("Validation %s: reviewed=%d, missing=%d, invalid=%d",
             v_status,
             validation_report["reviewed_count"],
             validation_report["missing_count"],
             validation_report["invalid_decisions"])

    try:
        local_aggregate, tracked_summary = aggregate_decisions(
            manifest,
            decision_files,
            validation_report,
            require_validation_pass=args.require_validation_pass,
        )
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)

    # Assert tracked summary has no private data
    try:
        _assert_summary_privacy(tracked_summary)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)

    # Write local aggregate
    aggregate_path = review_dir / f"filter_decisions_aggregate_{run_id}.json"
    _write_json_atomic(aggregate_path, local_aggregate)
    log.info("Local aggregate written: %s", aggregate_path)

    # Write tracked summary
    _write_json_atomic(args.summary_out, tracked_summary)
    log.info("Tracked summary written: %s", args.summary_out)

    total = local_aggregate["summary"]["total_reviewed"]
    eligible = local_aggregate["summary"]["training_eligible_count"]
    log.info("Aggregation complete: %d reviewed, %d training-eligible (confidence>=4)", total, eligible)


if __name__ == "__main__":
    main()
