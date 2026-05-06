"""
intake_review_validate.py — U4 Phase D: Validate completed decision files.

Full mode (default):
  Loads the local review manifest and all decision JSON files, then checks:
    - Coverage: every review ID in the manifest has exactly one decision
    - Uniqueness: no duplicate review IDs across or within decision files
    - Unknown IDs: no decision refers to an ID not in the manifest
    - Decision schema: all required fields present, enums valid, confidence in range
    - Decision consistency: KEEP / REMOVE / RELABEL / UNSURE rules enforced
    - Privacy: notes field passes privacy scan
  Exits 0 on PASS, 1 on any validation failure.

Partial mode (--partial):
  Validates only filled batches; classifies each batch as:
    - reviewed_valid: all records filled and valid
    - reviewed_invalid: filled but has schema/consistency errors
    - blank_unreviewed: template with all decision_type=None (not reviewed)
    - missing_file: expected decision file absent
  Blank templates are never counted as reviewed.
  Exits 0 only if all filled batches are valid (no reviewed_invalid / missing_file).

source=telegram_private_2026-04-24, license=private_training_only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intake_constants as C
from intake_review_schema import (
    validate_decision_file,
    validate_decision_file_records,
    validate_record_full,
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr)
log = logging.getLogger(__name__)


# ─── Batch status codes (partial mode) ───────────────────────────────────────

BATCH_STATUS_REVIEWED_VALID: str = "reviewed_valid"
BATCH_STATUS_REVIEWED_INVALID: str = "reviewed_invalid"
BATCH_STATUS_BLANK_UNREVIEWED: str = "blank_unreviewed"
BATCH_STATUS_MISSING_FILE: str = "missing_file"


# ─── I/O helpers ─────────────────────────────────────────────────────────────


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# ─── Manifest loading ─────────────────────────────────────────────────────────


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Review manifest not found: {manifest_path}")
    data = _load_json(manifest_path)
    if "run_id" not in data or "batches" not in data:
        raise ValueError(f"Manifest missing 'run_id' or 'batches': {manifest_path}")
    return data


def collect_expected_review_ids(manifest: dict) -> set[str]:
    """Return the full set of review_ids expected from the manifest."""
    ids: set[str] = set()
    for batch in manifest.get("batches", []):
        for rid in batch.get("review_ids", []):
            ids.add(rid)
    return ids


# ─── Decision file loading ────────────────────────────────────────────────────


def load_decision_files(review_dir: Path, run_id: str) -> list[tuple[Path, dict]]:
    """
    Load all decision files for a run. Returns list of (path, data) pairs.
    Only loads files matching the run_id pattern to avoid cross-run contamination.
    """
    pattern = f"filter_decisions_{run_id}_*.json"
    files = sorted(
        p for p in review_dir.glob(pattern)
        if "backup" not in p.name
    )
    if not files:
        return []
    result = []
    for p in files:
        try:
            data = _load_json(p)
            result.append((p, data))
        except (json.JSONDecodeError, OSError) as exc:
            log.error("Cannot load decision file %s: %s", p.name, exc)
            raise
    return result


# ─── Validation ───────────────────────────────────────────────────────────────


def validate(
    manifest: dict,
    decision_files: list[tuple[Path, dict]],
) -> dict:
    """
    Full validation pass. Returns a validation report dict.
    """
    errors: list[str] = []
    warnings: list[str] = []
    run_id = manifest["run_id"]
    expected_ids = collect_expected_review_ids(manifest)
    expected_count = manifest.get("total_records", len(expected_ids))

    # File-level checks
    for path, data in decision_files:
        file_errors = validate_decision_file(data)
        if file_errors:
            for e in file_errors:
                errors.append(f"[{path.name}] {e}")
        if data.get("run_id") != run_id:
            errors.append(
                f"[{path.name}] run_id mismatch: expected '{run_id}', got '{data.get('run_id')}'"
            )

    # Record-level aggregation
    seen_ids: dict[str, str] = {}  # review_id → filename
    all_records: list[dict] = []
    invalid_count = 0
    valid_count = 0

    for path, data in decision_files:
        fname = path.name
        for rec in data.get("records", []):
            rid = rec.get("review_id", "")
            if rid in seen_ids:
                errors.append(
                    f"[{fname}] duplicate review_id across files: '{rid}' "
                    f"(also in {seen_ids[rid]})"
                )
                invalid_count += 1
                continue
            seen_ids[rid] = fname
            all_records.append(rec)
            rec_errors = validate_record_full(rec)
            if rec_errors:
                for e in rec_errors:
                    errors.append(f"[{fname}:{rid}] {e}")
                invalid_count += 1
            else:
                valid_count += 1

    # Coverage check
    reviewed_ids = set(seen_ids.keys())
    missing_ids = expected_ids - reviewed_ids
    unknown_ids = reviewed_ids - expected_ids

    if missing_ids:
        missing_count = len(missing_ids)
        errors.append(
            f"Missing decisions for {missing_count} review_id(s) "
            f"(first 5: {sorted(missing_ids)[:5]})"
        )
    if unknown_ids:
        unknown_count = len(unknown_ids)
        errors.append(
            f"Unknown review_id(s) not in manifest: {unknown_count} "
            f"(first 5: {sorted(unknown_ids)[:5]})"
        )

    # Conflict resolution coverage
    conflict_review_ids: set[str] = set()
    for batch in manifest.get("batches", []):
        # We can't recover conflict flags from manifest alone (they're in Phase C data)
        # But we verify all conflict-priority records got a decision (covered by coverage check above)
        pass

    passed = len(errors) == 0

    report = {
        "run_id": run_id,
        "passed": passed,
        "expected_count": expected_count,
        "reviewed_count": len(reviewed_ids),
        "missing_count": len(missing_ids),
        "unknown_count": len(unknown_ids),
        "valid_decisions": valid_count,
        "invalid_decisions": invalid_count,
        "error_count": len(errors),
        "errors": errors,
        "warnings": warnings,
    }

    return report


# ─── Partial validation ───────────────────────────────────────────────────────


def is_blank_template(data: dict) -> bool:
    """
    Return True if every record in the decision file has decision_type=None (blank template).

    Returns False for corrupted files where 'records' is not a list (e.g. JSON null),
    so they are routed to validate() and reported as reviewed_invalid, not silently
    absorbed as blank_unreviewed.
    """
    records = data.get("records", [])
    if not isinstance(records, list):
        return False
    if not records:
        return True
    return all(r.get("decision_type") is None for r in records)


def validate_partial(
    manifest: dict,
    review_dir: Path,
    run_id: str,
) -> dict:
    """
    Partial validation: validate only filled batches, classify blank templates as unreviewed.

    Distinguishes:
      - reviewed_valid:    filled batch, all records pass full validation
      - reviewed_invalid:  filled batch with schema or consistency errors
      - blank_unreviewed:  file exists but all decision_type=None (not reviewed)
      - missing_file:      expected decision file not found

    Blank templates are NEVER counted as reviewed.
    Passes only when reviewed_invalid==0 and missing_file==0.
    """
    errors: list[str] = []
    batch_statuses: dict[str, str] = {}
    reviewed_valid_batches: list[str] = []
    reviewed_invalid_batches: list[str] = []
    blank_unreviewed_batches: list[str] = []
    missing_file_batches: list[str] = []
    reviewed_records = 0
    unreviewed_records = 0

    for batch in manifest.get("batches", []):
        batch_id = str(batch.get("batch_id", ""))
        decision_file = batch.get("decision_file", f"filter_decisions_{run_id}_{batch_id}.json")
        path = review_dir / decision_file
        record_count = batch.get("record_count", 0)

        if not path.exists():
            batch_statuses[batch_id] = BATCH_STATUS_MISSING_FILE
            missing_file_batches.append(batch_id)
            unreviewed_records += record_count
            continue

        try:
            data = _load_json(path)
        except (json.JSONDecodeError, OSError) as exc:
            errors.append(f"[batch {batch_id}] cannot load file: {exc}")
            batch_statuses[batch_id] = BATCH_STATUS_REVIEWED_INVALID
            reviewed_invalid_batches.append(batch_id)
            continue

        if is_blank_template(data):
            batch_statuses[batch_id] = BATCH_STATUS_BLANK_UNREVIEWED
            blank_unreviewed_batches.append(batch_id)
            unreviewed_records += record_count
            continue

        # Filled batch — build single-batch sub-manifest and validate strictly
        sub_manifest: dict = {
            "run_id": run_id,
            "source": manifest.get("source", C.SOURCE_TAG),
            "phase": manifest.get("phase", C.REVIEW_PHASE),
            "total_records": record_count,
            "batch_count": 1,
            "batches": [batch],
        }
        batch_report = validate(sub_manifest, [(path, data)])

        if batch_report["passed"]:
            batch_statuses[batch_id] = BATCH_STATUS_REVIEWED_VALID
            reviewed_valid_batches.append(batch_id)
            reviewed_records += len(data.get("records", []))
        else:
            batch_statuses[batch_id] = BATCH_STATUS_REVIEWED_INVALID
            reviewed_invalid_batches.append(batch_id)
            for e in batch_report["errors"]:
                errors.append(f"[batch {batch_id}] {e}")

    passed = len(reviewed_invalid_batches) == 0 and len(missing_file_batches) == 0
    total_batches = len(manifest.get("batches", []))

    return {
        "run_id": run_id,
        "mode": "partial",
        "passed": passed,
        "total_batches": total_batches,
        "reviewed_valid": len(reviewed_valid_batches),
        "reviewed_invalid": len(reviewed_invalid_batches),
        "blank_unreviewed": len(blank_unreviewed_batches),
        "missing_file": len(missing_file_batches),
        "reviewed_records": reviewed_records,
        "unreviewed_records": unreviewed_records,
        "reviewed_valid_batches": reviewed_valid_batches,
        "next_unreviewed_batches": blank_unreviewed_batches[:10],
        "invalid_batches": reviewed_invalid_batches,
        "error_count": len(errors),
        "errors": errors,
    }


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="U4 Phase D: Validate completed decision files.",
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
        help="Override path to manifest JSON (default: <review-dir>/filter_review_manifest_<run-id>.json)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output validation report as JSON to stdout",
    )
    p.add_argument(
        "--partial",
        action="store_true",
        dest="partial",
        help=(
            "Partial mode: validate only filled batches; classify blank templates as "
            "unreviewed (not invalid). Does not require full coverage. "
            "NOTE: exits 0 (PASS) even when reviewed_valid=0 (all batches still blank). "
            "Callers must check 'reviewed_valid' in --json output, not only the exit code."
        ),
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
        report = validate_partial(manifest, review_dir, run_id)
        if args.json_output:
            print(json.dumps(report, indent=2))
        else:
            status = "PASS" if report["passed"] else "FAIL"
            log.info(
                "Partial validation %s: %d reviewed_valid, %d reviewed_invalid, "
                "%d blank_unreviewed, %d missing_file, %d reviewed_records, %d unreviewed_records",
                status,
                report["reviewed_valid"],
                report["reviewed_invalid"],
                report["blank_unreviewed"],
                report["missing_file"],
                report["reviewed_records"],
                report["unreviewed_records"],
            )
            if report["next_unreviewed_batches"]:
                log.info("Next unreviewed batches: %s", report["next_unreviewed_batches"])
            for e in report["errors"]:
                log.error("  %s", e)
        sys.exit(0 if report["passed"] else 1)

    try:
        decision_files = load_decision_files(review_dir, run_id)
    except (OSError, ValueError) as exc:
        log.error("Failed to load decision files: %s", exc)
        sys.exit(1)

    if not decision_files:
        log.error("No decision files found in %s for run_id=%s", review_dir, run_id)
        sys.exit(1)

    report = validate(manifest, decision_files)

    if args.json_output:
        print(json.dumps(report, indent=2))
    else:
        status = "PASS" if report["passed"] else "FAIL"
        log.info("Validation %s: %d reviewed, %d missing, %d invalid, %d errors",
                 status,
                 report["reviewed_count"],
                 report["missing_count"],
                 report["invalid_decisions"],
                 report["error_count"])
        for e in report["errors"]:
            log.error("  %s", e)

    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
