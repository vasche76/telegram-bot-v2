"""
intake_review_ingest_export.py — U4 Phase D: Safely ingest an exported decision JSON.

Validates a downloaded decision file then copies it into the project review directory.
Backs up any existing blank template before overwriting.
Refuses to overwrite an already-filled (reviewed) batch unless --force is provided.

Usage:
  python3 scripts/intake_review_ingest_export.py \\
      --run-id rvrun_20260427T184629Z \\
      --batch-id 0002

Default source path:
  ~/Downloads/filter_decisions_<RUN_ID>_<BATCH_ID>.json

Use --source to specify an explicit path.

source=telegram_private_2026-04-24, license=private_training_only
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intake_constants as C
from intake_review_schema import validate_decision_file, validate_decision_file_records
from intake_review_validate import is_blank_template, validate_partial, load_manifest

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr)
log = logging.getLogger(__name__)


# ─── Validation ───────────────────────────────────────────────────────────────


def validate_export(path: Path, expected_run_id: str, expected_batch_id: str) -> list[str]:
    """
    Validate an exported decision JSON before ingestion.
    Returns list of error strings (empty = valid).
    Checks: file loads, schema_version, run_id, batch_id, record count > 0, all records valid.
    """
    errors: list[str] = []

    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        return [f"invalid JSON: {exc}"]
    except OSError as exc:
        return [f"cannot read file: {exc}"]

    # File-level schema
    file_errors = validate_decision_file(data)
    errors.extend(file_errors)

    # run_id match
    if data.get("run_id") != expected_run_id:
        errors.append(
            f"run_id mismatch: expected '{expected_run_id}', got '{data.get('run_id')}'"
        )

    # batch_id match
    if str(data.get("batch_id")) != str(expected_batch_id):
        errors.append(
            f"batch_id mismatch: expected '{expected_batch_id}', got '{data.get('batch_id')}'"
        )

    # Must have at least one record
    records = data.get("records", [])
    if not records:
        errors.append("exported file has no records — expected reviewed decisions")

    # All records filled (no null decision_type)
    if is_blank_template(data):
        errors.append("exported file is a blank template — all decision_type are null")

    # Record-level validation (schema + consistency)
    if not errors:
        rec_errors, invalid_count, _valid_count = validate_decision_file_records(data)
        if rec_errors:
            errors.extend(rec_errors[:10])  # cap at 10 to avoid flooding output
            if len(rec_errors) > 10:
                errors.append(f"... and {len(rec_errors) - 10} more record errors")
            errors.append(f"{invalid_count} record(s) failed validation")

    return errors


# ─── Privacy check ────────────────────────────────────────────────────────────


def _assert_no_private_paths(source_path: Path) -> None:
    """Fail loudly if the source path looks like it could expose private data elsewhere."""
    s = str(source_path).lower()
    forbidden = ["chatexport_", "telegram desktop", "/messages/", "manifest.jsonl"]
    for f in forbidden:
        if f in s:
            raise ValueError(
                f"SAFETY: source path contains suspicious segment '{f}' — "
                f"refusing to ingest from potentially private source: {source_path}"
            )


# ─── Ingest ───────────────────────────────────────────────────────────────────


def ingest_export(
    run_id: str,
    batch_id: str,
    source_path: Path,
    review_dir: Path,
    force: bool = False,
) -> Path:
    """
    Validate and copy an exported decision JSON into the project review directory.

    - Validates the source file (schema, run_id, batch_id, records, consistency).
    - Backs up existing blank template before overwriting.
    - Refuses to overwrite an already-filled batch unless force=True.
    - Returns the destination path on success.
    - Raises ValueError on validation failure or overwrite refusal.
    """
    _assert_no_private_paths(source_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_path}")

    # Validate the export
    errors = validate_export(source_path, run_id, batch_id)
    if errors:
        raise ValueError(
            f"Exported file failed validation ({len(errors)} error(s)):\n"
            + "\n".join(f"  {e}" for e in errors)
        )

    dest_path = review_dir / f"filter_decisions_{run_id}_{batch_id}.json"

    if dest_path.exists():
        with dest_path.open(encoding="utf-8") as fh:
            existing = json.load(fh)

        if not is_blank_template(existing):
            if not force:
                raise ValueError(
                    f"Destination is already a filled reviewed batch: {dest_path}\n"
                    f"Use --force to overwrite an existing reviewed batch (not recommended)."
                )
            # Back up filled reviewed batch before overwriting (reviewed decisions are high-value)
            from datetime import datetime, timezone
            ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            backup_path = dest_path.with_suffix(f".reviewed_backup_{ts}.json")
            shutil.copy2(dest_path, backup_path)
            log.warning("--force: reviewed batch backed up to %s before overwrite", backup_path)
        else:
            # Back up blank template before overwriting
            backup_path = dest_path.with_suffix(".blank_backup.json")
            shutil.copy2(dest_path, backup_path)
            log.info("Blank template backed up to: %s", backup_path)

    # Copy validated export into review dir
    review_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, dest_path)
    log.info("Ingested: %s → %s", source_path, dest_path)

    return dest_path


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="U4 Phase D: Safely ingest an exported decision JSON into the project.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Review run ID (e.g. rvrun_20260427T184629Z)",
    )
    p.add_argument(
        "--batch-id",
        type=str,
        required=True,
        help="Batch ID (e.g. 0002)",
    )
    p.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Explicit source path. "
            "Default: ~/Downloads/filter_decisions_<RUN_ID>_<BATCH_ID>.json"
        ),
    )
    p.add_argument(
        "--review-dir",
        type=Path,
        default=C.FILTER_REVIEW_DIR,
        help="Review artifacts directory (default: %(default)s)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Override path to manifest JSON for progress report after ingest",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an already-filled reviewed batch (not recommended)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_id: str = args.run_id
    batch_id: str = args.batch_id
    review_dir: Path = args.review_dir

    source_path = args.source or (
        Path.home() / "Downloads" / f"filter_decisions_{run_id}_{batch_id}.json"
    )

    log.info("Ingesting batch %s from: %s", batch_id, source_path)

    try:
        dest = ingest_export(run_id, batch_id, source_path, review_dir, force=args.force)
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        sys.exit(1)

    log.info("Ingest complete: %s", dest)

    # Print progress summary if manifest is available
    manifest_path = args.manifest or review_dir / f"filter_review_manifest_{run_id}.json"
    if manifest_path.exists():
        try:
            manifest = load_manifest(manifest_path)
            report = validate_partial(manifest, review_dir, run_id)
            total = report["total_batches"]
            done = report["reviewed_valid"]
            remaining = report["blank_unreviewed"]
            log.info(
                "Progress: %d/%d batches reviewed, %d remaining unreviewed",
                done, total, remaining,
            )
            if report["next_unreviewed_batches"]:
                next_batch = report["next_unreviewed_batches"][0]
                log.info(
                    "Next batch to review: batch %s  →  open review HTML and export JSON",
                    next_batch,
                )
        except Exception as exc:
            log.warning("Could not compute progress: %s", exc)


if __name__ == "__main__":
    main()
