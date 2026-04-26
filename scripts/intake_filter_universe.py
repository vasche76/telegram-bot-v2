"""
intake_filter_universe.py — Build the deduplicated unique image universe for U4 filtering.

Reads dedup_clusters.jsonl + audit.jsonl, excludes dedup non-keeps and corrupt images,
and writes a local-only filter_universe.jsonl plus a tracked filter_universe_summary.json.

Hard-fails if the dedup run is still provisional or review is incomplete — the universe
must only be built from a fully finalized dedup.

source=telegram_private_2026-04-24, license=private_training_only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import intake_constants as C

SCHEMA_VERSION = 1

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr)
log = logging.getLogger(__name__)


# ─── I/O helpers ─────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{lineno}: corrupt JSONL: {exc.msg}"
                    ) from exc
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON atomically via temp file + os.replace() to avoid partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".tmp_",
            suffix=".json",
            delete=False,
        ) as tmp:
            tmp_path = tmp.name
            json.dump(data, tmp, indent=2, ensure_ascii=False)
            tmp.flush()
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


# ─── Safety gates ─────────────────────────────────────────────────────────────


def _check_dedup_finalized(dedup_summary_path: Path, review_summary_path: Path) -> int:
    """
    Return the expected unique count from the finalized dedup summary.

    Exits 1 if provisional, manual_review_required, or review_complete checks fail.
    """
    if not dedup_summary_path.exists():
        log.error("dedup_summary.json not found: %s", dedup_summary_path)
        sys.exit(1)

    with dedup_summary_path.open(encoding="utf-8") as fh:
        dsummary = json.load(fh)

    if dsummary.get("provisional") is not False:
        log.error(
            "HARD FAIL: dedup_summary.provisional=%r — dedup is still provisional. "
            "Finalize dedup before building the filter universe.",
            dsummary.get("provisional"),
        )
        sys.exit(1)

    if dsummary.get("manual_review_required") is not False:
        log.error(
            "HARD FAIL: dedup_summary.manual_review_required=%r — manual review still pending.",
            dsummary.get("manual_review_required"),
        )
        sys.exit(1)

    if not review_summary_path.exists():
        log.error("dedup_review_summary.json not found: %s", review_summary_path)
        sys.exit(1)

    with review_summary_path.open(encoding="utf-8") as fh:
        rsummary = json.load(fh)

    if rsummary.get("review_complete") is not True:
        log.error(
            "HARD FAIL: dedup_review_summary.review_complete=%r — boundary review incomplete.",
            rsummary.get("review_complete"),
        )
        sys.exit(1)

    expected = dsummary.get("total_unique_after_dedup")
    if not isinstance(expected, int) or expected < 0:
        log.error("dedup_summary.total_unique_after_dedup is missing or invalid: %r", expected)
        sys.exit(1)

    log.info(
        "Dedup finalized: provisional=False, manual_review_required=False, "
        "review_complete=True, expected_unique=%d",
        expected,
    )
    return expected


# ─── Core logic ───────────────────────────────────────────────────────────────


def build_universe(
    clusters_path: Path,
    audit_path: Path,
    output_dir: Path,
    dedup_summary_path: Path,
    review_summary_path: Path,
    dry_run: bool = False,
) -> dict:
    """
    Build the deduplicated unique image universe.

    Returns the summary dict.
    """
    expected_unique = _check_dedup_finalized(dedup_summary_path, review_summary_path)

    # --- Load dedup clusters → collect non-keep filenames ---
    if not clusters_path.exists():
        log.error("dedup_clusters.jsonl not found: %s", clusters_path)
        sys.exit(1)

    clusters = _read_jsonl(clusters_path)
    non_keeps: set[str] = set()
    keep_filenames: set[str] = set()
    for cluster in clusters:
        keep_filenames.add(cluster["keep_filename"])
        for fn in cluster.get("duplicate_filenames") or []:
            non_keeps.add(fn)

    log.info("Clusters loaded: %d clusters, %d non-keep duplicates", len(clusters), len(non_keeps))

    # --- Load audit records ---
    if not audit_path.exists():
        log.error("audit.jsonl not found: %s", audit_path)
        sys.exit(1)

    audit_records = _read_jsonl(audit_path)
    input_records = len(audit_records)
    log.info("Audit records loaded: %d", input_records)

    # --- Validate all cluster filenames exist in audit ---
    audit_filenames: set[str] = {rec["filename"] for rec in audit_records}
    missing_in_audit = 0
    for cluster in clusters:
        for fn in [cluster["keep_filename"]] + list(cluster.get("duplicate_filenames") or []):
            if fn not in audit_filenames:
                log.warning("Cluster filename missing in audit: %s", fn)
                missing_in_audit += 1

    if missing_in_audit > 0:
        log.error(
            "HARD FAIL: %d cluster filename(s) not found in audit. Pipeline integrity violated.",
            missing_in_audit,
        )
        sys.exit(1)

    # --- Build universe ---
    universe: list[dict] = []
    dedup_duplicate_removed = 0
    corrupt_excluded = 0
    audit_non_corrupt_count = 0

    for rec in audit_records:
        fn = rec["filename"]
        is_corrupt = bool(rec.get("corrupt"))

        if not is_corrupt:
            audit_non_corrupt_count += 1

        if fn in non_keeps:
            # Dedup non-keep: excluded (corrupt non-keeps counted here, not in corrupt_excluded)
            dedup_duplicate_removed += 1
            continue

        if is_corrupt:
            corrupt_excluded += 1
            continue

        # Determine dedup role
        if fn in keep_filenames:
            dedup_role = "cluster_keep"
        else:
            dedup_role = "unique"

        universe.append({
            "filename": fn,
            "sha256": rec.get("sha256"),
            "width": rec.get("width"),
            "height": rec.get("height"),
            "max_side": rec.get("max_side"),
            "file_size": rec.get("file_size"),
            "low_res": rec.get("low_res", False),
            "dedup_role": dedup_role,
            "source": C.SOURCE_TAG,
        })

    unique_records = len(universe)
    low_res_count = sum(1 for r in universe if r.get("low_res"))
    consistency_ok = unique_records == expected_unique

    if not dry_run and unique_records == 0:
        log.error(
            "HARD FAIL: unique_records == 0 in real run. Refusing to write an empty universe."
        )
        sys.exit(1)

    # Hashes the sorted set of filenames in the universe — not image bytes
    universe_filename_set_sha256 = hashlib.sha256(
        "\n".join(sorted(r["filename"] for r in universe)).encode()
    ).hexdigest()

    log.info(
        "Universe built: %d unique, %d non-keeps excluded, %d corrupt excluded",
        unique_records, dedup_duplicate_removed, corrupt_excluded,
    )

    if not consistency_ok:
        if dry_run:
            log.warning(
                "[DRY RUN] Consistency check FAILED: unique_records=%d != expected_unique=%d "
                "(delta=%d). A real run would block here.",
                unique_records, expected_unique, unique_records - expected_unique,
            )
        else:
            log.error(
                "HARD FAIL: unique_records=%d != expected_unique=%d (delta=%d). "
                "Refusing to write outputs. Investigate pipeline integrity.",
                unique_records, expected_unique, unique_records - expected_unique,
            )
            sys.exit(1)
    else:
        log.info("Consistency OK: unique_records == expected_unique == %d", unique_records)

    integrity_sum = unique_records + dedup_duplicate_removed + corrupt_excluded
    if integrity_sum != input_records:
        log.warning(
            "Integrity sum mismatch: unique(%d) + dedup_removed(%d) + corrupt(%d) = %d != input(%d)",
            unique_records, dedup_duplicate_removed, corrupt_excluded, integrity_sum, input_records,
        )

    summary = {
        "input_records": input_records,
        "audit_non_corrupt_records": audit_non_corrupt_count,
        "dedup_duplicate_removed": dedup_duplicate_removed,
        "corrupt_excluded": corrupt_excluded,
        "missing_in_audit": missing_in_audit,
        "low_res_count": low_res_count,
        "unique_records": unique_records,
        "universe_filename_set_sha256": universe_filename_set_sha256,
        "expected_unique_records_from_dedup": expected_unique,
        "consistency_ok": consistency_ok,
        "source": C.SOURCE_TAG,
        "license": C.LICENSE_TAG,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
    }

    if dry_run:
        log.info("[DRY RUN] Would write %d records to filter_universe.jsonl", unique_records)
        log.info("[DRY RUN] Summary: %s", json.dumps(summary, indent=2))
        return summary

    universe_path = output_dir / "filter_universe.jsonl"
    summary_path = output_dir / "filter_universe_summary.json"

    _write_jsonl(universe_path, universe)
    log.info("Written: %s (%d records)", universe_path, unique_records)

    _write_json_atomic(summary_path, summary)
    log.info("Written: %s", summary_path)

    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build the deduplicated unique image universe for U4 filtering.",
    )
    p.add_argument(
        "--clusters",
        type=Path,
        default=C.DEDUP_PATH,
        help="Path to dedup_clusters.jsonl (default: %(default)s)",
    )
    p.add_argument(
        "--audit",
        type=Path,
        default=C.AUDIT_PATH,
        help="Path to audit.jsonl (default: %(default)s)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=C.INTAKE_META_ROOT,
        help="Directory for filter_universe.jsonl and filter_universe_summary.json",
    )
    p.add_argument(
        "--dedup-summary",
        type=Path,
        default=C.INTAKE_META_ROOT / "dedup_summary.json",
        help="Path to dedup_summary.json for finalization gate checks",
    )
    p.add_argument(
        "--review-summary",
        type=Path,
        default=C.INTAKE_META_ROOT / "dedup_review_summary.json",
        help="Path to dedup_review_summary.json for review_complete gate check",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        build_universe(
            clusters_path=args.clusters,
            audit_path=args.audit,
            output_dir=args.output_dir,
            dedup_summary_path=args.dedup_summary,
            review_summary_path=args.review_summary,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        log.error("Input file error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
