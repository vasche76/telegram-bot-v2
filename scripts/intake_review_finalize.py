#!/usr/bin/env python3
"""
intake_review_finalize.py — Ingest manual review decisions and finalize dedup.

Reads review_decisions.json produced by the intake_review_boundary.py HTML contact
sheet, validates completeness against dedup_clusters.jsonl, computes the false-positive
rate (FALSE_POSITIVE + MIXED decisions / total reviewed), and either:

  (a) Clears the provisional flag in dedup_summary.json and writes a tracked
      dedup_review_summary.json (counts only, no PII), or

  (b) Prints a threshold re-run recommendation with a suggested lower threshold
      and leaves dedup_summary.json untouched.

PRIVACY: review_decisions.json is gitignored (may contain reviewer notes).
dedup_review_summary.json is tracked — it contains only aggregate counts and
metadata; no filenames, captions, sender names, or per-cluster decisions.

source=telegram_private_2026-04-24, license=private_training_only

Exit codes:
  0 — complete final accepted state (dedup_summary.json updated, review_complete=true)
  1 — hard error: invalid input, FP threshold exceeded, unsafe overwrite attempt,
      or contradictory state (source already final but decisions can't finalize)
  2 — valid but non-final: --partial mode, incomplete review, UNSURE decisions
      remain, --dry-run preview, or idempotent re-run on already-finalized source

Usage:
    python3 scripts/intake_review_finalize.py \\
        --decisions <path/to/review_decisions.json> \\
        --clusters  <path/to/dedup_clusters.jsonl> \\
        --summary   <path/to/dedup_summary.json> \\
        --output    <path/to/dedup_review_summary.json> \\
        [--fp-threshold 0.15] \\
        [--partial] \\
        [--dry-run] \\
        [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "telegram_private_2026-04-24"
LICENSE = "private_training_only"

DEFAULT_FP_THRESHOLD = 0.15

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stderr,
)
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
                        f"{path}:{lineno}: corrupt JSONL: {exc.msg} at position {exc.pos}"
                    ) from exc
    return records


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


def _sha256_file(path: Path) -> str:
    """Return hex SHA256 of a file's contents for lightweight provenance tracking."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Cluster loading ──────────────────────────────────────────────────────────


def _load_boundary_clusters(clusters_path: Path, phash_threshold: int) -> list[dict]:
    """Return all perceptual clusters at the boundary hamming distance."""
    return [
        r for r in _read_jsonl(clusters_path)
        if r.get("cluster_type") == "perceptual" and r.get("hamming_distance") == phash_threshold
    ]


# ─── Validation ──────────────────────────────────────────────────────────────


def _validate_completeness(
    boundary: list[dict],
    decisions_by_id: dict[int, dict],
) -> list[int]:
    """Return list of cluster_ids missing from decisions."""
    return [c["cluster_id"] for c in boundary if c["cluster_id"] not in decisions_by_id]


# ─── Threshold suggestion ─────────────────────────────────────────────────────


def _suggest_threshold(fp_rate: float, current_threshold: int) -> int:
    """Directional heuristic: lower threshold proportional to observed FP rate."""
    return max(1, current_threshold - round(fp_rate * current_threshold))


# ─── Core logic ──────────────────────────────────────────────────────────────


def run(
    decisions_path: Path,
    clusters_path: Path,
    summary_path: Path,
    output_path: Path,
    fp_threshold: float = DEFAULT_FP_THRESHOLD,
    partial: bool = False,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    # ── Load inputs ───────────────────────────────────────────────────────────
    if not decisions_path.exists():
        log.error("Decisions file not found: %s", decisions_path)
        return 1
    if not clusters_path.exists():
        log.error("Clusters file not found: %s", clusters_path)
        return 1
    if not summary_path.exists():
        log.error("Dedup summary not found: %s", summary_path)
        return 1

    with decisions_path.open(encoding="utf-8") as fh:
        decisions_data = json.load(fh)

    decisions_by_id: dict[int, dict] = {}
    for _i, _d in enumerate(decisions_data.get("decisions", [])):
        if "cluster_id" not in _d:
            log.error("Decision entry #%d is missing 'cluster_id': %r", _i, _d)
            return 1
        _raw_cid = _d["cluster_id"]
        if _raw_cid is None or isinstance(_raw_cid, bool):
            log.error("Decision entry #%d has invalid cluster_id: %r", _i, _raw_cid)
            return 1
        if isinstance(_raw_cid, int):
            _cid = _raw_cid
        elif isinstance(_raw_cid, str):
            try:
                _cid = int(_raw_cid)
            except ValueError:
                log.error(
                    "Decision entry #%d has non-coercible string cluster_id: %r", _i, _raw_cid
                )
                return 1
        else:
            log.error(
                "Decision entry #%d has non-integer cluster_id (type=%s): %r",
                _i, type(_raw_cid).__name__, _raw_cid,
            )
            return 1
        _entry = dict(_d)
        _entry["cluster_id"] = _cid
        decisions_by_id[_cid] = _entry

    # Load dedup_summary early for guard checks and threshold resolution
    with summary_path.open(encoding="utf-8") as fh:
        dedup_summary = json.load(fh)
    source_already_final = not dedup_summary.get("provisional", True)

    phash_threshold = dedup_summary.get("phash_threshold")
    if phash_threshold is None or not isinstance(phash_threshold, int) or phash_threshold < 1:
        log.error(
            "dedup_summary.json missing or invalid 'phash_threshold' (got %r). "
            "Cannot determine boundary cluster scope.",
            phash_threshold,
        )
        return 1

    # ── ADV-001: Output overwrite protection ──────────────────────────────────
    if output_path.exists() and not force:
        try:
            existing_output = json.loads(output_path.read_text(encoding="utf-8"))
            if existing_output.get("review_complete") is True:
                log.error(
                    "Output %s already contains a finalized review (review_complete=true). "
                    "A previous finalization run may have completed successfully. "
                    "If an interrupted run left this file in an inconsistent state, "
                    "use --force to overwrite and re-finalize.",
                    output_path,
                )
                return 1
        except (json.JSONDecodeError, OSError):
            pass  # unreadable existing file — allow overwrite

    try:
        boundary = _load_boundary_clusters(clusters_path, phash_threshold)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    log.info("Boundary clusters (hamming==%d) in clusters file: %d", phash_threshold, len(boundary))
    log.info("Decisions loaded from file: %d", len(decisions_by_id))

    # ── ADV-003: Cluster mismatch detection ───────────────────────────────────
    boundary_ids = {c["cluster_id"] for c in boundary}
    unknown_ids = set(decisions_by_id) - boundary_ids
    if unknown_ids:
        log.warning(
            "Decisions reference %d cluster ID(s) not found in clusters file — "
            "possible wrong clusters file. Unknown IDs: %s",
            len(unknown_ids),
            sorted(unknown_ids)[:20],
        )

    # ── KP-01: Completeness check (no sys.exit in helpers) ───────────────────
    missing = _validate_completeness(boundary, decisions_by_id)
    if missing and not partial:
        log.error("Missing decisions for %d boundary cluster(s):", len(missing))
        for cid in missing[:20]:
            log.error("  cluster_id=%d", cid)
        if len(missing) > 20:
            log.error("  ... and %d more", len(missing) - 20)
        log.error(
            "Use --partial to proceed with incomplete decisions (provisional flag won't be cleared), "
            "or complete the review and re-run."
        )
        return 1
    if missing:
        log.warning(
            "--partial mode: %d cluster(s) missing decisions. provisional flag will NOT be cleared.",
            len(missing),
        )

    # ── Compute counts ────────────────────────────────────────────────────────
    reviewed = [decisions_by_id[c["cluster_id"]] for c in boundary if c["cluster_id"] in decisions_by_id]
    counts: Counter[str] = Counter(d.get("decision", "UNSURE") for d in reviewed)

    keep_count = counts["KEEP_DEDUP"]
    fp_count = counts["FALSE_POSITIVE"] + counts["MIXED"]
    unsure_count = counts["UNSURE"]
    mixed_count = counts["MIXED"]
    total_reviewed = sum(counts.values())

    if total_reviewed == 0:
        log.error("No valid decisions found in decisions file.")
        return 1

    fp_rate = fp_count / total_reviewed

    # ── MIXED cluster IDs to stderr for optional follow-up ───────────────────
    if mixed_count > 0:
        mixed_ids = [d["cluster_id"] for d in reviewed if d.get("decision") == "MIXED"]
        log.warning(
            "%d cluster(s) marked MIXED (treated as FALSE_POSITIVE conservatively). "
            "Consider per-image re-inspection for: %s",
            mixed_count,
            mixed_ids[:20],
        )

    # ── Determine outcome ─────────────────────────────────────────────────────
    fp_exceeded = fp_rate >= fp_threshold
    has_unsure = unsure_count > 0
    is_incomplete = bool(missing)

    can_finalize = (
        not fp_exceeded
        and not has_unsure
        and not is_incomplete
    )

    current_threshold = decisions_data.get("threshold_reviewed", phash_threshold)
    suggested_threshold = _suggest_threshold(fp_rate, current_threshold) if fp_exceeded else None

    threshold_recommendation = "validated" if can_finalize else "lower_threshold"
    review_complete = can_finalize

    # ── ADV-002: Contradictory state guard ────────────────────────────────────
    if source_already_final and not can_finalize and not force:
        log.error(
            "dedup_summary.json already has provisional=false but current decisions "
            "cannot satisfy finalization criteria (review_complete would be false). "
            "This would create a contradictory state. Use --force to override or "
            "--dry-run to inspect without side effects."
        )
        return 1

    # ── ADV-004: SHA256 provenance ────────────────────────────────────────────
    clusters_sha256 = _sha256_file(clusters_path)

    # ── Write dedup_review_summary.json (always, tracked, no PII) ─────────────
    review_summary = {
        "reviewed_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "threshold_reviewed": current_threshold,
        "boundary_clusters_total": len(boundary),
        "boundary_clusters_reviewed": total_reviewed,
        "keep_dedup_count": keep_count,
        "false_positive_count": counts["FALSE_POSITIVE"],
        "mixed_count": mixed_count,
        "unsure_count": unsure_count,
        "false_positive_rate": round(fp_rate, 4),
        "threshold_recommendation": threshold_recommendation,
        "suggested_lower_threshold": suggested_threshold,
        "review_complete": review_complete,
        "clusters_sha256": clusters_sha256,
        "source": SOURCE,
        "license": LICENSE,
    }
    _write_json_atomic(output_path, review_summary)
    log.info("Written: %s", output_path)

    # ── Human-readable summary ────────────────────────────────────────────────
    print("\n─── Dedup Review Summary ───────────────────────────────")
    print(f"  Boundary clusters (hamming={phash_threshold}):  {len(boundary)}")
    print(f"  Reviewed:                       {total_reviewed}")
    print(f"  KEEP_DEDUP:                     {keep_count}")
    print(f"  FALSE_POSITIVE:                 {counts['FALSE_POSITIVE']}")
    print(f"  MIXED (→ conservative FP):      {mixed_count}")
    print(f"  UNSURE:                         {unsure_count}")
    print(f"  FP rate (FP+MIXED / total):     {fp_rate:.1%}")
    print(f"  FP threshold:                   {fp_threshold:.1%}")
    print(f"  Recommendation:                 {threshold_recommendation}")
    if suggested_threshold:
        print(f"  Suggested lower threshold:      {suggested_threshold}")
    print("────────────────────────────────────────────────────────\n")

    # ── KP-02: Finalization or re-run path — explicit exit codes ─────────────
    if can_finalize:
        if source_already_final:
            log.warning(
                "dedup_summary.json already has provisional=false — skipping update."
            )
            return 2  # idempotent re-run, nothing new to finalize

        if dry_run:
            log.info(
                "--dry-run: would clear provisional=true in %s (skipped)", summary_path
            )
            return 2  # valid preview state

        dedup_summary["provisional"] = False
        dedup_summary["manual_review_required"] = False
        dedup_summary["manual_review_reason"] = (
            f"Manual review completed for boundary clusters at phash_threshold={phash_threshold}: "
            f"{keep_count}/{total_reviewed} KEEP_DEDUP, "
            f"{counts['FALSE_POSITIVE']} FALSE_POSITIVE, "
            f"{mixed_count} MIXED, "
            f"{unsure_count} UNSURE. "
            f"Dedup approved for downstream staging."
        )
        dedup_summary["review_completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_json_atomic(summary_path, dedup_summary)
        log.info("Cleared provisional=true in %s", summary_path)
        print("✓ dedup_summary.json updated: provisional=false")
        print("✓ Downstream stages (U4–U8) are now unblocked.")
        return 0  # complete final accepted state

    else:
        if fp_exceeded:
            print("⚠️  FP rate exceeds threshold — recommend re-running dedup at a lower threshold.")
            print("\nSuggested re-run command:")
            print(
                f"  python3 scripts/intake_telegram_dedup.py \\\n"
                f"    --export-dir \"<your-export-dir>\" \\\n"
                f"    --phash-threshold {suggested_threshold}"
            )
            print("\nAfter re-run: commit updated dedup_clusters.jsonl and dedup_summary.json,")
            print("then restart the review from intake_review_boundary.py.")
            return 1  # hard error: FP threshold exceeded

        if has_unsure:
            print("\n⚠️  UNSURE decisions remain — resolve them before finalizing.")
            print("  Run intake_review_boundary.py with --unsure-from to target only UNSURE clusters:")
            print(
                f"  python3 scripts/intake_review_boundary.py \\\n"
                f"    --clusters <clusters_path> \\\n"
                f"    --export-dir \"<your-export-dir>\" \\\n"
                f"    --unsure-from <decisions_path>"
            )
        if is_incomplete:
            print(f"\n⚠️  {len(missing)} clusters missing decisions (--partial mode).")
            print("  Complete the review and re-run without --partial to finalize.")

        return 2  # valid but non-final: UNSURE decisions or partial-incomplete


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest manual review decisions and finalize dedup provisional flag.",
    )
    p.add_argument("--decisions", required=True, type=Path, help="Path to review_decisions.json")
    p.add_argument("--clusters", required=True, type=Path, help="Path to dedup_clusters.jsonl")
    p.add_argument("--summary", required=True, type=Path, help="Path to dedup_summary.json")
    p.add_argument(
        "--output", required=True, type=Path,
        help="Output path for dedup_review_summary.json (tracked, no PII)",
    )
    p.add_argument(
        "--fp-threshold", type=float, default=DEFAULT_FP_THRESHOLD, metavar="RATE",
        help=f"FP rate above which threshold-lowering is recommended (default: {DEFAULT_FP_THRESHOLD})",
    )
    p.add_argument(
        "--partial", action="store_true",
        help="Allow incomplete decisions (provisional flag will NOT be cleared)",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Compute and print results without modifying dedup_summary.json",
    )
    p.add_argument(
        "--force", action="store_true",
        help=(
            "Override safety guards: allow overwriting a finalized output summary "
            "and allow writing contradictory state when source is already finalized."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return run(
        decisions_path=args.decisions,
        clusters_path=args.clusters,
        summary_path=args.summary,
        output_path=args.output,
        fp_threshold=args.fp_threshold,
        partial=args.partial,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
