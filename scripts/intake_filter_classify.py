"""
intake_filter_classify.py — Deterministic candidate classification (U4 Phase C).

Reads filter_signals.jsonl (Phase B output) and assigns each image a coarse
candidate_category using a conservative rule waterfall.

CANDIDATE-ONLY: No image is promoted to training staging by this script.
review_required=True for every record. Phase D manual review is authoritative.

Writes:
  filter_candidates.jsonl         — local-only (gitignored); one record per image
  filter_candidates_summary.json  — tracked (aggregate counts only, no PII)

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

SCHEMA_VERSION = 1
PROGRESS_EVERY = 1000

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr)
log = logging.getLogger(__name__)


# ─── I/O helpers (verbatim from intake_filter_universe.py) ───────────────────


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


# ─── Core classification logic ────────────────────────────────────────────────


def classify_record(rec: dict) -> dict:
    """
    Apply the Phase C rule waterfall to one signal record.

    Returns a candidate dict with candidate_category, confidence, review_required,
    reasons, conflicts, and pass-through provenance fields.
    """
    aspect_class: str = rec.get("aspect_class", "unknown")
    file_size_bucket: str = rec.get("file_size_bucket", "medium")
    low_res: bool = bool(rec.get("low_res", False))

    caption_lure: bool = bool(rec.get("caption_lure_keyword", False))
    caption_fish_part: bool = bool(rec.get("caption_fish_part_keyword", False))
    caption_fry: bool = bool(rec.get("caption_fry_keyword", False))
    caption_no_fish: bool = bool(rec.get("caption_no_fish_keyword", False))
    caption_text_heavy: bool = bool(rec.get("caption_text_heavy", False))

    extreme_aspect: bool = aspect_class in ("extreme_portrait", "extreme_landscape")

    # STEP 1 — count active caption keywords
    active_caption_kw_count = sum([caption_lure, caption_fish_part, caption_fry, caption_no_fish])

    # STEP 2 — detect conflicts (evaluated before category assignment)
    conflicts: list[str] = []

    # CONFLICT_A: text_heavy + lure + non-extreme aspect (catch report vs gear post)
    if caption_text_heavy and caption_lure and not extreme_aspect:
        conflicts.append(C.CONFLICT_AMBIGUOUS_TEXT_HEAVY_LURE)

    # CONFLICT_B: two or more distinct caption keyword signals
    if active_caption_kw_count >= 2:
        conflicts.append(C.CONFLICT_COMPETING_CAPTION_KEYWORDS)

    # STEP 3 — category assignment
    reasons: list[str] = []

    if conflicts:
        candidate_category = "unknown_needs_review"
        confidence = C.CONFIDENCE_LOW
    else:
        # RULE 1: poster/screenshot
        if caption_no_fish and caption_text_heavy:
            candidate_category = "poster_screenshot"
            confidence = C.CONFIDENCE_MEDIUM
            reasons.append(C.REASON_CAPTION_NO_FISH_KW)
            reasons.append(C.REASON_CAPTION_TEXT_HEAVY)

        # RULE 2: fish part
        elif caption_fish_part:
            candidate_category = "fish_part"
            confidence = C.CONFIDENCE_LOW
            reasons.append(C.REASON_CAPTION_FISH_PART_KW)

        # RULE 3: fry / juvenile
        elif caption_fry:
            candidate_category = "fry_juvenile"
            confidence = C.CONFIDENCE_LOW
            reasons.append(C.REASON_CAPTION_FRY_KW)

        # RULE 4: default unknown
        else:
            candidate_category = "unknown_needs_review"
            confidence = C.CONFIDENCE_LOW
            # lure hint is surfaced but does NOT assign lure_gear
            if caption_lure:
                reasons.append(C.REASON_CAPTION_LURE_HINT)
            else:
                reasons.append(C.REASON_NO_STRONG_SIGNAL)

    # STEP 4 — append quality/format reason tags (any category)
    if low_res:
        reasons.append(C.REASON_LOW_RES)
    if file_size_bucket == "tiny":
        reasons.append(C.REASON_TINY_FILE)
    if extreme_aspect:
        reasons.append(C.REASON_EXTREME_ASPECT)

    # When conflicted and lure was one of the signals, surface the hint explicitly
    if conflicts and caption_lure and C.CONFLICT_COMPETING_CAPTION_KEYWORDS not in conflicts:
        reasons.append(C.REASON_CAPTION_LURE_HINT)

    # Ensure reasons list is never empty
    if not reasons:
        reasons.append(C.REASON_NO_STRONG_SIGNAL)

    # STEP 5 — finalize
    return {
        "filename": rec.get("filename", ""),
        "sha256": rec.get("sha256"),
        "candidate_category": candidate_category,
        "confidence": confidence,
        "review_required": True,
        "reasons": reasons,
        "conflicts": conflicts,
        "source": rec.get("source", C.SOURCE_TAG),
        "schema_version": SCHEMA_VERSION,
    }


# ─── Orchestration ────────────────────────────────────────────────────────────


def classify_all(
    signals_path: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> dict:
    """
    Classify all records from signals_path and write output files.

    Returns the summary dict.
    """
    if not signals_path.exists():
        log.error("filter_signals.jsonl not found: %s", signals_path)
        sys.exit(1)

    signals = _read_jsonl(signals_path)
    log.info("Signals loaded: %d records from %s", len(signals), signals_path)

    candidates: list[dict] = []
    for i, rec in enumerate(signals):
        if i > 0 and i % PROGRESS_EVERY == 0:
            log.info("Progress: %d / %d records classified", i, len(signals))
        candidates.append(classify_record(rec))

    total_images = len(candidates)

    # --- Aggregate counts (no PII) ---
    by_category: dict[str, int] = {cat: 0 for cat in C.COARSE_CATEGORIES}
    by_confidence: dict[str, int] = {
        C.CONFIDENCE_HIGH: 0,
        C.CONFIDENCE_MEDIUM: 0,
        C.CONFIDENCE_LOW: 0,
    }
    review_required_count = 0
    conflict_flag_count = 0
    by_reason: dict[str, int] = {}

    for cand in candidates:
        by_category[cand["candidate_category"]] = by_category.get(cand["candidate_category"], 0) + 1
        by_confidence[cand["confidence"]] = by_confidence.get(cand["confidence"], 0) + 1
        if cand["review_required"]:
            review_required_count += 1
        if cand["conflicts"]:
            conflict_flag_count += 1
        for reason in cand["reasons"]:
            by_reason[reason] = by_reason.get(reason, 0) + 1

    summary = {
        "total_images": total_images,
        "by_candidate_category": {cat: by_category.get(cat, 0) for cat in C.COARSE_CATEGORIES},
        "by_confidence": by_confidence,
        "review_required_count": review_required_count,
        "conflict_flag_count": conflict_flag_count,
        "by_reason": by_reason,
        "source": C.SOURCE_TAG,
        "license": C.LICENSE_TAG,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
    }

    log.info(
        "Classification complete: total=%d, review_required=%d, conflicts=%d",
        total_images, review_required_count, conflict_flag_count,
    )
    log.info(
        "Categories: poster_screenshot=%d, fish_part=%d, fry_juvenile=%d, unknown=%d",
        by_category.get("poster_screenshot", 0),
        by_category.get("fish_part", 0),
        by_category.get("fry_juvenile", 0),
        by_category.get("unknown_needs_review", 0),
    )
    log.info(
        "Confidence: high=%d, medium=%d, low=%d",
        by_confidence[C.CONFIDENCE_HIGH],
        by_confidence[C.CONFIDENCE_MEDIUM],
        by_confidence[C.CONFIDENCE_LOW],
    )

    if dry_run:
        log.info("[DRY RUN] Would write %d records to filter_candidates.jsonl", total_images)
        log.info("[DRY RUN] Summary:\n%s", json.dumps(summary, indent=2))
        return summary

    candidates_path = output_dir / "filter_candidates.jsonl"
    summary_path = output_dir / "filter_candidates_summary.json"

    _write_jsonl(candidates_path, candidates)
    log.info("Written: %s (%d records)", candidates_path, total_images)

    _write_json_atomic(summary_path, summary)
    log.info("Written: %s", summary_path)

    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deterministic candidate classification (U4 Phase C).",
    )
    p.add_argument(
        "--signals",
        type=Path,
        default=C.FILTER_SIGNALS_PATH,
        help="Path to filter_signals.jsonl (default: %(default)s)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=C.INTAKE_META_ROOT,
        help="Output directory for filter_candidates.jsonl and filter_candidates_summary.json",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute classification but do not write any files; print summary to stderr",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        classify_all(
            signals_path=args.signals,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        log.error("Input file error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
