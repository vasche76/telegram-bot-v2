"""
intake_filter_heuristic.py — Per-image weak-signal heuristic extraction (U4 Phase B).

Joins filter_universe.jsonl (geometry signals) with manifest.jsonl (caption keywords)
to produce a deterministic signal vector for each of the 32,420 unique universe images.

Writes:
  filter_signals.jsonl          — local-only (gitignored); one record per image
  filter_signals_summary.json   — tracked (aggregate counts only, no PII)

Caption signals are WEAK HINTS only. They do NOT assign truth labels and MUST NOT
override visual/manual truth. Conflict resolution is the Phase C classifier's job.

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


# ─── Signal helpers ───────────────────────────────────────────────────────────


def _classify_aspect(ar: float) -> str:
    if ar < 0.5:
        return "extreme_portrait"
    if ar < 0.8:
        return "portrait"
    if ar < 1.25:
        return "square"
    if ar < 2.0:
        return "landscape"
    return "extreme_landscape"


def _classify_file_size(file_size: int) -> str:
    if file_size < C.FILE_SIZE_BUCKET_TINY_MAX:
        return "tiny"
    if file_size < C.FILE_SIZE_BUCKET_SMALL_MAX:
        return "small"
    if file_size < C.FILE_SIZE_BUCKET_MEDIUM_MAX:
        return "medium"
    return "large"


# ─── Core logic ───────────────────────────────────────────────────────────────


def build_signals(
    universe_path: Path,
    manifest_path: Path,
    output_dir: Path,
    dry_run: bool = False,
    with_image_stats: bool = False,
) -> dict:
    """
    Compute per-image signal vectors and write output files.

    Returns the summary dict.
    """
    if with_image_stats:
        log.warning(
            "--with-image-stats: image stats not yet implemented; "
            "image_stats_computed=False for all records."
        )

    # --- Load universe (hard fail if missing) ---
    if not universe_path.exists():
        log.error("filter_universe.jsonl not found: %s", universe_path)
        sys.exit(1)

    universe = _read_jsonl(universe_path)
    log.info("Universe loaded: %d records from %s", len(universe), universe_path)

    # --- Load manifest (soft fail if missing) ---
    manifest_by_fn: dict[str, dict] = {}
    duplicate_manifest_filename_count = 0
    if manifest_path.exists():
        manifest_records = _read_jsonl(manifest_path)
        for r in manifest_records:
            fn = r["filename"]
            if fn in manifest_by_fn:
                duplicate_manifest_filename_count += 1
            manifest_by_fn[fn] = r  # last-wins
        if duplicate_manifest_filename_count > 0:
            log.warning(
                "Manifest has %d duplicate filename records (last-wins); "
                "distinct filenames retained: %d",
                duplicate_manifest_filename_count,
                len(manifest_by_fn),
            )
        log.info(
            "Manifest loaded: %d records, %d distinct from %s",
            len(manifest_records), len(manifest_by_fn), manifest_path,
        )
    else:
        log.warning(
            "manifest.jsonl not found at %s — proceeding with has_manifest_record=False for all records",
            manifest_path,
        )

    # --- Compute per-record signal vectors ---
    signals: list[dict] = []

    for i, rec in enumerate(universe):
        if i > 0 and i % PROGRESS_EVERY == 0:
            log.info("Progress: %d / %d records processed", i, len(universe))

        # Geometry signals
        width = rec.get("width")
        height = rec.get("height")
        if width is not None and height is not None and width > 0 and height > 0:
            aspect_ratio: float | None = round(width / height, 2)
            aspect_class = _classify_aspect(aspect_ratio)
        else:
            aspect_ratio = None
            aspect_class = "unknown"

        try:
            file_size = rec["file_size"]
            low_res = rec["low_res"]
            dedup_role = rec["dedup_role"]
        except KeyError as exc:
            raise ValueError(
                f"{universe_path}: record {i + 1}: missing required field {exc}"
            ) from exc
        file_size_bucket = _classify_file_size(file_size)

        # Caption signals
        mfst = manifest_by_fn.get(rec["filename"])
        has_manifest_record = mfst is not None
        caption = (mfst.get("caption") or "") if mfst else ""
        caption_length = len(caption)
        caption_empty = caption_length == 0
        text_lower = caption.lower()
        caption_text_heavy = caption_length > C.CAPTION_TEXT_HEAVY_THRESHOLD
        caption_lure_keyword = any(kw in text_lower for kw in C.CAPTION_LURE_KEYWORD_HINTS)
        caption_fish_part_keyword = any(kw in text_lower for kw in C.CAPTION_FISH_PART_KEYWORD_HINTS)
        caption_fry_keyword = any(kw in text_lower for kw in C.CAPTION_FRY_KEYWORD_HINTS)
        caption_no_fish_keyword = any(kw in text_lower for kw in C.CAPTION_NO_FISH_KEYWORD_HINTS)

        # Optional image stats (scaffolded, not computed in Phase B)
        image_stats_computed = False
        mean_luminance = None
        is_grayscale_like = None
        edge_density = None

        signals.append({
            "filename": rec["filename"],
            "sha256": rec.get("sha256"),
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "aspect_class": aspect_class,
            "file_size": file_size,
            "file_size_bucket": file_size_bucket,
            "low_res": low_res,
            "dedup_role": dedup_role,
            "has_manifest_record": has_manifest_record,
            "caption_length": caption_length,
            "caption_empty": caption_empty,
            "caption_text_heavy": caption_text_heavy,
            "caption_lure_keyword": caption_lure_keyword,
            "caption_fish_part_keyword": caption_fish_part_keyword,
            "caption_fry_keyword": caption_fry_keyword,
            "caption_no_fish_keyword": caption_no_fish_keyword,
            "image_stats_computed": image_stats_computed,
            "mean_luminance": mean_luminance,
            "is_grayscale_like": is_grayscale_like,
            "edge_density": edge_density,
            "source": C.SOURCE_TAG,
            "schema_version": SCHEMA_VERSION,
        })

    total_images = len(signals)

    # --- Aggregate counts (no PII) ---
    no_manifest_record_count = sum(1 for s in signals if not s["has_manifest_record"])
    low_res_count = sum(1 for s in signals if s["low_res"])
    extreme_aspect_count = sum(
        1 for s in signals if s["aspect_class"] in {"extreme_portrait", "extreme_landscape"}
    )
    tiny_file_count = sum(1 for s in signals if s["file_size_bucket"] == "tiny")
    caption_lure_keyword_count = sum(1 for s in signals if s["caption_lure_keyword"])
    caption_fish_part_keyword_count = sum(1 for s in signals if s["caption_fish_part_keyword"])
    caption_fry_keyword_count = sum(1 for s in signals if s["caption_fry_keyword"])
    caption_no_fish_keyword_count = sum(1 for s in signals if s["caption_no_fish_keyword"])
    caption_text_heavy_count = sum(1 for s in signals if s["caption_text_heavy"])

    summary = {
        "total_images": total_images,
        "duplicate_manifest_filename_count": duplicate_manifest_filename_count,
        "no_manifest_record_count": no_manifest_record_count,
        "low_res_count": low_res_count,
        "extreme_aspect_count": extreme_aspect_count,
        "tiny_file_count": tiny_file_count,
        "caption_lure_keyword_count": caption_lure_keyword_count,
        "caption_fish_part_keyword_count": caption_fish_part_keyword_count,
        "caption_fry_keyword_count": caption_fry_keyword_count,
        "caption_no_fish_keyword_count": caption_no_fish_keyword_count,
        "caption_text_heavy_count": caption_text_heavy_count,
        "source": C.SOURCE_TAG,
        "license": C.LICENSE_TAG,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
    }

    log.info(
        "Signals computed: total=%d, low_res=%d, extreme_aspect=%d, tiny_file=%d",
        total_images, low_res_count, extreme_aspect_count, tiny_file_count,
    )
    log.info(
        "Caption signals: lure=%d, fish_part=%d, fry=%d, no_fish=%d, text_heavy=%d",
        caption_lure_keyword_count, caption_fish_part_keyword_count,
        caption_fry_keyword_count, caption_no_fish_keyword_count, caption_text_heavy_count,
    )

    if dry_run:
        log.info("[DRY RUN] Would write %d records to filter_signals.jsonl", total_images)
        log.info("[DRY RUN] Summary:\n%s", json.dumps(summary, indent=2))
        return summary

    signals_path = output_dir / "filter_signals.jsonl"
    summary_path = output_dir / "filter_signals_summary.json"

    _write_jsonl(signals_path, signals)
    log.info("Written: %s (%d records)", signals_path, total_images)

    _write_json_atomic(summary_path, summary)
    log.info("Written: %s", summary_path)

    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-image weak-signal heuristic extraction (U4 Phase B).",
    )
    p.add_argument(
        "--universe",
        type=Path,
        default=C.FILTER_UNIVERSE_PATH,
        help="Path to filter_universe.jsonl (default: %(default)s)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=C.MANIFEST_PATH,
        help="Path to manifest.jsonl (default: %(default)s)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=C.INTAKE_META_ROOT,
        help="Output directory for filter_signals.jsonl and filter_signals_summary.json",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute signals but do not write any files; print summary to stderr",
    )
    p.add_argument(
        "--with-image-stats",
        action="store_true",
        help="(Not yet implemented) Accept flag; image stats are nulled in Phase B",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        build_signals(
            universe_path=args.universe,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            with_image_stats=args.with_image_stats,
        )
    except ValueError as exc:
        log.error("Input file error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
