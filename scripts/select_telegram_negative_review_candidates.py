"""
select_telegram_negative_review_candidates.py

Active learning candidate selection for Telegram-domain negatives.
Read-only shadow inference + deterministic candidate selection → new HTML review batches.

OBJECTIVE:
  The MVP structural model (mvp_structural_v1.pt) was evaluated on 500 human-reviewed
  Telegram thumbnails and showed unacceptable false-accept rate for negatives (FAR ~90%).
  This script selects a focused manual review batch of Telegram-domain negatives / hard
  cases to collect the ~200-500 new human labels needed before v2 retraining.

WHAT THIS SCRIPT DOES:
  1. Loads Phase C filter_candidates.jsonl (32,420 records).
  2. Excludes already-reviewed records (Phase D batches 0001-0002, 500 records).
  3. Builds signal-based triage buckets from Phase C metadata.
  4. Runs shadow inference using mvp_structural_v1.pt on a bounded pool.
  5. Selects a compact, high-value review batch (target 250-500 records).
  6. Generates EXIF-stripped thumbnails and HTML review batches with annotation guide.
  7. Writes detailed predictions to private/ (gitignored).
  8. Writes a privacy-safe counts-only summary (tracked).

WHAT THIS SCRIPT DOES NOT DO:
  - Does NOT train, fine-tune, or retrain any model.
  - Does NOT write model predictions as final labels.
  - Does NOT mutate Phase C outputs or Phase D reviewed batches.
  - Does NOT overwrite existing reviewed batches (0001, 0002).
  - Does NOT upload or transmit any private images.
  - Does NOT use GPT Vision or any external API.

PRIVACY CONTRACT:
  - filename, sha256, captions are NEVER written to HTML or decision templates.
  - Review IDs are HMAC-derived using the same local secret as Phase D.
  - Model predictions are NOT shown in HTML review UI (prevents reviewer bias).
  - Candidates are marked: needs_human_review / model_prediction_triage_only / not_truth_label.
  - Detailed shadow outputs → private/active_learning/... (gitignored).
  - HTML batches → data/intake_meta/tg_2026-04-24/review/ (gitignored).
  - Tracked summary → docs/ml/ or data/fish_models/ (counts-only).

source=telegram_private_2026-04-24, license=private_training_only
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import html as html_module
import json
import logging
import os
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intake_constants as C

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr)
log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

SCHEMA_VERSION = "al_negative_selection_v1"
DECISION_SCHEMA_VERSION = "u4_phase_d_decisions_v1"
SOURCE_TAG = C.SOURCE_TAG

# Model class expectations (must match mvp_structural_v1.pt training)
MODEL_CLASS_FISH = "fish"
MODEL_CLASS_NOT_FISH = "not_fish_or_other"

# Active learning run prefix — distinguishes from regular Phase D review runs
AL_RUN_PREFIX = "alvrun"

# Selection modes
MODE_DEFAULT = "default"
MODE_NEGATIVE_FOCUSED_V2 = "negative-focused-v2"

# Candidate selection parameters
DEFAULT_TARGET_SIZE = 500
BATCH_SIZE = 250
RANDOM_SEED = 42

# Inference pool: run inference on at most this many records
# (covers all signal-bearing records + a random sample of no-signal records)
# Bounded to keep inference time reasonable: ~1500 signal + ~1000 no-signal ≈ 2500 total
MAX_INFERENCE_POOL = 2500

# Fish confidence thresholds for model-based bucketing
# Bucket D: fish_conf <= UNCERTAIN_FISH_CONF_MAX (inclusive boundary, no gap)
# Bucket E: fish_conf > UNCERTAIN_FISH_CONF_MAX (exclusive, no double-count)
UNCERTAIN_FISH_CONF_MAX = 0.80

# Bucket allocation caps — default mode (sum ~500)
BUCKET_A_CAP = 25    # Phase C non-fish categories (fish_part, fry_juvenile, poster_screenshot)
BUCKET_B_CAP = 200   # Caption keyword signals → likely negatives
BUCKET_C_CAP = 100   # Quality/conflict signals (low_res, extreme_aspect, tiny_file, conflict)
BUCKET_D_CAP = 125   # Model uncertainty (low fish_conf from inference)
BUCKET_E_CAP = 50    # Random controls (likely fish, for calibration)

# Bucket allocation caps — negative-focused-v2 mode (sum ~240, target 250)
# Pass 2 rationale:
#   A: all 22 Phase C negatives exhausted in pass 1 → cap at 5 (stragglers only)
#   B: sort ascending fish_conf (lowest first) — records where model agrees NOT fish
#      AND has a negative caption hint → stronger negative evidence than pass 1's
#      high-fish-conf sort
#   C: quality signals remain; keep a moderate allocation
#   D: emphasize strong model not_fish (fish_conf < V2_STRONG_NOT_FISH_CONF_MAX) before
#      medium uncertainty — these were under-represented in pass 1
#   E: minimal fish controls; we already have 845 fish labels
V2_BUCKET_A_CAP = 5
V2_BUCKET_B_CAP = 70
V2_BUCKET_C_CAP = 60
V2_BUCKET_D_CAP = 90
V2_BUCKET_E_CAP = 15
# Bucket D v2 sub-bucket boundary: fish_conf < 0.25 → strong not_fish (strict <).
# fish_conf == 0.25 goes to medium uncertainty (>=). Non-overlapping by design.
V2_STRONG_NOT_FISH_CONF_MAX = 0.25

# Phase C categories counted as candidate negatives/hard-cases
PHASE_C_NEGATIVE_CATEGORIES = frozenset({"fish_part", "fry_juvenile", "poster_screenshot"})

# Caption reason codes that signal likely negatives
CAPTION_NEGATIVE_REASONS = frozenset({
    "caption_lure_hint",
    "caption_no_fish_keyword",
    "caption_text_heavy",
    "caption_fish_part_keyword",
    "caption_fry_keyword",
})

# Quality signal reasons
QUALITY_SIGNAL_REASONS = frozenset({"low_res", "extreme_aspect", "tiny_file"})

# Thumbnail dimensions for review UI (match existing Phase D thumbnails)
THUMBNAIL_MAX_SIZE = (320, 320)
THUMBNAIL_QUALITY = 75

# ─── I/O helpers ─────────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: corrupt JSONL: {exc.msg}") from exc
    return records


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


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _append_jsonl_atomic(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─── Secret + Review ID ───────────────────────────────────────────────────────


def _load_secret(secret_path: Path) -> bytes:
    """Load existing review HMAC secret. Never creates a new one — must already exist."""
    if not secret_path.exists():
        raise FileNotFoundError(
            f"Review secret not found at {secret_path}. "
            "Run intake_review_prepare.py first to establish the secret."
        )
    raw = secret_path.read_bytes().strip()
    if len(raw) == 64 and all(c in b"0123456789abcdefABCDEF" for c in raw):
        return bytes.fromhex(raw.decode("ascii"))
    return raw


def compute_review_id(secret: bytes, sha256: str) -> str:
    """Compute deterministic HMAC-based review ID. Matches Phase D convention."""
    digest = hmac.new(secret, sha256.encode("utf-8"), hashlib.sha256).hexdigest()
    return "rv_" + digest[:16]


# ─── Candidate loading ────────────────────────────────────────────────────────


def load_candidates(candidates_path: Path) -> list[dict]:
    """Load Phase C filter_candidates.jsonl."""
    if not candidates_path.exists():
        raise FileNotFoundError(f"filter_candidates.jsonl not found: {candidates_path}")
    candidates = _read_jsonl(candidates_path)
    log.info("Loaded %d Phase C candidates from %s", len(candidates), candidates_path)
    return candidates


def load_reviewed_ids(aggregate_path: Path) -> set[str]:
    """Load review_ids already reviewed from the Phase D partial aggregate JSON."""
    if not aggregate_path.exists():
        log.warning("No Phase D aggregate found at %s — assuming 0 reviewed records.", aggregate_path)
        return set()
    with aggregate_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    records = data.get("records", [])
    ids = {r["review_id"] for r in records if r.get("review_id")}
    log.info("Loaded %d already-reviewed IDs from %s", len(ids), aggregate_path)
    return ids


def load_reviewed_ids_from_decisions_dir(review_dir: Path, run_id_prefix: str) -> set[str]:
    """
    Load all reviewed review_ids by scanning individual decision files whose name
    starts with run_id_prefix. Accepts both a full run_id (e.g. 'rvrun_20260427T184629Z')
    and a run-prefix (e.g. 'alvrun') for matching all active-learning runs.
    Returns a set of review_ids that have been reviewed (have a non-null final_category).
    """
    ids: set[str] = set()
    for dec_file in sorted(review_dir.glob(f"filter_decisions_{run_id_prefix}*.json")):
        if dec_file.name.endswith(".blank_backup.json"):
            continue
        try:
            with dec_file.open(encoding="utf-8") as fh:
                data = json.load(fh)
            for r in data.get("records", []):
                if r.get("final_category") and r.get("review_id"):
                    ids.add(r["review_id"])
        except Exception as exc:
            log.warning("Could not read %s: %s", dec_file.name, exc)
    return ids


# ─── Signal extraction ────────────────────────────────────────────────────────


def _get_signals(rec: dict) -> dict[str, bool]:
    reasons: set[str] = set(rec.get("reasons", []))
    conflicts: list = rec.get("conflicts", [])
    cat = rec.get("candidate_category", "unknown_needs_review")
    return {
        "is_phase_c_negative": cat in PHASE_C_NEGATIVE_CATEGORIES,
        "has_caption_negative": bool(CAPTION_NEGATIVE_REASONS & reasons),
        "has_quality_signal": bool(QUALITY_SIGNAL_REASONS & reasons),
        "has_conflict": bool(conflicts),
        "low_res": "low_res" in reasons,
        "extreme_aspect": "extreme_aspect" in reasons,
        "tiny_file": "tiny_file" in reasons,
        "caption_keyword_signal_present": bool(CAPTION_NEGATIVE_REASONS & reasons),
        "phase_c_conflict": bool(conflicts),
        "phase_c_candidate_category": cat,
    }


def _has_any_signal(signals: dict) -> bool:
    return (
        signals["is_phase_c_negative"]
        or signals["has_caption_negative"]
        or signals["has_quality_signal"]
        or signals["has_conflict"]
    )


# ─── Shadow inference ─────────────────────────────────────────────────────────


def run_shadow_inference(
    model_path: Path,
    image_paths: list[Path],
    device: str = "cpu",
) -> list[dict | None]:
    """
    Run YOLOv8 classification inference (shadow/read-only).
    Returns list of {predicted_class, fish_conf, not_fish_conf} or None per image.
    Never writes predictions as truth labels.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        raise ImportError("ultralytics not installed. Use venv_ml/bin/python3.")

    log.info("Loading model from %s (device=%s)...", model_path, device)
    model = YOLO(str(model_path))

    names = model.names
    if names.get(0) != MODEL_CLASS_FISH or names.get(1) != MODEL_CLASS_NOT_FISH:
        raise ValueError(
            f"Unexpected model class names: {names}. "
            f"Expected {{0: '{MODEL_CLASS_FISH}', 1: '{MODEL_CLASS_NOT_FISH}'}}"
        )

    results: list[dict | None] = []
    total = len(image_paths)
    ok_count = 0
    fail_count = 0
    for i, img_path in enumerate(image_paths):
        if i % 200 == 0:
            log.info("  Inference progress: %d/%d (ok=%d, fail=%d)", i, total, ok_count, fail_count)
        try:
            preds = model(str(img_path), verbose=False)
            if not preds:
                results.append(None)
                fail_count += 1
                continue
            pred = preds[0]
            probs = pred.probs
            fish_conf = float(probs.data[0])
            not_fish_conf = float(probs.data[1])
            top_idx = int(probs.top1)
            predicted_class = names[top_idx]
            results.append({
                "predicted_class": predicted_class,
                "fish_conf": round(fish_conf, 4),
                "not_fish_conf": round(not_fish_conf, 4),
                # Mark explicitly — these are NOT labels
                "triage_only": True,
                "not_truth_label": True,
            })
            ok_count += 1
        except Exception as exc:
            log.warning("Inference failed for %s: %s", img_path.name, exc)
            results.append(None)
            fail_count += 1

    log.info("Inference complete: %d ok, %d failed out of %d", ok_count, fail_count, total)
    return results


# ─── Candidate selection ──────────────────────────────────────────────────────


def select_candidates(
    unreviewed: list[dict],
    predictions: dict[str, dict | None],
    target_size: int,
    rng: random.Random,
    mode: str = MODE_DEFAULT,
) -> tuple[list[dict], dict[str, int]]:
    """
    Deterministic candidate selection across five buckets.

    Bucket A: Phase C non-fish categories (fish_part, fry_juvenile, poster_screenshot)
    Bucket B: Caption keyword negative signals
    Bucket C: Quality/conflict signals (low_res, extreme_aspect, tiny_file, conflict)
    Bucket D: Model uncertainty / strong not_fish (fish_conf < threshold from inference)
    Bucket E: Random controls (likely fish, small calibration set)

    mode=MODE_DEFAULT: original caps, Bucket B sorted DESC fish_conf.
    mode=MODE_NEGATIVE_FOCUSED_V2: reduced caps, Bucket B sorted ASC fish_conf,
      Bucket D prioritises strong not_fish (fish_conf < V2_STRONG_NOT_FISH_CONF_MAX).

    Returns (selected_records, bucket_counts).
    Each record gets: candidate_reason, candidate_bucket, model_prediction_triage_only.
    """
    v2 = (mode == MODE_NEGATIVE_FOCUSED_V2)
    cap_a = V2_BUCKET_A_CAP if v2 else BUCKET_A_CAP
    cap_b = V2_BUCKET_B_CAP if v2 else BUCKET_B_CAP
    cap_c = V2_BUCKET_C_CAP if v2 else BUCKET_C_CAP
    cap_d = V2_BUCKET_D_CAP if v2 else BUCKET_D_CAP
    cap_e = V2_BUCKET_E_CAP if v2 else BUCKET_E_CAP

    selected_ids: set[str] = set()
    bucket_counts: dict[str, int] = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}

    def _tag(rec: dict, bucket: str, reason: str) -> dict:
        rv_id = rec["review_id"]
        pred = predictions.get(rv_id)
        tagged = dict(rec)
        tagged["candidate_bucket"] = bucket
        tagged["candidate_reason"] = reason
        tagged["needs_human_review"] = True
        tagged["model_prediction_triage_only"] = pred  # NOT a label
        tagged["not_truth_label"] = True
        return tagged

    selected: list[dict] = []

    # ── Bucket A: Phase C non-fish categories ────────────────────────────────
    bucket_a = [
        r for r in unreviewed
        if r["review_id"] not in selected_ids
        and _get_signals(r)["is_phase_c_negative"]
    ]
    bucket_a.sort(key=lambda r: r["review_id"])
    for rec in bucket_a[:cap_a]:
        selected.append(_tag(rec, "A", f"phase_c_category:{rec.get('candidate_category')}"))
        selected_ids.add(rec["review_id"])
        bucket_counts["A"] += 1

    # ── Bucket B: Caption keyword negative signals ───────────────────────────
    bucket_b_all = [
        r for r in unreviewed
        if r["review_id"] not in selected_ids
        and _get_signals(r)["has_caption_negative"]
    ]

    if v2:
        # v2: ascending fish_conf — records where model predicts NOT fish AND has a
        # negative caption hint → stronger negative evidence. Records with no inference
        # (fish_conf=0.0 placeholder) appear first; stable secondary sort by review_id.
        def _b_sort_key_v2(r: dict) -> tuple:
            pred = predictions.get(r["review_id"])
            fish_conf = pred["fish_conf"] if pred else 0.0
            return (fish_conf, r["review_id"])
        bucket_b_all.sort(key=_b_sort_key_v2)
    else:
        # default: descending fish_conf — suspicious potential FPs (model says fish but
        # caption hints at lure/gear) are prioritised.
        def _b_sort_key(r: dict) -> tuple:
            pred = predictions.get(r["review_id"])
            fish_conf = pred["fish_conf"] if pred else 0.0
            return (-fish_conf, r["review_id"])
        bucket_b_all.sort(key=_b_sort_key)

    for rec in bucket_b_all[:cap_b]:
        reasons = set(rec.get("reasons", []))
        active_reasons = sorted(CAPTION_NEGATIVE_REASONS & reasons)
        selected.append(_tag(rec, "B", "caption_negative_signal:" + ",".join(active_reasons)))
        selected_ids.add(rec["review_id"])
        bucket_counts["B"] += 1

    # ── Bucket C: Quality/conflict signals ───────────────────────────────────
    bucket_c_all = [
        r for r in unreviewed
        if r["review_id"] not in selected_ids
        and (_get_signals(r)["has_quality_signal"] or _get_signals(r)["has_conflict"])
    ]
    bucket_c_all.sort(key=lambda r: r["review_id"])
    for rec in bucket_c_all[:cap_c]:
        signals = _get_signals(rec)
        qs = []
        if signals["low_res"]: qs.append("low_res")
        if signals["extreme_aspect"]: qs.append("extreme_aspect")
        if signals["tiny_file"]: qs.append("tiny_file")
        if signals["has_conflict"]: qs.append("conflict")
        selected.append(_tag(rec, "C", "quality_or_conflict:" + ",".join(qs)))
        selected_ids.add(rec["review_id"])
        bucket_counts["C"] += 1

    # ── Bucket D: Model uncertainty / strong not_fish from inference ─────────
    # Default: candidates where fish_conf <= UNCERTAIN_FISH_CONF_MAX, asc sort.
    # v2: strong not_fish (fish_conf < V2_STRONG_NOT_FISH_CONF_MAX) are taken first
    #     to prioritise the highest-confidence model negatives, then medium uncertainty
    #     (V2_STRONG_NOT_FISH_CONF_MAX <= fish_conf <= UNCERTAIN_FISH_CONF_MAX).
    bucket_d_all = [
        r for r in unreviewed
        if r["review_id"] not in selected_ids
        and r["review_id"] in predictions
        and predictions[r["review_id"]] is not None
        and predictions[r["review_id"]]["fish_conf"] <= UNCERTAIN_FISH_CONF_MAX
    ]

    if v2:
        # Sub-buckets: strong not_fish first (ascending), then medium uncertainty (ascending).
        bucket_d_strong = sorted(
            [r for r in bucket_d_all if predictions[r["review_id"]]["fish_conf"] < V2_STRONG_NOT_FISH_CONF_MAX],
            key=lambda r: (predictions[r["review_id"]]["fish_conf"], r["review_id"]),
        )
        bucket_d_medium = sorted(
            [r for r in bucket_d_all if predictions[r["review_id"]]["fish_conf"] >= V2_STRONG_NOT_FISH_CONF_MAX],
            key=lambda r: (predictions[r["review_id"]]["fish_conf"], r["review_id"]),
        )
        bucket_d_ordered = bucket_d_strong + bucket_d_medium
    else:
        bucket_d_ordered = sorted(
            bucket_d_all,
            key=lambda r: (predictions[r["review_id"]]["fish_conf"], r["review_id"]),
        )

    for rec in bucket_d_ordered[:cap_d]:
        pred = predictions[rec["review_id"]]
        fc = pred["fish_conf"]
        label = "strong_not_fish" if (v2 and fc < V2_STRONG_NOT_FISH_CONF_MAX) else "model_uncertainty"
        selected.append(_tag(rec, "D", f"{label}:fish_conf={fc:.3f}"))
        selected_ids.add(rec["review_id"])
        bucket_counts["D"] += 1

    # ── Bucket E: Random controls (high-fish-conf, no-signal candidates) ─────
    # Keep small — we already have enough fish. This is just for calibration.
    # In v2 mode, cap is much smaller (V2_BUCKET_E_CAP=15) so generic
    # unknown_needs_review without any negative signal stays ≤ 6% of batch.
    bucket_e_pool = [
        r for r in unreviewed
        if r["review_id"] not in selected_ids
        and not _has_any_signal(_get_signals(r))
        and (
            r["review_id"] not in predictions
            or predictions.get(r["review_id"]) is None
            or predictions[r["review_id"]]["fish_conf"] > UNCERTAIN_FISH_CONF_MAX
        )
    ]
    rng.shuffle(bucket_e_pool)
    for rec in bucket_e_pool[:cap_e]:
        selected.append(_tag(rec, "E", "random_control:likely_fish"))
        selected_ids.add(rec["review_id"])
        bucket_counts["E"] += 1

    log.info(
        "Candidate selection complete: total=%d | A=%d B=%d C=%d D=%d E=%d | mode=%s",
        len(selected), bucket_counts["A"], bucket_counts["B"],
        bucket_counts["C"], bucket_counts["D"], bucket_counts["E"], mode,
    )
    return selected, bucket_counts


# ─── Thumbnail generation ─────────────────────────────────────────────────────


def generate_thumbnail(src_path: Path, dst_path: Path) -> bool:
    """Generate EXIF-stripped thumbnail at dst_path. Returns True on success."""
    try:
        from PIL import Image  # type: ignore
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img.thumbnail(THUMBNAIL_MAX_SIZE, Image.LANCZOS)
            img.save(dst_path, "JPEG", quality=THUMBNAIL_QUALITY, optimize=True)
        return True
    except ImportError:
        log.warning("Pillow not installed — thumbnails skipped.")
        return False
    except Exception as exc:
        log.warning("Thumbnail failed for %s: %s", dst_path.name, exc)
        return False


# ─── Annotation guide HTML ────────────────────────────────────────────────────


def _build_annotation_guide_html() -> str:
    """Build the detailed annotation guide HTML block."""
    return """
<details class="annotation-guide" open>
  <summary class="guide-toggle">📋 ANNOTATION GUIDE — READ BEFORE REVIEWING (click to collapse)</summary>
  <div class="guide-body">

    <div class="guide-intro">
      <strong>Purpose of this batch:</strong> These images were selected by an active-learning pipeline
      focused on <em>Telegram-domain negatives and hard cases</em>. Many images here are NOT fish.
      Your task is to classify each image using the categories below.<br><br>
      <strong>⚠ IMPORTANT:</strong> Do NOT pre-assume images are fish. This batch was intentionally
      skewed toward non-fish, hard cases, lures, quality failures, and ambiguous photos.<br><br>
      <strong>Model triage was used ONLY to select this batch — no model predictions are shown.
      Classify based on what you SEE in the image.</strong>
    </div>

    <div class="guide-section">
      <h3>Decision Tree — Use This First</h3>
      <ol class="decision-tree">
        <li>Is the image <strong>unreadable</strong> due to quality (too blurry, too dark, corrupted)?
            → <code>bad_quality</code></li>
        <li>Is it a <strong>screenshot, poster, meme, app screen, text-heavy image, or social-media repost</strong>?
            → <code>poster_screenshot</code></li>
        <li>Is a <strong>real, whole or mostly-whole fish</strong> clearly visible?
            → <code>fish</code></li>
        <li>Is <strong>only part of a fish</strong> visible (head, tail, fillet, close-up)?
            → <code>fish_part</code></li>
        <li>Is the main subject <strong>fishing gear / lure / tackle / bait / rod / reel / net</strong>?
            → <code>lure_gear</code></li>
        <li>Is it a <strong>normal photo from a fishing/trip context but no fish is visible</strong>?
            → <code>no_fish</code></li>
        <li>Is it <strong>clear but unrelated to fish or fishing</strong>?
            → <code>out_of_scope</code></li>
        <li>Is it an <strong>obvious duplicate</strong> of another image in this batch?
            → <code>duplicate_suspect</code></li>
        <li>Does not fit any of the above clearly?
            → <code>unsure</code></li>
      </ol>
    </div>

    <div class="guide-section">
      <h3>Category Definitions</h3>

      <div class="cat-block">
        <div class="cat-name">fish</div>
        <div class="cat-body">
          Use when: A <em>real</em> fish is visible. Fish is whole or mostly whole. Fish is the main subject or one of the main subjects.
          Species unknown is still <code>fish</code>. Fish may be in hands, on ground, in a boat, near water, in a net, or on a table.
          <div class="cat-examples">✓ Person holding a full fish. Fish on grass/snow/deck/scale. Several whole fish. Fish in landing net (fish clearly visible). Partly occluded but mostly recognizable.</div>
          <div class="cat-warning">✗ Only a tiny fragment (tail, head, fillet) → use <code>fish_part</code>.</div>
        </div>
      </div>

      <div class="cat-block">
        <div class="cat-name">fish_part</div>
        <div class="cat-body">
          Use when: Only <em>part</em> of a fish is shown. Not enough for normal recognition.
          <div class="cat-examples">✓ Only fish head. Only tail. Only fillet. Only gills/scales. Very close-up of fish mouth or eye. Pile of fish parts.</div>
          <div class="cat-tie">Tie-break: most of body visible → <code>fish</code>. Only fragment → <code>fish_part</code>. Cooked dish where body is gone → <code>out_of_scope</code>.</div>
        </div>
      </div>

      <div class="cat-block">
        <div class="cat-name">no_fish</div>
        <div class="cat-body">
          Use when: Normal photo, no fish. Not gear. Not a screenshot. Could plausibly be from a fishing trip.
          <div class="cat-examples">✓ River/lake/forest/boat without fish. Empty net. Empty bucket. People or hands (no fish). Water surface without fish.</div>
          <div class="cat-tie">Tie-break: gear/lure → <code>lure_gear</code>. Screenshot/poster → <code>poster_screenshot</code>. Completely unrelated → <code>out_of_scope</code>.</div>
        </div>
      </div>

      <div class="cat-block">
        <div class="cat-name">lure_gear</div>
        <div class="cat-body">
          Use when: Main subject is fishing gear, lure, bait, hook, rod, reel, tackle, box, line, or net — NO real fish as main subject.
          <strong>Artificial lures that look like fish → still <code>lure_gear</code>, not <code>fish</code>.</strong>
          <div class="cat-examples">✓ Wobbler, spinner, spoon, fly, jig, soft plastic lure. Hook, tackle box, fishing rod, reel. Net only. Fishing bait.</div>
          <div class="cat-tie">Tie-break: both fish and gear visible → choose by main subject. Fish is main → <code>fish</code>. Lure is main, fish absent → <code>lure_gear</code>.</div>
        </div>
      </div>

      <div class="cat-block">
        <div class="cat-name">poster_screenshot</div>
        <div class="cat-body">
          Use when: Not a natural camera photo — it's a screenshot, poster, meme, app screen, website, video frame, social-media repost, or text-heavy image.
          <div class="cat-examples">✓ Telegram/website/app screenshot. Meme or poster with fish. Advertisement. Screenshot of a fish photo from another platform. Video thumbnail. Image with large text. Weather/fishing app screenshot.</div>
          <div class="cat-tie">Tie-break: real photo with small timestamp watermark only → classify by actual content, not <code>poster_screenshot</code>. If clearly a screenshot/repost even with a fish → <code>poster_screenshot</code>.</div>
        </div>
      </div>

      <div class="cat-block">
        <div class="cat-name">bad_quality</div>
        <div class="cat-body">
          Use when: Quality is too poor to make a reliable decision. You cannot tell if fish is present.
          <div class="cat-examples">✓ Severe motion blur. Nearly black image. Very tiny object. Corrupted or partially loaded image. Extreme compression artifacts.</div>
          <div class="cat-tie">Tie-break: can clearly identify content despite low quality → choose content category. If image is clear but irrelevant → <code>out_of_scope</code> or <code>no_fish</code>, not <code>bad_quality</code>.</div>
        </div>
      </div>

      <div class="cat-block">
        <div class="cat-name">out_of_scope</div>
        <div class="cat-body">
          Use when: Clear and understandable, but outside the fish-recognition task. Not useful for fish/not-fish training in a fishing bot context.
          <div class="cat-examples">✓ Food, drinks, people posing without fish, pets, cars, buildings, documents, household items. Random memes without fish. Non-fishing travel photos. Animals that are not fish. General screenshots unrelated to fishing.</div>
          <div class="cat-tie">Tie-break: screenshot/poster/app → <code>poster_screenshot</code>. Fishing trip image no fish → <code>no_fish</code>. Fishing gear → <code>lure_gear</code>. Clear and unrelated → <code>out_of_scope</code>.</div>
        </div>
      </div>

      <div class="cat-block">
        <div class="cat-name">duplicate_suspect</div>
        <div class="cat-body">
          Use ONLY when: Obviously the same photo as another reviewed image. Same fish/person/background, essentially identical.
          <div class="cat-warning">Do NOT overuse. If not sure → classify by content. Same fish from a different angle → classify by content (usually <code>fish</code>).</div>
        </div>
      </div>

      <div class="cat-block">
        <div class="cat-name">unsure</div>
        <div class="cat-body">
          Use when: Does not fit cleanly into any category and you genuinely cannot decide. Object might be fish or lure — impossible to tell. Image might be screenshot or photo — unclear.
          <div class="cat-tie">Prefer a specific category if reasonably possible. Use <code>unsure</code> only when a specific choice would be a guess.</div>
        </div>
      </div>
    </div>

    <div class="guide-section">
      <h3>Confidence Scale (1–5)</h3>
      <table class="conf-table">
        <tr><td><strong>5</strong></td><td>Completely clear — obvious case, no doubt.</td></tr>
        <tr><td><strong>4</strong></td><td>Confident — clear enough, minor uncertainty.</td></tr>
        <tr><td><strong>3</strong></td><td>Acceptable — moderate confidence, plausible classification.</td></tr>
        <tr><td><strong>2</strong></td><td>Somewhat unsure — could be wrong.</td></tr>
        <tr><td><strong>1</strong></td><td>Very unsure — guessing.</td></tr>
      </table>
      <p class="guide-note">
        Low confidence does NOT mean bad quality. Use <code>bad_quality</code> only when
        image quality prevents classification. Low confidence means <em>you</em> are uncertain.
      </p>
    </div>

    <div class="guide-section">
      <h3>Anti-Bias Reminder</h3>
      <ul>
        <li>This batch was <strong>actively selected to find negatives and hard cases</strong>. Expect more non-fish than usual.</li>
        <li>Model predictions are intentionally <strong>hidden</strong> from this interface. Do not try to infer them.</li>
        <li>Lures that look like fish are <strong>lure_gear</strong>, not fish.</li>
        <li>Screenshots of fish photos are <strong>poster_screenshot</strong>.</li>
        <li>When in doubt, use <code>unsure</code> with low confidence rather than guessing.</li>
      </ul>
    </div>

    <div class="guide-section">
      <h3>After Reviewing This Batch</h3>
      <ol>
        <li>Complete ALL images in this batch.</li>
        <li>Click <strong>Export Decisions JSON</strong> at the top.</li>
        <li>Save the downloaded JSON file to:
            <code>data/intake_meta/tg_2026-04-24/review/</code></li>
        <li>Do NOT modify the exported JSON file.</li>
        <li>The next Claude Code task: ingest the exported review JSON and validate labels.</li>
      </ol>
    </div>

  </div>
</details>
"""


def _build_annotation_guide_style() -> str:
    return """
    .annotation-guide { background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
      margin-bottom: 20px; padding: 0; }
    .guide-toggle { cursor: pointer; padding: 12px 16px; font-size: 0.95rem; font-weight: bold;
      color: #f0c040; list-style: none; }
    .guide-toggle::-webkit-details-marker { display: none; }
    .guide-body { padding: 16px; display: flex; flex-direction: column; gap: 16px; }
    .guide-intro { background: #1c2028; border-left: 4px solid #e63946; padding: 12px;
      border-radius: 4px; font-size: 0.85rem; line-height: 1.5; }
    .guide-section h3 { font-size: 0.9rem; color: #7ec8e3; border-bottom: 1px solid #30363d;
      padding-bottom: 6px; margin-bottom: 10px; }
    .decision-tree { font-size: 0.83rem; line-height: 1.8; padding-left: 20px; }
    .decision-tree li { margin-bottom: 4px; }
    .cat-block { background: #161b22; border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; }
    .cat-name { font-family: monospace; font-size: 0.95rem; font-weight: bold; color: #79c0ff;
      margin-bottom: 6px; }
    .cat-body { font-size: 0.8rem; line-height: 1.5; color: #c9d1d9; }
    .cat-examples { margin-top: 5px; color: #3fb950; font-style: italic; }
    .cat-warning { margin-top: 5px; color: #f85149; }
    .cat-tie { margin-top: 5px; color: #a5a5a5; border-top: 1px solid #30363d; padding-top: 4px; }
    .conf-table { width: 100%; font-size: 0.82rem; border-collapse: collapse; }
    .conf-table td { padding: 4px 8px; border-bottom: 1px solid #30363d; }
    .guide-note { font-size: 0.78rem; color: #8b949e; margin-top: 8px; font-style: italic; }
    .guide-section ul { font-size: 0.82rem; line-height: 1.7; padding-left: 20px; }
    .guide-section ol { font-size: 0.82rem; line-height: 1.7; padding-left: 20px; }
    .guide-section code { background: #21262d; padding: 2px 5px; border-radius: 3px;
      font-family: monospace; font-size: 0.85em; }
    .al-badge { background: #f0c040; color: #000; padding: 2px 8px; border-radius: 3px;
      font-size: 0.7rem; font-weight: bold; margin-left: 8px; }
    """


# ─── HTML batch generation ────────────────────────────────────────────────────


def _render_card(rec: dict, assets_rel_dir: str) -> str:
    """Render one review card. No private data exposed. Model prediction NOT shown."""
    review_id = rec["review_id"]
    signals = rec["signals"]
    phase_c_cat = signals["phase_c_candidate_category"]

    thumb_src = f"{assets_rel_dir}/{review_id}.jpg"
    badges_html = " ".join(
        f'<span class="badge {"badge-active" if signals.get(k) else "badge-inactive"}">'
        f'{html_module.escape(lbl)}</span>'
        for k, lbl in [
            ("low_res", "LOW RES"),
            ("extreme_aspect", "EXTREME ASPECT"),
            ("tiny_file", "TINY FILE"),
            ("caption_keyword_signal_present", "CAPTION KW"),
            ("phase_c_conflict", "CONFLICT"),
        ]
    )

    fc_options = "\n".join(
        f'<option value="{fc}">{html_module.escape(fc)}</option>'
        for fc in C.FINAL_CATEGORIES
    )

    # Bucket badge for context (safe to show — it's triage metadata, not private data)
    bucket = rec.get("candidate_bucket", "?")
    bucket_labels = {
        "A": "PHASE-C-NEG",
        "B": "CAPTION-SIG",
        "C": "QUALITY-SIG",
        "D": "MODEL-UNCERTAIN",
        "E": "CONTROL",
    }
    bucket_label = bucket_labels.get(bucket, bucket)

    return f"""
<div class="card" data-review-id="{html_module.escape(review_id)}" data-phase-c-cat="{html_module.escape(phase_c_cat)}">
  <div class="card-thumb">
    <img src="{thumb_src}" alt="{html_module.escape(review_id)}"
      onerror="this.parentElement.querySelector('.no-img').style.display='block';this.style.display='none'">
    <div class="no-img" style="display:none">&#10006; Image unavailable</div>
  </div>
  <div class="card-meta">
    <div class="review-id" title="{html_module.escape(review_id)}">{html_module.escape(review_id)}</div>
    <div class="phase-c-cat">Phase C: <strong>{html_module.escape(phase_c_cat)}</strong>
      <span class="al-badge">{html_module.escape(bucket_label)}</span>
    </div>
    <div class="badges">{badges_html}</div>
  </div>
  <div class="card-form">
    <div class="card-hint">Select what you see — use the annotation guide above.
      Decision type auto-derived. Model predictions are hidden.</div>
    <div class="decision-preview">Decision (auto): <strong class="f-decision-preview">—</strong></div>
    <label>Final Category
      <select class="f-final-category" name="final_category" required>
        <option value="">-- select --</option>
        {fc_options}
      </select>
    </label>
    <label>Confidence (1–5)
      <input class="f-confidence" type="number" name="human_confidence" min="1" max="5" placeholder="1-5">
    </label>
    <label>Notes (no personal data)
      <input class="f-notes" type="text" name="notes" maxlength="500" placeholder="optional">
    </label>
  </div>
</div>"""


def _build_batch_html(
    batch_records: list[dict],
    run_id: str,
    batch_id: str,
    source: str,
    assets_rel_dir: str,
    generated_at: str,
) -> str:
    """Build a complete HTML review batch with annotation guide."""
    cards_html = "\n".join(_render_card(r, assets_rel_dir) for r in batch_records)
    annotation_guide = _build_annotation_guide_html()
    guide_style = _build_annotation_guide_style()
    batch_num = int(batch_id)
    export_filename = f"filter_decisions_{run_id}_{batch_id}.json"

    decision_js_payload = json.dumps({
        "schema_version": DECISION_SCHEMA_VERSION,
        "source": source,
        "phase": "AL_NEGATIVE_REVIEW",
        "run_id": run_id,
        "batch_id": batch_id,
        "created_by": "local_manual_review",
    }).replace("'", "\\'")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Active Learning Review Batch {batch_num} — {run_id}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #0d1117; color: #c9d1d9; margin: 0; padding: 12px; }}
    h1 {{ font-size: 1rem; color: #aaa; margin-bottom: 4px; }}
    .meta {{ font-size: 0.75rem; color: #555; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }}
    .card {{ background: #161b22; border-radius: 8px; padding: 10px; display: flex; flex-direction: column; gap: 6px; }}
    .card-thumb img {{ width: 100%; max-height: 200px; object-fit: contain; border-radius: 4px; background: #0d1117; }}
    .no-img {{ text-align: center; color: #555; padding: 40px; font-size: 0.8rem; }}
    .review-id {{ font-family: monospace; font-size: 0.7rem; color: #7ec8e3; word-break: break-all; }}
    .phase-c-cat {{ font-size: 0.8rem; color: #ccc; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }}
    .badge {{ font-size: 0.65rem; padding: 2px 6px; border-radius: 3px; font-weight: bold; }}
    .badge-active {{ background: #e63946; color: #fff; }}
    .badge-inactive {{ background: #21262d; color: #555; }}
    .card-form {{ display: flex; flex-direction: column; gap: 4px; }}
    .card-form label {{ font-size: 0.75rem; color: #8b949e; display: flex; flex-direction: column; gap: 2px; }}
    .card-form select, .card-form input {{ background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 4px; padding: 4px; font-size: 0.8rem; }}
    .card-hint {{ font-size: 0.68rem; color: #6e7681; font-style: italic; line-height: 1.3; }}
    .decision-preview {{ font-size: 0.75rem; color: #8b949e; margin-bottom: 2px; }}
    .decision-preview strong {{ color: #7ec8e3; font-family: monospace; }}
    #export-btn {{ position: sticky; top: 8px; z-index: 100; background: #238636; color: #fff;
      border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 0.9rem; margin-bottom: 12px; }}
    #export-btn:hover {{ background: #2ea043; }}
    #status {{ font-size: 0.75rem; color: #aaa; margin-left: 10px; }}
    {guide_style}
  </style>
</head>
<body>
  <h1>Active Learning Review — Batch {batch_num} <span class="al-badge">AL-NEG-SELECTION</span></h1>
  <div class="meta">
    run_id: {html_module.escape(run_id)} &nbsp;|&nbsp;
    source: {html_module.escape(source)} &nbsp;|&nbsp;
    records: {len(batch_records)} &nbsp;|&nbsp;
    generated: {html_module.escape(generated_at)} &nbsp;|&nbsp;
    batch: {batch_num}
  </div>

  {annotation_guide}

  <button id="export-btn" onclick="exportDecisions()">Export Decisions JSON</button>
  <span id="status"></span>

  <div class="grid">
{cards_html}
  </div>

  <script>
  function deriveDecisionType(phaseCCat, finalCat, conf) {{
    if (!finalCat || conf === '' || conf === null) return null;
    if (finalCat === 'unsure') return 'UNSURE';
    if (phaseCCat === 'unknown_needs_review') return 'RELABEL';
    if (finalCat === phaseCCat && parseInt(conf, 10) >= 3) return 'KEEP';
    return 'RELABEL';
  }}

  document.addEventListener('DOMContentLoaded', function() {{
    document.querySelectorAll('.card').forEach(function(card) {{
      function updatePreview() {{
        var phaseC = card.dataset.phaseCCat;
        var fc = card.querySelector('.f-final-category').value;
        var conf = card.querySelector('.f-confidence').value;
        var derived = deriveDecisionType(phaseC, fc, conf) || '—';
        card.querySelector('.f-decision-preview').textContent = derived;
      }}
      card.querySelector('.f-final-category').addEventListener('change', updatePreview);
      card.querySelector('.f-confidence').addEventListener('input', updatePreview);
    }});
  }});

  function exportDecisions() {{
    var cards = document.querySelectorAll('.card');
    var records = [];
    var errors = [];
    var now = new Date().toISOString();
    cards.forEach(function(card) {{
      var rid = card.dataset.reviewId;
      var phaseC = card.dataset.phaseCCat;
      var fc = card.querySelector('.f-final-category').value;
      var conf = card.querySelector('.f-confidence').value;
      var notes = card.querySelector('.f-notes').value || null;
      if (!fc || !conf) {{ errors.push('Incomplete: ' + rid); return; }}
      var dt = deriveDecisionType(phaseC, fc, conf);
      if (!dt) {{ errors.push('Incomplete: ' + rid); return; }}
      records.push({{
        review_id: rid,
        decision_type: dt,
        phase_c_category: phaseC,
        final_category: fc,
        human_confidence: parseInt(conf, 10),
        refinement: {{ species: null, life_stage: 'unknown' }},
        notes: notes,
        reviewed_at: now
      }});
    }});
    if (errors.length > 0) {{
      document.getElementById('status').textContent = 'Incomplete: ' + errors.length + ' record(s) missing fields.';
      return;
    }}
    var payload = {{
      schema_version: '{DECISION_SCHEMA_VERSION}',
      source: '{source}',
      phase: 'AL_NEGATIVE_REVIEW',
      run_id: '{run_id}',
      batch_id: '{batch_id}',
      created_by: 'local_manual_review',
      records: records
    }};
    var blob = new Blob([JSON.stringify(payload, null, 2)], {{type: 'application/json'}});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = '{export_filename}';
    a.click();
    document.getElementById('status').textContent = 'Exported ' + records.length + ' records.';
  }}
  </script>
</body>
</html>"""


# ─── Decision template ────────────────────────────────────────────────────────


def _build_decision_template(batch_records: list[dict], run_id: str, batch_id: str, source: str) -> dict:
    """Build blank decision JSON template (matches Phase D schema)."""
    return {
        "schema_version": DECISION_SCHEMA_VERSION,
        "source": source,
        "phase": "AL_NEGATIVE_REVIEW",
        "run_id": run_id,
        "batch_id": batch_id,
        "created_by": "local_manual_review",
        "records": [
            {
                "review_id": r["review_id"],
                "decision_type": None,
                "phase_c_category": r.get("candidate_category", "unknown_needs_review"),
                "final_category": None,
                "human_confidence": None,
                "refinement": {"species": None, "life_stage": "unknown"},
                "notes": None,
                "reviewed_at": None,
            }
            for r in batch_records
        ],
    }


# ─── Privacy-safe summary ─────────────────────────────────────────────────────


def _build_tracked_summary(
    run_id: str,
    generated_at: str,
    total_unreviewed_scanned: int,
    total_inference_attempted: int,
    total_inference_ok: int,
    total_inference_failed: int,
    total_selected: int,
    bucket_counts: dict[str, int],
    skipped_missing: int,
    skipped_corrupt: int,
    batch_count: int,
    annotation_guide_embedded: bool,
    model_predictions_shown_in_ui: bool,
    mode: str = MODE_DEFAULT,
) -> dict:
    """Privacy-safe counts-only summary for tracked storage."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "source": SOURCE_TAG,
        "generated_at": generated_at,
        "selection_mode": mode,
        "pass_number": 2 if mode == MODE_NEGATIVE_FOCUSED_V2 else 1,
        "privacy_status": {
            "counts_only": True,
            "contains_filenames": False,
            "contains_review_ids": False,
            "contains_captions": False,
            "contains_paths": False,
        },
        "input": {
            "total_unreviewed_candidates_scanned": total_unreviewed_scanned,
            "inference_attempted": total_inference_attempted,
            "inference_ok": total_inference_ok,
            "inference_failed": total_inference_failed,
            "skipped_missing_image": skipped_missing,
            "skipped_corrupt_image": skipped_corrupt,
        },
        "selection": {
            "total_selected": total_selected,
            "batch_count": batch_count,
            "batch_size": BATCH_SIZE,
            "by_bucket": {
                "A_phase_c_negative_categories": bucket_counts.get("A", 0),
                "B_caption_keyword_signals": bucket_counts.get("B", 0),
                "C_quality_conflict_signals": bucket_counts.get("C", 0),
                "D_model_uncertainty": bucket_counts.get("D", 0),
                "E_random_controls": bucket_counts.get("E", 0),
            },
        },
        "annotation_guide": {
            "embedded_in_every_html_batch": annotation_guide_embedded,
            "model_predictions_shown_in_ui": model_predictions_shown_in_ui,
            "bias_mitigation": "model_predictions_hidden_from_reviewer",
            "categories_defined": [
                "fish", "fish_part", "no_fish", "lure_gear",
                "poster_screenshot", "bad_quality", "out_of_scope",
                "duplicate_suspect", "unsure",
            ],
        },
        "model": {
            "path_relative": "data/fish_models/mvp_structural_v1.pt",
            "type": "yolov8n-cls",
            "predictions_used_as": "triage_only_not_labels",
            "predictions_shown_in_review_ui": False,
        },
    }


# ─── Main orchestration ───────────────────────────────────────────────────────


def run(
    candidates_path: Path,
    aggregate_path: Path,
    model_path: Path,
    export_dir: Path,
    review_dir: Path,
    secret_path: Path,
    private_root: Path,
    target_size: int,
    batch_size: int,
    device: str,
    dry_run: bool,
    run_id: str | None,
    tracked_summary_path: Path | None,
    mode: str = MODE_DEFAULT,
) -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    if run_id is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{AL_RUN_PREFIX}_{ts}"
    log.info("Active learning run ID: %s", run_id)

    # ── Load HMAC secret ─────────────────────────────────────────────────────
    secret = _load_secret(secret_path)
    log.info("Review HMAC secret loaded.")

    # ── Load Phase C candidates ──────────────────────────────────────────────
    candidates = load_candidates(candidates_path)
    total_candidates = len(candidates)

    # Build review_id → candidate mapping using same HMAC as Phase D
    cand_by_rv: dict[str, dict] = {}
    for rec in candidates:
        sha = rec.get("sha256", "")
        if not sha:
            continue
        rv_id = compute_review_id(secret, sha)
        cand_by_rv[rv_id] = rec

    # ── Load already-reviewed IDs ─────────────────────────────────────────────
    # Always union the partial aggregate with ALL individual decision files so that
    # newly-reviewed batches are excluded even when the aggregate hasn't been rebuilt.
    reviewed_ids = load_reviewed_ids(aggregate_path)
    # Also scan individual decision files for any run_id that uses the same review_dir,
    # to catch reviews recorded after the aggregate was last generated.
    for pattern_run_id in ("rvrun_20260427T184629Z", AL_RUN_PREFIX):
        extra = load_reviewed_ids_from_decisions_dir(review_dir, pattern_run_id)
        reviewed_ids |= extra
    log.info("Total reviewed records to exclude: %d", len(reviewed_ids))

    # ── Filter to unreviewed pool ─────────────────────────────────────────────
    unreviewed_recs: list[dict] = []
    for rv_id, cand_rec in cand_by_rv.items():
        if rv_id in reviewed_ids:
            continue
        augmented = dict(cand_rec)
        augmented["review_id"] = rv_id
        augmented["signals"] = _get_signals(cand_rec)
        unreviewed_recs.append(augmented)

    log.info("Unreviewed pool: %d records (excluded %d reviewed)", len(unreviewed_recs), len(reviewed_ids))

    # ── Build inference pool ─────────────────────────────────────────────────
    # Run inference on: all signal-bearing records + random sample of no-signal records
    rng = random.Random(RANDOM_SEED)
    signal_recs = [r for r in unreviewed_recs if _has_any_signal(r["signals"])]
    no_signal_recs = [r for r in unreviewed_recs if not _has_any_signal(r["signals"])]

    # Shuffle no-signal sample deterministically
    no_signal_sample = list(no_signal_recs)
    rng_sample = random.Random(RANDOM_SEED + 1)
    rng_sample.shuffle(no_signal_sample)
    no_signal_for_inference = no_signal_sample[:max(0, MAX_INFERENCE_POOL - len(signal_recs))]

    inference_pool = signal_recs + no_signal_for_inference
    log.info(
        "Inference pool: %d total (signal=%d, no-signal-sample=%d)",
        len(inference_pool), len(signal_recs), len(no_signal_for_inference),
    )

    # ── Resolve image paths + check existence ─────────────────────────────────
    inference_image_paths: list[Path] = []
    inference_review_ids: list[str] = []
    skipped_missing = 0
    skipped_corrupt = 0

    for rec in inference_pool:
        filename = rec.get("filename", "")
        if not filename:
            skipped_missing += 1
            continue
        img_path = export_dir / filename
        if not img_path.exists():
            skipped_missing += 1
            continue
        inference_image_paths.append(img_path)
        inference_review_ids.append(rec["review_id"])

    log.info(
        "Resolved %d/%d image paths (skipped_missing=%d)",
        len(inference_image_paths), len(inference_pool), skipped_missing,
    )

    # ── Run shadow inference ─────────────────────────────────────────────────
    predictions: dict[str, dict | None] = {}
    inference_ok = 0
    inference_failed = 0

    if not dry_run and inference_image_paths:
        raw_preds = run_shadow_inference(model_path, inference_image_paths, device=device)
        for rv_id, pred in zip(inference_review_ids, raw_preds):
            predictions[rv_id] = pred
            if pred is not None:
                inference_ok += 1
            else:
                inference_failed += 1
    else:
        if dry_run:
            log.info("DRY RUN: skipping inference.")
        inference_ok = 0
        inference_failed = 0

    # ── Save private shadow predictions ─────────────────────────────────────
    if not dry_run:
        private_run_dir = private_root / run_id
        private_run_dir.mkdir(parents=True, exist_ok=True)

        shadow_pred_path = private_run_dir / "shadow_predictions.jsonl"
        # Write atomically (replace, not append) — prevents silent double-write on re-runs
        # with the same run_id.
        lines = [
            json.dumps(
                {"review_id": rv_id, "prediction": pred, "triage_only": True, "not_truth_label": True},
                ensure_ascii=False,
            )
            for rv_id, pred in predictions.items()
        ]
        _write_text_atomic(shadow_pred_path, "\n".join(lines) + ("\n" if lines else ""))
        log.info("Shadow predictions written to %s (%d records)", shadow_pred_path, len(lines))

    # ── Candidate selection ──────────────────────────────────────────────────
    selected_recs, bucket_counts = select_candidates(
        unreviewed_recs, predictions, target_size, rng, mode=mode
    )

    # Cap at target_size
    if len(selected_recs) > target_size:
        selected_recs = selected_recs[:target_size]
        log.info("Capped at target_size=%d", target_size)

    log.info("Final selection: %d records", len(selected_recs))

    if not selected_recs:
        log.error("No candidates selected. Check data and signals.")
        sys.exit(1)

    # ── Generate thumbnails ──────────────────────────────────────────────────
    assets_dir = review_dir / "assets" / run_id
    thumbnail_ok = 0
    thumbnail_failed = 0

    if not dry_run:
        assets_dir.mkdir(parents=True, exist_ok=True)
        log.info("Generating thumbnails in %s ...", assets_dir)
        for rec in selected_recs:
            filename = rec.get("filename", "")
            if not filename:
                thumbnail_failed += 1
                continue
            src_path = export_dir / filename
            dst_path = assets_dir / f"{rec['review_id']}.jpg"
            if dst_path.exists():
                thumbnail_ok += 1
                continue
            if generate_thumbnail(src_path, dst_path):
                thumbnail_ok += 1
            else:
                thumbnail_failed += 1
        log.info("Thumbnails: ok=%d, failed=%d", thumbnail_ok, thumbnail_failed)
    else:
        log.info("DRY RUN: thumbnails skipped.")

    # ── Generate review batches ──────────────────────────────────────────────
    batches: list[list[dict]] = []
    for i in range(0, len(selected_recs), batch_size):
        batches.append(selected_recs[i: i + batch_size])

    assets_rel_dir = f"assets/{run_id}"
    manifest_batches = []

    for batch_idx, batch_recs in enumerate(batches):
        batch_id = f"{batch_idx + 1:04d}"
        html_filename = f"filter_review_batch_{run_id}_{batch_id}.html"
        decision_filename = f"filter_decisions_{run_id}_{batch_id}.json"

        html_path = review_dir / html_filename
        decision_path = review_dir / decision_filename

        if not dry_run:
            html_content = _build_batch_html(
                batch_recs, run_id, batch_id, SOURCE_TAG, assets_rel_dir, generated_at
            )
            _write_text_atomic(html_path, html_content)
            log.info("HTML batch written: %s (%d records)", html_filename, len(batch_recs))

            decision_template = _build_decision_template(batch_recs, run_id, batch_id, SOURCE_TAG)
            _write_json_atomic(decision_path, decision_template)
            log.info("Decision template written: %s", decision_filename)
        else:
            log.info("DRY RUN: would write %s (%d records)", html_filename, len(batch_recs))

        manifest_batches.append({
            "batch_id": batch_id,
            "record_count": len(batch_recs),
            "html_file": html_filename,
            "decision_file": decision_filename,
            "review_ids": [r["review_id"] for r in batch_recs],
        })

    # ── Write review manifest ────────────────────────────────────────────────
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "source": SOURCE_TAG,
        "phase": "AL_NEGATIVE_REVIEW",
        "created_at": generated_at,
        "total_records": len(selected_recs),
        "batch_count": len(batches),
        "batch_size": batch_size,
        "bucket_counts": bucket_counts,
        "batches": manifest_batches,
    }
    if not dry_run:
        manifest_path = review_dir / f"filter_review_manifest_{run_id}.json"
        _write_json_atomic(manifest_path, manifest)
        log.info("Review manifest written: %s", manifest_path.name)

    # ── Write private candidate manifest ─────────────────────────────────────
    if not dry_run:
        private_run_dir = private_root / run_id
        private_manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "source": SOURCE_TAG,
            "generated_at": generated_at,
            "total_selected": len(selected_recs),
            "bucket_counts": bucket_counts,
            "candidates": [
                {
                    "review_id": r["review_id"],
                    "candidate_bucket": r.get("candidate_bucket"),
                    "candidate_reason": r.get("candidate_reason"),
                    "needs_human_review": True,
                    "model_prediction_triage_only": r.get("model_prediction_triage_only"),
                    "not_truth_label": True,
                }
                for r in selected_recs
            ],
        }
        _write_json_atomic(private_run_dir / "candidate_manifest.json", private_manifest)

        run_config = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "generated_at": generated_at,
            "parameters": {
                "target_size": target_size,
                "batch_size": batch_size,
                "max_inference_pool": MAX_INFERENCE_POOL,
                "random_seed": RANDOM_SEED,
                "device": device,
            },
            "input_counts": {
                "total_phase_c_candidates": total_candidates,
                "reviewed_excluded": len(reviewed_ids),
                "unreviewed_pool": len(unreviewed_recs),
                "signal_bearing": len(signal_recs),
                "no_signal": len(no_signal_recs),
                "inference_pool_size": len(inference_image_paths),
                "skipped_missing": skipped_missing,
            },
        }
        _write_json_atomic(private_run_dir / "run_config.json", run_config)
        log.info("Private candidate manifest and run config written to %s", private_run_dir)

    # ── Privacy-safe tracked summary ─────────────────────────────────────────
    tracked_summary = _build_tracked_summary(
        run_id=run_id,
        generated_at=generated_at,
        total_unreviewed_scanned=len(unreviewed_recs),
        total_inference_attempted=len(inference_image_paths),
        total_inference_ok=inference_ok,
        total_inference_failed=inference_failed,
        total_selected=len(selected_recs),
        bucket_counts=bucket_counts,
        skipped_missing=skipped_missing,
        skipped_corrupt=skipped_corrupt,
        batch_count=len(batches),
        annotation_guide_embedded=True,
        model_predictions_shown_in_ui=False,
        mode=mode,
    )

    if not dry_run:
        if tracked_summary_path is None:
            tracked_summary_path = (
                Path(__file__).resolve().parent.parent
                / "data" / "fish_models"
                / f"telegram_negative_candidate_selection_summary_{run_id}.json"
            )
        _write_json_atomic(tracked_summary_path, tracked_summary)
        log.info("Tracked summary written to %s", tracked_summary_path)

    # ── Final report ─────────────────────────────────────────────────────────
    pass_num = 2 if mode == MODE_NEGATIVE_FOCUSED_V2 else 1
    print("\n" + "=" * 70)
    print(f"ACTIVE LEARNING CANDIDATE SELECTION COMPLETE  [pass {pass_num} / mode={mode}]")
    print(f"Run ID: {run_id}")
    print(f"{'=' * 70}")
    print(f"Unreviewed pool scanned: {len(unreviewed_recs):,}")
    print(f"Inference pool:          {len(inference_image_paths):,}")
    print(f"Total selected:          {len(selected_recs)}")
    print(f"  Bucket A (Phase C neg):         {bucket_counts.get('A', 0)}")
    b_label = "Caption signal ASC fish_conf" if mode == MODE_NEGATIVE_FOCUSED_V2 else "Caption signal DESC fish_conf"
    print(f"  Bucket B ({b_label}): {bucket_counts.get('B', 0)}")
    print(f"  Bucket C (Quality/conflict):    {bucket_counts.get('C', 0)}")
    d_label = "Strong not_fish + uncertainty" if mode == MODE_NEGATIVE_FOCUSED_V2 else "Model uncertainty"
    print(f"  Bucket D ({d_label}): {bucket_counts.get('D', 0)}")
    print(f"  Bucket E (Controls, minimal):   {bucket_counts.get('E', 0)}")
    print(f"Review batches:          {len(batches)} × {batch_size} records")
    print(f"{'=' * 70}")
    print("FILES TO OPEN FOR REVIEW:")
    for b in manifest_batches:
        print(f"  {review_dir / b['html_file']}")
    print(f"{'=' * 70}")
    print("AFTER REVIEWING:")
    print(f"  1. Export decisions JSON from each HTML batch.")
    print(f"  2. Save exported files to: {review_dir}/")
    print(f"  3. Do NOT modify the exported JSON.")
    print(f"  4. Next step: ingest/validate exported review JSON.")
    print(f"{'=' * 70}\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Active learning: select Telegram-domain negative candidates for manual review.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--candidates",
        type=Path,
        default=C.INTAKE_META_ROOT / "filter_candidates.jsonl",
        help="Phase C filter_candidates.jsonl",
    )
    p.add_argument(
        "--aggregate",
        type=Path,
        default=C.FILTER_REVIEW_DIR / "filter_decisions_partial_aggregate_rvrun_20260427T184629Z.json",
        help="Phase D partial aggregate (to exclude already-reviewed IDs)",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=C.DATA_ROOT / "fish_models" / "mvp_structural_v1.pt",
        help="Path to mvp_structural_v1.pt (YOLOv8n-cls)",
    )
    p.add_argument(
        "--export-dir",
        type=Path,
        default=C.EXPORT_DIR,
        help="Telegram export directory containing original photos",
    )
    p.add_argument(
        "--review-dir",
        type=Path,
        default=C.FILTER_REVIEW_DIR,
        help="Review output directory (gitignored)",
    )
    p.add_argument(
        "--secret",
        type=Path,
        default=C.REVIEW_SECRET_PATH,
        help="Path to review HMAC secret",
    )
    p.add_argument(
        "--private-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "private" / "active_learning" / "telegram_negative_candidates",
        help="Private output root for shadow predictions (gitignored)",
    )
    p.add_argument(
        "--target-size",
        type=int,
        default=DEFAULT_TARGET_SIZE,
        metavar="N",
        help="Total candidates to select (split into batches of --batch-size)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Records per review batch",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: cpu, mps, cuda",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override run ID (default: auto-generated alvrun_<timestamp>)",
    )
    p.add_argument(
        "--tracked-summary",
        type=Path,
        default=None,
        help="Path to write privacy-safe tracked summary JSON (default: auto)",
    )
    p.add_argument(
        "--mode",
        type=str,
        default=MODE_DEFAULT,
        choices=[MODE_DEFAULT, MODE_NEGATIVE_FOCUSED_V2],
        help=(
            f"Candidate selection strategy. "
            f"'{MODE_DEFAULT}': original caps, Bucket B sorted DESC fish_conf. "
            f"'{MODE_NEGATIVE_FOCUSED_V2}': reduced caps, Bucket B sorted ASC fish_conf, "
            f"Bucket D prioritises strong model not_fish — use for pass 2."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and log selection plan without writing files",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        run(
            candidates_path=args.candidates,
            aggregate_path=args.aggregate,
            model_path=args.model,
            export_dir=args.export_dir,
            review_dir=args.review_dir,
            secret_path=args.secret,
            private_root=args.private_root,
            target_size=args.target_size,
            batch_size=args.batch_size,
            device=args.device,
            dry_run=args.dry_run,
            run_id=args.run_id,
            tracked_summary_path=args.tracked_summary,
            mode=args.mode,
        )
    except (ValueError, FileNotFoundError) as exc:
        log.error("Run failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
