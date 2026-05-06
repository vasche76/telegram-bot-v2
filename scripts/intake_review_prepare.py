"""
intake_review_prepare.py — U4 Phase D: Prepare manual review batches.

Reads Phase C filter_candidates.jsonl and generates:
  - HMAC-based anonymous review IDs (deterministic, local-secret-keyed)
  - Stripped JPEG thumbnails (EXIF removed, named by review_id)
  - Static HTML review sheets (privacy-safe: no filenames, captions, sender data)
  - Blank decision JSON templates (pre-populated with review_id + phase_c_category)
  - Local review manifest

PRIVACY CONTRACT:
  - filename, sha256, reasons, conflicts fields from Phase C are NOT written to HTML
    or decision templates.
  - Only the following are surfaced in HTML:
      review_id, phase_c_candidate_category, signal badges
      (low_res, extreme_aspect, tiny_file, caption_keyword_signal_present, phase_c_conflict)
  - Caption text is never shown. Only the boolean that a keyword signal exists.
  - review_id is derived from HMAC(secret, sha256) — sha256 is never written to review dir.

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
import secrets
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intake_constants as C

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr)
log = logging.getLogger(__name__)

PROGRESS_EVERY = 500

# Caption keyword reason codes that indicate a keyword signal (not the caption text itself)
_CAPTION_KW_REASONS: frozenset[str] = frozenset({
    "caption_lure_hint",
    "caption_fish_part_keyword",
    "caption_fry_keyword",
    "caption_no_fish_keyword",
})


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


# ─── Secret management ────────────────────────────────────────────────────────


def _load_or_create_secret(secret_path: Path) -> bytes:
    """Load existing review secret or create a new one. Never commits."""
    if secret_path.exists():
        raw = secret_path.read_bytes().strip()
        # Accept hex or raw bytes
        if len(raw) == 64 and all(c in b"0123456789abcdefABCDEF" for c in raw):
            return bytes.fromhex(raw.decode("ascii"))
        return raw
    secret_path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_bytes(32)
    secret_path.write_bytes(secret.hex().encode("ascii") + b"\n")
    secret_path.chmod(0o600)
    log.info("Created new review secret: %s", secret_path)
    return secret


# ─── Review ID computation ────────────────────────────────────────────────────


def compute_review_id(secret: bytes, internal_key: str) -> str:
    """Compute deterministic HMAC-based review ID. Never exposes internal_key."""
    digest = hmac.new(secret, internal_key.encode("utf-8"), hashlib.sha256).hexdigest()
    return "rv_" + digest[:16]


# ─── Phase C invariant checks ────────────────────────────────────────────────


def check_phase_c_invariants(candidates: list[dict]) -> None:
    """Fail loudly if Phase C output violates known invariants."""
    errors: list[str] = []

    seen_sha: set[str] = set()
    for i, rec in enumerate(candidates):
        sha = rec.get("sha256", "")
        if sha in seen_sha:
            errors.append(f"Record {i}: duplicate sha256 in Phase C output: {sha[:8]}...")
        seen_sha.add(sha)

        if not rec.get("review_required", False):
            errors.append(f"Record {i}: review_required must be True for all Phase C records")

        cat = rec.get("candidate_category", "")
        if cat not in C.COARSE_CATEGORIES_SET:
            errors.append(f"Record {i}: unknown candidate_category: {cat!r}")

        conf = rec.get("confidence", "")
        if conf not in {C.CONFIDENCE_HIGH, C.CONFIDENCE_MEDIUM, C.CONFIDENCE_LOW}:
            errors.append(f"Record {i}: unknown confidence: {conf!r}")

        if not isinstance(rec.get("reasons", []), list):
            errors.append(f"Record {i}: 'reasons' must be a list")

        if not isinstance(rec.get("conflicts", []), list):
            errors.append(f"Record {i}: 'conflicts' must be a list")

    if errors:
        for e in errors:
            log.error("Phase C invariant violation: %s", e)
        raise ValueError(
            f"Phase C invariant check failed with {len(errors)} error(s): {errors[0]}"
        )

    # Also cross-check against known summary if present
    log.info("Phase C invariants OK: %d records, %d unique sha256s", len(candidates), len(seen_sha))


# ─── Signal extraction (privacy-safe) ────────────────────────────────────────


def _extract_safe_signals(rec: dict) -> dict:
    """
    Extract only the privacy-safe boolean signals allowed in the HTML review UI.
    Never exposes filename, sha256, caption text, sender metadata.
    """
    reasons: list[str] = rec.get("reasons", [])
    conflicts: list[str] = rec.get("conflicts", [])
    return {
        "low_res": "low_res" in reasons,
        "extreme_aspect": "extreme_aspect" in reasons,
        "tiny_file": "tiny_file" in reasons,
        "caption_keyword_signal_present": bool(_CAPTION_KW_REASONS & set(reasons)),
        "phase_c_conflict": bool(conflicts),
        "phase_c_candidate_category": C.normalize_phase_c_category(
            rec.get("candidate_category", "unknown_needs_review")
        ),
    }


# ─── Thumbnail generation ─────────────────────────────────────────────────────


def _generate_thumbnail(
    src_path: Path,
    dst_path: Path,
    max_size: tuple[int, int] = (320, 320),
) -> bool:
    """
    Generate a stripped EXIF thumbnail at dst_path.
    Returns True on success, False on error (logs warning).
    """
    try:
        from PIL import Image  # type: ignore
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img.thumbnail(max_size, Image.LANCZOS)
            img.save(dst_path, "JPEG", quality=75, optimize=True)
        return True
    except ImportError:
        log.warning("Pillow not installed — thumbnails skipped. Install with: pip install Pillow")
        return False
    except Exception as exc:
        log.warning("Thumbnail failed for %s: %s", dst_path.name, exc)
        return False


# ─── Batching ─────────────────────────────────────────────────────────────────


def _priority_bucket(rec: dict) -> int:
    """
    Priority ordering (lower = earlier batch):
    1. phase_c_conflict
    2. fish_part / fry_juvenile / poster_screenshot
    3. low_res / extreme_aspect / tiny_file (quality signals)
    4. remaining unknown_needs_review
    """
    signals = _extract_safe_signals(rec)
    if signals["phase_c_conflict"]:
        return 0
    cat = signals["phase_c_candidate_category"]
    if cat in {"fish_part", "fry_juvenile", "poster_screenshot"}:
        return 1
    if signals["low_res"] or signals["extreme_aspect"] or signals["tiny_file"]:
        return 2
    return 3


def build_batches(review_records: list[dict], batch_size: int) -> list[list[dict]]:
    """
    Sort records by (priority_bucket, review_id) and chunk into batches.
    Deterministic: same input → same batches.
    """
    sorted_records = sorted(
        review_records,
        key=lambda r: (_priority_bucket(r), r["review_id"]),
    )
    batches: list[list[dict]] = []
    for i in range(0, len(sorted_records), batch_size):
        batches.append(sorted_records[i : i + batch_size])
    return batches


# ─── HTML generation ──────────────────────────────────────────────────────────


_SIGNAL_LABELS: dict[str, str] = {
    "low_res": "LOW RES",
    "extreme_aspect": "EXTREME ASPECT",
    "tiny_file": "TINY FILE",
    "caption_keyword_signal_present": "CAPTION KW",
    "phase_c_conflict": "CONFLICT",
}


def _render_badge(key: str, active: bool) -> str:
    cls = "badge-active" if active else "badge-inactive"
    label = _SIGNAL_LABELS.get(key, key)
    return f'<span class="badge {cls}">{html_module.escape(label)}</span>'


def _render_record_card(rec: dict, assets_rel_dir: str) -> str:
    """Render one review card. No private data exposed."""
    review_id = rec["review_id"]
    signals = rec["signals"]
    phase_c_cat = signals["phase_c_candidate_category"]

    thumb_src = f"{assets_rel_dir}/{review_id}.jpg"
    thumb_html = (
        f'<img src="{thumb_src}" alt="{html_module.escape(review_id)}" '
        f'onerror="this.parentElement.querySelector(\'.no-img\').style.display=\'block\';this.style.display=\'none\'">'
    )
    badges_html = " ".join(
        _render_badge(k, signals[k])
        for k in ("low_res", "extreme_aspect", "tiny_file", "caption_keyword_signal_present", "phase_c_conflict")
    )

    fc_options = "\n".join(
        f'<option value="{fc}">{html_module.escape(fc)}</option>'
        for fc in C.FINAL_CATEGORIES
    )

    if phase_c_cat == "unknown_needs_review":
        hint_html = (
            '<div class="card-hint">'
            "Phase C could not classify this image — KEEP is not available. "
            "Select what you see; decision type is auto-derived."
            "</div>"
        )
    else:
        hint_html = (
            '<div class="card-hint">'
            "Decision type is auto-derived: KEEP when category matches Phase C "
            "and confidence ≥ 3, UNSURE when you select <em>unsure</em>, "
            "otherwise RELABEL."
            "</div>"
        )

    return f"""
<div class="card" data-review-id="{html_module.escape(review_id)}" data-phase-c-cat="{html_module.escape(phase_c_cat)}">
  <div class="card-thumb">
    {thumb_html}
    <div class="no-img" style="display:none">&#10006; Image unavailable</div>
  </div>
  <div class="card-meta">
    <div class="review-id" title="{html_module.escape(review_id)}">{html_module.escape(review_id[:20])}</div>
    <div class="phase-c-cat">Phase C: <strong>{html_module.escape(phase_c_cat)}</strong></div>
    <div class="badges">{badges_html}</div>
  </div>
  <div class="card-form">
    {hint_html}
    <div class="decision-preview">Decision (auto): <strong class="f-decision-preview">—</strong></div>
    <label>Final Category
      <select class="f-final-category" name="final_category" required>
        <option value="">-- select --</option>
        {fc_options}
      </select>
    </label>
    <label>Confidence (1-5)
      <input class="f-confidence" type="number" name="human_confidence" min="1" max="5" placeholder="1-5">
    </label>
    <label>Notes (no personal data)
      <input class="f-notes" type="text" name="notes" maxlength="500" placeholder="optional">
    </label>
  </div>
</div>"""


def _build_html(
    batch_records: list[dict],
    run_id: str,
    batch_id: str,
    source: str,
    assets_rel_dir: str,
) -> str:
    cards = "\n".join(_render_record_card(r, assets_rel_dir) for r in batch_records)
    now_str = datetime.now(timezone.utc).isoformat()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Review Batch {html_module.escape(batch_id)} — {html_module.escape(run_id)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 12px; }}
    h1 {{ font-size: 1rem; color: #aaa; margin-bottom: 8px; }}
    .meta {{ font-size: 0.75rem; color: #666; margin-bottom: 16px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 12px; }}
    .card {{ background: #16213e; border-radius: 8px; padding: 10px; display: flex; flex-direction: column; gap: 6px; }}
    .card-thumb img {{ width: 100%; max-height: 200px; object-fit: contain; border-radius: 4px; background: #0f0e17; }}
    .no-img {{ text-align: center; color: #666; padding: 40px; font-size: 0.8rem; }}
    .review-id {{ font-family: monospace; font-size: 0.7rem; color: #7ec8e3; word-break: break-all; }}
    .phase-c-cat {{ font-size: 0.8rem; color: #ccc; }}
    .badges {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }}
    .badge {{ font-size: 0.65rem; padding: 2px 6px; border-radius: 3px; font-weight: bold; }}
    .badge-active {{ background: #e63946; color: #fff; }}
    .badge-inactive {{ background: #2a2a4a; color: #555; }}
    .card-form {{ display: flex; flex-direction: column; gap: 4px; }}
    .card-form label {{ font-size: 0.75rem; color: #aaa; display: flex; flex-direction: column; gap: 2px; }}
    .card-form select, .card-form input {{ background: #0f0e17; color: #eee; border: 1px solid #333; border-radius: 4px; padding: 4px; font-size: 0.8rem; }}
    .card-hint {{ font-size: 0.68rem; color: #888; font-style: italic; line-height: 1.3; }}
    .decision-preview {{ font-size: 0.75rem; color: #aaa; margin-bottom: 2px; }}
    .decision-preview strong {{ color: #7ec8e3; font-family: monospace; }}
    #export-btn {{ position: sticky; top: 8px; z-index: 100; background: #4caf50; color: #fff;
      border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-size: 0.9rem; margin-bottom: 12px; }}
    #export-btn:hover {{ background: #45a049; }}
    #status {{ font-size: 0.75rem; color: #aaa; margin-left: 10px; }}
  </style>
</head>
<body>
  <h1>Phase D Manual Review — Batch {html_module.escape(batch_id)}</h1>
  <div class="meta">
    run_id: {html_module.escape(run_id)} &nbsp;|&nbsp;
    source: {html_module.escape(source)} &nbsp;|&nbsp;
    records: {len(batch_records)} &nbsp;|&nbsp;
    generated: {html_module.escape(now_str)}
  </div>
  <button id="export-btn" onclick="exportDecisions()">Export Decisions JSON</button>
  <span id="status"></span>
  <div class="grid">
{cards}
  </div>
  <script>
  // Auto-derive decision_type from phase_c_category, final_category, and confidence.
  // KEEP is only valid when phase_c_category is known (not unknown_needs_review),
  // final_category matches phase_c_category, and confidence >= 3.
  function deriveDecisionType(phaseCCat, finalCat, conf) {{
    if (!finalCat || conf === '' || conf === null) return null;
    if (finalCat === 'unsure') return 'UNSURE';
    if (phaseCCat === 'unknown_needs_review') return 'RELABEL';
    if (finalCat === phaseCCat && parseInt(conf, 10) >= 3) return 'KEEP';
    return 'RELABEL';
  }}

  // Update the read-only decision preview on each card.
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
      if (!fc || !conf) {{
        errors.push('Incomplete: ' + rid);
        return;
      }}
      var dt = deriveDecisionType(phaseC, fc, conf);
      if (!dt) {{
        errors.push('Incomplete: ' + rid);
        return;
      }}
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
      schema_version: '{C.REVIEW_SCHEMA_VERSION}',
      source: '{html_module.escape(source)}',
      phase: '{C.REVIEW_PHASE}',
      run_id: '{html_module.escape(run_id)}',
      batch_id: '{html_module.escape(batch_id)}',
      created_by: '{C.REVIEW_CREATED_BY}',
      records: records
    }};
    var blob = new Blob([JSON.stringify(payload, null, 2)], {{type: 'application/json'}});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'filter_decisions_{html_module.escape(run_id)}_{html_module.escape(batch_id)}.json';
    a.click();
    document.getElementById('status').textContent = 'Exported ' + records.length + ' records.';
  }}
  </script>
</body>
</html>"""


# ─── Decision template generation ────────────────────────────────────────────


def _build_decision_template(
    batch_records: list[dict],
    run_id: str,
    batch_id: str,
    source: str,
) -> dict:
    """Build a blank decision JSON template for a batch (all fields pre-populated, values empty)."""
    return {
        "schema_version": C.REVIEW_SCHEMA_VERSION,
        "source": source,
        "phase": C.REVIEW_PHASE,
        "run_id": run_id,
        "batch_id": batch_id,
        "created_by": C.REVIEW_CREATED_BY,
        "records": [
            {
                "review_id": rec["review_id"],
                "decision_type": None,
                "phase_c_category": rec["signals"]["phase_c_candidate_category"],
                "final_category": None,
                "human_confidence": None,
                "refinement": {"species": None, "life_stage": "unknown"},
                "notes": None,
                "reviewed_at": None,
            }
            for rec in batch_records
        ],
    }


# ─── Main prepare orchestration ───────────────────────────────────────────────


def prepare(
    candidates_path: Path,
    review_dir: Path,
    secret_path: Path,
    assets_base: Path,
    export_dir: Path | None,
    run_id: str,
    batch_size: int,
    source: str,
    dry_run: bool = False,
    limit: int | None = None,
    patch_html_only: bool = False,
) -> dict:
    """
    Full prepare run. Returns summary dict.
    """
    if not candidates_path.exists():
        log.error("filter_candidates.jsonl not found: %s", candidates_path)
        sys.exit(1)

    log.info("Loading Phase C candidates: %s", candidates_path)
    candidates = _read_jsonl(candidates_path)
    log.info("Loaded %d Phase C records", len(candidates))

    check_phase_c_invariants(candidates)

    # In dry_run, load existing secret or use an ephemeral one (never write to disk)
    if dry_run and not secret_path.exists():
        secret = secrets.token_bytes(32)
        log.info("[DRY RUN] Using ephemeral secret (no file written)")
    else:
        secret = _load_or_create_secret(secret_path)

    # Build review records with privacy-safe signals and HMAC IDs
    review_records: list[dict] = []
    seen_review_ids: set[str] = set()
    for rec in candidates:
        # internal_key = sha256; never written to review artifacts
        rid = compute_review_id(secret, rec["sha256"])
        if rid in seen_review_ids:
            raise ValueError(f"HMAC collision detected for review_id {rid} — check secret integrity")
        seen_review_ids.add(rid)
        signals = _extract_safe_signals(rec)
        review_records.append({
            "review_id": rid,
            "signals": signals,
            # Store source path only in memory for thumbnail generation; not written anywhere
            "_src_filename": rec.get("filename"),
        })

    log.info("Review IDs computed: %d unique IDs", len(review_records))

    if limit is not None:
        review_records = sorted(
            review_records,
            key=lambda r: (_priority_bucket(r), r["review_id"]),
        )[:limit]
        log.info("[LIMIT] Smoke-test mode: using first %d records by priority", len(review_records))

    batches = build_batches(review_records, batch_size)
    log.info("Batches: %d (size=%d)", len(batches), batch_size)

    if dry_run:
        log.info("[DRY RUN] Would generate %d batches in %s", len(batches), review_dir)
        return {
            "run_id": run_id,
            "source": source,
            "total_records": len(review_records),
            "batch_count": len(batches),
            "batch_size": batch_size,
            "dry_run": True,
        }

    assets_run_dir = assets_base / run_id

    # Thumbnail generation
    thumb_available = export_dir is not None and export_dir.exists()
    if not thumb_available:
        log.warning(
            "Export dir not found (%s) — thumbnails will be skipped. "
            "HTML will show 'Image unavailable' placeholders.",
            export_dir,
        )
    thumb_success = 0
    thumb_failed = 0

    thumb_skipped = 0
    for i, rec in enumerate(review_records):
        if i > 0 and i % PROGRESS_EVERY == 0:
            log.info(
                "Thumbnails: %d / %d (ok=%d, skip=%d, existing=%d)",
                i, len(review_records), thumb_success, thumb_failed, thumb_skipped,
            )
        rid = rec["review_id"]
        dst = assets_run_dir / f"{rid}.jpg"
        if thumb_available:
            if patch_html_only and dst.exists():
                thumb_skipped += 1
                continue
            src = export_dir / rec["_src_filename"]  # type: ignore[arg-type]
            ok = _generate_thumbnail(src, dst)
            if ok:
                thumb_success += 1
            else:
                thumb_failed += 1
        # else: leave dst absent; HTML will show placeholder

    if thumb_available:
        log.info(
            "Thumbnails complete: ok=%d, failed=%d, existing_skipped=%d",
            thumb_success, thumb_failed, thumb_skipped,
        )

    # Generate batch artifacts
    batch_manifest_entries: list[dict] = []
    assets_rel = f"assets/{run_id}"  # relative from review_dir

    for batch_idx, batch_recs in enumerate(batches):
        batch_id_str = f"{batch_idx + 1:04d}"
        html_path = review_dir / f"filter_review_batch_{run_id}_{batch_id_str}.html"
        decision_path = review_dir / f"filter_decisions_{run_id}_{batch_id_str}.json"

        html_content = _build_html(batch_recs, run_id, batch_id_str, source, assets_rel)
        _write_text_atomic(html_path, html_content)

        if not patch_html_only:
            decision_template = _build_decision_template(batch_recs, run_id, batch_id_str, source)
            _write_json_atomic(decision_path, decision_template)

        batch_manifest_entries.append({
            "batch_id": batch_id_str,
            "record_count": len(batch_recs),
            "html_file": html_path.name,
            "decision_file": decision_path.name,
            "review_ids": [r["review_id"] for r in batch_recs],
        })

        if (batch_idx + 1) % 10 == 0 or batch_idx == len(batches) - 1:
            log.info("Batch %s/%d written", batch_id_str, len(batches))

    manifest = {
        "schema_version": C.REVIEW_SCHEMA_VERSION,
        "run_id": run_id,
        "source": source,
        "phase": C.REVIEW_PHASE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_records": len(review_records),
        "batch_count": len(batches),
        "batch_size": batch_size,
        "batches": batch_manifest_entries,
    }
    manifest_path = review_dir / f"filter_review_manifest_{run_id}.json"
    if not patch_html_only:
        _write_json_atomic(manifest_path, manifest)
        log.info("Manifest written: %s", manifest_path)
    else:
        log.info("patch_html_only: manifest not overwritten (%s)", manifest_path)

    # Safety check: no private data in any generated HTML
    _assert_no_private_data_in_html(review_dir, run_id)

    return manifest


def _assert_no_private_data_in_html(review_dir: Path, run_id: str) -> None:
    """Scan generated HTML files for any forbidden metadata patterns."""
    import re
    forbidden = [
        re.compile(r"photos/photo_", re.IGNORECASE),
        re.compile(r"photo_\d+@\d{2}-\d{2}-\d{4}", re.IGNORECASE),
        re.compile(r"\b[0-9a-f]{64}\b"),  # sha256
    ]
    for html_path in review_dir.glob(f"filter_review_batch_{run_id}_*.html"):
        content = html_path.read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern.search(content):
                raise ValueError(
                    f"PRIVACY VIOLATION: forbidden pattern '{pattern.pattern}' found in {html_path.name}"
                )
    log.info("Privacy scan passed: no forbidden metadata in HTML files")


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="U4 Phase D: Prepare manual review batches.",
    )
    p.add_argument(
        "--candidates",
        type=Path,
        default=C.FILTER_CANDIDATES_PATH,
        help="Path to filter_candidates.jsonl (default: %(default)s)",
    )
    p.add_argument(
        "--review-dir",
        type=Path,
        default=C.FILTER_REVIEW_DIR,
        help="Output directory for review artifacts (default: %(default)s)",
    )
    p.add_argument(
        "--secret",
        type=Path,
        default=C.REVIEW_SECRET_PATH,
        help="Path to review HMAC secret (created if absent; default: %(default)s)",
    )
    p.add_argument(
        "--assets-base",
        type=Path,
        default=C.REVIEW_ASSETS_BASE,
        help="Base directory for thumbnail assets (default: %(default)s)",
    )
    p.add_argument(
        "--export-dir",
        type=Path,
        default=C.EXPORT_DIR,
        help="Telegram export directory containing original photos (default: %(default)s)",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Review run ID (default: auto-generated rvrun_<timestamp>)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=C.REVIEW_BATCH_SIZE_DEFAULT,
        help="Images per review batch (default: %(default)s)",
    )
    p.add_argument(
        "--source",
        type=str,
        default=C.SOURCE_TAG,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and compute batches without writing any files",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N records by priority (smoke/dev runs only; omit for production)",
    )
    p.add_argument(
        "--patch-html-only",
        action="store_true",
        help=(
            "Regenerate HTML batch files only — decision JSON templates and manifest are NOT "
            "overwritten. Safe to run after manual decisions have been recorded in JSON files."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_id = args.run_id or f"rvrun_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    try:
        prepare(
            candidates_path=args.candidates,
            review_dir=args.review_dir,
            secret_path=args.secret,
            assets_base=args.assets_base,
            export_dir=args.export_dir,
            run_id=run_id,
            batch_size=args.batch_size,
            source=args.source,
            dry_run=args.dry_run,
            limit=args.limit,
            patch_html_only=args.patch_html_only,
        )
    except ValueError as exc:
        log.error("Prepare failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
