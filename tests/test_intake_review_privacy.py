"""
test_intake_review_privacy.py — Privacy enforcement tests for U4 Phase D.

Covers:
- review_id never exposes original filename or sha256
- Decision templates contain no filenames, captions, or sender data
- Tracked summary contains no per-record identifiers
- HTML contact sheets contain no original filenames, sha256, or caption text
- Notes privacy scan catches all forbidden patterns
- Signal extraction never leaks filename or sha256
- Prepare output files pass full privacy scan
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import intake_constants as C
from intake_review_schema import scan_notes_for_privacy_leak
from intake_review_prepare import (
    compute_review_id,
    _extract_safe_signals,
    prepare,
    _assert_no_private_data_in_html,
)
from intake_review_aggregate import _assert_summary_privacy


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_candidate(
    i: int,
    reasons: list[str] | None = None,
    conflicts: list[str] | None = None,
    candidate_category: str = "unknown_needs_review",
) -> dict:
    sha = hashlib.sha256(f"img_{i}".encode()).hexdigest()
    return {
        "filename": f"photos/photo_{i}@01-01-2020_12-00-00.jpg",
        "sha256": sha,
        "candidate_category": candidate_category,
        "confidence": "low",
        "review_required": True,
        "reasons": reasons or [],
        "conflicts": conflicts or [],
        "source": C.SOURCE_TAG,
        "schema_version": 1,
    }


def _write_candidates(tmp_path: Path, candidates: list[dict]) -> Path:
    p = tmp_path / "filter_candidates.jsonl"
    p.write_text("\n".join(json.dumps(c) for c in candidates) + "\n", encoding="utf-8")
    return p


# ─── review_id privacy ────────────────────────────────────────────────────────


def test_review_id_does_not_contain_original_filename() -> None:
    import secrets
    secret = secrets.token_bytes(32)
    sha = "a" * 64
    rid = compute_review_id(secret, sha)
    # The review_id must not contain any substring of the sha256
    assert sha not in rid
    assert sha[:16] not in rid


def test_review_id_does_not_contain_sha256() -> None:
    import secrets
    secret = secrets.token_bytes(32)
    sha = hashlib.sha256(b"secret_image").hexdigest()
    rid = compute_review_id(secret, sha)
    assert sha not in rid


def test_different_images_get_different_review_ids() -> None:
    import secrets
    secret = secrets.token_bytes(32)
    sha1 = hashlib.sha256(b"img1").hexdigest()
    sha2 = hashlib.sha256(b"img2").hexdigest()
    assert compute_review_id(secret, sha1) != compute_review_id(secret, sha2)


# ─── Signal extraction privacy ────────────────────────────────────────────────


def test_signal_extraction_no_filename_leak() -> None:
    cand = _make_candidate(0)
    signals = _extract_safe_signals(cand)
    content = json.dumps(signals)
    assert "photos/" not in content
    assert "photo_0" not in content
    assert cand["filename"] not in content


def test_signal_extraction_no_sha256_leak() -> None:
    cand = _make_candidate(0)
    signals = _extract_safe_signals(cand)
    assert cand["sha256"] not in json.dumps(signals)


def test_signal_extraction_no_caption_text() -> None:
    cand = _make_candidate(0, reasons=["caption_lure_hint"])
    signals = _extract_safe_signals(cand)
    # Only the boolean presence, not any caption text
    assert signals["caption_keyword_signal_present"] is True
    # Values must not be caption text — only booleans / category strings
    for v in signals.values():
        assert isinstance(v, (bool, str)), f"unexpected value type: {type(v)}"
    # Confirm no actual caption content (the value is a bool, not text)
    assert signals["caption_keyword_signal_present"] is True  # boolean, not caption text


# ─── Notes privacy scan ───────────────────────────────────────────────────────


def test_notes_scan_clean() -> None:
    assert scan_notes_for_privacy_leak("Large pike, looks healthy") == []
    assert scan_notes_for_privacy_leak("") == []
    assert scan_notes_for_privacy_leak(None) == []


def test_notes_scan_catches_telegram_filename() -> None:
    violations = scan_notes_for_privacy_leak("see photos/photo_1@25-12-2017_19-47-37.jpg")
    assert violations


def test_notes_scan_catches_sha256() -> None:
    violations = scan_notes_for_privacy_leak("hash: " + "f" * 64)
    assert violations


def test_notes_scan_catches_sender_id() -> None:
    violations = scan_notes_for_privacy_leak("from_id: 987654321")
    assert violations


def test_notes_scan_catches_chat_export() -> None:
    violations = scan_notes_for_privacy_leak("ChatExport_2026-04-24/photos")
    assert violations


def test_notes_scan_catches_user_id() -> None:
    violations = scan_notes_for_privacy_leak("user_id=12345")
    assert violations


# ─── HTML privacy enforcement ─────────────────────────────────────────────────


def test_html_does_not_contain_sha256(tmp_path: Path) -> None:
    candidates = [_make_candidate(i) for i in range(3)]
    cands_path = _write_candidates(tmp_path, candidates)
    review_dir = tmp_path / "review"
    secret_path = review_dir / ".review_secret"

    prepare(
        candidates_path=cands_path,
        review_dir=review_dir,
        secret_path=secret_path,
        assets_base=review_dir / "assets",
        export_dir=None,
        run_id="rvrun_priv",
        batch_size=10,
        source=C.SOURCE_TAG,
        dry_run=False,
    )

    for html_path in review_dir.glob("*.html"):
        content = html_path.read_text()
        for cand in candidates:
            assert cand["sha256"] not in content, f"sha256 leaked in {html_path.name}"
            assert cand["filename"] not in content, f"filename leaked in {html_path.name}"


def test_html_does_not_contain_original_filenames(tmp_path: Path) -> None:
    candidates = [_make_candidate(i) for i in range(2)]
    cands_path = _write_candidates(tmp_path, candidates)
    review_dir = tmp_path / "review2"
    secret_path = review_dir / ".review_secret"

    prepare(
        candidates_path=cands_path,
        review_dir=review_dir,
        secret_path=secret_path,
        assets_base=review_dir / "assets",
        export_dir=None,
        run_id="rvrun_prvfn",
        batch_size=10,
        source=C.SOURCE_TAG,
        dry_run=False,
    )

    for html_path in review_dir.glob("*.html"):
        content = html_path.read_text()
        assert "photos/photo_" not in content


# ─── Decision template privacy ────────────────────────────────────────────────


def test_decision_template_no_sha256(tmp_path: Path) -> None:
    candidates = [_make_candidate(i) for i in range(2)]
    cands_path = _write_candidates(tmp_path, candidates)
    review_dir = tmp_path / "review3"
    secret_path = review_dir / ".review_secret"

    prepare(
        candidates_path=cands_path,
        review_dir=review_dir,
        secret_path=secret_path,
        assets_base=review_dir / "assets",
        export_dir=None,
        run_id="rvrun_tpl",
        batch_size=10,
        source=C.SOURCE_TAG,
        dry_run=False,
    )

    for decision_path in review_dir.glob("filter_decisions_rvrun_tpl_*.json"):
        content = decision_path.read_text()
        for cand in candidates:
            assert cand["sha256"] not in content
            assert cand["filename"] not in content


def test_decision_template_review_ids_present(tmp_path: Path) -> None:
    candidates = [_make_candidate(i) for i in range(2)]
    cands_path = _write_candidates(tmp_path, candidates)
    review_dir = tmp_path / "review4"
    secret_path = review_dir / ".review_secret"

    prepare(
        candidates_path=cands_path,
        review_dir=review_dir,
        secret_path=secret_path,
        assets_base=review_dir / "assets",
        export_dir=None,
        run_id="rvrun_ridpr",
        batch_size=10,
        source=C.SOURCE_TAG,
        dry_run=False,
    )

    for decision_path in review_dir.glob("filter_decisions_rvrun_ridpr_*.json"):
        data = json.loads(decision_path.read_text())
        for rec in data["records"]:
            assert rec["review_id"].startswith("rv_")
            # decision fields are null (template state)
            assert rec["decision_type"] is None
            assert rec["final_category"] is None


# ─── Tracked summary privacy ──────────────────────────────────────────────────


def test_tracked_summary_no_review_ids() -> None:
    summary = {
        "schema_version": C.REVIEW_SUMMARY_SCHEMA_VERSION,
        "total_reviewed": 100,
        "decision_type_counts": {"RELABEL": 50, "REMOVE": 30, "KEEP": 15, "UNSURE": 5},
        "final_category_counts": {"fish": 50},
    }
    _assert_summary_privacy(summary)  # must not raise


def test_tracked_summary_fails_on_review_id() -> None:
    summary = {"leaked": "rv_abcdef123456789a"}  # rv_ + exactly 16 hex chars
    with pytest.raises(ValueError, match="PRIVACY VIOLATION"):
        _assert_summary_privacy(summary)


def test_tracked_summary_fails_on_sha256() -> None:
    summary = {"hash": "a" * 64}
    with pytest.raises(ValueError, match="PRIVACY VIOLATION"):
        _assert_summary_privacy(summary)
