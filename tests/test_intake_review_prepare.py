"""
test_intake_review_prepare.py — Tests for intake_review_prepare.py (U4 Phase D).

Covers:
- HMAC review ID: deterministic, correct prefix, different secrets → different IDs
- Phase C invariant checks: duplicate sha256, review_required=False, unknown category
- Signal extraction: privacy-safe signals from candidate record
- Batching: deterministic order (priority bucket, then review_id), chunking
- Batch priority: conflicts first, then special cats, then quality, then default
- Dry-run: no files written
- HTML generation: no forbidden private data
- Integration: full prepare on synthetic mini-dataset
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import intake_constants as C
from intake_review_prepare import (
    compute_review_id,
    check_phase_c_invariants,
    build_batches,
    _extract_safe_signals,
    _priority_bucket,
    prepare,
    _build_html,
    _assert_no_private_data_in_html,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _make_candidate(
    filename: str = "photos/photo_1@01-01-2020_00-00-00.jpg",
    sha256: str = "a" * 64,
    candidate_category: str = "unknown_needs_review",
    confidence: str = "low",
    review_required: bool = True,
    reasons: list[str] | None = None,
    conflicts: list[str] | None = None,
) -> dict:
    return {
        "filename": filename,
        "sha256": sha256,
        "candidate_category": candidate_category,
        "confidence": confidence,
        "review_required": review_required,
        "reasons": reasons or [],
        "conflicts": conflicts or [],
        "source": C.SOURCE_TAG,
        "schema_version": 1,
    }


def _make_candidates(n: int) -> list[dict]:
    return [
        _make_candidate(
            filename=f"photos/photo_{i}@01-01-2020.jpg",
            sha256=hashlib.sha256(f"image_{i}".encode()).hexdigest(),
        )
        for i in range(n)
    ]


def _make_signals_file(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "filter_candidates.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return p


# ─── HMAC review ID ──────────────────────────────────────────────────────────


def test_compute_review_id_prefix() -> None:
    secret = secrets.token_bytes(32)
    rid = compute_review_id(secret, "abc123")
    assert rid.startswith("rv_")
    assert len(rid) == 3 + 16  # "rv_" + 16 hex chars


def test_compute_review_id_deterministic() -> None:
    secret = secrets.token_bytes(32)
    rid1 = compute_review_id(secret, "abc123")
    rid2 = compute_review_id(secret, "abc123")
    assert rid1 == rid2


def test_compute_review_id_different_key() -> None:
    secret = secrets.token_bytes(32)
    rid1 = compute_review_id(secret, "key_a")
    rid2 = compute_review_id(secret, "key_b")
    assert rid1 != rid2


def test_compute_review_id_different_secret() -> None:
    key = "same_key"
    rid1 = compute_review_id(secrets.token_bytes(32), key)
    rid2 = compute_review_id(secrets.token_bytes(32), key)
    assert rid1 != rid2


# ─── Phase C invariant checks ────────────────────────────────────────────────


def test_invariant_check_ok() -> None:
    candidates = _make_candidates(3)
    check_phase_c_invariants(candidates)  # should not raise


def test_invariant_duplicate_sha256_fails() -> None:
    candidates = _make_candidates(2)
    candidates[1]["sha256"] = candidates[0]["sha256"]
    with pytest.raises(ValueError, match="duplicate sha256"):
        check_phase_c_invariants(candidates)


def test_invariant_review_required_false_fails() -> None:
    candidates = _make_candidates(1)
    candidates[0]["review_required"] = False
    with pytest.raises(ValueError, match="review_required"):
        check_phase_c_invariants(candidates)


def test_invariant_unknown_category_fails() -> None:
    candidates = _make_candidates(1)
    candidates[0]["candidate_category"] = "not_a_real_category"
    with pytest.raises(ValueError, match="candidate_category"):
        check_phase_c_invariants(candidates)


# ─── Signal extraction ────────────────────────────────────────────────────────


def test_extract_signals_default() -> None:
    rec = _make_candidate()
    signals = _extract_safe_signals(rec)
    assert signals == {
        "low_res": False,
        "extreme_aspect": False,
        "tiny_file": False,
        "caption_keyword_signal_present": False,
        "phase_c_conflict": False,
        "phase_c_candidate_category": "unknown_needs_review",
    }


def test_extract_signals_low_res() -> None:
    rec = _make_candidate(reasons=["low_res"])
    signals = _extract_safe_signals(rec)
    assert signals["low_res"] is True
    assert signals["extreme_aspect"] is False


def test_extract_signals_caption_kw() -> None:
    rec = _make_candidate(reasons=["caption_lure_hint"])
    signals = _extract_safe_signals(rec)
    assert signals["caption_keyword_signal_present"] is True


def test_extract_signals_conflict() -> None:
    rec = _make_candidate(conflicts=["conflict_a"])
    signals = _extract_safe_signals(rec)
    assert signals["phase_c_conflict"] is True


def test_extract_signals_no_filename_in_output() -> None:
    rec = _make_candidate(filename="photos/secret_name.jpg")
    signals = _extract_safe_signals(rec)
    assert "filename" not in signals
    assert "secret_name" not in str(signals)


def test_extract_signals_no_sha256_in_output() -> None:
    sha = "b" * 64
    rec = _make_candidate(sha256=sha)
    signals = _extract_safe_signals(rec)
    assert sha not in str(signals)


# ─── Category normalization (lure_fishing_gear → lure_gear) ──────────────────


def test_normalize_lure_fishing_gear_to_lure_gear() -> None:
    assert C.normalize_phase_c_category("lure_fishing_gear") == "lure_gear"


def test_normalize_unknown_passthrough() -> None:
    assert C.normalize_phase_c_category("fish") == "fish"
    assert C.normalize_phase_c_category("unknown_needs_review") == "unknown_needs_review"


def test_extract_signals_lure_fishing_gear_normalized() -> None:
    rec = _make_candidate(candidate_category="lure_fishing_gear")
    signals = _extract_safe_signals(rec)
    assert signals["phase_c_candidate_category"] == "lure_gear"
    assert signals["phase_c_candidate_category"] in C.FINAL_CATEGORIES_SET


def test_keep_decision_lure_gear_valid() -> None:
    """KEEP with final_category=lure_gear and phase_c_category=lure_gear must pass consistency."""
    from intake_review_schema import validate_record_consistency
    rec = {
        "review_id": "rv_abc1234567890123",
        "decision_type": C.DECISION_TYPE_KEEP,
        "phase_c_category": "lure_gear",
        "final_category": "lure_gear",
        "human_confidence": 4,
        "refinement": None,
        "notes": None,
    }
    errors = validate_record_consistency(rec)
    assert errors == [], f"Unexpected errors: {errors}"


def test_decision_template_phase_c_category_normalized(tmp_path: Path) -> None:
    """Decision template must store lure_gear (not lure_fishing_gear) as phase_c_category."""
    candidates = [_make_candidate(
        sha256="d" * 64,
        candidate_category="lure_fishing_gear",
    )]
    candidates_path = _make_signals_file(tmp_path, candidates)
    secret_path = tmp_path / "review" / ".review_secret"
    review_dir = tmp_path / "review"
    assets_base = tmp_path / "review" / "assets"

    manifest = prepare(
        candidates_path=candidates_path,
        review_dir=review_dir,
        secret_path=secret_path,
        assets_base=assets_base,
        export_dir=None,
        run_id="rvrun_normtest",
        batch_size=10,
        source=C.SOURCE_TAG,
        dry_run=False,
    )
    decision_files = list(review_dir.glob("filter_decisions_rvrun_normtest_*.json"))
    assert len(decision_files) == 1
    data = json.loads(decision_files[0].read_text())
    assert data["records"][0]["phase_c_category"] == "lure_gear"


# ─── Batching ─────────────────────────────────────────────────────────────────


def _make_review_rec(review_id: str, signals: dict | None = None) -> dict:
    return {
        "review_id": review_id,
        "signals": signals or {
            "low_res": False,
            "extreme_aspect": False,
            "tiny_file": False,
            "caption_keyword_signal_present": False,
            "phase_c_conflict": False,
            "phase_c_candidate_category": "unknown_needs_review",
        },
        "_src_filename": "photos/x.jpg",
    }


def test_build_batches_chunking() -> None:
    recs = [_make_review_rec(f"rv_{i:016x}") for i in range(10)]
    batches = build_batches(recs, batch_size=4)
    assert len(batches) == 3
    assert len(batches[0]) == 4
    assert len(batches[1]) == 4
    assert len(batches[2]) == 2


def test_build_batches_deterministic() -> None:
    recs = [_make_review_rec(f"rv_{i:016x}") for i in range(6)]
    b1 = build_batches(recs, batch_size=3)
    b2 = build_batches(recs, batch_size=3)
    assert [[r["review_id"] for r in b] for b in b1] == [[r["review_id"] for r in b] for b in b2]


def test_batch_priority_conflict_first() -> None:
    conflict_rec = _make_review_rec("rv_conflict0000000a", signals={
        "low_res": False, "extreme_aspect": False, "tiny_file": False,
        "caption_keyword_signal_present": False,
        "phase_c_conflict": True,
        "phase_c_candidate_category": "unknown_needs_review",
    })
    normal_rec = _make_review_rec("rv_normal00000000a")
    recs = [normal_rec, conflict_rec]
    batches = build_batches(recs, batch_size=10)
    assert batches[0][0]["review_id"] == "rv_conflict0000000a"


def test_batch_priority_special_cats_before_quality() -> None:
    quality_rec = _make_review_rec("rv_quality00000000", signals={
        "low_res": True, "extreme_aspect": False, "tiny_file": False,
        "caption_keyword_signal_present": False, "phase_c_conflict": False,
        "phase_c_candidate_category": "unknown_needs_review",
    })
    fish_part_rec = _make_review_rec("rv_fishpart0000000", signals={
        "low_res": False, "extreme_aspect": False, "tiny_file": False,
        "caption_keyword_signal_present": False, "phase_c_conflict": False,
        "phase_c_candidate_category": "fish_part",
    })
    recs = [quality_rec, fish_part_rec]
    batches = build_batches(recs, batch_size=10)
    ids = [r["review_id"] for r in batches[0]]
    assert ids.index("rv_fishpart0000000") < ids.index("rv_quality00000000")


# ─── HTML privacy check ───────────────────────────────────────────────────────


def test_html_no_private_data(tmp_path: Path) -> None:
    sha = "c" * 64
    rid = "rv_test1234567890"
    batch_recs = [
        {
            "review_id": rid,
            "signals": {
                "low_res": True, "extreme_aspect": False, "tiny_file": False,
                "caption_keyword_signal_present": True, "phase_c_conflict": False,
                "phase_c_candidate_category": "unknown_needs_review",
            },
            "_src_filename": f"photos/photo_1@01-01-2020.jpg",
        }
    ]
    html = _build_html(batch_recs, "rvrun_test", "0001", C.SOURCE_TAG, "assets/rvrun_test")
    # SHA256 must not appear
    assert sha not in html
    # Original filename must not appear
    assert "photos/photo_1@01-01-2020" not in html
    # review_id is allowed (it's anonymized)
    assert rid in html


def test_assert_no_private_data_passes(tmp_path: Path) -> None:
    html_content = "<html><body>rv_abc123 fish_part</body></html>"
    html_path = tmp_path / "filter_review_batch_rvrun_x_0001.html"
    html_path.write_text(html_content)
    _assert_no_private_data_in_html(tmp_path, "rvrun_x")  # should not raise


def test_assert_no_private_data_fails_on_filename(tmp_path: Path) -> None:
    html_content = "<html><body>photos/photo_1@25-12-2017_19-47-37.jpg</body></html>"
    html_path = tmp_path / "filter_review_batch_rvrun_bad_0001.html"
    html_path.write_text(html_content)
    import pytest
    with pytest.raises(ValueError, match="PRIVACY VIOLATION"):
        _assert_no_private_data_in_html(tmp_path, "rvrun_bad")


# ─── Dry-run integration ──────────────────────────────────────────────────────


def test_prepare_dry_run(tmp_path: Path) -> None:
    candidates = _make_candidates(5)
    candidates_path = _make_signals_file(tmp_path, candidates)
    secret_path = tmp_path / "review" / ".review_secret"
    review_dir = tmp_path / "review"
    assets_base = tmp_path / "review" / "assets"

    result = prepare(
        candidates_path=candidates_path,
        review_dir=review_dir,
        secret_path=secret_path,
        assets_base=assets_base,
        export_dir=None,
        run_id="rvrun_drytest",
        batch_size=2,
        source=C.SOURCE_TAG,
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["total_records"] == 5
    assert result["batch_count"] == 3  # ceil(5/2)
    # No files should have been written
    assert not review_dir.exists() or not any(review_dir.iterdir())


def test_prepare_writes_artifacts(tmp_path: Path) -> None:
    candidates = _make_candidates(4)
    candidates_path = _make_signals_file(tmp_path, candidates)
    secret_path = tmp_path / "review" / ".review_secret"
    review_dir = tmp_path / "review"
    assets_base = tmp_path / "review" / "assets"

    manifest = prepare(
        candidates_path=candidates_path,
        review_dir=review_dir,
        secret_path=secret_path,
        assets_base=assets_base,
        export_dir=None,
        run_id="rvrun_writetest",
        batch_size=3,
        source=C.SOURCE_TAG,
        dry_run=False,
    )

    assert manifest["run_id"] == "rvrun_writetest"
    assert manifest["batch_count"] == 2

    # Manifest file written
    manifest_path = review_dir / "filter_review_manifest_rvrun_writetest.json"
    assert manifest_path.exists()

    # HTML files written
    html_files = list(review_dir.glob("filter_review_batch_rvrun_writetest_*.html"))
    assert len(html_files) == 2

    # Decision template files written
    decision_files = list(review_dir.glob("filter_decisions_rvrun_writetest_*.json"))
    assert len(decision_files) == 2

    # Decision templates have null decision_type (template, not filled)
    for df in decision_files:
        data = json.loads(df.read_text())
        assert data["schema_version"] == C.REVIEW_SCHEMA_VERSION
        for rec in data["records"]:
            assert rec["decision_type"] is None
            assert rec["review_id"].startswith("rv_")
            assert "filename" not in rec
            assert "sha256" not in rec


def test_prepare_html_no_sha256(tmp_path: Path) -> None:
    candidates = _make_candidates(2)
    sha = candidates[0]["sha256"]
    candidates_path = _make_signals_file(tmp_path, candidates)
    secret_path = tmp_path / "review" / ".review_secret"
    review_dir = tmp_path / "review"

    prepare(
        candidates_path=candidates_path,
        review_dir=review_dir,
        secret_path=secret_path,
        assets_base=tmp_path / "review" / "assets",
        export_dir=None,
        run_id="rvrun_prv",
        batch_size=10,
        source=C.SOURCE_TAG,
        dry_run=False,
    )

    for html_path in review_dir.glob("*.html"):
        content = html_path.read_text()
        assert sha not in content
        assert "photos/photo_" not in content


# ─── --limit smoke-test mode ──────────────────────────────────────────────────


def test_prepare_limit_dry_run(tmp_path: Path) -> None:
    candidates = _make_candidates(10)
    candidates_path = _make_signals_file(tmp_path, candidates)
    secret_path = tmp_path / "review" / ".review_secret"
    review_dir = tmp_path / "review"
    assets_base = tmp_path / "review" / "assets"

    result = prepare(
        candidates_path=candidates_path,
        review_dir=review_dir,
        secret_path=secret_path,
        assets_base=assets_base,
        export_dir=None,
        run_id="rvrun_limitdry",
        batch_size=10,
        source=C.SOURCE_TAG,
        dry_run=True,
        limit=3,
    )

    assert result["total_records"] == 3
    assert result["batch_count"] == 1


def test_prepare_limit_writes_only_n_records(tmp_path: Path) -> None:
    candidates = _make_candidates(10)
    candidates_path = _make_signals_file(tmp_path, candidates)
    secret_path = tmp_path / "review" / ".review_secret"
    review_dir = tmp_path / "review"
    assets_base = tmp_path / "review" / "assets"

    manifest = prepare(
        candidates_path=candidates_path,
        review_dir=review_dir,
        secret_path=secret_path,
        assets_base=assets_base,
        export_dir=None,
        run_id="rvrun_limitwrite",
        batch_size=10,
        source=C.SOURCE_TAG,
        dry_run=False,
        limit=5,
    )

    assert manifest["total_records"] == 5
    decision_files = list(review_dir.glob("filter_decisions_rvrun_limitwrite_*.json"))
    assert len(decision_files) == 1
    data = json.loads(decision_files[0].read_text())
    assert len(data["records"]) == 5


def test_prepare_limit_deterministic(tmp_path: Path) -> None:
    candidates = _make_candidates(10)
    candidates_path = _make_signals_file(tmp_path, candidates)

    def _run_limited(run_id: str, td: Path) -> list[str]:
        result = prepare(
            candidates_path=candidates_path,
            review_dir=td / "review",
            secret_path=td / ".secret",
            assets_base=td / "assets",
            export_dir=None,
            run_id=run_id,
            batch_size=10,
            source=C.SOURCE_TAG,
            dry_run=True,
            limit=4,
        )
        return []

    import tempfile
    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        r1 = prepare(
            candidates_path=candidates_path,
            review_dir=Path(d1) / "review",
            secret_path=Path(d1) / ".secret",
            assets_base=Path(d1) / "assets",
            export_dir=None,
            run_id="rvrun_det1",
            batch_size=10,
            source=C.SOURCE_TAG,
            dry_run=True,
            limit=4,
        )
        r2 = prepare(
            candidates_path=candidates_path,
            review_dir=Path(d2) / "review",
            secret_path=Path(d2) / ".secret",
            assets_base=Path(d2) / "assets",
            export_dir=None,
            run_id="rvrun_det2",
            batch_size=10,
            source=C.SOURCE_TAG,
            dry_run=True,
            limit=4,
        )
        assert r1["total_records"] == r2["total_records"] == 4


def test_prepare_no_limit_processes_all(tmp_path: Path) -> None:
    candidates = _make_candidates(7)
    candidates_path = _make_signals_file(tmp_path, candidates)
    secret_path = tmp_path / "review" / ".review_secret"
    review_dir = tmp_path / "review"

    result = prepare(
        candidates_path=candidates_path,
        review_dir=review_dir,
        secret_path=secret_path,
        assets_base=tmp_path / "review" / "assets",
        export_dir=None,
        run_id="rvrun_nolimit",
        batch_size=10,
        source=C.SOURCE_TAG,
        dry_run=True,
    )
    assert result["total_records"] == 7


# ─── HTML export: decision_type auto-derivation rules ────────────────────────
#
# These tests validate the JS deriveDecisionType logic indirectly by checking
# that the generated HTML:
#   1. Contains no f-decision-type select (removed from UI)
#   2. Contains the correct hint for unknown_needs_review cards
#   3. Contains the deriveDecisionType function with all required branches
#
# They also validate the schema consistency rules that the auto-derived values
# must satisfy (using validate_record_consistency directly as the ground truth).


def test_html_no_decision_type_dropdown() -> None:
    """Generated HTML must not contain a decision_type select — it is auto-derived."""
    batch_recs = [
        {
            "review_id": "rv_test1234567890aa",
            "signals": {
                "low_res": False, "extreme_aspect": False, "tiny_file": False,
                "caption_keyword_signal_present": False, "phase_c_conflict": False,
                "phase_c_candidate_category": "unknown_needs_review",
            },
            "_src_filename": "photos/x.jpg",
        }
    ]
    html = _build_html(batch_recs, "rvrun_test2", "0001", C.SOURCE_TAG, "assets/rvrun_test2")
    assert "f-decision-type" not in html
    assert 'name="decision_type"' not in html


def test_html_unknown_needs_review_hint_present() -> None:
    """unknown_needs_review cards must show a hint explaining KEEP is not available."""
    batch_recs = [
        {
            "review_id": "rv_test1234567890bb",
            "signals": {
                "low_res": False, "extreme_aspect": False, "tiny_file": False,
                "caption_keyword_signal_present": False, "phase_c_conflict": False,
                "phase_c_candidate_category": "unknown_needs_review",
            },
            "_src_filename": "photos/x.jpg",
        }
    ]
    html = _build_html(batch_recs, "rvrun_test3", "0001", C.SOURCE_TAG, "assets/rvrun_test3")
    assert "KEEP is not available" in html
    assert "auto-derived" in html


def test_html_known_category_hint_present() -> None:
    """Cards with a known Phase C category must show a KEEP-eligible hint."""
    batch_recs = [
        {
            "review_id": "rv_test1234567890cc",
            "signals": {
                "low_res": False, "extreme_aspect": False, "tiny_file": False,
                "caption_keyword_signal_present": False, "phase_c_conflict": False,
                "phase_c_candidate_category": "fish",
            },
            "_src_filename": "photos/x.jpg",
        }
    ]
    html = _build_html(batch_recs, "rvrun_test4", "0001", C.SOURCE_TAG, "assets/rvrun_test4")
    # Must not show the unknown_needs_review warning
    assert "KEEP is not available" not in html
    assert "auto-derived" in html


def test_html_contains_derive_decision_type_function() -> None:
    """Generated HTML must embed the deriveDecisionType JS function with all branches."""
    batch_recs = [
        {
            "review_id": "rv_test1234567890dd",
            "signals": {
                "low_res": False, "extreme_aspect": False, "tiny_file": False,
                "caption_keyword_signal_present": False, "phase_c_conflict": False,
                "phase_c_candidate_category": "unknown_needs_review",
            },
            "_src_filename": "photos/x.jpg",
        }
    ]
    html = _build_html(batch_recs, "rvrun_test5", "0001", C.SOURCE_TAG, "assets/rvrun_test5")
    assert "deriveDecisionType" in html
    assert "'UNSURE'" in html
    assert "'RELABEL'" in html
    assert "'KEEP'" in html
    assert "unknown_needs_review" in html


# ─── Schema consistency: auto-derived values must pass validator ──────────────


def _make_decision_record(
    phase_c_category: str,
    final_category: str,
    decision_type: str,
    human_confidence: int = 3,
) -> dict:
    return {
        "review_id": "rv_abc1234567890123",
        "decision_type": decision_type,
        "phase_c_category": phase_c_category,
        "final_category": final_category,
        "human_confidence": human_confidence,
        "refinement": None,
        "notes": None,
        "reviewed_at": "2026-04-27T20:00:00+00:00",
    }


def test_unknown_needs_review_fish_exports_relabel() -> None:
    """unknown_needs_review + final_category=fish must produce RELABEL (not KEEP)."""
    from intake_review_schema import validate_record_consistency
    rec = _make_decision_record(
        phase_c_category="unknown_needs_review",
        final_category="fish",
        decision_type="RELABEL",
    )
    errors = validate_record_consistency(rec)
    assert errors == [], f"RELABEL for unknown_needs_review+fish should be valid: {errors}"


def test_unknown_needs_review_no_fish_exports_relabel() -> None:
    """unknown_needs_review + final_category=no_fish must produce RELABEL."""
    from intake_review_schema import validate_record_consistency
    rec = _make_decision_record(
        phase_c_category="unknown_needs_review",
        final_category="no_fish",
        decision_type="RELABEL",
    )
    errors = validate_record_consistency(rec)
    assert errors == [], f"RELABEL for unknown_needs_review+no_fish should be valid: {errors}"


def test_unknown_needs_review_unsure_exports_unsure() -> None:
    """unknown_needs_review + final_category=unsure must produce UNSURE."""
    from intake_review_schema import validate_record_consistency
    rec = _make_decision_record(
        phase_c_category="unknown_needs_review",
        final_category="unsure",
        decision_type="UNSURE",
        human_confidence=2,
    )
    errors = validate_record_consistency(rec)
    assert errors == [], f"UNSURE for unknown_needs_review+unsure should be valid: {errors}"


def test_unknown_needs_review_keep_is_invalid() -> None:
    """KEEP with phase_c_category=unknown_needs_review must fail the consistency check."""
    from intake_review_schema import validate_record_consistency
    rec = _make_decision_record(
        phase_c_category="unknown_needs_review",
        final_category="fish",
        decision_type="KEEP",
        human_confidence=4,
    )
    errors = validate_record_consistency(rec)
    assert any("unknown_needs_review" in e for e in errors), (
        f"KEEP for unknown_needs_review must be rejected; got errors: {errors}"
    )


def test_known_category_matching_fc_high_conf_exports_keep() -> None:
    """Known phase_c_category + matching final_category + conf>=3 must produce KEEP."""
    from intake_review_schema import validate_record_consistency
    rec = _make_decision_record(
        phase_c_category="fish",
        final_category="fish",
        decision_type="KEEP",
        human_confidence=4,
    )
    errors = validate_record_consistency(rec)
    assert errors == [], f"KEEP for fish+fish+conf4 should be valid: {errors}"


def test_known_category_different_fc_exports_relabel() -> None:
    """Known phase_c_category + different final_category must produce RELABEL."""
    from intake_review_schema import validate_record_consistency
    rec = _make_decision_record(
        phase_c_category="fish",
        final_category="fish_part",
        decision_type="RELABEL",
        human_confidence=3,
    )
    errors = validate_record_consistency(rec)
    assert errors == [], f"RELABEL for fish+fish_part should be valid: {errors}"


# ─── patch_html_only: decision JSON is NOT overwritten ───────────────────────


def test_patch_html_only_does_not_overwrite_decision_json(tmp_path: Path) -> None:
    """patch_html_only=True must regenerate HTML but leave decision JSON files untouched."""
    import hashlib

    candidates = [
        _make_candidate(
            sha256=hashlib.sha256(f"img_{i}".encode()).hexdigest(),
            filename=f"photos/photo_{i}.jpg",
        )
        for i in range(3)
    ]
    candidates_path = _make_signals_file(tmp_path, candidates)
    review_dir = tmp_path / "review"
    secret_path = review_dir / ".review_secret"
    assets_base = review_dir / "assets"

    # Initial prepare — writes HTML + decision JSON + manifest
    prepare(
        candidates_path=candidates_path,
        review_dir=review_dir,
        secret_path=secret_path,
        assets_base=assets_base,
        export_dir=None,
        run_id="rvrun_patchtest",
        batch_size=10,
        source=C.SOURCE_TAG,
        dry_run=False,
    )

    # Simulate manual decisions written into the decision JSON
    decision_files = list(review_dir.glob("filter_decisions_rvrun_patchtest_*.json"))
    assert len(decision_files) == 1
    sentinel_content = '{"sentinel": "manual_decision_must_not_be_overwritten"}'
    decision_files[0].write_text(sentinel_content)

    # patch_html_only=True — must NOT overwrite the decision JSON
    prepare(
        candidates_path=candidates_path,
        review_dir=review_dir,
        secret_path=secret_path,
        assets_base=assets_base,
        export_dir=None,
        run_id="rvrun_patchtest",
        batch_size=10,
        source=C.SOURCE_TAG,
        dry_run=False,
        patch_html_only=True,
    )

    assert decision_files[0].read_text() == sentinel_content, (
        "patch_html_only must not overwrite existing decision JSON"
    )

    # HTML must have been regenerated with the new decision_type-free form
    html_files = list(review_dir.glob("filter_review_batch_rvrun_patchtest_*.html"))
    assert len(html_files) == 1
    html = html_files[0].read_text()
    assert "f-decision-type" not in html
    assert "deriveDecisionType" in html
