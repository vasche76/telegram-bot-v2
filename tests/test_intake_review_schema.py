"""
test_intake_review_schema.py — Tests for intake_review_schema.py (U4 Phase D).

Covers:
- Schema validation: required fields, enum checks, confidence range
- Decision consistency: KEEP, REMOVE, RELABEL, UNSURE rules
- Privacy: notes field forbidden pattern detection
- File-level validation: schema_version, phase, records type
- Duplicate review ID within file
- Edge cases: null notes, null refinement
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import intake_constants as C
from intake_review_schema import (
    validate_record_schema,
    validate_record_consistency,
    validate_record_full,
    validate_decision_file,
    validate_decision_file_records,
    scan_notes_for_privacy_leak,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _rec(
    review_id: str = "rv_abcdef1234567890",
    decision_type: str = "RELABEL",
    phase_c_category: str = "unknown_needs_review",
    final_category: str = "fish",
    human_confidence: int = 4,
    refinement: dict | None = None,
    notes: str | None = None,
    reviewed_at: str = "2026-04-27T20:00:00+03:00",
) -> dict:
    return {
        "review_id": review_id,
        "decision_type": decision_type,
        "phase_c_category": phase_c_category,
        "final_category": final_category,
        "human_confidence": human_confidence,
        "refinement": refinement or {"species": None, "life_stage": "unknown"},
        "notes": notes,
        "reviewed_at": reviewed_at,
    }


def _decision_file(records: list[dict], run_id: str = "rvrun_test") -> dict:
    return {
        "schema_version": C.REVIEW_SCHEMA_VERSION,
        "source": C.SOURCE_TAG,
        "phase": C.REVIEW_PHASE,
        "run_id": run_id,
        "batch_id": "0001",
        "created_by": C.REVIEW_CREATED_BY,
        "records": records,
    }


# ─── Schema validation: happy paths ──────────────────────────────────────────


def test_valid_relabel_record() -> None:
    errors = validate_record_schema(_rec())
    assert errors == []


def test_valid_remove_record() -> None:
    errors = validate_record_schema(_rec(decision_type="REMOVE", final_category="no_fish"))
    assert errors == []


def test_valid_keep_record() -> None:
    errors = validate_record_schema(
        _rec(decision_type="KEEP", phase_c_category="fish_part", final_category="fish_part", human_confidence=3)
    )
    assert errors == []


def test_valid_unsure_record() -> None:
    errors = validate_record_schema(
        _rec(decision_type="UNSURE", final_category="unsure", human_confidence=2)
    )
    assert errors == []


def test_null_notes_allowed() -> None:
    errors = validate_record_schema(_rec(notes=None))
    assert errors == []


def test_null_refinement_allowed() -> None:
    errors = validate_record_schema(_rec(refinement=None))
    assert errors == []


# ─── Schema validation: error paths ──────────────────────────────────────────


def test_missing_required_field_review_id() -> None:
    rec = _rec()
    del rec["review_id"]
    errors = validate_record_schema(rec)
    assert any("review_id" in e for e in errors)


def test_missing_required_field_decision_type() -> None:
    rec = _rec()
    del rec["decision_type"]
    errors = validate_record_schema(rec)
    assert any("decision_type" in e for e in errors)


def test_invalid_decision_type() -> None:
    rec = _rec(decision_type="APPROVE")
    errors = validate_record_schema(rec)
    assert any("decision_type" in e for e in errors)


def test_invalid_final_category() -> None:
    rec = _rec(final_category="some_garbage")
    errors = validate_record_schema(rec)
    assert any("final_category" in e for e in errors)


def test_confidence_too_low() -> None:
    rec = _rec(human_confidence=0)
    errors = validate_record_schema(rec)
    assert any("human_confidence" in e for e in errors)


def test_confidence_too_high() -> None:
    rec = _rec(human_confidence=6)
    errors = validate_record_schema(rec)
    assert any("human_confidence" in e for e in errors)


def test_review_id_bad_prefix() -> None:
    rec = _rec(review_id="xx_abcdef1234567890")
    errors = validate_record_schema(rec)
    assert any("review_id" in e for e in errors)


def test_non_dict_refinement_invalid() -> None:
    rec = _rec()
    rec["refinement"] = "bad"
    errors = validate_record_schema(rec)
    assert any("refinement" in e for e in errors)


def test_non_string_notes_invalid() -> None:
    rec = _rec()
    rec["notes"] = 123
    errors = validate_record_schema(rec)
    assert any("notes" in e for e in errors)


# ─── Decision consistency: KEEP ───────────────────────────────────────────────


def test_keep_unknown_needs_review_fails() -> None:
    rec = _rec(decision_type="KEEP", phase_c_category="unknown_needs_review",
               final_category="unknown_needs_review", human_confidence=3)
    errors = validate_record_consistency(rec)
    assert any("unknown_needs_review" in e for e in errors)


def test_keep_category_mismatch_fails() -> None:
    rec = _rec(decision_type="KEEP", phase_c_category="fish_part",
               final_category="fish", human_confidence=4)
    errors = validate_record_consistency(rec)
    assert any("final_category must equal phase_c_category" in e for e in errors)


def test_keep_low_confidence_fails() -> None:
    rec = _rec(decision_type="KEEP", phase_c_category="fish_part",
               final_category="fish_part", human_confidence=2)
    errors = validate_record_consistency(rec)
    assert any("confidence" in e.lower() for e in errors)


def test_keep_valid_passes() -> None:
    rec = _rec(decision_type="KEEP", phase_c_category="fish_part",
               final_category="fish_part", human_confidence=3)
    errors = validate_record_consistency(rec)
    assert errors == []


# ─── Decision consistency: REMOVE ────────────────────────────────────────────


def test_remove_with_fish_category_fails() -> None:
    rec = _rec(decision_type="REMOVE", final_category="fish")
    errors = validate_record_consistency(rec)
    assert any("REMOVE" in e for e in errors)


def test_remove_with_no_fish_passes() -> None:
    rec = _rec(decision_type="REMOVE", final_category="no_fish", human_confidence=1)
    errors = validate_record_consistency(rec)
    assert errors == []


def test_remove_with_bad_quality_passes() -> None:
    rec = _rec(decision_type="REMOVE", final_category="bad_quality", human_confidence=2)
    errors = validate_record_consistency(rec)
    assert errors == []


def test_remove_with_duplicate_suspect_passes() -> None:
    rec = _rec(decision_type="REMOVE", final_category="duplicate_suspect", human_confidence=1)
    errors = validate_record_consistency(rec)
    assert errors == []


# ─── Decision consistency: RELABEL ───────────────────────────────────────────


def test_relabel_same_category_fails() -> None:
    rec = _rec(decision_type="RELABEL", phase_c_category="fish_part", final_category="fish_part")
    errors = validate_record_consistency(rec)
    assert any("differ from phase_c_category" in e for e in errors)


def test_relabel_to_unsure_fails() -> None:
    rec = _rec(decision_type="RELABEL", phase_c_category="fish_part", final_category="unsure")
    errors = validate_record_consistency(rec)
    assert any("unsure" in e for e in errors)


def test_relabel_valid_passes() -> None:
    rec = _rec(decision_type="RELABEL", phase_c_category="unknown_needs_review", final_category="fish")
    errors = validate_record_consistency(rec)
    assert errors == []


# ─── Decision consistency: UNSURE ────────────────────────────────────────────


def test_unsure_wrong_final_category_fails() -> None:
    rec = _rec(decision_type="UNSURE", final_category="fish", human_confidence=1)
    errors = validate_record_consistency(rec)
    assert any("unsure" in e for e in errors)


def test_unsure_high_confidence_passes() -> None:
    # High confidence + UNSURE is valid: reviewer may be certain the image is genuinely ambiguous.
    rec = _rec(decision_type="UNSURE", final_category="unsure", human_confidence=5)
    errors = validate_record_consistency(rec)
    assert errors == []


def test_unsure_valid_passes() -> None:
    rec = _rec(decision_type="UNSURE", final_category="unsure", human_confidence=2)
    errors = validate_record_consistency(rec)
    assert errors == []


# ─── Privacy scan ─────────────────────────────────────────────────────────────


def test_privacy_no_violation() -> None:
    assert scan_notes_for_privacy_leak("Looks like a pike") == []


def test_privacy_null_notes_ok() -> None:
    assert scan_notes_for_privacy_leak(None) == []


def test_privacy_filename_in_notes_fails() -> None:
    violations = scan_notes_for_privacy_leak("see photos/photo_1@25-12-2017_19-47-37.jpg")
    assert len(violations) > 0


def test_privacy_sha256_in_notes_fails() -> None:
    sha = "a" * 64
    violations = scan_notes_for_privacy_leak(f"hash is {sha}")
    assert len(violations) > 0


def test_privacy_sender_id_in_notes_fails() -> None:
    violations = scan_notes_for_privacy_leak("sender_id is 123456")
    assert len(violations) > 0


# ─── File-level validation ────────────────────────────────────────────────────


def test_valid_decision_file() -> None:
    data = _decision_file([_rec()])
    errors = validate_decision_file(data)
    assert errors == []


def test_decision_file_wrong_schema_version() -> None:
    data = _decision_file([])
    data["schema_version"] = "wrong"
    errors = validate_decision_file(data)
    assert any("schema_version" in e for e in errors)


def test_decision_file_wrong_phase() -> None:
    data = _decision_file([])
    data["phase"] = "WRONG"
    errors = validate_decision_file(data)
    assert any("phase" in e for e in errors)


def test_decision_file_al_negative_review_phase_accepted() -> None:
    data = _decision_file([_rec()])
    data["phase"] = C.REVIEW_PHASE_AL
    errors = validate_decision_file(data)
    assert errors == []


def test_decision_file_missing_records_field() -> None:
    data = _decision_file([])
    del data["records"]
    errors = validate_decision_file(data)
    assert any("records" in e for e in errors)


# ─── Duplicate review ID within file ─────────────────────────────────────────


def test_duplicate_review_id_within_file_fails() -> None:
    rec1 = _rec(review_id="rv_abcdef1234567890")
    rec2 = _rec(review_id="rv_abcdef1234567890")
    data = _decision_file([rec1, rec2])
    errors, invalid_count, valid_count = validate_decision_file_records(data)
    assert any("duplicate" in e for e in errors)
    assert invalid_count >= 1


# ─── Full validation (schema + consistency) ───────────────────────────────────


def test_full_validate_valid_record_passes() -> None:
    errors = validate_record_full(_rec())
    assert errors == []


def test_full_validate_invalid_record_fails() -> None:
    rec = _rec(decision_type="KEEP", phase_c_category="unknown_needs_review",
               final_category="unknown_needs_review")
    errors = validate_record_full(rec)
    assert len(errors) > 0
