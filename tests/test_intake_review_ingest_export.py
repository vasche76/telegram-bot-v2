"""
test_intake_review_ingest_export.py — Tests for intake_review_ingest_export.py (U4 Phase D).

Covers:
- ingest_export copies valid downloaded JSON into review dir
- ingest_export validates schema and rejects invalid JSON
- ingest_export rejects mismatched run_id
- ingest_export rejects mismatched batch_id
- ingest_export rejects blank template (no decisions filled)
- ingest_export refuses to overwrite already-filled batch without --force
- ingest_export backs up blank template before overwrite
- validate_export returns no errors for valid export
- validate_export catches record-level errors
- privacy: suspicious source paths are rejected
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import intake_constants as C
from intake_review_ingest_export import ingest_export, validate_export, _assert_no_private_paths


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _valid_export(
    run_id: str,
    batch_id: str,
    review_ids: list[str],
    path: Path,
) -> Path:
    """Write a filled decision export to path."""
    records = [
        {
            "review_id": rid,
            "decision_type": "RELABEL",
            "phase_c_category": "unknown_needs_review",
            "final_category": "fish",
            "human_confidence": 4,
            "refinement": {"species": None, "life_stage": "unknown"},
            "notes": None,
            "reviewed_at": "2026-04-27T20:00:00+03:00",
        }
        for rid in review_ids
    ]
    data = {
        "schema_version": C.REVIEW_SCHEMA_VERSION,
        "source": C.SOURCE_TAG,
        "phase": C.REVIEW_PHASE,
        "run_id": run_id,
        "batch_id": batch_id,
        "created_by": C.REVIEW_CREATED_BY,
        "records": records,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _blank_template(
    run_id: str,
    batch_id: str,
    review_ids: list[str],
    path: Path,
) -> Path:
    """Write a blank decision template to path."""
    records = [
        {
            "review_id": rid,
            "decision_type": None,
            "phase_c_category": "unknown_needs_review",
            "final_category": None,
            "human_confidence": None,
            "refinement": None,
            "notes": None,
            "reviewed_at": None,
        }
        for rid in review_ids
    ]
    data = {
        "schema_version": C.REVIEW_SCHEMA_VERSION,
        "source": C.SOURCE_TAG,
        "phase": C.REVIEW_PHASE,
        "run_id": run_id,
        "batch_id": batch_id,
        "created_by": C.REVIEW_CREATED_BY,
        "records": records,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# ─── validate_export ──────────────────────────────────────────────────────────


def test_validate_export_valid_returns_no_errors(tmp_path: Path) -> None:
    p = tmp_path / "export.json"
    _valid_export("rvrun_ie1", "0001", ["rv_aaa1111111111111"], p)
    errors = validate_export(p, "rvrun_ie1", "0001")
    assert errors == []


def test_validate_export_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("not json {{{ }", encoding="utf-8")
    errors = validate_export(p, "rvrun_ie2", "0001")
    assert any("invalid JSON" in e for e in errors)


def test_validate_export_run_id_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "export.json"
    _valid_export("rvrun_ie3", "0001", ["rv_aaa1111111111111"], p)
    errors = validate_export(p, "rvrun_WRONG", "0001")
    assert any("run_id mismatch" in e for e in errors)


def test_validate_export_batch_id_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "export.json"
    _valid_export("rvrun_ie4", "0001", ["rv_aaa1111111111111"], p)
    errors = validate_export(p, "rvrun_ie4", "0002")
    assert any("batch_id mismatch" in e for e in errors)


def test_validate_export_rejects_blank_template(tmp_path: Path) -> None:
    p = tmp_path / "export.json"
    _blank_template("rvrun_ie5", "0001", ["rv_aaa1111111111111"], p)
    errors = validate_export(p, "rvrun_ie5", "0001")
    assert any("blank template" in e for e in errors)


def test_validate_export_catches_record_errors(tmp_path: Path) -> None:
    """A record with an invalid decision_type should be caught."""
    p = tmp_path / "export.json"
    data = {
        "schema_version": C.REVIEW_SCHEMA_VERSION,
        "source": C.SOURCE_TAG,
        "phase": C.REVIEW_PHASE,
        "run_id": "rvrun_ie6",
        "batch_id": "0001",
        "created_by": C.REVIEW_CREATED_BY,
        "records": [{
            "review_id": "rv_aaa1111111111111",
            "decision_type": "INVALID_TYPE",
            "phase_c_category": "unknown_needs_review",
            "final_category": "fish",
            "human_confidence": 4,
            "refinement": None,
            "notes": None,
            "reviewed_at": "2026-04-27T20:00:00+03:00",
        }],
    }
    p.write_text(json.dumps(data), encoding="utf-8")
    errors = validate_export(p, "rvrun_ie6", "0001")
    assert len(errors) > 0


# ─── ingest_export ────────────────────────────────────────────────────────────


def test_ingest_copies_valid_json(tmp_path: Path) -> None:
    """ingest_export copies a valid export into review_dir."""
    src = tmp_path / "downloads" / "export.json"
    src.parent.mkdir()
    review_dir = tmp_path / "review"
    review_dir.mkdir()

    _valid_export("rvrun_in1", "0001", ["rv_aaa1111111111111"], src)
    dest = ingest_export("rvrun_in1", "0001", src, review_dir)

    assert dest.exists()
    with dest.open() as f:
        data = json.load(f)
    assert data["run_id"] == "rvrun_in1"
    assert data["batch_id"] == "0001"
    assert len(data["records"]) == 1


def test_ingest_refuses_invalid_json(tmp_path: Path) -> None:
    """ingest_export raises ValueError on schema errors."""
    src = tmp_path / "export.json"
    src.write_text("not json", encoding="utf-8")
    review_dir = tmp_path / "review"
    review_dir.mkdir()

    with pytest.raises(ValueError, match="failed validation"):
        ingest_export("rvrun_in2", "0001", src, review_dir)


def test_ingest_refuses_mismatched_run_id(tmp_path: Path) -> None:
    """ingest_export raises ValueError if run_id in file doesn't match argument."""
    src = tmp_path / "export.json"
    _valid_export("rvrun_in3_WRONG", "0001", ["rv_aaa1111111111111"], src)
    review_dir = tmp_path / "review"
    review_dir.mkdir()

    with pytest.raises(ValueError, match="failed validation"):
        ingest_export("rvrun_in3", "0001", src, review_dir)


def test_ingest_refuses_mismatched_batch_id(tmp_path: Path) -> None:
    """ingest_export raises ValueError if batch_id in file doesn't match argument."""
    src = tmp_path / "export.json"
    _valid_export("rvrun_in4", "0001", ["rv_aaa1111111111111"], src)
    review_dir = tmp_path / "review"
    review_dir.mkdir()

    with pytest.raises(ValueError, match="failed validation"):
        ingest_export("rvrun_in4", "0002", src, review_dir)


def test_ingest_refuses_overwrite_without_force(tmp_path: Path) -> None:
    """ingest_export refuses to overwrite a filled reviewed batch without --force."""
    src = tmp_path / "export.json"
    _valid_export("rvrun_in5", "0001", ["rv_aaa1111111111111"], src)
    review_dir = tmp_path / "review"
    review_dir.mkdir()

    # Place a filled file in review_dir already
    dest = review_dir / "filter_decisions_rvrun_in5_0001.json"
    _valid_export("rvrun_in5", "0001", ["rv_aaa1111111111111"], dest)

    with pytest.raises(ValueError, match="already.*filled|--force"):
        ingest_export("rvrun_in5", "0001", src, review_dir, force=False)


def test_ingest_force_overwrites_filled_batch(tmp_path: Path) -> None:
    """ingest_export with force=True overwrites an existing filled batch."""
    src = tmp_path / "export.json"
    ids_new = ["rv_aaa1111111111111"]
    _valid_export("rvrun_in6", "0001", ids_new, src)
    review_dir = tmp_path / "review"
    review_dir.mkdir()

    dest = review_dir / "filter_decisions_rvrun_in6_0001.json"
    _valid_export("rvrun_in6", "0001", ["rv_aaa1111111111111"], dest)

    result = ingest_export("rvrun_in6", "0001", src, review_dir, force=True)
    assert result.exists()


def test_ingest_force_backs_up_filled_batch(tmp_path: Path) -> None:
    """ingest_export --force creates a timestamped backup of the existing filled batch."""
    src = tmp_path / "export.json"
    _valid_export("rvrun_in9", "0001", ["rv_aaa1111111111111"], src)
    review_dir = tmp_path / "review"
    review_dir.mkdir()

    dest = review_dir / "filter_decisions_rvrun_in9_0001.json"
    _valid_export("rvrun_in9", "0001", ["rv_aaa1111111111111"], dest)

    ingest_export("rvrun_in9", "0001", src, review_dir, force=True)

    backups = list(review_dir.glob("filter_decisions_rvrun_in9_0001.reviewed_backup_*.json"))
    assert len(backups) == 1, "A timestamped backup of the reviewed batch should have been created"


def test_ingest_backs_up_blank_template(tmp_path: Path) -> None:
    """ingest_export creates a .blank_backup.json before overwriting a blank template."""
    src = tmp_path / "export.json"
    _valid_export("rvrun_in7", "0001", ["rv_aaa1111111111111"], src)
    review_dir = tmp_path / "review"
    review_dir.mkdir()

    dest = review_dir / "filter_decisions_rvrun_in7_0001.json"
    _blank_template("rvrun_in7", "0001", ["rv_aaa1111111111111"], dest)

    ingest_export("rvrun_in7", "0001", src, review_dir)

    backup = dest.with_suffix(".blank_backup.json")
    assert backup.exists(), "Blank template backup should have been created"


def test_ingest_missing_source_raises(tmp_path: Path) -> None:
    """ingest_export raises FileNotFoundError if source path does not exist."""
    src = tmp_path / "nonexistent.json"
    review_dir = tmp_path / "review"
    review_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        ingest_export("rvrun_in8", "0001", src, review_dir)


# ─── Privacy: source path safety ─────────────────────────────────────────────


def test_assert_no_private_paths_safe(tmp_path: Path) -> None:
    """Normal Downloads path should pass."""
    _assert_no_private_paths(Path("/Users/user/Downloads/filter_decisions_rvrun_x_0001.json"))


def test_assert_no_private_paths_chatexport_rejected() -> None:
    """ChatExport_ path should be rejected."""
    with pytest.raises(ValueError, match="SAFETY"):
        _assert_no_private_paths(Path("/Users/user/Downloads/ChatExport_2026-04-24/export.json"))


# ─── AL phase acceptance ──────────────────────────────────────────────────────


def test_validate_export_al_negative_review_phase_accepted(tmp_path: Path) -> None:
    """AL_NEGATIVE_REVIEW phase must be accepted (used by select_telegram_negative_review_candidates.py)."""
    p = tmp_path / "export.json"
    records = [{
        "review_id": "rv_aaa1111111111111",
        "decision_type": "RELABEL",
        "phase_c_category": "unknown_needs_review",
        "final_category": "no_fish",
        "human_confidence": 4,
        "refinement": {"species": None, "life_stage": "unknown"},
        "notes": None,
        "reviewed_at": "2026-05-05T20:00:00.000Z",
    }]
    data = {
        "schema_version": C.REVIEW_SCHEMA_VERSION,
        "source": C.SOURCE_TAG,
        "phase": C.REVIEW_PHASE_AL,
        "run_id": "alvrun_test",
        "batch_id": "0001",
        "created_by": C.REVIEW_CREATED_BY,
        "records": records,
    }
    p.write_text(__import__("json").dumps(data), encoding="utf-8")
    errors = validate_export(p, "alvrun_test", "0001")
    assert errors == [], f"AL phase export should be valid; got: {errors}"
