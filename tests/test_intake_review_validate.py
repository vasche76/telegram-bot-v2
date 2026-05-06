"""
test_intake_review_validate.py — Tests for intake_review_validate.py (U4 Phase D).

Full-mode covers:
- Missing decision fails (coverage check)
- Duplicate decision fails (uniqueness check)
- Unknown review ID fails
- Valid complete set passes
- Missing decision count in report
- Schema error in decision record caught
- KEEP unknown_needs_review caught in validate
- UNSURE with wrong final_category caught
- REMOVE with fish caught
- RELABEL without category change caught

Partial-mode covers:
- Partial validation accepts filled batches and reports remaining unreviewed
- Blank templates are not counted as reviewed
- Invalid filled batch is reported as reviewed_invalid
- Missing decision_type in otherwise-filled batch fails
- Missing batch file is reported as missing_file
- Full validation behavior is not weakened by partial mode existence
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import intake_constants as C
from intake_review_validate import (
    validate,
    collect_expected_review_ids,
    validate_partial,
    is_blank_template,
    BATCH_STATUS_REVIEWED_VALID,
    BATCH_STATUS_REVIEWED_INVALID,
    BATCH_STATUS_BLANK_UNREVIEWED,
    BATCH_STATUS_MISSING_FILE,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _manifest(run_id: str, review_ids: list[list[str]]) -> dict:
    """review_ids: list of batches, each batch a list of ids."""
    batches = []
    for i, batch_ids in enumerate(review_ids):
        batches.append({
            "batch_id": f"{i + 1:04d}",
            "record_count": len(batch_ids),
            "html_file": f"filter_review_batch_{run_id}_{i + 1:04d}.html",
            "decision_file": f"filter_decisions_{run_id}_{i + 1:04d}.json",
            "review_ids": batch_ids,
        })
    all_ids = [rid for batch in review_ids for rid in batch]
    return {
        "schema_version": C.REVIEW_SCHEMA_VERSION,
        "run_id": run_id,
        "source": C.SOURCE_TAG,
        "phase": C.REVIEW_PHASE,
        "total_records": len(all_ids),
        "batch_count": len(batches),
        "batches": batches,
    }


def _rec(
    review_id: str,
    decision_type: str = "RELABEL",
    phase_c_category: str = "unknown_needs_review",
    final_category: str = "fish",
    human_confidence: int = 4,
) -> dict:
    return {
        "review_id": review_id,
        "decision_type": decision_type,
        "phase_c_category": phase_c_category,
        "final_category": final_category,
        "human_confidence": human_confidence,
        "refinement": {"species": None, "life_stage": "unknown"},
        "notes": None,
        "reviewed_at": "2026-04-27T20:00:00+03:00",
    }


def _decision_file(run_id: str, batch_id: str, records: list[dict], path: Path) -> tuple[Path, dict]:
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
    return path, data


# ─── Valid complete set passes ────────────────────────────────────────────────


def test_valid_complete_set_passes(tmp_path: Path) -> None:
    ids = ["rv_aaa1111111111111", "rv_bbb2222222222222"]
    manifest = _manifest("rvrun_t", [ids])
    p, data = _decision_file("rvrun_t", "0001", [_rec(r) for r in ids], tmp_path / "d0001.json")
    report = validate(manifest, [(p, data)])
    assert report["passed"] is True
    assert report["missing_count"] == 0
    assert report["unknown_count"] == 0
    assert report["invalid_decisions"] == 0


# ─── Missing decision fails ───────────────────────────────────────────────────


def test_missing_decision_fails(tmp_path: Path) -> None:
    ids = ["rv_aaa1111111111111", "rv_bbb2222222222222"]
    manifest = _manifest("rvrun_t2", [ids])
    # Only submit decision for first ID
    p, data = _decision_file("rvrun_t2", "0001", [_rec(ids[0])], tmp_path / "d0001.json")
    report = validate(manifest, [(p, data)])
    assert report["passed"] is False
    assert report["missing_count"] == 1
    assert any("Missing decisions" in e for e in report["errors"])


# ─── Duplicate decision fails ─────────────────────────────────────────────────


def test_duplicate_decision_fails(tmp_path: Path) -> None:
    ids = ["rv_aaa1111111111111"]
    manifest = _manifest("rvrun_t3", [ids])
    rid = ids[0]
    records = [_rec(rid), _rec(rid)]  # same ID twice
    p, data = _decision_file("rvrun_t3", "0001", records, tmp_path / "d0001.json")
    report = validate(manifest, [(p, data)])
    assert report["passed"] is False
    assert any("duplicate" in e.lower() for e in report["errors"])


# ─── Unknown review ID fails ──────────────────────────────────────────────────


def test_unknown_review_id_fails(tmp_path: Path) -> None:
    ids = ["rv_aaa1111111111111"]
    manifest = _manifest("rvrun_t4", [ids])
    unknown = "rv_zzzzzzzzzzzzzzzz"
    records = [_rec(ids[0]), _rec(unknown)]
    p, data = _decision_file("rvrun_t4", "0001", records, tmp_path / "d0001.json")
    report = validate(manifest, [(p, data)])
    assert report["passed"] is False
    assert any("Unknown review_id" in e or "unknown_id" in e.lower() for e in report["errors"])


# ─── Schema errors caught ─────────────────────────────────────────────────────


def test_keep_unknown_needs_review_caught_in_validate(tmp_path: Path) -> None:
    ids = ["rv_aaa1111111111111"]
    manifest = _manifest("rvrun_t5", [ids])
    rec = _rec(ids[0], decision_type="KEEP",
               phase_c_category="unknown_needs_review",
               final_category="unknown_needs_review")
    p, data = _decision_file("rvrun_t5", "0001", [rec], tmp_path / "d0001.json")
    report = validate(manifest, [(p, data)])
    assert report["passed"] is False
    assert report["invalid_decisions"] >= 1


def test_unsure_wrong_final_category_caught(tmp_path: Path) -> None:
    ids = ["rv_aaa1111111111111"]
    manifest = _manifest("rvrun_t6", [ids])
    rec = _rec(ids[0], decision_type="UNSURE", final_category="fish", human_confidence=1)
    p, data = _decision_file("rvrun_t6", "0001", [rec], tmp_path / "d0001.json")
    report = validate(manifest, [(p, data)])
    assert report["passed"] is False


def test_remove_with_fish_caught(tmp_path: Path) -> None:
    ids = ["rv_aaa1111111111111"]
    manifest = _manifest("rvrun_t7", [ids])
    rec = _rec(ids[0], decision_type="REMOVE", final_category="fish")
    p, data = _decision_file("rvrun_t7", "0001", [rec], tmp_path / "d0001.json")
    report = validate(manifest, [(p, data)])
    assert report["passed"] is False


def test_relabel_no_category_change_caught(tmp_path: Path) -> None:
    ids = ["rv_aaa1111111111111"]
    manifest = _manifest("rvrun_t8", [ids])
    rec = _rec(ids[0], decision_type="RELABEL",
               phase_c_category="fish_part", final_category="fish_part")
    p, data = _decision_file("rvrun_t8", "0001", [rec], tmp_path / "d0001.json")
    report = validate(manifest, [(p, data)])
    assert report["passed"] is False


# ─── Multi-batch validation ───────────────────────────────────────────────────


def test_multi_batch_valid(tmp_path: Path) -> None:
    ids_b1 = ["rv_aaa1111111111111", "rv_bbb2222222222222"]
    ids_b2 = ["rv_ccc3333333333333", "rv_ddd4444444444444"]
    manifest = _manifest("rvrun_mb", [ids_b1, ids_b2])
    p1, d1 = _decision_file("rvrun_mb", "0001", [_rec(r) for r in ids_b1], tmp_path / "d1.json")
    p2, d2 = _decision_file("rvrun_mb", "0002", [_rec(r) for r in ids_b2], tmp_path / "d2.json")
    report = validate(manifest, [(p1, d1), (p2, d2)])
    assert report["passed"] is True
    assert report["reviewed_count"] == 4


def test_duplicate_review_id_across_batches_fails(tmp_path: Path) -> None:
    ids_b1 = ["rv_aaa1111111111111"]
    ids_b2 = ["rv_bbb2222222222222"]
    manifest = _manifest("rvrun_cx", [ids_b1, ids_b2])
    shared_id = "rv_aaa1111111111111"
    p1, d1 = _decision_file("rvrun_cx", "0001", [_rec(shared_id)], tmp_path / "d1.json")
    # Second batch uses same ID (duplicate across files)
    p2, d2 = _decision_file("rvrun_cx", "0002", [_rec(shared_id)], tmp_path / "d2.json")
    report = validate(manifest, [(p1, d1), (p2, d2)])
    assert report["passed"] is False
    assert any("duplicate" in e.lower() for e in report["errors"])


# ─── collect_expected_review_ids ─────────────────────────────────────────────


def test_collect_expected_review_ids() -> None:
    m = _manifest("rvrun_x", [["rv_a0000000000000a0", "rv_b0000000000000b0"], ["rv_c0000000000000c0"]])
    ids = collect_expected_review_ids(m)
    assert ids == {"rv_a0000000000000a0", "rv_b0000000000000b0", "rv_c0000000000000c0"}


# ─── is_blank_template ────────────────────────────────────────────────────────


def test_is_blank_template_true_all_null() -> None:
    data = {"records": [
        {"review_id": "rv_a", "decision_type": None},
        {"review_id": "rv_b", "decision_type": None},
    ]}
    assert is_blank_template(data) is True


def test_is_blank_template_false_when_filled() -> None:
    data = {"records": [
        {"review_id": "rv_a", "decision_type": "RELABEL"},
        {"review_id": "rv_b", "decision_type": None},
    ]}
    assert is_blank_template(data) is False


def test_is_blank_template_empty_records_is_blank() -> None:
    data = {"records": []}
    assert is_blank_template(data) is True


def test_is_blank_template_records_null_is_not_blank() -> None:
    """records=null (JSON null deserialized to None) must NOT be treated as blank template.
    Such a file is corrupted/invalid and should be routed to validate() as reviewed_invalid."""
    data = {"records": None}
    assert is_blank_template(data) is False


# ─── Partial validation helpers ───────────────────────────────────────────────


def _blank_template_file(run_id: str, batch_id: str, review_ids: list[str], path: Path) -> Path:
    """Write a blank decision template (all decision_type=None) to path."""
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


def _manifest_with_decision_files(run_id: str, batches_spec: list[dict]) -> dict:
    """
    batches_spec: list of dicts with keys: batch_id, review_ids
    Generates manifest with decision_file field per batch.
    """
    batches = []
    all_ids = []
    for spec in batches_spec:
        bid = spec["batch_id"]
        rids = spec["review_ids"]
        all_ids.extend(rids)
        batches.append({
            "batch_id": bid,
            "record_count": len(rids),
            "html_file": f"filter_review_batch_{run_id}_{bid}.html",
            "decision_file": f"filter_decisions_{run_id}_{bid}.json",
            "review_ids": rids,
        })
    return {
        "schema_version": C.REVIEW_SCHEMA_VERSION,
        "run_id": run_id,
        "source": C.SOURCE_TAG,
        "phase": C.REVIEW_PHASE,
        "total_records": len(all_ids),
        "batch_count": len(batches),
        "batches": batches,
    }


# ─── Partial validation: core scenarios ──────────────────────────────────────


def test_partial_accepts_filled_batches_reports_unreviewed(tmp_path: Path) -> None:
    """validate_partial: filled batch → reviewed_valid, blank batch → blank_unreviewed."""
    run_id = "rvrun_pp1"
    ids_b1 = ["rv_aaa1111111111111"]
    ids_b2 = ["rv_bbb2222222222222"]
    manifest = _manifest_with_decision_files(run_id, [
        {"batch_id": "0001", "review_ids": ids_b1},
        {"batch_id": "0002", "review_ids": ids_b2},
    ])
    # Write filled batch 0001
    p1 = tmp_path / f"filter_decisions_{run_id}_0001.json"
    _decision_file(run_id, "0001", [_rec(ids_b1[0])], p1)
    # Write blank template batch 0002
    p2 = tmp_path / f"filter_decisions_{run_id}_0002.json"
    _blank_template_file(run_id, "0002", ids_b2, p2)

    report = validate_partial(manifest, tmp_path, run_id)

    assert report["passed"] is True
    assert report["reviewed_valid"] == 1
    assert report["blank_unreviewed"] == 1
    assert report["reviewed_invalid"] == 0
    assert report["missing_file"] == 0
    assert report["reviewed_records"] == 1
    assert report["unreviewed_records"] == 1
    assert "0001" in report["reviewed_valid_batches"]
    assert "0002" in report["next_unreviewed_batches"]


def test_partial_blank_template_not_counted_as_reviewed(tmp_path: Path) -> None:
    """Blank templates must never appear in reviewed_valid_batches."""
    run_id = "rvrun_pp2"
    ids = ["rv_aaa1111111111111"]
    manifest = _manifest_with_decision_files(run_id, [
        {"batch_id": "0001", "review_ids": ids},
    ])
    p = tmp_path / f"filter_decisions_{run_id}_0001.json"
    _blank_template_file(run_id, "0001", ids, p)

    report = validate_partial(manifest, tmp_path, run_id)

    assert report["reviewed_valid"] == 0
    assert report["blank_unreviewed"] == 1
    assert "0001" not in report.get("reviewed_valid_batches", [])


def test_partial_invalid_filled_batch_fails(tmp_path: Path) -> None:
    """validate_partial: a filled batch with schema errors is reported as reviewed_invalid."""
    run_id = "rvrun_pp3"
    ids = ["rv_aaa1111111111111"]
    manifest = _manifest_with_decision_files(run_id, [
        {"batch_id": "0001", "review_ids": ids},
    ])
    # RELABEL with no category change (invalid)
    rec = _rec(ids[0], decision_type="RELABEL",
               phase_c_category="fish_part", final_category="fish_part")
    p = tmp_path / f"filter_decisions_{run_id}_0001.json"
    _decision_file(run_id, "0001", [rec], p)

    report = validate_partial(manifest, tmp_path, run_id)

    assert report["passed"] is False
    assert report["reviewed_invalid"] == 1
    assert report["reviewed_valid"] == 0
    assert "0001" in report["invalid_batches"]


def test_partial_missing_decision_type_in_filled_batch_fails(tmp_path: Path) -> None:
    """A record with null decision_type in an otherwise non-blank file fails validation."""
    run_id = "rvrun_pp4"
    ids = ["rv_aaa1111111111111", "rv_bbb2222222222222"]
    manifest = _manifest_with_decision_files(run_id, [
        {"batch_id": "0001", "review_ids": ids},
    ])
    records = [
        _rec(ids[0]),  # valid
        {   # null decision_type in an otherwise non-blank file
            "review_id": ids[1],
            "decision_type": None,
            "phase_c_category": "unknown_needs_review",
            "final_category": None,
            "human_confidence": None,
            "refinement": None,
            "notes": None,
            "reviewed_at": None,
        },
    ]
    p = tmp_path / f"filter_decisions_{run_id}_0001.json"
    _decision_file(run_id, "0001", records, p)

    # This file is NOT a blank template (first record is filled)
    # so validate_partial must validate it and catch the invalid record
    report = validate_partial(manifest, tmp_path, run_id)

    assert report["reviewed_invalid"] == 1


def test_partial_missing_file_reported(tmp_path: Path) -> None:
    """Expected decision file that doesn't exist is reported as missing_file."""
    run_id = "rvrun_pp5"
    ids = ["rv_aaa1111111111111"]
    manifest = _manifest_with_decision_files(run_id, [
        {"batch_id": "0001", "review_ids": ids},
    ])
    # No file written — expect missing_file

    report = validate_partial(manifest, tmp_path, run_id)

    assert report["missing_file"] == 1
    assert report["passed"] is False


def test_partial_does_not_weaken_full_validate(tmp_path: Path) -> None:
    """Full validate still requires complete coverage — partial mode doesn't affect it."""
    ids = ["rv_aaa1111111111111", "rv_bbb2222222222222"]
    manifest = _manifest("rvrun_fw", [ids])
    # Only provide decision for first ID
    p, data = _decision_file("rvrun_fw", "0001", [_rec(ids[0])], tmp_path / "d0001.json")

    report = validate(manifest, [(p, data)])

    assert report["passed"] is False
    assert report["missing_count"] == 1
