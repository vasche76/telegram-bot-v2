"""
test_intake_review_aggregate.py — Tests for intake_review_aggregate.py (U4 Phase D).

Full-mode covers:
- Valid aggregate produces correct counts
- Tracked summary contains counts only (no review IDs, filenames, etc.)
- Aggregation fails when validation_report.passed=False and require_validation_pass=True
- Aggregation proceeds with require_validation_pass=False even on failed validation
- training_eligible_count correct (confidence >= 4, not UNSURE)
- UNSURE count correct
- Summary schema version and fields correct
- Integration test on synthetic mini dataset

Partial-mode covers:
- aggregate_partial aggregates only reviewed_valid batches
- unreviewed records never counted as training-ready
- partial summary contains no private data
- training eligibility only from reviewed records with confidence >= 4
- high_confidence_fish vs non_fish buckets correct
- reviewed_invalid blocks aggregation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import intake_constants as C
from intake_review_aggregate import aggregate_decisions, _assert_summary_privacy, aggregate_partial


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _manifest(run_id: str, review_ids: list[str]) -> dict:
    return {
        "schema_version": C.REVIEW_SCHEMA_VERSION,
        "run_id": run_id,
        "source": C.SOURCE_TAG,
        "phase": C.REVIEW_PHASE,
        "total_records": len(review_ids),
        "batch_count": 1,
        "batches": [{"review_ids": review_ids}],
    }


def _rec(
    review_id: str = "rv_aaa1111111111111",
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


def _decision_file(run_id: str, records: list[dict], path: Path) -> tuple[Path, dict]:
    data = {
        "schema_version": C.REVIEW_SCHEMA_VERSION,
        "source": C.SOURCE_TAG,
        "phase": C.REVIEW_PHASE,
        "run_id": run_id,
        "batch_id": "0001",
        "created_by": C.REVIEW_CREATED_BY,
        "records": records,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path, data


def _validation_pass(run_id: str, count: int) -> dict:
    return {
        "run_id": run_id,
        "passed": True,
        "expected_count": count,
        "reviewed_count": count,
        "missing_count": 0,
        "unknown_count": 0,
        "valid_decisions": count,
        "invalid_decisions": 0,
        "error_count": 0,
        "errors": [],
        "warnings": [],
    }


def _validation_fail(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "passed": False,
        "expected_count": 10,
        "reviewed_count": 5,
        "missing_count": 5,
        "unknown_count": 0,
        "valid_decisions": 5,
        "invalid_decisions": 0,
        "error_count": 1,
        "errors": ["Missing decisions for 5 review_id(s)"],
        "warnings": [],
    }


# ─── Happy path ───────────────────────────────────────────────────────────────


def test_aggregate_produces_decision_type_counts(tmp_path: Path) -> None:
    ids = ["rv_aaa1111111111111", "rv_bbb2222222222222", "rv_ccc3333333333333"]
    manifest = _manifest("rvrun_t", ids)
    records = [
        _rec(ids[0], decision_type="RELABEL", final_category="fish", human_confidence=4),
        _rec(ids[1], decision_type="REMOVE", final_category="no_fish", human_confidence=3),
        _rec(ids[2], decision_type="UNSURE", final_category="unsure", human_confidence=2),
    ]
    p, data = _decision_file("rvrun_t", records, tmp_path / "d.json")
    agg, summary = aggregate_decisions(manifest, [(p, data)], _validation_pass("rvrun_t", 3), True)

    assert agg["summary"]["decision_type_counts"]["RELABEL"] == 1
    assert agg["summary"]["decision_type_counts"]["REMOVE"] == 1
    assert agg["summary"]["decision_type_counts"]["UNSURE"] == 1


def test_aggregate_training_eligible_count(tmp_path: Path) -> None:
    ids = ["rv_aaa1111111111111", "rv_bbb2222222222222", "rv_ccc3333333333333"]
    manifest = _manifest("rvrun_te", ids)
    records = [
        _rec(ids[0], human_confidence=5),   # eligible (RELABEL, conf>=4)
        _rec(ids[1], human_confidence=4),   # eligible
        _rec(ids[2], human_confidence=3),   # not eligible (conf < 4)
    ]
    p, data = _decision_file("rvrun_te", records, tmp_path / "d.json")
    agg, summary = aggregate_decisions(manifest, [(p, data)], _validation_pass("rvrun_te", 3), True)

    assert agg["summary"]["training_eligible_count"] == 2


def test_aggregate_unsure_count(tmp_path: Path) -> None:
    ids = ["rv_aaa1111111111111", "rv_bbb2222222222222"]
    manifest = _manifest("rvrun_uc", ids)
    records = [
        _rec(ids[0], decision_type="UNSURE", final_category="unsure", human_confidence=1),
        _rec(ids[1]),  # RELABEL
    ]
    p, data = _decision_file("rvrun_uc", records, tmp_path / "d.json")
    agg, summary = aggregate_decisions(manifest, [(p, data)], _validation_pass("rvrun_uc", 2), True)

    assert agg["summary"]["unsure_count"] == 1


def test_aggregate_final_category_counts(tmp_path: Path) -> None:
    ids = ["rv_aaa1111111111111", "rv_bbb2222222222222"]
    manifest = _manifest("rvrun_fc", ids)
    records = [
        _rec(ids[0], final_category="fish"),
        _rec(ids[1], final_category="no_fish", decision_type="REMOVE"),
    ]
    p, data = _decision_file("rvrun_fc", records, tmp_path / "d.json")
    agg, summary = aggregate_decisions(manifest, [(p, data)], _validation_pass("rvrun_fc", 2), True)

    assert summary["final_category_counts"]["fish"] == 1
    assert summary["final_category_counts"]["no_fish"] == 1


# ─── Validation gate ──────────────────────────────────────────────────────────


def test_aggregate_fails_when_validation_failed(tmp_path: Path) -> None:
    manifest = _manifest("rvrun_gf", ["rv_aaa1111111111111"])
    p, data = _decision_file("rvrun_gf", [_rec()], tmp_path / "d.json")
    with pytest.raises(ValueError, match="Validation FAILED"):
        aggregate_decisions(manifest, [(p, data)], _validation_fail("rvrun_gf"), True)


def test_aggregate_proceeds_with_validation_bypass(tmp_path: Path) -> None:
    manifest = _manifest("rvrun_gb", ["rv_aaa1111111111111"])
    p, data = _decision_file("rvrun_gb", [_rec()], tmp_path / "d.json")
    # Should not raise even though validation failed
    agg, summary = aggregate_decisions(manifest, [(p, data)], _validation_fail("rvrun_gb"), False)
    assert agg["summary"]["total_reviewed"] == 1


# ─── Tracked summary privacy ──────────────────────────────────────────────────


def test_summary_contains_no_review_ids(tmp_path: Path) -> None:
    ids = ["rv_aaa1111111111111"]
    manifest = _manifest("rvrun_sp", ids)
    p, data = _decision_file("rvrun_sp", [_rec(ids[0])], tmp_path / "d.json")
    _agg, summary = aggregate_decisions(manifest, [(p, data)], _validation_pass("rvrun_sp", 1), True)

    summary_json = json.dumps(summary)
    # review_ids must not appear
    for rid in ids:
        assert rid not in summary_json


def test_summary_contains_no_filenames(tmp_path: Path) -> None:
    ids = ["rv_aaa1111111111111"]
    manifest = _manifest("rvrun_sf", ids)
    p, data = _decision_file("rvrun_sf", [_rec(ids[0])], tmp_path / "d.json")
    _agg, summary = aggregate_decisions(manifest, [(p, data)], _validation_pass("rvrun_sf", 1), True)

    summary_json = json.dumps(summary)
    assert "photos/" not in summary_json
    assert "photo_" not in summary_json


def test_summary_schema_version_and_phase(tmp_path: Path) -> None:
    ids = ["rv_aaa1111111111111"]
    manifest = _manifest("rvrun_sv", ids)
    p, data = _decision_file("rvrun_sv", [_rec(ids[0])], tmp_path / "d.json")
    _agg, summary = aggregate_decisions(manifest, [(p, data)], _validation_pass("rvrun_sv", 1), True)

    assert summary["schema_version"] == C.REVIEW_SUMMARY_SCHEMA_VERSION
    assert summary["phase"] == C.REVIEW_PHASE
    assert summary["source"] == C.SOURCE_TAG


def test_summary_privacy_status_flags(tmp_path: Path) -> None:
    ids = ["rv_aaa1111111111111"]
    manifest = _manifest("rvrun_ps", ids)
    p, data = _decision_file("rvrun_ps", [_rec(ids[0])], tmp_path / "d.json")
    _agg, summary = aggregate_decisions(manifest, [(p, data)], _validation_pass("rvrun_ps", 1), True)

    ps = summary["privacy_status"]
    assert ps["contains_review_ids"] is False
    assert ps["contains_filenames"] is False
    assert ps["contains_paths"] is False
    assert ps["contains_captions"] is False
    assert ps["contains_sender_metadata"] is False


# ─── _assert_summary_privacy ─────────────────────────────────────────────────


def test_assert_summary_privacy_passes() -> None:
    summary = {"total": 42, "fish": 10, "no_fish": 32}
    _assert_summary_privacy(summary)  # should not raise


def test_assert_summary_privacy_fails_on_review_id() -> None:
    # review_id value (rv_ + 16 hex chars) must be detected
    summary = {"total": 1, "id": "rv_abcdef123456789a"}  # rv_ + 16 hex chars
    with pytest.raises(ValueError, match="PRIVACY VIOLATION"):
        _assert_summary_privacy(summary)


def test_assert_summary_privacy_fails_on_filename() -> None:
    summary = {"path": "photos/photo_1@01-01-2020.jpg"}
    with pytest.raises(ValueError, match="PRIVACY VIOLATION"):
        _assert_summary_privacy(summary)


# ─── Partial aggregation helpers ─────────────────────────────────────────────


def _manifest_partial(run_id: str, batches_spec: list[dict]) -> dict:
    """Build a manifest with explicit decision_file per batch."""
    batches = []
    all_ids: list[str] = []
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


def _write_filled(run_id: str, batch_id: str, records: list[dict], review_dir: Path) -> None:
    data = {
        "schema_version": C.REVIEW_SCHEMA_VERSION,
        "source": C.SOURCE_TAG,
        "phase": C.REVIEW_PHASE,
        "run_id": run_id,
        "batch_id": batch_id,
        "created_by": C.REVIEW_CREATED_BY,
        "records": records,
    }
    p = review_dir / f"filter_decisions_{run_id}_{batch_id}.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_blank(run_id: str, batch_id: str, review_ids: list[str], review_dir: Path) -> None:
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
    p = review_dir / f"filter_decisions_{run_id}_{batch_id}.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ─── Partial aggregation: core scenarios ─────────────────────────────────────


def test_partial_aggregates_only_reviewed_batches(tmp_path: Path) -> None:
    """aggregate_partial counts only records from reviewed_valid batches."""
    run_id = "rvrun_pa1"
    ids_b1 = ["rv_aaa1111111111111"]
    ids_b2 = ["rv_bbb2222222222222"]
    manifest = _manifest_partial(run_id, [
        {"batch_id": "0001", "review_ids": ids_b1},
        {"batch_id": "0002", "review_ids": ids_b2},
    ])
    _write_filled(run_id, "0001", [_rec(ids_b1[0], final_category="fish", human_confidence=4)], tmp_path)
    _write_blank(run_id, "0002", ids_b2, tmp_path)

    local_agg, summary = aggregate_partial(manifest, tmp_path, run_id)

    assert local_agg["summary"]["total_reviewed"] == 1
    assert summary["progress"]["reviewed_valid_batches"] == 1
    assert summary["progress"]["blank_unreviewed_batches"] == 1
    assert summary["progress"]["reviewed_records"] == 1
    assert summary["progress"]["unreviewed_records"] == 1


def test_partial_unreviewed_never_training_ready(tmp_path: Path) -> None:
    """unreviewed_not_eligible equals the count of all unreviewed records."""
    run_id = "rvrun_pa2"
    ids_b1 = ["rv_aaa1111111111111"]
    ids_b2 = ["rv_bbb2222222222222", "rv_ccc3333333333333"]
    manifest = _manifest_partial(run_id, [
        {"batch_id": "0001", "review_ids": ids_b1},
        {"batch_id": "0002", "review_ids": ids_b2},
    ])
    _write_filled(run_id, "0001", [_rec(ids_b1[0], final_category="fish", human_confidence=5)], tmp_path)
    _write_blank(run_id, "0002", ids_b2, tmp_path)

    _local, summary = aggregate_partial(manifest, tmp_path, run_id)

    assert summary["training_eligibility"]["unreviewed_not_eligible"] == 2
    # training_eligible_count must NOT include unreviewed
    assert summary["training_eligibility"]["training_eligible_count"] == 1


def test_partial_training_eligibility_confidence_threshold(tmp_path: Path) -> None:
    """Only reviewed records with confidence >= 4 and non-excluded category are eligible."""
    run_id = "rvrun_pa3"
    ids = ["rv_aaa1111111111111", "rv_bbb2222222222222", "rv_ccc3333333333333"]
    manifest = _manifest_partial(run_id, [{"batch_id": "0001", "review_ids": ids}])
    records = [
        _rec(ids[0], final_category="fish", human_confidence=5),   # eligible
        _rec(ids[1], final_category="fish", human_confidence=3),   # low_conf
        _rec(ids[2], decision_type="UNSURE", final_category="unsure", human_confidence=2),  # unsure
    ]
    _write_filled(run_id, "0001", records, tmp_path)

    _local, summary = aggregate_partial(manifest, tmp_path, run_id)
    tb = summary["training_eligibility"]

    assert tb["training_eligible_count"] == 1
    assert tb["reviewed_high_confidence_fish"] == 1
    assert tb["reviewed_low_confidence"] == 2  # conf<4: both conf=3 and conf=2 UNSURE


def test_partial_fish_vs_non_fish_buckets(tmp_path: Path) -> None:
    """reviewed_high_confidence_fish vs non_fish_or_excluded buckets are correctly split."""
    run_id = "rvrun_pa4"
    ids = ["rv_aaa1111111111111", "rv_bbb2222222222222", "rv_ccc3333333333333",
           "rv_ddd4444444444444"]
    manifest = _manifest_partial(run_id, [{"batch_id": "0001", "review_ids": ids}])
    records = [
        _rec(ids[0], final_category="fish", human_confidence=4),        # fish bucket
        _rec(ids[1], final_category="fish_part", human_confidence=4),   # fish bucket
        _rec(ids[2], decision_type="REMOVE", final_category="no_fish", human_confidence=4),  # non-fish
        _rec(ids[3], final_category="out_of_scope", human_confidence=4,
             decision_type="RELABEL", phase_c_category="unknown_needs_review"),  # excluded bucket
    ]
    _write_filled(run_id, "0001", records, tmp_path)

    _local, summary = aggregate_partial(manifest, tmp_path, run_id)
    tb = summary["training_eligibility"]

    assert tb["reviewed_high_confidence_fish"] == 2
    # no_fish and out_of_scope both go to non_fish_or_excluded
    assert tb["reviewed_high_confidence_non_fish_or_excluded"] == 2
    assert tb["training_eligible_count"] == 3  # fish(2) + no_fish(1), out_of_scope excluded


def test_partial_summary_contains_no_private_data(tmp_path: Path) -> None:
    """Partial tracked summary must not contain review IDs, filenames, or paths."""
    run_id = "rvrun_pa5"
    ids = ["rv_aaa1111111111111"]
    manifest = _manifest_partial(run_id, [{"batch_id": "0001", "review_ids": ids}])
    _write_filled(run_id, "0001", [_rec(ids[0])], tmp_path)

    _local, summary = aggregate_partial(manifest, tmp_path, run_id)

    summary_json = json.dumps(summary)
    for rid in ids:
        assert rid not in summary_json, f"review_id '{rid}' leaked into tracked summary"
    assert "photos/" not in summary_json
    assert "photo_" not in summary_json
    # Assert the privacy helper also passes
    _assert_summary_privacy(summary)


def test_partial_invalid_batch_blocks_aggregation(tmp_path: Path) -> None:
    """aggregate_partial raises if any reviewed batch is invalid."""
    run_id = "rvrun_pa6"
    ids = ["rv_aaa1111111111111"]
    manifest = _manifest_partial(run_id, [{"batch_id": "0001", "review_ids": ids}])
    # RELABEL with no category change — invalid
    rec = _rec(ids[0], decision_type="RELABEL", phase_c_category="fish_part", final_category="fish_part")
    _write_filled(run_id, "0001", [rec], tmp_path)

    with pytest.raises(ValueError, match="failed validation"):
        aggregate_partial(manifest, tmp_path, run_id)
