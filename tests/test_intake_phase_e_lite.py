"""
Tests for intake_phase_e_lite.py — U4 Phase E Lite reviewed-only seed materialization.

Verifies:
  - Only reviewed records enter the seed
  - Unreviewed records are excluded
  - Summaries are counts-only (no private fields)
  - Class mapping is deterministic
  - Excluded records are tracked
  - Privacy: no disallowed fields in tracked outputs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from intake_phase_e_lite import (
    assign_mvp_class,
    build_summary,
    build_manifest,
    process_batch,
    _DISALLOWED_TRACKED_FIELDS,
    _assert_summary_privacy,
    MVP_CONFIDENCE_MIN,
    SEED_SUMMARY_SCHEMA_VERSION,
)
import intake_constants as C


# ─── assign_mvp_class tests ───────────────────────────────────────────────────


class TestAssignMvpClass:
    def test_fish_high_confidence_is_fish(self):
        assert assign_mvp_class("fish", 3) == "fish"
        assert assign_mvp_class("fish", 5) == "fish"

    def test_fish_part_is_fish(self):
        assert assign_mvp_class("fish_part", 4) == "fish"

    def test_no_fish_is_not_fish(self):
        assert assign_mvp_class("no_fish", 4) == "not_fish_or_other"

    def test_lure_gear_is_not_fish(self):
        assert assign_mvp_class("lure_gear", 4) == "not_fish_or_other"

    def test_poster_screenshot_is_not_fish(self):
        assert assign_mvp_class("poster_screenshot", 4) == "not_fish_or_other"

    def test_bad_quality_is_needs_review(self):
        assert assign_mvp_class("bad_quality", 4) == "needs_human_review"

    def test_out_of_scope_is_needs_review(self):
        assert assign_mvp_class("out_of_scope", 4) == "needs_human_review"

    def test_low_confidence_overrides_to_needs_review(self):
        # Even fish at low confidence → needs_human_review
        assert assign_mvp_class("fish", 1) == "needs_human_review"
        assert assign_mvp_class("fish", 2) == "needs_human_review"

    def test_boundary_confidence(self):
        # MVP_CONFIDENCE_MIN = 3 → conf=3 is eligible
        assert assign_mvp_class("fish", MVP_CONFIDENCE_MIN) == "fish"
        assert assign_mvp_class("fish", MVP_CONFIDENCE_MIN - 1) == "needs_human_review"

    def test_none_confidence_is_needs_review(self):
        assert assign_mvp_class("fish", None) == "needs_human_review"

    def test_unknown_category_is_needs_review(self):
        assert assign_mvp_class("something_unknown", 5) == "needs_human_review"

    def test_fry_juvenile_is_needs_review(self):
        assert assign_mvp_class("fry_juvenile", 5) == "needs_human_review"

    def test_unsure_is_needs_review(self):
        assert assign_mvp_class("unsure", 5) == "needs_human_review"

    def test_duplicate_suspect_is_needs_review(self):
        assert assign_mvp_class("duplicate_suspect", 5) == "needs_human_review"


# ─── process_batch tests ──────────────────────────────────────────────────────


def _make_batch_file(tmp_path: Path, records: list[dict], batch_id: str = "0001") -> Path:
    data = {
        "schema_version": "u4_phase_d_decisions_v1",
        "source": "telegram_private_2026-04-24",
        "phase": "U4_PHASE_D_MANUAL_REVIEW",
        "run_id": "rvrun_test",
        "batch_id": batch_id,
        "created_by": "test",
        "records": records,
    }
    p = tmp_path / f"filter_decisions_rvrun_test_{batch_id}.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _make_record(
    review_id: str = "abc123",
    decision_type: str | None = "RELABEL",
    final_category: str = "fish",
    human_confidence: int = 4,
) -> dict:
    return {
        "review_id": review_id,
        "decision_type": decision_type,
        "phase_c_category": "unknown_needs_review",
        "final_category": final_category,
        "human_confidence": human_confidence,
        "refinement": None,
        "notes": None,
        "reviewed_at": "2026-04-27T18:00:00Z",
    }


class TestProcessBatch:
    def test_fish_record_enters_seed(self, tmp_path):
        batch = _make_batch_file(tmp_path, [_make_record("r1", "RELABEL", "fish", 4)])
        seed, excluded = process_batch(batch)
        assert len(seed) == 1
        assert len(excluded) == 0
        assert seed[0]["mvp_class"] == "fish"

    def test_low_confidence_goes_to_excluded(self, tmp_path):
        batch = _make_batch_file(tmp_path, [_make_record("r1", "RELABEL", "fish", 2)])
        seed, excluded = process_batch(batch)
        assert len(seed) == 0
        assert len(excluded) == 1
        assert excluded[0]["mvp_class"] == "needs_human_review"

    def test_blank_record_skipped(self, tmp_path):
        batch = _make_batch_file(tmp_path, [_make_record("r1", decision_type=None)])
        seed, excluded = process_batch(batch)
        assert len(seed) == 0
        assert len(excluded) == 0

    def test_out_of_scope_excluded(self, tmp_path):
        batch = _make_batch_file(tmp_path, [_make_record("r1", "RELABEL", "out_of_scope", 5)])
        seed, excluded = process_batch(batch)
        assert len(seed) == 0
        assert len(excluded) == 1

    def test_no_fish_in_seed_as_not_fish(self, tmp_path):
        batch = _make_batch_file(tmp_path, [_make_record("r1", "RELABEL", "no_fish", 4)])
        seed, excluded = process_batch(batch)
        assert len(seed) == 1
        assert seed[0]["mvp_class"] == "not_fish_or_other"

    def test_seed_records_have_required_fields(self, tmp_path):
        batch = _make_batch_file(tmp_path, [_make_record("r1", "RELABEL", "fish", 4)])
        seed, _ = process_batch(batch)
        assert "review_id" in seed[0]
        assert "final_category" in seed[0]
        assert "human_confidence" in seed[0]
        assert "mvp_class" in seed[0]
        assert "batch_id" in seed[0]

    def test_excluded_records_have_exclude_reason(self, tmp_path):
        batch = _make_batch_file(tmp_path, [_make_record("r1", "RELABEL", "fish", 1)])
        _, excluded = process_batch(batch)
        assert "exclude_reason" in excluded[0]
        assert "low_confidence" in excluded[0]["exclude_reason"]


# ─── build_summary tests ──────────────────────────────────────────────────────


class TestBuildSummary:
    def _make_seed(self, n_fish=3, n_notfish=1, n_excluded=2) -> tuple[list, list]:
        seed = (
            [{"review_id": f"f{i}", "mvp_class": "fish", "batch_id": "0001"} for i in range(n_fish)] +
            [{"review_id": f"n{i}", "mvp_class": "not_fish_or_other", "batch_id": "0001"} for i in range(n_notfish)]
        )
        excluded = [
            {"review_id": f"e{i}", "mvp_class": "needs_human_review", "exclude_reason": "low_confidence_2", "batch_id": "0001"}
            for i in range(n_excluded)
        ]
        return seed, excluded

    def test_summary_counts_correct(self):
        seed, excluded = self._make_seed(3, 1, 2)
        s = build_summary(seed, excluded, 100, "run1", ["0001"], "2026-05-01T00:00:00Z")
        assert s["counts"]["total_reviewed_in_seed"] == 4
        assert s["counts"]["total_excluded"] == 2
        assert s["counts"]["unreviewed_not_eligible"] == 100
        assert s["mvp_class_counts"]["fish"] == 3
        assert s["mvp_class_counts"]["not_fish_or_other"] == 1

    def test_summary_no_private_fields(self):
        seed, excluded = self._make_seed()
        s = build_summary(seed, excluded, 0, "run1", ["0001"], "2026-05-01T00:00:00Z")
        s_str = json.dumps(s)
        # "review_id" appears in seed/excluded data structures but should NOT be in summary
        assert '"review_id"' not in s_str
        assert '"filename"' not in s_str
        assert '"sha256"' not in s_str
        assert '"caption"' not in s_str

    def test_summary_has_schema_version(self):
        seed, excluded = self._make_seed()
        s = build_summary(seed, excluded, 0, "run1", ["0001"], "2026-05-01T00:00:00Z")
        assert s["schema_version"] == SEED_SUMMARY_SCHEMA_VERSION

    def test_privacy_violation_raises(self):
        """Injecting a disallowed field into summary raises ValueError."""
        seed = [{"review_id": "x", "mvp_class": "fish", "batch_id": "0001"}]
        s = build_summary(seed, [], 0, "run1", ["0001"], "2026-05-01T00:00:00Z")
        # Manually inject a disallowed field
        s["review_id"] = "leaked_id"
        with pytest.raises(ValueError, match="Privacy violation"):
            _assert_summary_privacy(s)


# ─── Class mapping determinism ────────────────────────────────────────────────


class TestClassMappingDeterminism:
    """Verify the mapping is deterministic across multiple calls."""

    def test_same_input_same_output(self):
        for fc in ["fish", "no_fish", "lure_gear", "fish_part", "out_of_scope", "bad_quality"]:
            for conf in [1, 2, 3, 4, 5]:
                r1 = assign_mvp_class(fc, conf)
                r2 = assign_mvp_class(fc, conf)
                assert r1 == r2, f"Non-deterministic for {fc}, {conf}"

    def test_all_final_categories_have_mapping(self):
        from intake_phase_e_lite import _CATEGORY_TO_MVP
        for cat in C.FINAL_CATEGORIES:
            assert cat in _CATEGORY_TO_MVP, f"Category '{cat}' has no MVP mapping"


# ─── Disallowed fields in tracked outputs ─────────────────────────────────────


class TestPrivacySafety:
    """
    Tracked summary must not contain private identifiers.
    """

    def test_disallowed_field_set_covers_key_private_fields(self):
        for field in ["review_id", "filename", "sha256", "caption", "sender"]:
            assert field in _DISALLOWED_TRACKED_FIELDS

    def test_assert_summary_privacy_passes_on_clean_summary(self):
        clean = {"counts": {"fish": 5}, "schema_version": "v1"}
        _assert_summary_privacy(clean)  # Should not raise

    def test_assert_summary_privacy_fails_on_leakage(self):
        leaky = {"filename": "secret_photo.jpg", "counts": {"fish": 5}}
        with pytest.raises(ValueError):
            _assert_summary_privacy(leaky)
