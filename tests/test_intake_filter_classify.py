"""
test_intake_filter_classify.py — Tests for intake_filter_classify.py (U4 Phase C).

Covers:
- Rule waterfall: poster_screenshot, fish_part, fry_juvenile, default_unknown
- Conflict detection: CONFLICT_A (text_heavy+lure+non-extreme), CONFLICT_B (2+ caption kws)
- Conflict A exemption: extreme aspect suppresses CONFLICT_A
- Quality flags: low_res, tiny_file, extreme_aspect appended to any category
- Lure hint: caption_lure_keyword alone routes to unknown with REASON_CAPTION_LURE_HINT
- Invariants: fish never assigned, high confidence never assigned, review_required always True
- Summary accuracy: sum of by_candidate_category == total_images, review_required_count == total
- Dry-run: no files written
- Privacy: summary has no filenames, captions, senders
- Error paths: missing signals exits 1, corrupt JSONL raises ValueError
- Real-data integration (skipif data absent)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import intake_constants as C
from intake_filter_classify import classify_record, classify_all


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _sig(
    filename: str = "photos/a.jpg",
    sha256: str = "abc123",
    aspect_class: str = "landscape",
    file_size_bucket: str = "medium",
    low_res: bool = False,
    caption_lure_keyword: bool = False,
    caption_fish_part_keyword: bool = False,
    caption_fry_keyword: bool = False,
    caption_no_fish_keyword: bool = False,
    caption_text_heavy: bool = False,
    source: str = "telegram_private_2026-04-24",
) -> dict:
    return {
        "filename": filename,
        "sha256": sha256,
        "aspect_class": aspect_class,
        "file_size_bucket": file_size_bucket,
        "low_res": low_res,
        "caption_lure_keyword": caption_lure_keyword,
        "caption_fish_part_keyword": caption_fish_part_keyword,
        "caption_fry_keyword": caption_fry_keyword,
        "caption_no_fish_keyword": caption_no_fish_keyword,
        "caption_text_heavy": caption_text_heavy,
        "source": source,
    }


def _make_signals_file(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "filter_signals.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return p


# ─── Rule waterfall: happy paths ─────────────────────────────────────────────


def test_poster_screenshot_rule() -> None:
    """caption_no_fish_keyword + caption_text_heavy → poster_screenshot / medium."""
    rec = _sig(caption_no_fish_keyword=True, caption_text_heavy=True)
    cand = classify_record(rec)
    assert cand["candidate_category"] == "poster_screenshot"
    assert cand["confidence"] == C.CONFIDENCE_MEDIUM
    assert cand["review_required"] is True
    assert cand["conflicts"] == []
    assert C.REASON_CAPTION_NO_FISH_KW in cand["reasons"]
    assert C.REASON_CAPTION_TEXT_HEAVY in cand["reasons"]


def test_fish_part_rule() -> None:
    """caption_fish_part_keyword alone → fish_part / low."""
    rec = _sig(caption_fish_part_keyword=True)
    cand = classify_record(rec)
    assert cand["candidate_category"] == "fish_part"
    assert cand["confidence"] == C.CONFIDENCE_LOW
    assert cand["review_required"] is True
    assert C.REASON_CAPTION_FISH_PART_KW in cand["reasons"]
    assert cand["conflicts"] == []


def test_fry_juvenile_rule() -> None:
    """caption_fry_keyword alone → fry_juvenile / low."""
    rec = _sig(caption_fry_keyword=True)
    cand = classify_record(rec)
    assert cand["candidate_category"] == "fry_juvenile"
    assert cand["confidence"] == C.CONFIDENCE_LOW
    assert cand["review_required"] is True
    assert C.REASON_CAPTION_FRY_KW in cand["reasons"]
    assert cand["conflicts"] == []


def test_default_unknown_empty_caption() -> None:
    """All caption booleans False, normal geometry → unknown_needs_review / low."""
    rec = _sig()
    cand = classify_record(rec)
    assert cand["candidate_category"] == "unknown_needs_review"
    assert cand["confidence"] == C.CONFIDENCE_LOW
    assert cand["review_required"] is True
    assert C.REASON_NO_STRONG_SIGNAL in cand["reasons"]
    assert cand["conflicts"] == []


# ─── Conflict A ───────────────────────────────────────────────────────────────


def test_conflict_a_text_heavy_lure_normal_aspect() -> None:
    """text_heavy + lure + non-extreme aspect → CONFLICT_A → unknown_needs_review."""
    rec = _sig(caption_text_heavy=True, caption_lure_keyword=True, aspect_class="landscape")
    cand = classify_record(rec)
    assert cand["candidate_category"] == "unknown_needs_review"
    assert C.CONFLICT_AMBIGUOUS_TEXT_HEAVY_LURE in cand["conflicts"]
    assert cand["review_required"] is True


def test_conflict_a_text_heavy_lure_square_aspect() -> None:
    """text_heavy + lure + square aspect also triggers CONFLICT_A."""
    rec = _sig(caption_text_heavy=True, caption_lure_keyword=True, aspect_class="square")
    cand = classify_record(rec)
    assert C.CONFLICT_AMBIGUOUS_TEXT_HEAVY_LURE in cand["conflicts"]


def test_conflict_a_not_fired_for_extreme_landscape() -> None:
    """CONFLICT_A is suppressed when aspect_class=extreme_landscape."""
    rec = _sig(caption_text_heavy=True, caption_lure_keyword=True, aspect_class="extreme_landscape")
    cand = classify_record(rec)
    assert C.CONFLICT_AMBIGUOUS_TEXT_HEAVY_LURE not in cand["conflicts"]
    # Single caption keyword → no CONFLICT_B either; lure hint surfaced
    assert cand["candidate_category"] == "unknown_needs_review"
    assert C.REASON_CAPTION_LURE_HINT in cand["reasons"]
    assert C.REASON_EXTREME_ASPECT in cand["reasons"]


def test_conflict_a_not_fired_for_extreme_portrait() -> None:
    """CONFLICT_A is suppressed when aspect_class=extreme_portrait."""
    rec = _sig(caption_text_heavy=True, caption_lure_keyword=True, aspect_class="extreme_portrait")
    cand = classify_record(rec)
    assert C.CONFLICT_AMBIGUOUS_TEXT_HEAVY_LURE not in cand["conflicts"]


# ─── Conflict B ───────────────────────────────────────────────────────────────


def test_conflict_b_two_caption_keywords_fish_part_fry() -> None:
    """fish_part + fry → CONFLICT_B → unknown_needs_review (not fish_part)."""
    rec = _sig(caption_fish_part_keyword=True, caption_fry_keyword=True)
    cand = classify_record(rec)
    assert cand["candidate_category"] == "unknown_needs_review"
    assert C.CONFLICT_COMPETING_CAPTION_KEYWORDS in cand["conflicts"]


def test_conflict_b_lure_and_fish_part() -> None:
    """lure + fish_part → CONFLICT_B → unknown_needs_review (not fish_part)."""
    rec = _sig(caption_lure_keyword=True, caption_fish_part_keyword=True)
    cand = classify_record(rec)
    assert cand["candidate_category"] == "unknown_needs_review"
    assert C.CONFLICT_COMPETING_CAPTION_KEYWORDS in cand["conflicts"]


def test_conflict_b_lure_and_fry() -> None:
    """lure + fry → CONFLICT_B."""
    rec = _sig(caption_lure_keyword=True, caption_fry_keyword=True)
    cand = classify_record(rec)
    assert C.CONFLICT_COMPETING_CAPTION_KEYWORDS in cand["conflicts"]


def test_conflict_b_no_fish_and_lure() -> None:
    """no_fish + lure → CONFLICT_B."""
    rec = _sig(caption_no_fish_keyword=True, caption_lure_keyword=True)
    cand = classify_record(rec)
    assert C.CONFLICT_COMPETING_CAPTION_KEYWORDS in cand["conflicts"]


# ─── Quality flags ────────────────────────────────────────────────────────────


def test_low_res_appended_to_unknown() -> None:
    rec = _sig(low_res=True)
    cand = classify_record(rec)
    assert C.REASON_LOW_RES in cand["reasons"]
    assert cand["candidate_category"] == "unknown_needs_review"


def test_tiny_file_appended_to_unknown() -> None:
    rec = _sig(file_size_bucket="tiny")
    cand = classify_record(rec)
    assert C.REASON_TINY_FILE in cand["reasons"]


def test_extreme_aspect_appended_to_unknown() -> None:
    rec = _sig(aspect_class="extreme_portrait")
    cand = classify_record(rec)
    assert C.REASON_EXTREME_ASPECT in cand["reasons"]
    assert cand["candidate_category"] == "unknown_needs_review"


def test_quality_flags_do_not_override_fish_part_category() -> None:
    """A fish_part record that is also low_res stays fish_part."""
    rec = _sig(caption_fish_part_keyword=True, low_res=True)
    cand = classify_record(rec)
    assert cand["candidate_category"] == "fish_part"
    assert C.REASON_CAPTION_FISH_PART_KW in cand["reasons"]
    assert C.REASON_LOW_RES in cand["reasons"]


def test_poster_with_quality_flags() -> None:
    """poster_screenshot record with low_res still gets both tags."""
    rec = _sig(caption_no_fish_keyword=True, caption_text_heavy=True, low_res=True)
    cand = classify_record(rec)
    assert cand["candidate_category"] == "poster_screenshot"
    assert cand["confidence"] == C.CONFIDENCE_MEDIUM
    assert C.REASON_LOW_RES in cand["reasons"]
    assert C.REASON_CAPTION_NO_FISH_KW in cand["reasons"]


def test_all_quality_flags() -> None:
    """All three quality flags appear together."""
    rec = _sig(low_res=True, file_size_bucket="tiny", aspect_class="extreme_landscape")
    cand = classify_record(rec)
    assert C.REASON_LOW_RES in cand["reasons"]
    assert C.REASON_TINY_FILE in cand["reasons"]
    assert C.REASON_EXTREME_ASPECT in cand["reasons"]


# ─── Lure hint (no conflict) ──────────────────────────────────────────────────


def test_lure_hint_no_conflict_routes_to_unknown_with_hint() -> None:
    """caption_lure alone (no text_heavy, no other kw) → unknown + REASON_CAPTION_LURE_HINT."""
    rec = _sig(caption_lure_keyword=True, caption_text_heavy=False, aspect_class="landscape")
    cand = classify_record(rec)
    assert cand["candidate_category"] == "unknown_needs_review"
    assert C.REASON_CAPTION_LURE_HINT in cand["reasons"]
    assert "lure_fishing_gear" != cand["candidate_category"]
    assert cand["conflicts"] == []


def test_lure_hint_not_lure_gear() -> None:
    """lure keyword never produces lure_fishing_gear candidate_category."""
    rec = _sig(caption_lure_keyword=True)
    cand = classify_record(rec)
    assert cand["candidate_category"] != "lure_fishing_gear"


# ─── Invariants ───────────────────────────────────────────────────────────────


# Representative set of signal combos for parametrized invariant tests
_INVARIANT_CASES = [
    _sig(),
    _sig(caption_lure_keyword=True),
    _sig(caption_fish_part_keyword=True),
    _sig(caption_fry_keyword=True),
    _sig(caption_no_fish_keyword=True, caption_text_heavy=True),
    _sig(caption_text_heavy=True, caption_lure_keyword=True),
    _sig(caption_fish_part_keyword=True, caption_fry_keyword=True),
    _sig(caption_lure_keyword=True, caption_fish_part_keyword=True),
    _sig(low_res=True),
    _sig(file_size_bucket="tiny"),
    _sig(aspect_class="extreme_portrait"),
    _sig(aspect_class="extreme_landscape"),
    _sig(caption_text_heavy=True, caption_lure_keyword=True, aspect_class="extreme_landscape"),
    _sig(caption_no_fish_keyword=True, caption_text_heavy=True, low_res=True),
    _sig(caption_fish_part_keyword=True, low_res=True, file_size_bucket="tiny"),
    _sig(caption_lure_keyword=True, aspect_class="extreme_portrait"),
    _sig(caption_fry_keyword=True, aspect_class="extreme_landscape"),
    _sig(caption_no_fish_keyword=True, caption_lure_keyword=True),
    _sig(caption_text_heavy=True),
    _sig(caption_lure_keyword=True, caption_fry_keyword=True),
]


@pytest.mark.parametrize("rec", _INVARIANT_CASES)
def test_fish_never_assigned(rec: dict) -> None:
    cand = classify_record(rec)
    assert cand["candidate_category"] != "fish", (
        f"fish was assigned for rec={rec}"
    )


@pytest.mark.parametrize("rec", _INVARIANT_CASES)
def test_high_confidence_never_assigned(rec: dict) -> None:
    cand = classify_record(rec)
    assert cand["confidence"] != C.CONFIDENCE_HIGH, (
        f"high confidence assigned for rec={rec}"
    )


@pytest.mark.parametrize("rec", _INVARIANT_CASES)
def test_review_required_always_true(rec: dict) -> None:
    cand = classify_record(rec)
    assert cand["review_required"] is True, (
        f"review_required was not True for rec={rec}"
    )


@pytest.mark.parametrize("rec", _INVARIANT_CASES)
def test_lure_gear_never_assigned(rec: dict) -> None:
    cand = classify_record(rec)
    assert cand["candidate_category"] != "lure_fishing_gear"


@pytest.mark.parametrize("rec", _INVARIANT_CASES)
def test_no_fish_never_assigned(rec: dict) -> None:
    cand = classify_record(rec)
    assert cand["candidate_category"] != "no_fish"


@pytest.mark.parametrize("rec", _INVARIANT_CASES)
def test_reasons_never_empty(rec: dict) -> None:
    cand = classify_record(rec)
    assert len(cand["reasons"]) > 0, f"reasons was empty for rec={rec}"


# ─── Summary accuracy ─────────────────────────────────────────────────────────


def test_summary_category_sum_equals_total_images(tmp_path: Path) -> None:
    """sum(by_candidate_category.values()) == total_images."""
    records = [
        _sig("photos/a.jpg", caption_no_fish_keyword=True, caption_text_heavy=True),
        _sig("photos/b.jpg", caption_fish_part_keyword=True),
        _sig("photos/c.jpg", caption_fry_keyword=True),
        _sig("photos/d.jpg"),
        _sig("photos/e.jpg", caption_text_heavy=True, caption_lure_keyword=True),
    ]
    signals_path = _make_signals_file(tmp_path, records)
    summary = classify_all(signals_path, tmp_path)
    assert sum(summary["by_candidate_category"].values()) == summary["total_images"] == 5


def test_summary_review_required_count_equals_total(tmp_path: Path) -> None:
    """review_required_count == total_images (all records in Phase C)."""
    records = [_sig(f"photos/{i}.jpg") for i in range(10)]
    signals_path = _make_signals_file(tmp_path, records)
    summary = classify_all(signals_path, tmp_path)
    assert summary["review_required_count"] == summary["total_images"] == 10


def test_summary_fish_zero(tmp_path: Path) -> None:
    """by_candidate_category.fish == 0 always."""
    records = [_sig(f"photos/{i}.jpg") for i in range(5)]
    signals_path = _make_signals_file(tmp_path, records)
    summary = classify_all(signals_path, tmp_path)
    assert summary["by_candidate_category"]["fish"] == 0


def test_summary_high_confidence_zero(tmp_path: Path) -> None:
    """by_confidence.high == 0 always."""
    records = [_sig(f"photos/{i}.jpg") for i in range(5)]
    signals_path = _make_signals_file(tmp_path, records)
    summary = classify_all(signals_path, tmp_path)
    assert summary["by_confidence"][C.CONFIDENCE_HIGH] == 0


def test_summary_conflict_flag_count(tmp_path: Path) -> None:
    """conflict_flag_count equals count of records that have at least one conflict."""
    records = [
        _sig("photos/conflict.jpg", caption_text_heavy=True, caption_lure_keyword=True),
        _sig("photos/clean.jpg"),
    ]
    signals_path = _make_signals_file(tmp_path, records)
    summary = classify_all(signals_path, tmp_path)
    assert summary["conflict_flag_count"] == 1


def test_summary_privacy_no_filenames(tmp_path: Path) -> None:
    """filter_candidates_summary.json must not contain any filename strings."""
    fn = "photos/secret_name_xyz.jpg"
    signals_path = _make_signals_file(tmp_path, [_sig(fn)])
    classify_all(signals_path, tmp_path)
    summary_text = (tmp_path / "filter_candidates_summary.json").read_text(encoding="utf-8")
    assert "secret_name_xyz" not in summary_text
    assert "photos/" not in summary_text


def test_summary_privacy_no_pii_fields(tmp_path: Path) -> None:
    """Summary must not contain sender IDs, file:// paths, or per-image fields.
    Note: reason code keys like 'caption_lure_hint' are aggregate signal identifiers,
    not PII — 'caption' as a substring of a key name is not a privacy leak."""
    signals_path = _make_signals_file(tmp_path, [_sig()])
    classify_all(signals_path, tmp_path)
    summary_text = (tmp_path / "filter_candidates_summary.json").read_text(encoding="utf-8")
    for forbidden in ("sender", "from_id", "user_id", "file://"):
        assert forbidden not in summary_text, f"Summary contains forbidden PII field: {forbidden!r}"


# ─── Dry-run: no files written ────────────────────────────────────────────────


def test_dry_run_writes_no_files(tmp_path: Path) -> None:
    signals_path = _make_signals_file(tmp_path, [_sig()])
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    classify_all(signals_path, out_dir, dry_run=True)
    assert not (out_dir / "filter_candidates.jsonl").exists()
    assert not (out_dir / "filter_candidates_summary.json").exists()


def test_dry_run_returns_summary_dict(tmp_path: Path) -> None:
    signals_path = _make_signals_file(tmp_path, [_sig()])
    summary = classify_all(signals_path, tmp_path, dry_run=True)
    assert isinstance(summary, dict)
    assert summary["total_images"] == 1


# ─── Error paths ──────────────────────────────────────────────────────────────


def test_missing_signals_exits(tmp_path: Path) -> None:
    """Missing filter_signals.jsonl → sys.exit(1)."""
    with pytest.raises(SystemExit) as exc:
        classify_all(tmp_path / "no_signals.jsonl", tmp_path)
    assert exc.value.code == 1


def test_corrupt_signals_raises_value_error(tmp_path: Path) -> None:
    """Corrupt JSONL in signals raises ValueError."""
    p = tmp_path / "filter_signals.jsonl"
    p.write_text('{"ok": 1}\n{CORRUPT\n', encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt JSONL"):
        classify_all(p, tmp_path)


# ─── Schema fields ────────────────────────────────────────────────────────────


def test_candidate_record_has_required_fields() -> None:
    """classify_record output contains all required schema fields."""
    cand = classify_record(_sig())
    required = {"filename", "sha256", "candidate_category", "confidence",
                 "review_required", "reasons", "conflicts", "source", "schema_version"}
    assert required.issubset(cand.keys())


def test_schema_version_is_1() -> None:
    cand = classify_record(_sig())
    assert cand["schema_version"] == 1


def test_candidate_category_in_coarse_categories() -> None:
    """candidate_category must always be a member of COARSE_CATEGORIES."""
    for rec in _INVARIANT_CASES:
        cand = classify_record(rec)
        assert cand["candidate_category"] in C.COARSE_CATEGORIES_SET, (
            f"Unknown category: {cand['candidate_category']!r}"
        )


# ─── Real-data integration (skipif data absent) ──────────────────────────────


SIGNALS_PATH = REPO_ROOT / "data/intake_meta/tg_2026-04-24/filter_signals.jsonl"

_real_data_available = SIGNALS_PATH.exists()


@pytest.mark.skipif(not _real_data_available, reason="Real filter_signals.jsonl not available")
def test_real_total_images_equals_32420(tmp_path: Path) -> None:
    summary = classify_all(SIGNALS_PATH, tmp_path, dry_run=True)
    assert summary["total_images"] == 32420


@pytest.mark.skipif(not _real_data_available, reason="Real filter_signals.jsonl not available")
def test_real_fish_zero(tmp_path: Path) -> None:
    summary = classify_all(SIGNALS_PATH, tmp_path, dry_run=True)
    assert summary["by_candidate_category"]["fish"] == 0


@pytest.mark.skipif(not _real_data_available, reason="Real filter_signals.jsonl not available")
def test_real_lure_gear_zero(tmp_path: Path) -> None:
    summary = classify_all(SIGNALS_PATH, tmp_path, dry_run=True)
    assert summary["by_candidate_category"]["lure_fishing_gear"] == 0


@pytest.mark.skipif(not _real_data_available, reason="Real filter_signals.jsonl not available")
def test_real_no_fish_zero(tmp_path: Path) -> None:
    summary = classify_all(SIGNALS_PATH, tmp_path, dry_run=True)
    assert summary["by_candidate_category"]["no_fish"] == 0


@pytest.mark.skipif(not _real_data_available, reason="Real filter_signals.jsonl not available")
def test_real_high_confidence_zero(tmp_path: Path) -> None:
    summary = classify_all(SIGNALS_PATH, tmp_path, dry_run=True)
    assert summary["by_confidence"][C.CONFIDENCE_HIGH] == 0


@pytest.mark.skipif(not _real_data_available, reason="Real filter_signals.jsonl not available")
def test_real_review_required_equals_total(tmp_path: Path) -> None:
    summary = classify_all(SIGNALS_PATH, tmp_path, dry_run=True)
    assert summary["review_required_count"] == summary["total_images"]


@pytest.mark.skipif(not _real_data_available, reason="Real filter_signals.jsonl not available")
def test_real_conflict_flag_count_ge_770(tmp_path: Path) -> None:
    """CONFLICT_A fires for text_heavy+lure+non-extreme; ~7 of 778 have extreme aspect and are exempt.
    Actual observed: 771. Safe lower bound: 770."""
    summary = classify_all(SIGNALS_PATH, tmp_path, dry_run=True)
    assert summary["conflict_flag_count"] >= 770


@pytest.mark.skipif(not _real_data_available, reason="Real filter_signals.jsonl not available")
def test_real_unknown_needs_review_dominant(tmp_path: Path) -> None:
    summary = classify_all(SIGNALS_PATH, tmp_path, dry_run=True)
    assert summary["by_candidate_category"]["unknown_needs_review"] > 32300


@pytest.mark.skipif(not _real_data_available, reason="Real filter_signals.jsonl not available")
def test_real_poster_screenshot_le_10(tmp_path: Path) -> None:
    summary = classify_all(SIGNALS_PATH, tmp_path, dry_run=True)
    count = summary["by_candidate_category"]["poster_screenshot"]
    assert count <= 10, f"poster_screenshot={count} exceeds expected max of 10"
