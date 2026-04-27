"""
test_intake_filter_heuristic.py — Tests for intake_filter_heuristic.py (U4 Phase B).

Covers:
- U1 constants: ordering invariants, frozenset types, non-empty sets
- Geometry signals: aspect_class, file_size_bucket (happy path + boundaries)
- Caption signals: keyword matching (case-insensitive), text_heavy, empty caption
- Edge cases: zero/None dimensions, no manifest record, conflicting signals
- Error paths: missing universe exits 1, corrupt JSONL raises ValueError, missing manifest continues
- dry_run: writes no files
- Privacy/summary integrity: no filenames, no captions, no sender names in summary
- Pipeline invariants: bijective filename mapping, count consistency
- Real-data integration (skipif data absent): 32420 records, known counts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import intake_constants as C
from intake_filter_heuristic import _classify_aspect, _classify_file_size, build_signals


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _make_universe(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "filter_universe.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return p


def _make_manifest(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "manifest.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return p


def _universe_rec(
    filename: str = "photos/a.jpg",
    width: int = 1280,
    height: int = 960,
    file_size: int = 300_000,
    low_res: bool = False,
    dedup_role: str = "unique",
) -> dict:
    return {
        "filename": filename,
        "sha256": "abc123",
        "width": width,
        "height": height,
        "max_side": max(width, height),
        "file_size": file_size,
        "low_res": low_res,
        "dedup_role": dedup_role,
        "source": "telegram_private_2026-04-24",
    }


def _manifest_rec(filename: str, caption: str = "") -> dict:
    return {"filename": filename, "caption": caption}


# ─── U1 constants tests ───────────────────────────────────────────────────────


def test_file_size_bucket_ordering() -> None:
    """FILE_SIZE_BUCKET thresholds must be strictly increasing."""
    assert C.FILE_SIZE_BUCKET_TINY_MAX < C.FILE_SIZE_BUCKET_SMALL_MAX
    assert C.FILE_SIZE_BUCKET_SMALL_MAX < C.FILE_SIZE_BUCKET_MEDIUM_MAX


def test_caption_text_heavy_threshold_positive_int() -> None:
    assert isinstance(C.CAPTION_TEXT_HEAVY_THRESHOLD, int)
    assert C.CAPTION_TEXT_HEAVY_THRESHOLD > 0


def test_keyword_frozensets_are_frozensets() -> None:
    assert isinstance(C.CAPTION_LURE_KEYWORD_HINTS, frozenset)
    assert isinstance(C.CAPTION_FISH_PART_KEYWORD_HINTS, frozenset)
    assert isinstance(C.CAPTION_FRY_KEYWORD_HINTS, frozenset)
    assert isinstance(C.CAPTION_NO_FISH_KEYWORD_HINTS, frozenset)


def test_keyword_frozensets_non_empty() -> None:
    assert len(C.CAPTION_LURE_KEYWORD_HINTS) > 0
    assert len(C.CAPTION_FISH_PART_KEYWORD_HINTS) > 0
    assert len(C.CAPTION_FRY_KEYWORD_HINTS) > 0
    assert len(C.CAPTION_NO_FISH_KEYWORD_HINTS) > 0


def test_fry_keyword_not_in_lure_set() -> None:
    """Sanity: 'малёк' must not appear in the lure keyword set."""
    assert "малёк" not in C.CAPTION_LURE_KEYWORD_HINTS


# ─── _classify_aspect happy path ─────────────────────────────────────────────


@pytest.mark.parametrize("width,height,expected_class", [
    (1280, 960,  "landscape"),         # 1.33
    (960,  1280, "portrait"),          # 0.75
    (1000, 200,  "extreme_landscape"), # 5.0
    (200,  1000, "extreme_portrait"),  # 0.20
    (1000, 1000, "square"),            # 1.0
])
def test_classify_aspect_happy(width: int, height: int, expected_class: str) -> None:
    ar = round(width / height, 2)
    assert _classify_aspect(ar) == expected_class


@pytest.mark.parametrize("ar,expected", [
    (0.499, "extreme_portrait"),
    (0.5,   "portrait"),        # boundary: not extreme
    (0.799, "portrait"),
    (0.8,   "square"),          # boundary: not portrait
    (1.249, "square"),
    (1.25,  "landscape"),       # boundary: not square
    (1.999, "landscape"),
    (2.0,   "extreme_landscape"),# boundary: extreme
])
def test_classify_aspect_boundaries(ar: float, expected: str) -> None:
    assert _classify_aspect(ar) == expected


# ─── _classify_file_size happy path + boundaries ──────────────────────────────


@pytest.mark.parametrize("file_size,expected_bucket", [
    (20_000,  "tiny"),
    (80_000,  "small"),
    (200_000, "medium"),
    (500_000, "large"),
    (49_999,  "tiny"),          # just below tiny boundary
    (50_000,  "small"),         # boundary: not tiny
    (149_999, "small"),
    (150_000, "medium"),        # boundary: not small
    (399_999, "medium"),
    (400_000, "large"),         # boundary: not medium
])
def test_classify_file_size(file_size: int, expected_bucket: str) -> None:
    assert _classify_file_size(file_size) == expected_bucket


# ─── Caption keyword signals ──────────────────────────────────────────────────


@pytest.mark.parametrize("caption,expected_lure", [
    ("балансир со дна",  True),
    ("БАЛАНСИР СО ДНА",  True),   # case-insensitive
    ("поймали на воблер", True),
    ("обычный день",     False),
    ("",                 False),
])
def test_caption_lure_keyword(tmp_path: Path, caption: str, expected_lure: bool) -> None:
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    mfst = _make_manifest(tmp_path, [_manifest_rec(fn, caption)])

    summary = build_signals(univ, mfst, tmp_path, dry_run=False)
    signals = [
        json.loads(l)
        for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines()
        if l.strip()
    ]
    assert signals[0]["caption_lure_keyword"] is expected_lure


def test_caption_fry_keyword(tmp_path: Path) -> None:
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    mfst = _make_manifest(tmp_path, [_manifest_rec(fn, "мальки у берега")])

    build_signals(univ, mfst, tmp_path, dry_run=False)
    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    assert signals[0]["caption_fry_keyword"] is True
    assert signals[0]["caption_lure_keyword"] is False


def test_caption_fish_part_keyword(tmp_path: Path) -> None:
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    mfst = _make_manifest(tmp_path, [_manifest_rec(fn, "разделка улова на берегу")])

    build_signals(univ, mfst, tmp_path, dry_run=False)
    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    assert signals[0]["caption_fish_part_keyword"] is True


def test_caption_no_fish_keyword(tmp_path: Path) -> None:
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    mfst = _make_manifest(tmp_path, [_manifest_rec(fn, "расписание на сезон")])

    build_signals(univ, mfst, tmp_path, dry_run=False)
    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    assert signals[0]["caption_no_fish_keyword"] is True


def test_caption_empty(tmp_path: Path) -> None:
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    mfst = _make_manifest(tmp_path, [_manifest_rec(fn, "")])

    build_signals(univ, mfst, tmp_path, dry_run=False)
    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    s = signals[0]
    assert s["caption_empty"] is True
    assert s["caption_lure_keyword"] is False
    assert s["caption_fry_keyword"] is False
    assert s["caption_fish_part_keyword"] is False
    assert s["caption_no_fish_keyword"] is False


def test_caption_text_heavy_boundary(tmp_path: Path) -> None:
    """caption_text_heavy = True when length > 200, False when exactly 200."""
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn), _universe_rec("photos/b.jpg")])
    caption_201 = "а" * 201
    caption_200 = "а" * 200
    mfst = _make_manifest(tmp_path, [
        _manifest_rec(fn, caption_201),
        _manifest_rec("photos/b.jpg", caption_200),
    ])

    build_signals(univ, mfst, tmp_path, dry_run=False)
    signals = {
        r["filename"]: r
        for r in [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    }
    assert signals[fn]["caption_text_heavy"] is True
    assert signals["photos/b.jpg"]["caption_text_heavy"] is False


def test_conflicting_caption_signals_both_true(tmp_path: Path) -> None:
    """Both lure and fry keyword signals can be True simultaneously — conflicts left for Phase C."""
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    mfst = _make_manifest(tmp_path, [_manifest_rec(fn, "балансир и мальки")])

    build_signals(univ, mfst, tmp_path, dry_run=False)
    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    assert signals[0]["caption_lure_keyword"] is True
    assert signals[0]["caption_fry_keyword"] is True


def test_conflicting_lure_and_fish_part_both_true(tmp_path: Path) -> None:
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    mfst = _make_manifest(tmp_path, [_manifest_rec(fn, "воблер и разделка")])

    build_signals(univ, mfst, tmp_path, dry_run=False)
    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    assert signals[0]["caption_lure_keyword"] is True
    assert signals[0]["caption_fish_part_keyword"] is True


# ─── Edge cases ───────────────────────────────────────────────────────────────


def test_zero_dimensions_aspect_unknown(tmp_path: Path) -> None:
    """width=0 or height=0 → aspect_ratio=None, aspect_class='unknown'."""
    fn = "photos/a.jpg"
    rec = _universe_rec(fn)
    rec["width"] = 0
    rec["height"] = 960
    univ = _make_universe(tmp_path, [rec])
    mfst = _make_manifest(tmp_path, [])

    build_signals(univ, mfst, tmp_path, dry_run=False)
    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    assert signals[0]["aspect_ratio"] is None
    assert signals[0]["aspect_class"] == "unknown"


def test_none_dimensions_aspect_unknown(tmp_path: Path) -> None:
    """width=None or height=None → aspect_ratio=None, aspect_class='unknown'."""
    fn = "photos/a.jpg"
    rec = _universe_rec(fn)
    rec["width"] = None
    rec["height"] = None
    univ = _make_universe(tmp_path, [rec])
    mfst = _make_manifest(tmp_path, [])

    build_signals(univ, mfst, tmp_path, dry_run=False)
    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    assert signals[0]["aspect_ratio"] is None
    assert signals[0]["aspect_class"] == "unknown"


def test_no_manifest_record_for_filename(tmp_path: Path) -> None:
    """No manifest record → has_manifest_record=False, caption_empty=True, all keyword signals False."""
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    mfst = _make_manifest(tmp_path, [])  # empty manifest

    build_signals(univ, mfst, tmp_path, dry_run=False)
    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    s = signals[0]
    assert s["has_manifest_record"] is False
    assert s["caption_length"] == 0
    assert s["caption_empty"] is True
    assert s["caption_lure_keyword"] is False
    assert s["caption_fry_keyword"] is False
    assert s["caption_fish_part_keyword"] is False
    assert s["caption_no_fish_keyword"] is False


def test_dedup_role_cluster_keep_carried_through(tmp_path: Path) -> None:
    """dedup_role='cluster_keep' is preserved in signal record."""
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn, dedup_role="cluster_keep")])
    mfst = _make_manifest(tmp_path, [])

    build_signals(univ, mfst, tmp_path, dry_run=False)
    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    assert signals[0]["dedup_role"] == "cluster_keep"


def test_image_stats_scaffolded_null(tmp_path: Path) -> None:
    """image_stats_computed=False and stat fields are null in Phase B."""
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    mfst = _make_manifest(tmp_path, [])

    build_signals(univ, mfst, tmp_path, dry_run=False)
    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    s = signals[0]
    assert s["image_stats_computed"] is False
    assert s["mean_luminance"] is None
    assert s["is_grayscale_like"] is None
    assert s["edge_density"] is None


# ─── Error paths ─────────────────────────────────────────────────────────────


def test_missing_universe_exits(tmp_path: Path) -> None:
    """filter_universe.jsonl missing → sys.exit(1)."""
    mfst = _make_manifest(tmp_path, [])
    with pytest.raises(SystemExit) as exc:
        build_signals(tmp_path / "no_universe.jsonl", mfst, tmp_path)
    assert exc.value.code == 1


def test_corrupt_universe_raises_value_error(tmp_path: Path) -> None:
    """Corrupt JSONL in universe raises ValueError with file:line context."""
    p = tmp_path / "filter_universe.jsonl"
    p.write_text('{"ok": 1}\n{CORRUPT\n', encoding="utf-8")
    mfst = _make_manifest(tmp_path, [])

    with pytest.raises(ValueError, match="corrupt JSONL"):
        build_signals(p, mfst, tmp_path)


def test_missing_manifest_continues(tmp_path: Path) -> None:
    """manifest.jsonl missing → warning, continues; all records get has_manifest_record=False."""
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    # No manifest file created

    summary = build_signals(univ, tmp_path / "no_manifest.jsonl", tmp_path, dry_run=False)

    assert summary["total_images"] == 1
    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    assert signals[0]["has_manifest_record"] is False


# ─── Duplicate manifest filename handling ─────────────────────────────────────


def test_duplicate_manifest_filename_counted_in_summary(tmp_path: Path) -> None:
    """Duplicate manifest filenames are counted and the count appears in the summary."""
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    # Two records with the same filename → 1 duplicate
    mfst = _make_manifest(tmp_path, [
        _manifest_rec(fn, "first caption"),
        _manifest_rec(fn, "second caption"),
    ])

    summary = build_signals(univ, mfst, tmp_path, dry_run=False)

    assert summary["duplicate_manifest_filename_count"] == 1


def test_duplicate_manifest_last_wins(tmp_path: Path) -> None:
    """When a manifest filename is duplicated, the last record's caption is used."""
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    first_kw = "расписание"   # triggers caption_no_fish_keyword
    last_kw = "воблер"        # triggers caption_lure_keyword
    mfst = _make_manifest(tmp_path, [
        _manifest_rec(fn, first_kw),
        _manifest_rec(fn, last_kw),
    ])

    build_signals(univ, mfst, tmp_path, dry_run=False)
    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    s = signals[0]
    assert s["caption_lure_keyword"] is True      # last caption wins
    assert s["caption_no_fish_keyword"] is False


def test_no_duplicates_has_zero_count(tmp_path: Path) -> None:
    """When manifest has no duplicates, duplicate_manifest_filename_count == 0."""
    records = [_universe_rec(f"photos/{i}.jpg") for i in range(3)]
    univ = _make_universe(tmp_path, records)
    mfst = _make_manifest(tmp_path, [_manifest_rec(r["filename"]) for r in records])

    summary = build_signals(univ, mfst, tmp_path, dry_run=False)

    assert summary["duplicate_manifest_filename_count"] == 0


def test_missing_manifest_has_zero_duplicate_count(tmp_path: Path) -> None:
    """When manifest is absent, duplicate_manifest_filename_count == 0."""
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])

    summary = build_signals(univ, tmp_path / "no_manifest.jsonl", tmp_path, dry_run=False)

    assert summary["duplicate_manifest_filename_count"] == 0


# ─── Missing required universe fields ────────────────────────────────────────


@pytest.mark.parametrize("missing_field", ["file_size", "low_res", "dedup_role"])
def test_missing_required_universe_field_raises_value_error(
    tmp_path: Path, missing_field: str
) -> None:
    """A universe record missing a required field raises ValueError with field name in message."""
    rec = _universe_rec("photos/a.jpg")
    del rec[missing_field]
    univ = _make_universe(tmp_path, [rec])
    mfst = _make_manifest(tmp_path, [])

    with pytest.raises(ValueError, match=missing_field):
        build_signals(univ, mfst, tmp_path)


@pytest.mark.parametrize("missing_field", ["file_size", "low_res", "dedup_role"])
def test_missing_required_universe_field_exits_rc1_via_main(
    tmp_path: Path, missing_field: str
) -> None:
    """main() exits rc=1 when a universe record is missing a required field."""
    import intake_filter_heuristic as M

    rec = _universe_rec("photos/a.jpg")
    del rec[missing_field]
    univ = _make_universe(tmp_path, [rec])
    mfst = _make_manifest(tmp_path, [])

    with pytest.raises(SystemExit) as exc:
        M.main([
            "--universe", str(univ),
            "--manifest", str(mfst),
            "--output-dir", str(tmp_path),
        ])
    assert exc.value.code == 1


# ─── dry_run writes no files ─────────────────────────────────────────────────


def test_dry_run_writes_no_files(tmp_path: Path) -> None:
    """--dry-run must write no files to the output directory."""
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    mfst = _make_manifest(tmp_path, [_manifest_rec(fn, "балансир")])
    out_dir = tmp_path / "output"
    out_dir.mkdir()

    build_signals(univ, mfst, out_dir, dry_run=True)

    assert not (out_dir / "filter_signals.jsonl").exists()
    assert not (out_dir / "filter_signals_summary.json").exists()


def test_dry_run_returns_summary_dict(tmp_path: Path) -> None:
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    mfst = _make_manifest(tmp_path, [_manifest_rec(fn, "")])

    summary = build_signals(univ, mfst, tmp_path, dry_run=True)

    assert isinstance(summary, dict)
    assert summary["total_images"] == 1


# ─── Summary privacy integrity ────────────────────────────────────────────────


def test_summary_has_no_filenames(tmp_path: Path) -> None:
    """filter_signals_summary.json must not contain any filename strings."""
    fn = "photos/secret_name_abc.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    mfst = _make_manifest(tmp_path, [_manifest_rec(fn, "воблер")])

    build_signals(univ, mfst, tmp_path, dry_run=False)

    summary_text = (tmp_path / "filter_signals_summary.json").read_text(encoding="utf-8")
    assert "secret_name_abc" not in summary_text
    assert "photos/" not in summary_text


def test_summary_has_no_captions_or_senders(tmp_path: Path) -> None:
    """filter_signals_summary.json must not contain caption text, sender names, or file:// paths."""
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    mfst = _make_manifest(tmp_path, [_manifest_rec(fn, "секретная подпись")])

    build_signals(univ, mfst, tmp_path, dry_run=False)

    summary_text = (tmp_path / "filter_signals_summary.json").read_text(encoding="utf-8")
    for forbidden in ("секретная подпись", "sender", "from_id", "file://"):
        assert forbidden not in summary_text, (
            f"Summary contains forbidden value: {forbidden!r}"
        )


def test_summary_counts_all_le_total_images(tmp_path: Path) -> None:
    """All count fields in summary must be ≤ total_images."""
    records = [_universe_rec(f"photos/{i}.jpg") for i in range(5)]
    univ = _make_universe(tmp_path, records)
    mfst = _make_manifest(tmp_path, [])

    summary = build_signals(univ, mfst, tmp_path, dry_run=False)

    count_fields = [
        "no_manifest_record_count", "low_res_count", "extreme_aspect_count",
        "tiny_file_count", "caption_lure_keyword_count", "caption_fish_part_keyword_count",
        "caption_fry_keyword_count", "caption_no_fish_keyword_count", "caption_text_heavy_count",
    ]
    for field in count_fields:
        assert summary[field] <= summary["total_images"], (
            f"{field}={summary[field]} > total_images={summary['total_images']}"
        )


# ─── Pipeline invariants ──────────────────────────────────────────────────────


def test_bijective_filename_mapping(tmp_path: Path) -> None:
    """Every filename in filter_universe.jsonl appears exactly once in filter_signals.jsonl."""
    records = [_universe_rec(f"photos/{i}.jpg") for i in range(5)]
    univ = _make_universe(tmp_path, records)
    mfst = _make_manifest(tmp_path, [])

    build_signals(univ, mfst, tmp_path, dry_run=False)

    universe_fns = [r["filename"] for r in records]
    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    signal_fns = [s["filename"] for s in signals]

    assert set(universe_fns) == set(signal_fns)
    assert len(signal_fns) == len(set(signal_fns)), "Duplicate filenames in filter_signals.jsonl"


def test_total_images_matches_signal_record_count(tmp_path: Path) -> None:
    """summary.total_images == len(filter_signals.jsonl records)."""
    records = [_universe_rec(f"photos/{i}.jpg") for i in range(7)]
    univ = _make_universe(tmp_path, records)
    mfst = _make_manifest(tmp_path, [])

    summary = build_signals(univ, mfst, tmp_path, dry_run=False)

    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    assert summary["total_images"] == len(signals) == 7


def test_low_res_count_consistent(tmp_path: Path) -> None:
    """summary.low_res_count == count of records with low_res=True in signals."""
    records = [
        _universe_rec("photos/a.jpg", low_res=True),
        _universe_rec("photos/b.jpg", low_res=True),
        _universe_rec("photos/c.jpg", low_res=False),
    ]
    univ = _make_universe(tmp_path, records)
    mfst = _make_manifest(tmp_path, [])

    summary = build_signals(univ, mfst, tmp_path, dry_run=False)

    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    computed_low_res = sum(1 for s in signals if s["low_res"])
    assert summary["low_res_count"] == computed_low_res == 2


def test_extreme_aspect_count_consistent(tmp_path: Path) -> None:
    """summary.extreme_aspect_count == count of extreme_portrait/extreme_landscape in signals."""
    records = [
        _universe_rec("photos/ep.jpg", width=200, height=1000),   # extreme_portrait
        _universe_rec("photos/el.jpg", width=1000, height=200),   # extreme_landscape
        _universe_rec("photos/ls.jpg", width=1280, height=960),   # landscape
    ]
    univ = _make_universe(tmp_path, records)
    mfst = _make_manifest(tmp_path, [])

    summary = build_signals(univ, mfst, tmp_path, dry_run=False)

    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    computed = sum(1 for s in signals if s["aspect_class"] in {"extreme_portrait", "extreme_landscape"})
    assert summary["extreme_aspect_count"] == computed == 2


def test_tiny_file_count_consistent(tmp_path: Path) -> None:
    """summary.tiny_file_count == count of file_size_bucket=='tiny' in signals."""
    records = [
        _universe_rec("photos/tiny.jpg", file_size=20_000),
        _universe_rec("photos/small.jpg", file_size=80_000),
    ]
    univ = _make_universe(tmp_path, records)
    mfst = _make_manifest(tmp_path, [])

    summary = build_signals(univ, mfst, tmp_path, dry_run=False)

    signals = [json.loads(l) for l in (tmp_path / "filter_signals.jsonl").read_text().splitlines() if l.strip()]
    computed = sum(1 for s in signals if s["file_size_bucket"] == "tiny")
    assert summary["tiny_file_count"] == computed == 1


def test_no_consistency_ok_in_signals_summary(tmp_path: Path) -> None:
    """filter_signals_summary.json must not contain 'consistency_ok' (that belongs in universe summary)."""
    fn = "photos/a.jpg"
    univ = _make_universe(tmp_path, [_universe_rec(fn)])
    mfst = _make_manifest(tmp_path, [])

    build_signals(univ, mfst, tmp_path, dry_run=False)

    summary = json.loads((tmp_path / "filter_signals_summary.json").read_text())
    assert "consistency_ok" not in summary


# ─── Real-data integration (skipif data absent) ──────────────────────────────


UNIVERSE_PATH = REPO_ROOT / "data/intake_meta/tg_2026-04-24/filter_universe.jsonl"
MANIFEST_PATH = REPO_ROOT / "data/intake_meta/tg_2026-04-24/manifest.jsonl"

_real_data_available = UNIVERSE_PATH.exists()


@pytest.mark.skipif(not _real_data_available, reason="Real filter_universe.jsonl not available")
def test_real_total_images_equals_32420(tmp_path: Path) -> None:
    """Integration: real pipeline must produce exactly 32,420 signal records."""
    summary = build_signals(UNIVERSE_PATH, MANIFEST_PATH, tmp_path, dry_run=False)
    assert summary["total_images"] == 32420, (
        f"Expected 32420, got {summary['total_images']}"
    )


@pytest.mark.skipif(not _real_data_available, reason="Real filter_universe.jsonl not available")
def test_real_low_res_count_equals_371(tmp_path: Path) -> None:
    summary = build_signals(UNIVERSE_PATH, MANIFEST_PATH, tmp_path, dry_run=False)
    assert summary["low_res_count"] == 371, (
        f"Expected 371, got {summary['low_res_count']}"
    )


@pytest.mark.skipif(not _real_data_available, reason="Real filter_universe.jsonl not available")
def test_real_extreme_aspect_count_equals_281(tmp_path: Path) -> None:
    summary = build_signals(UNIVERSE_PATH, MANIFEST_PATH, tmp_path, dry_run=False)
    assert summary["extreme_aspect_count"] == 281, (
        f"Expected 281, got {summary['extreme_aspect_count']}"
    )


@pytest.mark.skipif(not _real_data_available, reason="Real filter_universe.jsonl not available")
def test_real_tiny_file_count_equals_137(tmp_path: Path) -> None:
    summary = build_signals(UNIVERSE_PATH, MANIFEST_PATH, tmp_path, dry_run=False)
    assert summary["tiny_file_count"] == 137, (
        f"Expected 137, got {summary['tiny_file_count']}"
    )


@pytest.mark.skipif(not _real_data_available, reason="Real filter_universe.jsonl not available")
def test_real_no_consistency_ok_field(tmp_path: Path) -> None:
    """filter_signals_summary.json must not have consistency_ok (belongs in universe summary)."""
    summary = build_signals(UNIVERSE_PATH, MANIFEST_PATH, tmp_path, dry_run=False)
    assert "consistency_ok" not in summary
