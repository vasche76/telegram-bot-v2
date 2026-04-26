"""
test_intake_filter_universe.py — Tests for intake_filter_universe.py (U4 Phase A).

Covers:
- Safety gate: refuses provisional dedup
- Safety gate: refuses incomplete review
- Happy path: unique record derivation
- Duplicate removal correctness
- Cluster keep representative preserved
- Summary has NO filenames, captions, senders, file:// paths
- Count consistency: unique_records == 32420 (real data, if available)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from intake_filter_universe import build_universe


# ─── Fixtures ────────────────────────────────────────────────────────────────


def _make_dedup_summary(
    tmp_path: Path,
    provisional: bool = False,
    manual_review_required: bool = False,
    total_unique: int = 3,
) -> Path:
    p = tmp_path / "dedup_summary.json"
    p.write_text(
        json.dumps({
            "provisional": provisional,
            "manual_review_required": manual_review_required,
            "total_unique_after_dedup": total_unique,
        }),
        encoding="utf-8",
    )
    return p


def _make_review_summary(tmp_path: Path, review_complete: bool = True) -> Path:
    p = tmp_path / "dedup_review_summary.json"
    p.write_text(
        json.dumps({"review_complete": review_complete}),
        encoding="utf-8",
    )
    return p


def _make_clusters(tmp_path: Path, clusters: list[dict]) -> Path:
    p = tmp_path / "dedup_clusters.jsonl"
    lines = [json.dumps(c) for c in clusters]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _make_audit(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "audit.jsonl"
    lines = [json.dumps(r) for r in records]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def _audit_rec(filename: str, corrupt: bool = False, low_res: bool = False) -> dict:
    return {
        "filename": filename,
        "sha256": "abc123",
        "width": 1280,
        "height": 960,
        "max_side": 1280,
        "file_size": 300000,
        "low_res": low_res,
        "corrupt": corrupt,
    }


# ─── Safety gate tests ────────────────────────────────────────────────────────


def test_refuse_provisional_dedup(tmp_path: Path) -> None:
    """Hard fail if dedup_summary.provisional is True."""
    ds = _make_dedup_summary(tmp_path, provisional=True)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [])
    au = _make_audit(tmp_path, [_audit_rec("photos/a.jpg")])

    with pytest.raises(SystemExit) as exc:
        build_universe(cl, au, tmp_path, ds, rs)
    assert exc.value.code == 1


def test_refuse_manual_review_required(tmp_path: Path) -> None:
    """Hard fail if dedup_summary.manual_review_required is True."""
    ds = _make_dedup_summary(tmp_path, manual_review_required=True)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [])
    au = _make_audit(tmp_path, [_audit_rec("photos/a.jpg")])

    with pytest.raises(SystemExit) as exc:
        build_universe(cl, au, tmp_path, ds, rs)
    assert exc.value.code == 1


def test_refuse_incomplete_review(tmp_path: Path) -> None:
    """Hard fail if dedup_review_summary.review_complete is False."""
    ds = _make_dedup_summary(tmp_path)
    rs = _make_review_summary(tmp_path, review_complete=False)
    cl = _make_clusters(tmp_path, [])
    au = _make_audit(tmp_path, [_audit_rec("photos/a.jpg")])

    with pytest.raises(SystemExit) as exc:
        build_universe(cl, au, tmp_path, ds, rs)
    assert exc.value.code == 1


# ─── Missing file error paths ─────────────────────────────────────────────────


def test_missing_audit_exits(tmp_path: Path) -> None:
    """Exit 1 when audit.jsonl is missing."""
    ds = _make_dedup_summary(tmp_path)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [])
    # No audit file created

    with pytest.raises(SystemExit) as exc:
        build_universe(cl, tmp_path / "no_audit.jsonl", tmp_path, ds, rs)
    assert exc.value.code == 1


def test_missing_clusters_exits(tmp_path: Path) -> None:
    """Exit 1 when dedup_clusters.jsonl is missing."""
    ds = _make_dedup_summary(tmp_path)
    rs = _make_review_summary(tmp_path)
    au = _make_audit(tmp_path, [_audit_rec("photos/a.jpg")])

    with pytest.raises(SystemExit) as exc:
        build_universe(tmp_path / "no_clusters.jsonl", au, tmp_path, ds, rs)
    assert exc.value.code == 1


# ─── Happy path tests ─────────────────────────────────────────────────────────


def test_all_non_corrupt_non_duplicate_included(tmp_path: Path) -> None:
    """All audit records not in non_keeps and not corrupt appear in the universe."""
    ds = _make_dedup_summary(tmp_path, total_unique=3)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [])  # no clusters → no non-keeps
    records = [_audit_rec(f"photos/{i}.jpg") for i in range(3)]
    au = _make_audit(tmp_path, records)

    summary = build_universe(cl, au, tmp_path, ds, rs, dry_run=False)

    assert summary["unique_records"] == 3
    assert summary["dedup_duplicate_removed"] == 0
    assert summary["corrupt_excluded"] == 0
    assert (tmp_path / "filter_universe.jsonl").exists()


def test_arithmetic_identity(tmp_path: Path) -> None:
    """unique_records + dedup_duplicate_removed + corrupt_excluded == input_records."""
    ds = _make_dedup_summary(tmp_path, total_unique=2)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [{
        "cluster_id": 1,
        "keep_filename": "photos/keep.jpg",
        "duplicate_filenames": ["photos/dup.jpg"],
    }])
    records = [
        _audit_rec("photos/keep.jpg"),
        _audit_rec("photos/dup.jpg"),
        _audit_rec("photos/corrupt.jpg", corrupt=True),
        _audit_rec("photos/singleton.jpg"),
    ]
    # total=4, non_keep=1, corrupt=1 (not non-keep), unique=2
    au = _make_audit(tmp_path, records)

    summary = build_universe(cl, au, tmp_path, ds, rs)

    assert summary["input_records"] == 4
    assert summary["dedup_duplicate_removed"] == 1
    assert summary["corrupt_excluded"] == 1
    assert summary["unique_records"] == 2
    total = summary["unique_records"] + summary["dedup_duplicate_removed"] + summary["corrupt_excluded"]
    assert total == summary["input_records"]


def test_duplicate_removed(tmp_path: Path) -> None:
    """Filenames listed in duplicate_filenames are excluded from the universe."""
    ds = _make_dedup_summary(tmp_path, total_unique=1)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [{
        "cluster_id": 1,
        "keep_filename": "photos/keep.jpg",
        "duplicate_filenames": ["photos/dup1.jpg", "photos/dup2.jpg"],
    }])
    records = [
        _audit_rec("photos/keep.jpg"),
        _audit_rec("photos/dup1.jpg"),
        _audit_rec("photos/dup2.jpg"),
    ]
    au = _make_audit(tmp_path, records)

    summary = build_universe(cl, au, tmp_path, ds, rs)

    assert summary["unique_records"] == 1
    assert summary["dedup_duplicate_removed"] == 2

    universe = [
        json.loads(l)
        for l in (tmp_path / "filter_universe.jsonl").read_text().splitlines()
        if l.strip()
    ]
    filenames = {r["filename"] for r in universe}
    assert "photos/keep.jpg" in filenames
    assert "photos/dup1.jpg" not in filenames
    assert "photos/dup2.jpg" not in filenames


def test_cluster_keep_is_included(tmp_path: Path) -> None:
    """The keep_filename of a cluster is included in the universe with dedup_role=cluster_keep."""
    ds = _make_dedup_summary(tmp_path, total_unique=1)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [{
        "cluster_id": 1,
        "keep_filename": "photos/keep.jpg",
        "duplicate_filenames": ["photos/dup.jpg"],
    }])
    records = [_audit_rec("photos/keep.jpg"), _audit_rec("photos/dup.jpg")]
    au = _make_audit(tmp_path, records)

    build_universe(cl, au, tmp_path, ds, rs)

    universe = [
        json.loads(l)
        for l in (tmp_path / "filter_universe.jsonl").read_text().splitlines()
        if l.strip()
    ]
    keeps = [r for r in universe if r["filename"] == "photos/keep.jpg"]
    assert len(keeps) == 1
    assert keeps[0]["dedup_role"] == "cluster_keep"


def test_singleton_dedup_role(tmp_path: Path) -> None:
    """Filenames not in any cluster get dedup_role=unique."""
    ds = _make_dedup_summary(tmp_path, total_unique=1)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [])  # no clusters
    records = [_audit_rec("photos/solo.jpg")]
    au = _make_audit(tmp_path, records)

    build_universe(cl, au, tmp_path, ds, rs)

    universe = [
        json.loads(l)
        for l in (tmp_path / "filter_universe.jsonl").read_text().splitlines()
        if l.strip()
    ]
    assert universe[0]["dedup_role"] == "unique"


def test_corrupt_cluster_keep_excluded(tmp_path: Path) -> None:
    """A corrupt record that is a cluster keep is excluded from the universe (corrupt wins)."""
    # corrupt=True non-keep → counted as non_keep
    # corrupt=True keep → counted as corrupt_excluded
    # Use dry_run=True to avoid the zero-universe hard fail.
    ds = _make_dedup_summary(tmp_path, total_unique=0)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [{
        "cluster_id": 1,
        "keep_filename": "photos/corrupt_keep.jpg",
        "duplicate_filenames": ["photos/dup.jpg"],
    }])
    records = [
        _audit_rec("photos/corrupt_keep.jpg", corrupt=True),
        _audit_rec("photos/dup.jpg"),  # non-keep
    ]
    au = _make_audit(tmp_path, records)

    summary = build_universe(cl, au, tmp_path, ds, rs, dry_run=True)

    assert summary["unique_records"] == 0
    assert summary["dedup_duplicate_removed"] == 1  # dup.jpg → non-keep
    assert summary["corrupt_excluded"] == 1          # corrupt_keep.jpg → corrupt (not in non_keeps)


def test_empty_clusters_all_singletons(tmp_path: Path) -> None:
    """Empty clusters file → all non-corrupt audit records enter universe as unique."""
    ds = _make_dedup_summary(tmp_path, total_unique=2)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [])
    records = [_audit_rec("photos/a.jpg"), _audit_rec("photos/b.jpg")]
    au = _make_audit(tmp_path, records)

    summary = build_universe(cl, au, tmp_path, ds, rs)

    assert summary["unique_records"] == 2
    assert summary["dedup_duplicate_removed"] == 0


def test_empty_duplicate_filenames_no_exclusion(tmp_path: Path) -> None:
    """Cluster with empty duplicate_filenames list excludes nothing."""
    ds = _make_dedup_summary(tmp_path, total_unique=1)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [{
        "cluster_id": 1,
        "keep_filename": "photos/keep.jpg",
        "duplicate_filenames": [],
    }])
    records = [_audit_rec("photos/keep.jpg")]
    au = _make_audit(tmp_path, records)

    summary = build_universe(cl, au, tmp_path, ds, rs)

    assert summary["unique_records"] == 1
    assert summary["dedup_duplicate_removed"] == 0


# ─── Summary privacy checks ───────────────────────────────────────────────────


def test_summary_has_no_filenames(tmp_path: Path) -> None:
    """filter_universe_summary.json must not contain any filename strings."""
    ds = _make_dedup_summary(tmp_path, total_unique=1)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [])
    au = _make_audit(tmp_path, [_audit_rec("photos/secret_name.jpg")])

    build_universe(cl, au, tmp_path, ds, rs)

    summary_text = (tmp_path / "filter_universe_summary.json").read_text(encoding="utf-8")
    # Must not contain any photo filename
    assert "secret_name" not in summary_text
    assert "photos/" not in summary_text


def test_summary_has_no_captions_or_senders(tmp_path: Path) -> None:
    """Summary must not contain caption or sender fields."""
    ds = _make_dedup_summary(tmp_path, total_unique=1)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [])
    au = _make_audit(tmp_path, [_audit_rec("photos/a.jpg")])

    build_universe(cl, au, tmp_path, ds, rs)

    summary_text = (tmp_path / "filter_universe_summary.json").read_text(encoding="utf-8")
    for forbidden_key in ("caption", "sender", "from_id", "file://"):
        assert forbidden_key not in summary_text, (
            f"Summary contains forbidden key/value: {forbidden_key!r}"
        )


def test_dry_run_writes_no_files(tmp_path: Path) -> None:
    """--dry-run must not write any files to output_dir."""
    ds = _make_dedup_summary(tmp_path, total_unique=1)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [])
    au = _make_audit(tmp_path, [_audit_rec("photos/a.jpg")])

    out_dir = tmp_path / "output"
    out_dir.mkdir()

    build_universe(cl, au, out_dir, ds, rs, dry_run=True)

    assert not (out_dir / "filter_universe.jsonl").exists()
    assert not (out_dir / "filter_universe_summary.json").exists()


# ─── New hardening tests ──────────────────────────────────────────────────────


def test_no_duplicate_filenames_in_universe(tmp_path: Path) -> None:
    """filter_universe.jsonl must not contain any filename more than once."""
    ds = _make_dedup_summary(tmp_path, total_unique=2)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [{
        "cluster_id": 1,
        "keep_filename": "photos/keep.jpg",
        "duplicate_filenames": ["photos/dup.jpg"],
    }])
    records = [_audit_rec("photos/keep.jpg"), _audit_rec("photos/dup.jpg"), _audit_rec("photos/solo.jpg")]
    au = _make_audit(tmp_path, records)

    build_universe(cl, au, tmp_path, ds, rs)

    lines = [
        l for l in (tmp_path / "filter_universe.jsonl").read_text().splitlines() if l.strip()
    ]
    filenames = [json.loads(l)["filename"] for l in lines]
    assert len(filenames) == len(set(filenames)), (
        f"Duplicate filenames in universe: {[f for f in filenames if filenames.count(f) > 1]}"
    )


def test_missing_keep_in_audit_exits(tmp_path: Path) -> None:
    """Hard fail when a cluster keep_filename is absent from audit."""
    ds = _make_dedup_summary(tmp_path, total_unique=1)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [{
        "cluster_id": 1,
        "keep_filename": "photos/ghost.jpg",  # not in audit
        "duplicate_filenames": [],
    }])
    au = _make_audit(tmp_path, [_audit_rec("photos/a.jpg")])

    with pytest.raises(SystemExit) as exc:
        build_universe(cl, au, tmp_path, ds, rs)
    assert exc.value.code == 1


def test_missing_dup_in_audit_exits(tmp_path: Path) -> None:
    """Hard fail when a cluster duplicate_filename is absent from audit."""
    ds = _make_dedup_summary(tmp_path, total_unique=1)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [{
        "cluster_id": 1,
        "keep_filename": "photos/keep.jpg",
        "duplicate_filenames": ["photos/phantom.jpg"],  # not in audit
    }])
    au = _make_audit(tmp_path, [_audit_rec("photos/keep.jpg")])

    with pytest.raises(SystemExit) as exc:
        build_universe(cl, au, tmp_path, ds, rs)
    assert exc.value.code == 1


def test_missing_in_audit_zero_when_all_present(tmp_path: Path) -> None:
    """missing_in_audit is 0 when all cluster filenames exist in audit."""
    ds = _make_dedup_summary(tmp_path, total_unique=1)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [{
        "cluster_id": 1,
        "keep_filename": "photos/keep.jpg",
        "duplicate_filenames": ["photos/dup.jpg"],
    }])
    au = _make_audit(tmp_path, [_audit_rec("photos/keep.jpg"), _audit_rec("photos/dup.jpg")])

    summary = build_universe(cl, au, tmp_path, ds, rs)

    assert summary["missing_in_audit"] == 0


def test_universe_filename_set_sha256_present_and_deterministic(tmp_path: Path) -> None:
    """summary contains universe_filename_set_sha256 and two identical runs produce the same hash."""
    ds = _make_dedup_summary(tmp_path, total_unique=2)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [])
    records = [_audit_rec("photos/a.jpg"), _audit_rec("photos/b.jpg")]
    au = _make_audit(tmp_path, records)

    s1 = build_universe(cl, au, tmp_path / "run1", ds, rs)
    s2 = build_universe(cl, au, tmp_path / "run2", ds, rs)

    assert "universe_filename_set_sha256" in s1
    assert len(s1["universe_filename_set_sha256"]) == 64  # SHA-256 hex digest
    assert s1["universe_filename_set_sha256"] == s2["universe_filename_set_sha256"]


def test_zero_unique_real_run_exits(tmp_path: Path) -> None:
    """Hard fail when real run would produce an empty universe (unique_records == 0)."""
    # corrupt keep (excluded as corrupt) + non-keep dup (excluded as duplicate) → 0 unique records
    ds = _make_dedup_summary(tmp_path, total_unique=0)
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [{
        "cluster_id": 1,
        "keep_filename": "photos/keep.jpg",
        "duplicate_filenames": ["photos/dup.jpg"],
    }])
    au = _make_audit(tmp_path, [
        _audit_rec("photos/keep.jpg", corrupt=True),
        _audit_rec("photos/dup.jpg"),
    ])

    with pytest.raises(SystemExit) as exc:
        build_universe(cl, au, tmp_path, ds, rs, dry_run=False)
    assert exc.value.code == 1


def test_consistency_mismatch_exits_non_dry_run(tmp_path: Path) -> None:
    """Non-dry-run hard fails when unique_records != expected_unique_records_from_dedup."""
    ds = _make_dedup_summary(tmp_path, total_unique=5)  # expect 5
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [])
    records = [_audit_rec(f"photos/{i}.jpg") for i in range(3)]  # only 3 unique
    au = _make_audit(tmp_path, records)

    with pytest.raises(SystemExit) as exc:
        build_universe(cl, au, tmp_path, ds, rs, dry_run=False)
    assert exc.value.code == 1


def test_consistency_mismatch_dry_run_continues(tmp_path: Path) -> None:
    """Dry-run does NOT exit on consistency mismatch — returns summary with consistency_ok=False."""
    ds = _make_dedup_summary(tmp_path, total_unique=5)  # expect 5
    rs = _make_review_summary(tmp_path)
    cl = _make_clusters(tmp_path, [])
    records = [_audit_rec(f"photos/{i}.jpg") for i in range(3)]  # only 3 unique
    au = _make_audit(tmp_path, records)

    summary = build_universe(cl, au, tmp_path, ds, rs, dry_run=True)

    assert summary["consistency_ok"] is False
    assert summary["unique_records"] == 3
    assert summary["expected_unique_records_from_dedup"] == 5


def test_corrupt_clusters_jsonl_raises_value_error(tmp_path: Path) -> None:
    """Corrupt JSONL in clusters raises ValueError with file:line context."""
    ds = _make_dedup_summary(tmp_path)
    rs = _make_review_summary(tmp_path)
    cl = tmp_path / "dedup_clusters.jsonl"
    cl.write_text('{"ok": 1}\n{CORRUPT\n', encoding="utf-8")
    au = _make_audit(tmp_path, [_audit_rec("photos/a.jpg")])

    with pytest.raises(ValueError, match="corrupt JSONL"):
        build_universe(cl, au, tmp_path, ds, rs)


# ─── Real-data count consistency (skipped if data absent) ─────────────────────


DEDUP_SUMMARY_PATH = REPO_ROOT / "data/intake_meta/tg_2026-04-24/dedup_summary.json"
REVIEW_SUMMARY_PATH = REPO_ROOT / "data/intake_meta/tg_2026-04-24/dedup_review_summary.json"
CLUSTERS_PATH = REPO_ROOT / "data/intake_meta/tg_2026-04-24/dedup_clusters.jsonl"
AUDIT_PATH = REPO_ROOT / "data/intake_meta/tg_2026-04-24/audit.jsonl"

_real_data_available = all(
    p.exists() for p in [DEDUP_SUMMARY_PATH, REVIEW_SUMMARY_PATH, CLUSTERS_PATH, AUDIT_PATH]
)


@pytest.mark.skipif(not _real_data_available, reason="Real intake data not available")
def test_real_unique_count_equals_32420(tmp_path: Path) -> None:
    """
    Integration: real pipeline must produce exactly 32,420 unique records.

    This is the post-dedup count from U3 (total_unique_after_dedup in dedup_summary.json).
    """
    summary = build_universe(
        clusters_path=CLUSTERS_PATH,
        audit_path=AUDIT_PATH,
        output_dir=tmp_path,
        dedup_summary_path=DEDUP_SUMMARY_PATH,
        review_summary_path=REVIEW_SUMMARY_PATH,
        dry_run=False,
    )

    assert summary["unique_records"] == 32420, (
        f"Expected 32420 unique records, got {summary['unique_records']}. "
        f"Summary: {json.dumps(summary, indent=2)}"
    )
    assert summary["consistency_ok"] is True
    assert summary["expected_unique_records_from_dedup"] == 32420
