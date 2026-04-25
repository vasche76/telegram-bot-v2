"""
test_intake_review_boundary.py — Tests for scripts/intake_review_boundary.py (U2).

Tests cover:
- Happy path HTML generation from synthetic cluster data
- Multi-member cluster MIXED option presence
- --sample flag reproducibility and count
- --unsure-from filtering
- file:// URI percent-encoding for paths with spaces
- Missing image graceful handling
- Non-boundary records excluded
- Empty filter result
- Error path for missing --clusters file
- Exact clusters never appear in output
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from intake_review_boundary import (  # noqa: E402
    _file_uri,
    _filter_boundary,
    _load_clusters,
    main,
    SAMPLE_SEED,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _write_clusters(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _make_perceptual(cid: int, hamming: int, n_dups: int = 1) -> dict:
    return {
        "cluster_id": cid,
        "cluster_type": "perceptual",
        "keep_filename": f"photos/keep_{cid}.jpg",
        "duplicate_filenames": [f"photos/dup_{cid}_{i}.jpg" for i in range(n_dups)],
        "hamming_distance": hamming,
        "reason": f"phash_hamming<={hamming}",
    }


def _make_exact(cid: int) -> dict:
    return {
        "cluster_id": cid,
        "cluster_type": "exact",
        "keep_filename": f"photos/keep_{cid}.jpg",
        "duplicate_filenames": [f"photos/dup_{cid}_0.jpg"],
        "hamming_distance": None,
        "reason": "sha256=abc123",
    }


# ─── _file_uri tests ──────────────────────────────────────────────────────────


def test_file_uri_encodes_spaces() -> None:
    p = Path("/Users/imac/Downloads/Telegram Desktop/ChatExport/photos/test.jpg")
    uri = _file_uri(p)
    assert "%20" in uri, f"Expected %20 in URI for path with space, got: {uri}"
    assert "Telegram%20Desktop" in uri


def test_file_uri_no_double_slash() -> None:
    p = Path("/Users/imac/photos/fish.jpg")
    uri = _file_uri(p)
    assert uri.startswith("file:///Users/"), f"Unexpected URI prefix: {uri}"
    assert "//" not in uri[7:], f"Double-slash after file:// in: {uri}"


# ─── _filter_boundary tests ──────────────────────────────────────────────────


def test_filter_boundary_keeps_hamming_8_only() -> None:
    clusters = [
        _make_perceptual(1, 5),
        _make_perceptual(2, 8),
        _make_perceptual(3, 8),
        _make_exact(4),
    ]
    result = _filter_boundary(clusters, 8, 8)
    assert len(result) == 2
    assert all(c["hamming_distance"] == 8 for c in result)


def test_filter_boundary_excludes_exact() -> None:
    clusters = [_make_exact(1), _make_perceptual(2, 8)]
    result = _filter_boundary(clusters, 8, 8)
    assert len(result) == 1
    assert result[0]["cluster_id"] == 2


def test_filter_boundary_range() -> None:
    clusters = [_make_perceptual(i, i) for i in range(10)]
    result = _filter_boundary(clusters, 4, 6)
    assert {c["hamming_distance"] for c in result} == {4, 5, 6}


# ─── main() HTML generation tests ─────────────────────────────────────────────


def test_basic_html_generation(tmp_path: Path) -> None:
    clusters_file = tmp_path / "dedup_clusters.jsonl"
    _write_clusters(clusters_file, [
        _make_perceptual(1, 8),
        _make_perceptual(2, 8),
        _make_perceptual(3, 8),
    ])
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    output = tmp_path / "review" / "test.html"

    rc = main([
        "--clusters", str(clusters_file),
        "--export-dir", str(export_dir),
        "--output", str(output),
    ])
    assert rc == 0
    assert output.exists()
    html_text = output.read_text(encoding="utf-8")
    assert "KEEP_DEDUP" in html_text
    assert "FALSE_POSITIVE" in html_text
    assert "UNSURE" in html_text
    assert "keep_1.jpg" in html_text
    assert "file://" in html_text
    # T-01: verify exactly 3 cluster divs rendered (not just keyword presence)
    assert html_text.count('data-cluster-id=') == 3, (
        "Expected exactly 3 cluster divs for 3 input clusters"
    )


def test_multi_member_shows_mixed_option(tmp_path: Path) -> None:
    clusters_file = tmp_path / "dedup_clusters.jsonl"
    _write_clusters(clusters_file, [
        _make_perceptual(1, 8, n_dups=3),  # multi-member
        _make_perceptual(2, 8, n_dups=1),  # single-dup
    ])
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    output = tmp_path / "review" / "test.html"

    rc = main(["--clusters", str(clusters_file), "--export-dir", str(export_dir), "--output", str(output)])
    assert rc == 0
    html = output.read_text(encoding="utf-8")
    assert "MIXED" in html, "MIXED option should appear for multi-member cluster"
    assert "multi-member" in html


def test_single_dup_no_mixed_option(tmp_path: Path) -> None:
    clusters_file = tmp_path / "dedup_clusters.jsonl"
    _write_clusters(clusters_file, [_make_perceptual(1, 8, n_dups=1)])
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    output = tmp_path / "test.html"

    rc = main(["--clusters", str(clusters_file), "--export-dir", str(export_dir), "--output", str(output)])
    assert rc == 0
    html = output.read_text(encoding="utf-8")
    assert "MIXED" not in html, "MIXED should not appear for single-dup cluster"


def test_sample_count(tmp_path: Path) -> None:
    clusters_file = tmp_path / "dedup_clusters.jsonl"
    _write_clusters(clusters_file, [_make_perceptual(i, 8) for i in range(1, 11)])
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    output = tmp_path / "sample.html"

    rc = main([
        "--clusters", str(clusters_file),
        "--export-dir", str(export_dir),
        "--output", str(output),
        "--sample", "3",
    ])
    assert rc == 0
    html = output.read_text(encoding="utf-8")
    # Exactly 3 cluster divs
    assert html.count('data-cluster-id=') == 3
    assert "SAMPLE MODE" in html


def test_sample_reproducibility(tmp_path: Path) -> None:
    clusters_file = tmp_path / "dedup_clusters.jsonl"
    _write_clusters(clusters_file, [_make_perceptual(i, 8) for i in range(1, 21)])
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    out1 = tmp_path / "run1.html"
    out2 = tmp_path / "run2.html"
    main(["--clusters", str(clusters_file), "--export-dir", str(export_dir), "--output", str(out1), "--sample", "5"])
    main(["--clusters", str(clusters_file), "--export-dir", str(export_dir), "--output", str(out2), "--sample", "5"])
    assert out1.read_text() == out2.read_text(), "Sampling must be deterministic with fixed seed"


def test_missing_image_renders_missing_placeholder(tmp_path: Path) -> None:
    clusters_file = tmp_path / "dedup_clusters.jsonl"
    _write_clusters(clusters_file, [{
        "cluster_id": 1,
        "cluster_type": "perceptual",
        "keep_filename": "photos/nonexistent.jpg",
        "duplicate_filenames": ["photos/also_missing.jpg"],
        "hamming_distance": 8,
        "reason": "phash_hamming<=8",
    }])
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    output = tmp_path / "test.html"

    rc = main(["--clusters", str(clusters_file), "--export-dir", str(export_dir), "--output", str(output)])
    assert rc == 0
    html = output.read_text(encoding="utf-8")
    assert "img-missing" in html, "Missing image should render placeholder, not <img>"
    assert "<img " not in html or "file://" not in html.split("img-missing")[0].split("<img ")[-1]


def test_non_boundary_excluded(tmp_path: Path) -> None:
    clusters_file = tmp_path / "dedup_clusters.jsonl"
    _write_clusters(clusters_file, [
        _make_perceptual(1, 5),
        _make_perceptual(2, 8),
    ])
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    output = tmp_path / "test.html"

    rc = main(["--clusters", str(clusters_file), "--export-dir", str(export_dir), "--output", str(output)])
    assert rc == 0
    html = output.read_text(encoding="utf-8")
    assert 'data-cluster-id="1"' not in html, "hamming=5 cluster should be excluded"
    assert 'data-cluster-id="2"' in html


def test_empty_result_no_crash(tmp_path: Path) -> None:
    clusters_file = tmp_path / "dedup_clusters.jsonl"
    _write_clusters(clusters_file, [_make_perceptual(1, 4)])  # no hamming=8
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    output = tmp_path / "test.html"

    rc = main(["--clusters", str(clusters_file), "--export-dir", str(export_dir), "--output", str(output)])
    assert rc == 0
    html = output.read_text(encoding="utf-8")
    assert "No clusters to review" in html


def test_missing_clusters_file_exits_nonzero(tmp_path: Path) -> None:
    rc = main([
        "--clusters", str(tmp_path / "nonexistent.jsonl"),
        "--export-dir", str(tmp_path),
    ])
    assert rc != 0


def test_exact_clusters_excluded(tmp_path: Path) -> None:
    clusters_file = tmp_path / "dedup_clusters.jsonl"
    _write_clusters(clusters_file, [
        _make_exact(1),
        _make_perceptual(2, 8),
    ])
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    output = tmp_path / "test.html"

    rc = main([
        "--clusters", str(clusters_file),
        "--export-dir", str(export_dir),
        "--output", str(output),
        "--hamming-min", "0",
        "--hamming-max", "8",
    ])
    assert rc == 0
    html = output.read_text(encoding="utf-8")
    assert 'data-cluster-id="1"' not in html, "Exact cluster should never appear regardless of hamming filter"
    assert 'data-cluster-id="2"' in html


def test_unsure_from_filter(tmp_path: Path) -> None:
    clusters_file = tmp_path / "dedup_clusters.jsonl"
    _write_clusters(clusters_file, [_make_perceptual(i, 8) for i in range(1, 6)])

    decisions_file = tmp_path / "decisions.json"
    decisions_file.write_text(json.dumps({
        "schema_version": 1,
        "decisions": [
            {"cluster_id": 2, "decision": "UNSURE"},
            {"cluster_id": 4, "decision": "KEEP_DEDUP"},
        ],
    }), encoding="utf-8")

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    output = tmp_path / "unsure.html"

    rc = main([
        "--clusters", str(clusters_file),
        "--export-dir", str(export_dir),
        "--output", str(output),
        "--unsure-from", str(decisions_file),
    ])
    assert rc == 0
    html = output.read_text(encoding="utf-8")
    assert 'data-cluster-id="2"' in html
    assert 'data-cluster-id="4"' not in html
    assert 'data-cluster-id="1"' not in html
    assert "UNSURE RE-PASS" in html


def test_unsure_from_no_unsure_entries(tmp_path: Path) -> None:
    # T-02: --unsure-from with a decisions file that has no UNSURE entries
    # → boundary list becomes empty, HTML shows "No clusters to review"
    clusters_file = tmp_path / "clusters.jsonl"
    _write_clusters(clusters_file, [_make_perceptual(i, 8) for i in range(1, 4)])

    decisions_file = tmp_path / "decisions.json"
    decisions_file.write_text(json.dumps({
        "schema_version": 1,
        "decisions": [
            {"cluster_id": 1, "decision": "KEEP_DEDUP"},
            {"cluster_id": 2, "decision": "FALSE_POSITIVE"},
            {"cluster_id": 3, "decision": "KEEP_DEDUP"},
        ],
    }), encoding="utf-8")

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    output = tmp_path / "unsure_empty.html"

    rc = main([
        "--clusters", str(clusters_file),
        "--export-dir", str(export_dir),
        "--output", str(output),
        "--unsure-from", str(decisions_file),
    ])
    assert rc == 0
    html_text = output.read_text(encoding="utf-8")
    assert "No clusters to review" in html_text
    assert "data-cluster-id=" not in html_text
    assert "UNSURE RE-PASS" in html_text  # notice still shown even when empty


def test_sample_ignored_when_unsure_from_used(tmp_path: Path) -> None:
    # T-03: --sample is ignored when --unsure-from is active.
    # All UNSURE clusters should appear; no SAMPLE MODE notice.
    clusters_file = tmp_path / "clusters.jsonl"
    _write_clusters(clusters_file, [_make_perceptual(i, 8) for i in range(1, 11)])

    decisions_file = tmp_path / "decisions.json"
    decisions_file.write_text(json.dumps({
        "schema_version": 1,
        "decisions": [
            {"cluster_id": i, "decision": "UNSURE"} for i in range(1, 4)
        ],
    }), encoding="utf-8")

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    output = tmp_path / "unsure_sample.html"

    rc = main([
        "--clusters", str(clusters_file),
        "--export-dir", str(export_dir),
        "--output", str(output),
        "--unsure-from", str(decisions_file),
        "--sample", "1",  # should be ignored
    ])
    assert rc == 0
    html_text = output.read_text(encoding="utf-8")
    # All 3 UNSURE clusters shown, not just 1
    assert html_text.count("data-cluster-id=") == 3, (
        "Expected 3 UNSURE clusters; --sample should be ignored with --unsure-from"
    )
    assert "SAMPLE MODE" not in html_text
    assert "UNSURE RE-PASS" in html_text


def test_file_uri_with_space_in_export_dir(tmp_path: Path) -> None:
    # Export dir contains a space ("Telegram Desktop") — path must be percent-encoded in URIs
    space_dir = tmp_path / "Telegram Desktop" / "export"
    photos_dir = space_dir / "photos"
    photos_dir.mkdir(parents=True)
    # Create a real (dummy) file so _file_uri is called instead of rendering img-missing
    (photos_dir / "keep_1.jpg").write_bytes(b"dummy")

    clusters_file = tmp_path / "clusters.jsonl"
    _write_clusters(clusters_file, [_make_perceptual(1, 8)])
    output = tmp_path / "out.html"

    rc = main([
        "--clusters", str(clusters_file),
        "--export-dir", str(space_dir),
        "--output", str(output),
    ])
    assert rc == 0
    html = output.read_text(encoding="utf-8")
    assert "%20" in html, "Spaces in export-dir must be percent-encoded in file:// URIs"
    assert "Telegram%20Desktop" in html
