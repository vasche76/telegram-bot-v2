"""
test_intake_audit.py — Unit tests for intake_telegram_audit.py (U2).

Uses PIL to write real (minimal) JPEG files so that PIL's verify() and .size
work against actual valid image content.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from intake_telegram_audit import (  # noqa: E402
    LOW_RES_THRESHOLD,
    _sha256,
    audit_image,
    run,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _write_jpeg(path: Path, width: int, height: int) -> None:
    """Write a minimal valid JPEG at the given dimensions."""
    from PIL import Image  # noqa: PLC0415
    img = Image.new("RGB", (width, height), color=(128, 64, 32))
    img.save(path, format="JPEG")


def _write_corrupt(path: Path) -> None:
    """Write a file that is not a valid JPEG."""
    path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 20)  # truncated JPEG header


def _make_manifest(tmp_path: Path, filenames: list[str]) -> Path:
    manifest = tmp_path / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for fn in filenames:
            fh.write(json.dumps({"filename": fn}) + "\n")
    return manifest


# ─── _sha256 ──────────────────────────────────────────────────────────────────


def test_sha256_matches_hashlib(tmp_path: Path) -> None:
    p = tmp_path / "file.bin"
    p.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert _sha256(p) == expected


# ─── audit_image ──────────────────────────────────────────────────────────────


def test_happy_path_valid_image(tmp_path: Path) -> None:
    """Valid 1280×960 JPEG → correct dimensions, low_res=false, corrupt=false."""
    p = tmp_path / "photo.jpg"
    _write_jpeg(p, 1280, 960)

    rec = audit_image(p)

    assert rec["corrupt"] is False
    assert rec["width"] == 1280
    assert rec["height"] == 960
    assert rec["max_side"] == 1280
    assert rec["low_res"] is False
    assert rec["sha256"] is not None
    assert len(rec["sha256"]) == 64
    assert rec["file_size"] == p.stat().st_size


def test_low_res_flag_below_threshold(tmp_path: Path) -> None:
    """600×400 image → low_res=true (max_side=600 < 800)."""
    p = tmp_path / "small.jpg"
    _write_jpeg(p, 600, 400)
    rec = audit_image(p)
    assert rec["low_res"] is True
    assert rec["max_side"] == 600


def test_low_res_flag_at_boundary(tmp_path: Path) -> None:
    """800×600 image → low_res=false (max_side=800 is exactly at threshold, OK)."""
    p = tmp_path / "boundary.jpg"
    _write_jpeg(p, 800, 600)
    rec = audit_image(p)
    assert rec["low_res"] is False
    assert rec["max_side"] == 800


def test_corrupt_file(tmp_path: Path) -> None:
    """Truncated JPEG → corrupt=true, no exception propagated."""
    p = tmp_path / "corrupt.jpg"
    _write_corrupt(p)
    rec = audit_image(p)
    assert rec["corrupt"] is True
    assert rec["width"] is None
    assert rec["height"] is None


def test_two_open_pattern_dimensions_correct(tmp_path: Path) -> None:
    """Ensure dimensions come from a fresh handle (not the verify() handle)."""
    p = tmp_path / "photo.jpg"
    _write_jpeg(p, 400, 300)
    rec = audit_image(p)
    # If the two-open pattern is wrong, width/height would be None or stale
    assert rec["width"] == 400
    assert rec["height"] == 300
    assert rec["corrupt"] is False


# ─── run() integration ────────────────────────────────────────────────────────


def test_run_all_records_written(tmp_path: Path) -> None:
    """audit.jsonl record count must equal manifest.jsonl record count."""
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()

    fnames = []
    for i in range(1, 6):
        fname = f"photos/photo_{i}.jpg"
        _write_jpeg(tmp_path / fname, 1000, 800)
        fnames.append(fname)

    manifest = _make_manifest(tmp_path, fnames)
    output_dir = tmp_path / "out"
    n = run(manifest, tmp_path, output_dir)

    assert n == 5
    audit = output_dir / "audit.jsonl"
    lines = audit.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5


def test_run_corrupt_continues(tmp_path: Path) -> None:
    """A corrupt file does not abort; processing continues for other files."""
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()

    _write_jpeg(tmp_path / "photos/good.jpg", 1000, 800)
    _write_corrupt(tmp_path / "photos/bad.jpg")
    _write_jpeg(tmp_path / "photos/also_good.jpg", 1200, 900)

    fnames = ["photos/good.jpg", "photos/bad.jpg", "photos/also_good.jpg"]
    manifest = _make_manifest(tmp_path, fnames)
    output_dir = tmp_path / "out"
    n = run(manifest, tmp_path, output_dir)

    assert n == 3
    lines = (output_dir / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    recs = [json.loads(l) for l in lines]
    corrupt_recs = [r for r in recs if r["corrupt"]]
    assert len(corrupt_recs) == 1
    assert corrupt_recs[0]["filename"] == "photos/bad.jpg"


def test_run_sha256_unique_distinct_images(tmp_path: Path) -> None:
    """Distinct images produce distinct SHA-256 hashes."""
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()

    # Use different sizes to guarantee different content
    sizes = [(100, 100), (200, 200), (300, 300), (400, 400), (500, 500)]
    fnames = []
    for i, (w, h) in enumerate(sizes, 1):
        fname = f"photos/photo_{i}.jpg"
        _write_jpeg(tmp_path / fname, w, h)
        fnames.append(fname)

    manifest = _make_manifest(tmp_path, fnames)
    output_dir = tmp_path / "out"
    run(manifest, tmp_path, output_dir)

    lines = (output_dir / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    hashes = [json.loads(l)["sha256"] for l in lines]
    assert len(set(hashes)) == len(hashes), "Duplicate SHA-256 for distinct images"


def test_run_progress_large_set(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """Script logs progress every 1000 images (check it doesn't crash on a 50-image set)."""
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()

    fnames = []
    for i in range(1, 51):
        fname = f"photos/photo_{i}.jpg"
        _write_jpeg(tmp_path / fname, 500, 500)
        fnames.append(fname)

    manifest = _make_manifest(tmp_path, fnames)
    output_dir = tmp_path / "out"
    n = run(manifest, tmp_path, output_dir)
    assert n == 50


def test_run_missing_manifest_exits(tmp_path: Path) -> None:
    """Missing manifest.jsonl causes sys.exit(1)."""
    with pytest.raises(SystemExit) as exc_info:
        run(tmp_path / "nonexistent.jsonl", tmp_path, tmp_path / "out")
    assert exc_info.value.code == 1
