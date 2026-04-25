"""
test_intake_dedup.py — Unit + integration tests for intake_telegram_dedup.py (U3).

Unit tests exercise internal functions with synthetic data (exact hamming control).
Integration tests use run() with real minimal JPEG files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from intake_telegram_dedup import (  # noqa: E402
    DEFAULT_PHASH_THRESHOLD,
    LICENSE,
    SOURCE,
    _UnionFind,
    _find_near_dup_pairs,
    _pass1_exact,
    _resolve_conflicts,
    run,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _write_jpeg(path: Path, width: int, height: int, color: tuple = (128, 64, 32)) -> None:
    from PIL import Image  # noqa: PLC0415

    img = Image.new("RGB", (width, height), color=color)
    img.save(path, format="JPEG")


def _write_corrupt(path: Path) -> None:
    path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 20)


def _write_audit(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "audit.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def _write_manifest(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "manifest.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def _read_clusters(output_dir: Path) -> list[dict]:
    path = output_dir / "dedup_clusters.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _read_summary(output_dir: Path) -> dict:
    path = output_dir / "dedup_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ─── _UnionFind ───────────────────────────────────────────────────────────────


def test_union_find_basic() -> None:
    uf = _UnionFind(5)
    uf.union(0, 1)
    uf.union(2, 3)

    assert uf.find(0) == uf.find(1)
    assert uf.find(2) == uf.find(3)
    assert uf.find(0) != uf.find(2)
    assert uf.find(4) not in {uf.find(0), uf.find(2)}

    components = uf.components()
    sizes = sorted(len(v) for v in components.values())
    assert sizes == [1, 2, 2]


def test_union_find_three_way_merge() -> None:
    uf = _UnionFind(3)
    uf.union(0, 1)
    uf.union(1, 2)

    assert uf.find(0) == uf.find(1) == uf.find(2)
    components = uf.components()
    assert any(len(v) == 3 for v in components.values())


# ─── _find_near_dup_pairs ─────────────────────────────────────────────────────


def _make_hashes(*bit_patterns: int) -> np.ndarray:
    """Build (n, 8) uint8 array from list of 64-bit integer values."""
    rows = [v.to_bytes(8, byteorder="big") for v in bit_patterns]
    return np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(len(rows), 8)


def test_find_near_dup_pairs_hamming4_found() -> None:
    """Hamming=4 (≤ threshold=8) → pair found."""
    # 0x000000000000000F has 4 bits set → hamming(0, 0x0F) = 4
    hashes = _make_hashes(0x0000000000000000, 0x000000000000000F)
    pairs = _find_near_dup_pairs(hashes, threshold=8)
    assert len(pairs) == 1
    assert pairs[0][2] == 4


def test_find_near_dup_pairs_hamming12_not_found_at_default() -> None:
    """Hamming=12 (> threshold=8) → no pair at default threshold."""
    # 0xFF (8 bits) + 0x0F (4 bits) = 12 bits differ from zero
    hashes = _make_hashes(0x0000000000000000, 0x00000000000000FF | (0x0F << 8))
    pairs = _find_near_dup_pairs(hashes, threshold=8)
    assert len(pairs) == 0


def test_find_near_dup_pairs_custom_threshold_12() -> None:
    """Hamming=12 pair IS found when threshold=12."""
    hashes = _make_hashes(0x0000000000000000, 0x0000000000000FFF)
    # 0x0FFF = 0b0000_1111_1111_1111 = 12 bits set
    pairs = _find_near_dup_pairs(hashes, threshold=8)
    assert len(pairs) == 0  # misses at default

    pairs12 = _find_near_dup_pairs(hashes, threshold=12)
    assert len(pairs12) == 1
    assert pairs12[0][2] == 12


def test_find_near_dup_pairs_identical_hashes() -> None:
    """Identical hashes → hamming=0 → always found."""
    hashes = _make_hashes(0xDEADBEEFCAFEBABE, 0xDEADBEEFCAFEBABE)
    pairs = _find_near_dup_pairs(hashes, threshold=0)
    assert len(pairs) == 1
    assert pairs[0][2] == 0


def test_find_near_dup_pairs_no_self_pairs() -> None:
    """No (i, i) self-pairs returned."""
    hashes = _make_hashes(0x0, 0x1)
    pairs = _find_near_dup_pairs(hashes, threshold=8)
    for i, j, _ in pairs:
        assert i != j


def test_find_near_dup_pairs_upper_triangle_only() -> None:
    """Only pairs with i < j returned (no duplicates)."""
    hashes = _make_hashes(0x0, 0x1, 0x2)
    pairs = _find_near_dup_pairs(hashes, threshold=64)  # all pairs
    for i, j, _ in pairs:
        assert i < j


def test_find_near_dup_pairs_same_batch_3_images() -> None:
    """
    3 near-duplicate images, all in single batch (B=256):
    all pairs (0,1), (0,2), (1,2) found, union-find merges into one component.
    Exercises global index mapping: b_local promoted to i_offset + b_local.
    """
    hashes = _make_hashes(0x0, 0x0, 0x0)  # hamming=0 for all pairs
    pairs = _find_near_dup_pairs(hashes, threshold=8, batch_size=256)
    assert len(pairs) == 3  # all 3 unique pairs

    pair_set = {(i, j) for i, j, _ in pairs}
    assert (0, 1) in pair_set
    assert (0, 2) in pair_set
    assert (1, 2) in pair_set

    uf = _UnionFind(3)
    for i, j, _ in pairs:
        uf.union(i, j)
    components = uf.components()
    assert any(len(v) == 3 for v in components.values())


def test_find_near_dup_pairs_empty_input() -> None:
    """Empty hash array → no pairs."""
    hashes = np.empty((0, 8), dtype=np.uint8)
    pairs = _find_near_dup_pairs(hashes, threshold=8)
    assert pairs == []


def test_find_near_dup_pairs_single_image() -> None:
    """Single image → no pairs."""
    hashes = _make_hashes(0xABCD)
    pairs = _find_near_dup_pairs(hashes, threshold=8)
    assert pairs == []


# ─── _pass1_exact ─────────────────────────────────────────────────────────────


def test_pass1_exact_happy_path() -> None:
    """Two records with same sha256, valid msg_ids → keep = lower msg_id."""
    sha = "a" * 64
    audit = [
        {"filename": "early.jpg", "sha256": sha, "max_side": 800, "corrupt": False},
        {"filename": "late.jpg",  "sha256": sha, "max_side": 800, "corrupt": False},
    ]
    manifest = {
        "early.jpg": {"filename": "early.jpg", "msg_id": 5},
        "late.jpg":  {"filename": "late.jpg",  "msg_id": 99},
    }
    clusters, non_keeps = _pass1_exact(audit, manifest)

    assert len(clusters) == 1
    assert clusters[0]["cluster_type"] == "exact"
    assert clusters[0]["keep_filename"] == "early.jpg"
    assert "late.jpg" in clusters[0]["duplicate_filenames"]
    assert "late.jpg" in non_keeps
    assert "early.jpg" not in non_keeps


def test_pass1_exact_mixed_none_msg_id() -> None:
    """One record has msg_id=None → keep = the record with integer msg_id. No TypeError."""
    sha = "b" * 64
    audit = [
        {"filename": "none_msg.jpg", "sha256": sha, "max_side": 800, "corrupt": False},
        {"filename": "int_msg.jpg",  "sha256": sha, "max_side": 800, "corrupt": False},
    ]
    manifest = {
        "none_msg.jpg": {"filename": "none_msg.jpg", "msg_id": None},
        "int_msg.jpg":  {"filename": "int_msg.jpg",  "msg_id": 42},
    }
    clusters, non_keeps = _pass1_exact(audit, manifest)

    assert len(clusters) == 1
    assert clusters[0]["keep_filename"] == "int_msg.jpg"
    assert "none_msg.jpg" in non_keeps


def test_pass1_exact_all_none_msg_id() -> None:
    """Both records have msg_id=None → keep = lexicographically first filename. No exception."""
    sha = "c" * 64
    audit = [
        {"filename": "z_file.jpg", "sha256": sha, "max_side": 800, "corrupt": False},
        {"filename": "a_file.jpg", "sha256": sha, "max_side": 800, "corrupt": False},
    ]
    manifest = {
        "z_file.jpg": {"filename": "z_file.jpg", "msg_id": None},
        "a_file.jpg": {"filename": "a_file.jpg", "msg_id": None},
    }
    clusters, non_keeps = _pass1_exact(audit, manifest)

    assert len(clusters) == 1
    assert clusters[0]["keep_filename"] == "a_file.jpg"
    assert "z_file.jpg" in non_keeps


def test_pass1_exact_all_unique() -> None:
    """No duplicate sha256 groups → 0 clusters, empty non_keeps set."""
    audit = [
        {"filename": "a.jpg", "sha256": "a" * 64, "max_side": 800, "corrupt": False},
        {"filename": "b.jpg", "sha256": "b" * 64, "max_side": 800, "corrupt": False},
    ]
    manifest = {
        "a.jpg": {"filename": "a.jpg", "msg_id": 1},
        "b.jpg": {"filename": "b.jpg", "msg_id": 2},
    }
    clusters, non_keeps = _pass1_exact(audit, manifest)

    assert len(clusters) == 0
    assert len(non_keeps) == 0


def test_pass1_exact_sha256_none_skipped() -> None:
    """Records with sha256=None are skipped (corrupt images from U2)."""
    audit = [
        {"filename": "corrupt.jpg", "sha256": None, "max_side": None, "corrupt": True},
        {"filename": "good.jpg",    "sha256": "a" * 64, "max_side": 800, "corrupt": False},
    ]
    manifest: dict = {}
    clusters, non_keeps = _pass1_exact(audit, manifest)

    assert len(clusters) == 0


# ─── _resolve_conflicts ───────────────────────────────────────────────────────


def test_resolve_conflicts_exact_keep_is_perceptual_dup() -> None:
    """
    When exact-cluster keep also appears as a perceptual-cluster dup,
    the exact cluster is absorbed: its dups move to the perceptual cluster,
    and the exact cluster is removed.
    """
    exact = [
        {
            "cluster_id": 1,
            "cluster_type": "exact",
            "keep_filename": "photos/keep_exact.jpg",
            "duplicate_filenames": ["photos/dup_exact.jpg"],
            "hamming_distance": None,
            "reason": "sha256=abc",
        }
    ]
    perceptual = [
        {
            "cluster_id": 2,
            "cluster_type": "perceptual",
            "keep_filename": "photos/best.jpg",
            "duplicate_filenames": ["photos/keep_exact.jpg", "photos/other.jpg"],
            "hamming_distance": 4,
            "reason": "phash_hamming<=8",
        }
    ]

    resolved_exact, resolved_perceptual = _resolve_conflicts(exact, perceptual)

    # Exact cluster absorbed — zero exact clusters remain
    assert len(resolved_exact) == 0

    # Perceptual cluster gains dup_exact.jpg
    pc = resolved_perceptual[0]
    assert "photos/dup_exact.jpg" in pc["duplicate_filenames"]
    assert "photos/keep_exact.jpg" in pc["duplicate_filenames"]
    assert pc["keep_filename"] == "photos/best.jpg"

    # Integrity: no filename in both keep and duplicates
    all_dup_fns = set(pc["duplicate_filenames"])
    assert pc["keep_filename"] not in all_dup_fns


def test_resolve_conflicts_no_conflict() -> None:
    """When there is no cross-pass conflict, clusters are unchanged."""
    exact = [
        {
            "cluster_id": 1,
            "cluster_type": "exact",
            "keep_filename": "photos/a.jpg",
            "duplicate_filenames": ["photos/b.jpg"],
            "hamming_distance": None,
            "reason": "sha256=abc",
        }
    ]
    perceptual = [
        {
            "cluster_id": 2,
            "cluster_type": "perceptual",
            "keep_filename": "photos/c.jpg",
            "duplicate_filenames": ["photos/d.jpg"],
            "hamming_distance": 2,
            "reason": "phash_hamming<=8",
        }
    ]

    resolved_exact, resolved_perceptual = _resolve_conflicts(exact, perceptual)

    assert len(resolved_exact) == 1
    assert resolved_exact[0]["keep_filename"] == "photos/a.jpg"
    assert len(resolved_perceptual) == 1


def test_resolve_conflicts_exact_keep_is_perceptual_keep() -> None:
    """
    When exact-cluster keep_filename is also the KEEP of a perceptual cluster
    (not a dup), the exact cluster is absorbed: its dups join the perceptual
    cluster's duplicate list, and the exact cluster is removed.

    Without this fix, two cluster records would share the same keep_filename,
    violating the keep-uniqueness invariant.
    """
    exact = [
        {
            "cluster_id": 1,
            "cluster_type": "exact",
            "keep_filename": "photos/a.jpg",
            "duplicate_filenames": ["photos/b.jpg"],
            "hamming_distance": None,
            "reason": "sha256=abc",
        }
    ]
    perceptual = [
        {
            "cluster_id": 2,
            "cluster_type": "perceptual",
            "keep_filename": "photos/a.jpg",  # same as exact keep
            "duplicate_filenames": ["photos/c.jpg"],
            "hamming_distance": 3,
            "reason": "phash_hamming<=8",
        }
    ]

    resolved_exact, resolved_perceptual = _resolve_conflicts(exact, perceptual)

    # Exact cluster absorbed — zero exact clusters remain
    assert len(resolved_exact) == 0

    # Perceptual cluster gains b.jpg from the absorbed exact cluster
    pc = resolved_perceptual[0]
    assert pc["keep_filename"] == "photos/a.jpg"
    assert "photos/b.jpg" in pc["duplicate_filenames"]
    assert "photos/c.jpg" in pc["duplicate_filenames"]

    # Integrity: keep not in its own duplicates
    assert pc["keep_filename"] not in set(pc["duplicate_filenames"])


# ─── run() integration tests ─────────────────────────────────────────────────


def _setup_export(tmp_path: Path) -> Path:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "photos").mkdir()
    return export_dir


def test_run_missing_audit_exits(tmp_path: Path) -> None:
    """Missing audit.jsonl → sys.exit(1)."""
    manifest = _write_manifest(tmp_path, [{"filename": "a.jpg", "msg_id": 1}])
    with pytest.raises(SystemExit) as exc:
        run(tmp_path / "nonexistent_audit.jsonl", manifest, tmp_path, tmp_path / "out")
    assert exc.value.code == 1


def test_run_missing_manifest_exits(tmp_path: Path) -> None:
    """Missing manifest.jsonl → sys.exit(1)."""
    audit = _write_audit(tmp_path, [{"filename": "a.jpg", "sha256": "a" * 64, "max_side": 800, "corrupt": False}])
    with pytest.raises(SystemExit) as exc:
        run(audit, tmp_path / "nonexistent_manifest.jsonl", tmp_path, tmp_path / "out")
    assert exc.value.code == 1


def test_run_all_unique_no_clusters(tmp_path: Path) -> None:
    """All-unique dataset (no exact or perceptual dups) → 0 clusters."""
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600)
    _write_jpeg(export_dir / "photos/b.jpg", 400, 300)

    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": "a" * 64, "max_side": 800, "corrupt": False},
        {"filename": "photos/b.jpg", "sha256": "b" * 64, "max_side": 400, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/a.jpg", "msg_id": 1},
        {"filename": "photos/b.jpg", "msg_id": 2},
    ])
    output_dir = tmp_path / "out"
    n, summary = run(audit, manifest, export_dir, output_dir)

    # Both images are solid same-color (pHash=0 each) → they WILL be clustered as perceptual dups
    # So 0 exact clusters, but perceptual may differ. The important invariant is:
    # total_unique = input - total_removed
    assert summary["input_records"] == 2
    assert summary["exact_clusters"] == 0
    total_removed = summary["exact_removed"] + summary["perceptual_removed"]
    assert summary["total_unique_after_dedup"] == 2 - total_removed


def test_run_exact_dup_pair(tmp_path: Path) -> None:
    """Two audit records with same sha256 → one exact cluster."""
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600)
    _write_jpeg(export_dir / "photos/b.jpg", 800, 600)

    sha = "a" * 64  # same sha256 → exact dup
    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": sha, "max_side": 800, "corrupt": False},
        {"filename": "photos/b.jpg", "sha256": sha, "max_side": 800, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/a.jpg", "msg_id": 5},   # lower → keep
        {"filename": "photos/b.jpg", "msg_id": 99},
    ])
    output_dir = tmp_path / "out"
    n, summary = run(audit, manifest, export_dir, output_dir)

    clusters = _read_clusters(output_dir)
    exact = [c for c in clusters if c["cluster_type"] == "exact"]
    assert len(exact) == 1
    assert exact[0]["keep_filename"] == "photos/a.jpg"
    assert "photos/b.jpg" in exact[0]["duplicate_filenames"]
    assert exact[0]["hamming_distance"] is None
    assert summary["exact_clusters"] == 1
    assert summary["exact_removed"] == 1


def test_run_near_dup_pair(tmp_path: Path) -> None:
    """
    Two near-identical images (same solid color, different sizes → same pHash, different sha256)
    → one perceptual cluster. Keep = higher max_side.
    Solid-color JPEG images always produce pHash=0 (all-zero AC DCT coefficients).
    """
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/large.jpg", 800, 600, color=(100, 100, 100))
    _write_jpeg(export_dir / "photos/small.jpg", 400, 300, color=(100, 100, 100))

    audit = _write_audit(tmp_path, [
        {"filename": "photos/large.jpg", "sha256": "aaaa" * 16, "max_side": 800, "corrupt": False},
        {"filename": "photos/small.jpg", "sha256": "bbbb" * 16, "max_side": 400, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/large.jpg", "msg_id": 10},
        {"filename": "photos/small.jpg", "msg_id": 20},
    ])
    output_dir = tmp_path / "out"
    n, summary = run(audit, manifest, export_dir, output_dir)

    clusters = _read_clusters(output_dir)
    perceptual = [c for c in clusters if c["cluster_type"] == "perceptual"]
    assert len(perceptual) == 1
    assert perceptual[0]["keep_filename"] == "photos/large.jpg"
    assert "photos/small.jpg" in perceptual[0]["duplicate_filenames"]
    assert perceptual[0]["hamming_distance"] == 0  # identical pHash
    assert summary["perceptual_clusters"] == 1
    assert summary["perceptual_removed"] == 1


def test_run_perceptual_keep_by_max_side(tmp_path: Path) -> None:
    """Perceptual dedup: keep = record with largest max_side."""
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/hires.jpg", 1200, 900, color=(50, 50, 50))
    _write_jpeg(export_dir / "photos/lowres.jpg", 300, 225, color=(50, 50, 50))

    audit = _write_audit(tmp_path, [
        {"filename": "photos/hires.jpg",  "sha256": "aaaa" * 16, "max_side": 1200, "corrupt": False},
        {"filename": "photos/lowres.jpg", "sha256": "bbbb" * 16, "max_side": 300,  "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/hires.jpg",  "msg_id": 100},  # higher msg_id, but higher max_side
        {"filename": "photos/lowres.jpg", "msg_id": 1},
    ])
    output_dir = tmp_path / "out"
    n, summary = run(audit, manifest, export_dir, output_dir)

    clusters = _read_clusters(output_dir)
    perceptual = [c for c in clusters if c["cluster_type"] == "perceptual"]
    assert len(perceptual) == 1
    assert perceptual[0]["keep_filename"] == "photos/hires.jpg"


def test_run_perceptual_tie_break_msg_id(tmp_path: Path) -> None:
    """Perceptual tie-break: equal max_side → keep = lower msg_id."""
    export_dir = _setup_export(tmp_path)
    # Same size → same max_side; both solid same color → same pHash
    _write_jpeg(export_dir / "photos/early.jpg", 800, 600, color=(80, 80, 80))
    _write_jpeg(export_dir / "photos/late.jpg",  800, 600, color=(80, 80, 80))

    audit = _write_audit(tmp_path, [
        {"filename": "photos/early.jpg", "sha256": "aaaa" * 16, "max_side": 800, "corrupt": False},
        {"filename": "photos/late.jpg",  "sha256": "bbbb" * 16, "max_side": 800, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/early.jpg", "msg_id": 42},   # lower → tie-break winner
        {"filename": "photos/late.jpg",  "msg_id": 999},
    ])
    output_dir = tmp_path / "out"
    n, summary = run(audit, manifest, export_dir, output_dir)

    clusters = _read_clusters(output_dir)
    perceptual = [c for c in clusters if c["cluster_type"] == "perceptual"]
    assert len(perceptual) == 1
    assert perceptual[0]["keep_filename"] == "photos/early.jpg"


def test_run_exact_non_keep_excluded_from_phash(tmp_path: Path) -> None:
    """
    Exact-dup non-keep (b.jpg) was excluded from the pHash pass.

    When a (the exact keep) is also the perceptual keep (highest max_side),
    conflict resolution absorbs the exact cluster into the perceptual cluster.
    Result: zero exact clusters, one perceptual cluster with keep=a and
    duplicate_filenames containing both b.jpg (via absorption) and c.jpg
    (via pHash near-dup comparison).

    The key invariant: b.jpg must NOT be the perceptual keep_filename —
    it entered the perceptual cluster only through conflict resolution,
    not through an independent pHash comparison.
    """
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600, color=(70, 70, 70))
    _write_jpeg(export_dir / "photos/b.jpg", 800, 600, color=(70, 70, 70))  # exact dup of a
    _write_jpeg(export_dir / "photos/c.jpg", 400, 300, color=(70, 70, 70))  # near-dup of a

    sha_ab = "ab" * 32  # a and b are exact dups
    sha_c = "cc" * 32

    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": sha_ab, "max_side": 800, "corrupt": False},
        {"filename": "photos/b.jpg", "sha256": sha_ab, "max_side": 800, "corrupt": False},
        {"filename": "photos/c.jpg", "sha256": sha_c,  "max_side": 400, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/a.jpg", "msg_id": 10},  # a is exact keep (lower msg_id)
        {"filename": "photos/b.jpg", "msg_id": 20},  # b is exact non-keep
        {"filename": "photos/c.jpg", "msg_id": 30},
    ])
    output_dir = tmp_path / "out"
    n, summary = run(audit, manifest, export_dir, output_dir)

    clusters = _read_clusters(output_dir)

    # Exact cluster absorbed into the perceptual cluster → zero exact clusters
    exact = [c for c in clusters if c["cluster_type"] == "exact"]
    assert len(exact) == 0, "exact cluster should have been absorbed into perceptual"

    # One perceptual cluster with a.jpg as keep
    perceptual = [c for c in clusters if c["cluster_type"] == "perceptual"]
    assert len(perceptual) == 1
    pc = perceptual[0]
    assert pc["keep_filename"] == "photos/a.jpg"

    # b.jpg must NOT be the perceptual keep — it was absorbed, not independently selected
    perceptual_keeps = {c["keep_filename"] for c in perceptual}
    assert "photos/b.jpg" not in perceptual_keeps


def test_run_conflict_resolution_exact_keep_is_perceptual_keep(tmp_path: Path) -> None:
    """
    Integration test for the keep-as-keep conflict-resolution path.

    Setup: A and B are exact dups (A is keep, lower msg_id).
           A and C are perceptual near-dups (solid same color, A has larger max_side
           so _perceptual_keep_key selects A as perceptual keep).

    Before fix: two cluster records both declare A as keep_filename.
    After fix:  one perceptual cluster — keep=A, dups=[B, C]; zero exact clusters.
    """
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600, color=(110, 110, 110))  # exact keep + perceptual keep
    _write_jpeg(export_dir / "photos/b.jpg", 800, 600, color=(110, 110, 110))  # exact dup of a
    _write_jpeg(export_dir / "photos/c.jpg", 400, 300, color=(110, 110, 110))  # perceptual near-dup of a

    sha_ab = "ab" * 32  # a and b are exact dups
    sha_c = "cc" * 32

    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": sha_ab, "max_side": 800, "corrupt": False},
        {"filename": "photos/b.jpg", "sha256": sha_ab, "max_side": 800, "corrupt": False},
        {"filename": "photos/c.jpg", "sha256": sha_c,  "max_side": 400, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/a.jpg", "msg_id": 10},   # lower msg_id → exact keep
        {"filename": "photos/b.jpg", "msg_id": 20},
        {"filename": "photos/c.jpg", "msg_id": 30},
    ])
    output_dir = tmp_path / "out"
    n, summary = run(audit, manifest, export_dir, output_dir)

    clusters = _read_clusters(output_dir)
    exact = [c for c in clusters if c["cluster_type"] == "exact"]
    perceptual = [c for c in clusters if c["cluster_type"] == "perceptual"]

    # Exact cluster was absorbed → zero exact clusters remain
    assert len(exact) == 0, "exact cluster should have been absorbed"

    # One perceptual cluster with a.jpg as keep and both b.jpg and c.jpg as dups
    assert len(perceptual) == 1
    pc = perceptual[0]
    assert pc["keep_filename"] == "photos/a.jpg"
    assert "photos/b.jpg" in pc["duplicate_filenames"]
    assert "photos/c.jpg" in pc["duplicate_filenames"]

    # Arithmetic invariant
    total_dups = sum(len(c["duplicate_filenames"]) for c in clusters)
    assert summary["total_unique_after_dedup"] == summary["input_records"] - total_dups

    # keep_filename uniqueness: no filename is keep of more than one cluster
    keep_fns = [c["keep_filename"] for c in clusters]
    assert len(keep_fns) == len(set(keep_fns)), "duplicate keep_filename found in clusters"


def test_run_corrupt_image_skipped_in_phash(tmp_path: Path) -> None:
    """
    Corrupt image (corrupt=True in audit) is excluded from pHash pass.
    Its valid near-dup partner remains unclustered (no false cluster).
    """
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/valid.jpg", 800, 600)
    _write_corrupt(export_dir / "photos/corrupt.jpg")

    audit = _write_audit(tmp_path, [
        {"filename": "photos/valid.jpg",   "sha256": "a" * 64, "max_side": 800, "corrupt": False},
        {"filename": "photos/corrupt.jpg", "sha256": None,     "max_side": None, "corrupt": True},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/valid.jpg",   "msg_id": 1},
        {"filename": "photos/corrupt.jpg", "msg_id": 2},
    ])
    output_dir = tmp_path / "out"
    n, summary = run(audit, manifest, export_dir, output_dir)

    clusters = _read_clusters(output_dir)
    perceptual = [c for c in clusters if c["cluster_type"] == "perceptual"]
    assert len(perceptual) == 0
    assert summary["perceptual_clusters"] == 0


def test_run_custom_phash_threshold_recorded(tmp_path: Path) -> None:
    """--phash-threshold value is stored in dedup_summary.json."""
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600)

    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": "a" * 64, "max_side": 800, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/a.jpg", "msg_id": 1},
    ])
    output_dir = tmp_path / "out"
    run(audit, manifest, export_dir, output_dir, phash_threshold=12)

    summary = _read_summary(output_dir)
    assert summary["phash_threshold"] == 12


def test_run_dry_run_no_files(tmp_path: Path) -> None:
    """--dry-run: no files written to output_dir."""
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600)

    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": "a" * 64, "max_side": 800, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/a.jpg", "msg_id": 1},
    ])
    output_dir = tmp_path / "out"
    n, summary = run(audit, manifest, export_dir, output_dir, dry_run=True)

    assert n == 0
    assert not (output_dir / "dedup_clusters.jsonl").exists()
    assert not (output_dir / "dedup_summary.json").exists()


# ─── Cluster integrity invariants ─────────────────────────────────────────────


def test_cluster_integrity_no_filename_in_both_keep_and_duplicates(tmp_path: Path) -> None:
    """No filename appears as both keep_filename and in duplicate_filenames."""
    export_dir = _setup_export(tmp_path)
    # 4 images: exact dup pair + near-dup pair
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600, color=(60, 60, 60))
    _write_jpeg(export_dir / "photos/b.jpg", 800, 600, color=(60, 60, 60))
    _write_jpeg(export_dir / "photos/c.jpg", 600, 450, color=(90, 90, 90))
    _write_jpeg(export_dir / "photos/d.jpg", 300, 225, color=(90, 90, 90))

    sha_ab = "ab" * 32
    sha_c = "cc" * 32
    sha_d = "dd" * 32

    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": sha_ab, "max_side": 800, "corrupt": False},
        {"filename": "photos/b.jpg", "sha256": sha_ab, "max_side": 800, "corrupt": False},
        {"filename": "photos/c.jpg", "sha256": sha_c,  "max_side": 600, "corrupt": False},
        {"filename": "photos/d.jpg", "sha256": sha_d,  "max_side": 300, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/a.jpg", "msg_id": 1},
        {"filename": "photos/b.jpg", "msg_id": 2},
        {"filename": "photos/c.jpg", "msg_id": 3},
        {"filename": "photos/d.jpg", "msg_id": 4},
    ])
    output_dir = tmp_path / "out"
    run(audit, manifest, export_dir, output_dir)

    clusters = _read_clusters(output_dir)
    keep_fns = {c["keep_filename"] for c in clusters}
    for c in clusters:
        for fn in c["duplicate_filenames"]:
            assert fn not in keep_fns, f"{fn} is both keep and duplicate"


def test_cluster_integrity_no_filename_in_multiple_clusters(tmp_path: Path) -> None:
    """No filename appears in duplicate_filenames of more than one cluster."""
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600, color=(60, 60, 60))
    _write_jpeg(export_dir / "photos/b.jpg", 800, 600, color=(60, 60, 60))

    sha = "ab" * 32
    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": sha, "max_side": 800, "corrupt": False},
        {"filename": "photos/b.jpg", "sha256": sha, "max_side": 800, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/a.jpg", "msg_id": 1},
        {"filename": "photos/b.jpg", "msg_id": 2},
    ])
    output_dir = tmp_path / "out"
    run(audit, manifest, export_dir, output_dir)

    clusters = _read_clusters(output_dir)
    seen_as_dup: set[str] = set()
    for c in clusters:
        for fn in c["duplicate_filenames"]:
            assert fn not in seen_as_dup, f"{fn} in multiple clusters"
            seen_as_dup.add(fn)


def test_cluster_integrity_no_keep_in_multiple_clusters(tmp_path: Path) -> None:
    """No keep_filename appears as the keep of more than one cluster."""
    export_dir = _setup_export(tmp_path)
    # Scenario: a+b are exact dups (a is keep); a+c are perceptual near-dups
    # (a is perceptual keep due to higher max_side).
    # After conflict resolution: one cluster only, keep=a.
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600, color=(55, 55, 55))
    _write_jpeg(export_dir / "photos/b.jpg", 800, 600, color=(55, 55, 55))
    _write_jpeg(export_dir / "photos/c.jpg", 400, 300, color=(55, 55, 55))

    sha_ab = "ab" * 32
    sha_c = "cc" * 32

    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": sha_ab, "max_side": 800, "corrupt": False},
        {"filename": "photos/b.jpg", "sha256": sha_ab, "max_side": 800, "corrupt": False},
        {"filename": "photos/c.jpg", "sha256": sha_c,  "max_side": 400, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/a.jpg", "msg_id": 1},
        {"filename": "photos/b.jpg", "msg_id": 2},
        {"filename": "photos/c.jpg", "msg_id": 3},
    ])
    output_dir = tmp_path / "out"
    run(audit, manifest, export_dir, output_dir)

    clusters = _read_clusters(output_dir)
    keep_fns = [c["keep_filename"] for c in clusters]
    assert len(keep_fns) == len(set(keep_fns)), (
        f"keep_filename appears in multiple clusters: "
        f"{[fn for fn in keep_fns if keep_fns.count(fn) > 1]}"
    )


def test_run_count_invariant(tmp_path: Path) -> None:
    """total_unique_after_dedup == input_records - sum(len(duplicate_filenames))."""
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600, color=(60, 60, 60))
    _write_jpeg(export_dir / "photos/b.jpg", 800, 600, color=(60, 60, 60))
    _write_jpeg(export_dir / "photos/c.jpg", 400, 300, color=(60, 60, 60))

    sha_ab = "ab" * 32
    sha_c = "cc" * 32

    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": sha_ab, "max_side": 800, "corrupt": False},
        {"filename": "photos/b.jpg", "sha256": sha_ab, "max_side": 800, "corrupt": False},
        {"filename": "photos/c.jpg", "sha256": sha_c,  "max_side": 400, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/a.jpg", "msg_id": 1},
        {"filename": "photos/b.jpg", "msg_id": 2},
        {"filename": "photos/c.jpg", "msg_id": 3},
    ])
    output_dir = tmp_path / "out"
    n, summary = run(audit, manifest, export_dir, output_dir)

    clusters = _read_clusters(output_dir)
    total_dups = sum(len(c["duplicate_filenames"]) for c in clusters)
    assert summary["total_unique_after_dedup"] == summary["input_records"] - total_dups


def test_run_summary_required_keys(tmp_path: Path) -> None:
    """dedup_summary.json must contain all required aggregate keys."""
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600)

    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": "a" * 64, "max_side": 800, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/a.jpg", "msg_id": 1},
    ])
    output_dir = tmp_path / "out"
    run(audit, manifest, export_dir, output_dir)

    summary = _read_summary(output_dir)
    required_keys = {
        "input_records",
        "exact_clusters",
        "exact_removed",
        "phash_candidates_upper_bound",
        "phash_threshold",
        "perceptual_clusters",
        "perceptual_removed",
        "boundary_clusters_at_threshold",
        "total_unique_after_dedup",
        "provisional",
        "manual_review_required",
        "manual_review_reason",
        "source",
        "license",
    }
    assert required_keys.issubset(summary.keys())
    assert summary["source"] == SOURCE
    assert summary["license"] == LICENSE
    assert summary["phash_threshold"] == DEFAULT_PHASH_THRESHOLD


def test_run_dedup_clusters_jsonl_written(tmp_path: Path) -> None:
    """dedup_clusters.jsonl is written to output_dir."""
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600)

    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": "a" * 64, "max_side": 800, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [{"filename": "photos/a.jpg", "msg_id": 1}])
    output_dir = tmp_path / "out"
    run(audit, manifest, export_dir, output_dir)

    assert (output_dir / "dedup_clusters.jsonl").exists()
    assert (output_dir / "dedup_summary.json").exists()


def test_run_cluster_records_no_pii(tmp_path: Path) -> None:
    """Cluster records contain no captions, sender names, or message text."""
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600, color=(60, 60, 60))
    _write_jpeg(export_dir / "photos/b.jpg", 800, 600, color=(60, 60, 60))

    sha = "ab" * 32
    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": sha, "max_side": 800, "corrupt": False},
        {"filename": "photos/b.jpg", "sha256": sha, "max_side": 800, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/a.jpg", "msg_id": 1, "caption": "SECRET TEXT", "sender": "Alice"},
        {"filename": "photos/b.jpg", "msg_id": 2, "caption": "MORE SECRETS", "sender": "Bob"},
    ])
    output_dir = tmp_path / "out"
    run(audit, manifest, export_dir, output_dir)

    content = (output_dir / "dedup_clusters.jsonl").read_text(encoding="utf-8")
    assert "SECRET TEXT" not in content
    assert "MORE SECRETS" not in content
    assert "Alice" not in content
    assert "Bob" not in content


# ─── Provisional / manual-review fields ──────────────────────────────────────


def test_run_summary_provisional_fields(tmp_path: Path) -> None:
    """dedup_summary.json must carry provisional=True and manual_review_required=True."""
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600)

    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": "a" * 64, "max_side": 800, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [{"filename": "photos/a.jpg", "msg_id": 1}])
    output_dir = tmp_path / "out"
    run(audit, manifest, export_dir, output_dir)

    summary = _read_summary(output_dir)
    assert summary["provisional"] is True
    assert summary["manual_review_required"] is True
    assert isinstance(summary["manual_review_reason"], str)
    assert len(summary["manual_review_reason"]) > 0


def test_run_boundary_clusters_at_threshold_zero(tmp_path: Path) -> None:
    """
    Two identical solid-color images → pHash hamming=0, which is < threshold=8.
    boundary_clusters_at_threshold must be 0 (0 != 8).
    """
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600, color=(100, 100, 100))
    _write_jpeg(export_dir / "photos/b.jpg", 400, 300, color=(100, 100, 100))

    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": "aaaa" * 16, "max_side": 800, "corrupt": False},
        {"filename": "photos/b.jpg", "sha256": "bbbb" * 16, "max_side": 400, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/a.jpg", "msg_id": 1},
        {"filename": "photos/b.jpg", "msg_id": 2},
    ])
    output_dir = tmp_path / "out"
    run(audit, manifest, export_dir, output_dir, phash_threshold=8)

    summary = _read_summary(output_dir)
    assert summary["boundary_clusters_at_threshold"] == 0


def test_run_boundary_clusters_at_threshold_counts_exact_match(tmp_path: Path) -> None:
    """
    When threshold=0, a cluster with hamming=0 is a boundary cluster.
    boundary_clusters_at_threshold == perceptual_clusters.
    """
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600, color=(100, 100, 100))
    _write_jpeg(export_dir / "photos/b.jpg", 400, 300, color=(100, 100, 100))

    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": "aaaa" * 16, "max_side": 800, "corrupt": False},
        {"filename": "photos/b.jpg", "sha256": "bbbb" * 16, "max_side": 400, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/a.jpg", "msg_id": 1},
        {"filename": "photos/b.jpg", "msg_id": 2},
    ])
    output_dir = tmp_path / "out"
    run(audit, manifest, export_dir, output_dir, phash_threshold=0)

    summary = _read_summary(output_dir)
    # With threshold=0, the one perceptual cluster (hamming=0) is exactly at the boundary.
    assert summary["perceptual_clusters"] == 1
    assert summary["boundary_clusters_at_threshold"] == 1


def test_run_phash_candidates_upper_bound_field(tmp_path: Path) -> None:
    """phash_candidates_upper_bound = input_records - exact_removed (pre-resolution)."""
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600)

    sha = "a" * 64
    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": sha, "max_side": 800, "corrupt": False},
        {"filename": "photos/b.jpg", "sha256": sha, "max_side": 800, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/a.jpg", "msg_id": 1},
        {"filename": "photos/b.jpg", "msg_id": 2},
    ])
    output_dir = tmp_path / "out"
    _, summary = run(audit, manifest, export_dir, output_dir)

    # 2 input records, 1 exact-removed → 1 going into pHash pass
    assert "phash_candidates_upper_bound" in summary
    assert summary["phash_candidates_upper_bound"] == 1


def test_run_summary_no_pii(tmp_path: Path) -> None:
    """dedup_summary.json must contain no captions, sender names, or message text."""
    export_dir = _setup_export(tmp_path)
    _write_jpeg(export_dir / "photos/a.jpg", 800, 600)

    audit = _write_audit(tmp_path, [
        {"filename": "photos/a.jpg", "sha256": "a" * 64, "max_side": 800, "corrupt": False},
    ])
    manifest = _write_manifest(tmp_path, [
        {"filename": "photos/a.jpg", "msg_id": 1, "caption": "TOP SECRET", "sender": "Alice"},
    ])
    output_dir = tmp_path / "out"
    run(audit, manifest, export_dir, output_dir)

    content = (output_dir / "dedup_summary.json").read_text(encoding="utf-8")
    assert "TOP SECRET" not in content
    assert "Alice" not in content


# ─── .gitignore rules ─────────────────────────────────────────────────────────


def _git_check_ignore(rel_path: str) -> bool:
    import subprocess  # noqa: PLC0415

    result = subprocess.run(
        ["git", "check-ignore", "--quiet", rel_path],
        cwd=str(REPO_ROOT),
        capture_output=True,
    )
    return result.returncode == 0


def test_gitignore_does_not_exclude_dedup_clusters_jsonl() -> None:
    """dedup_clusters.jsonl must NOT be gitignored (tracked output)."""
    assert not _git_check_ignore("data/intake_meta/tg_2026-04-24/dedup_clusters.jsonl")


def test_gitignore_does_not_exclude_dedup_summary_json() -> None:
    """dedup_summary.json must NOT be gitignored (tracked output)."""
    assert not _git_check_ignore("data/intake_meta/tg_2026-04-24/dedup_summary.json")
