#!/usr/bin/env python3
"""
intake_telegram_dedup.py — Exact + perceptual dedup for the Telegram export intake pipeline.

Pass 1: SHA-256 exact dedup from audit.jsonl (no image re-reads).
Pass 2: pHash perceptual near-duplicate clustering using imagehash.phash() +
        numpy vectorized chunked brute force comparison.

source=telegram_private_2026-04-24, license=private_training_only

Usage:
    python3 scripts/intake_telegram_dedup.py [options]

Outputs (tracked, privacy-safe — no captions, sender names, or photo bytes):
    dedup_clusters.jsonl  — one record per cluster (exact or perceptual)
    dedup_summary.json    — aggregate counts + provenance
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

PROGRESS_EVERY = 1000
BATCH_SIZE = 256
DEFAULT_PHASH_THRESHOLD = 8

SOURCE = "telegram_private_2026-04-24"
LICENSE = "private_training_only"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


# ─── Popcount lookup table ────────────────────────────────────────────────────


def _make_popcount_lut() -> np.ndarray:
    """256-entry LUT: lut[b] = popcount(b) for b in 0..255."""
    lut = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        lut[i] = bin(i).count("1")
    return lut


_POPCOUNT_LUT = _make_popcount_lut()


# ─── Union-Find ───────────────────────────────────────────────────────────────


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))
        self._rank = [0] * n

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self._rank[rx] < self._rank[ry]:
            rx, ry = ry, rx
        self._parent[ry] = rx
        if self._rank[rx] == self._rank[ry]:
            self._rank[rx] += 1

    def components(self) -> dict[int, list[int]]:
        """Return dict root → [member indices]."""
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self._parent)):
            groups[self.find(i)].append(i)
        return dict(groups)


# ─── pHash helpers ────────────────────────────────────────────────────────────


def _phash_to_bytes(ph) -> bytes:
    """Convert imagehash.ImageHash to 8 uint8 bytes (64-bit DCT pHash)."""
    return int(str(ph), 16).to_bytes(8, byteorder="big")


def _compute_phashes(
    filenames: list[str],
    export_dir: Path,
) -> tuple[list[str], np.ndarray]:
    """
    Compute pHash for each image file. Returns (valid_filenames, hash_array).
    hash_array shape: (len(valid_filenames), 8) uint8.
    Skips files that fail to open; logs a warning.
    """
    import imagehash  # noqa: PLC0415
    from PIL import Image, UnidentifiedImageError  # noqa: PLC0415

    valid: list[str] = []
    rows: list[bytes] = []
    total = len(filenames)

    for idx, fn in enumerate(filenames, start=1):
        if idx % PROGRESS_EVERY == 0 or idx == total:
            log.info("pHash progress: %d / %d", idx, total)

        path = export_dir / fn
        try:
            with Image.open(path) as img:
                ph = imagehash.phash(img)
            rows.append(_phash_to_bytes(ph))
            valid.append(fn)
        except Exception as exc:
            log.warning("pHash failed for %s: %s", fn, exc)

    if not rows:
        return [], np.empty((0, 8), dtype=np.uint8)

    arr = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(len(rows), 8)
    return valid, arr


# ─── Numpy vectorized chunked comparison ─────────────────────────────────────


def _find_near_dup_pairs(
    hashes: np.ndarray,
    threshold: int,
    batch_size: int = BATCH_SIZE,
) -> list[tuple[int, int, int]]:
    """
    Return list of (i, j, hamming_distance) for all unique pairs where
    hamming_distance <= threshold and i < j.

    Uses numpy vectorized chunked brute force: zero false negatives.
    Memory per batch: ~67 MB at batch_size=256 for n=32K.
    """
    n = len(hashes)
    pairs: list[tuple[int, int, int]] = []
    lut = _POPCOUNT_LUT

    for i_offset in range(0, n, batch_size):
        batch = hashes[i_offset : i_offset + batch_size]  # (B, 8)

        # XOR batch vs all hashes → (B, n, 8) uint8
        xor = batch[:, np.newaxis, :] ^ hashes[np.newaxis, :, :]

        # Popcount via LUT → (B, n) hamming distances
        dist = lut[xor].sum(axis=-1)  # shape (B, n), dtype promoted to int

        # Collect upper-triangle pairs only (global i < global j)
        b_locals, js = np.where(dist <= threshold)
        for b_local, j in zip(b_locals.tolist(), js.tolist()):
            i_global = i_offset + b_local
            if i_global < j:
                pairs.append((i_global, j, int(dist[b_local, j])))

    return pairs


# ─── Keep selection sort keys ─────────────────────────────────────────────────


def _exact_keep_key(rec: dict) -> tuple:
    """Lowest msg_id first; tie-break: ascending filename. None msg_id sorts last."""
    msg_id = rec.get("msg_id")
    return (msg_id if msg_id is not None else float("inf"), rec.get("filename", ""))


def _perceptual_keep_key(rec: dict) -> tuple:
    """Highest max_side first (negated); tie-break: lowest msg_id; second tie-break: filename."""
    max_side = rec.get("max_side") or 0
    msg_id = rec.get("msg_id")
    return (-max_side, msg_id if msg_id is not None else float("inf"), rec.get("filename", ""))


# ─── Pass 1: exact dedup ──────────────────────────────────────────────────────


def _pass1_exact(
    audit_records: list[dict],
    manifest_by_fn: dict[str, dict],
) -> tuple[list[dict], set[str]]:
    """
    Group audit records by SHA-256. For each group of size > 1, select keep
    (lowest msg_id; tie-break: filename). Remainder are exact-dup non-keeps.

    Returns:
        exact_clusters   — list of cluster dicts
        exact_non_keeps  — set of filenames excluded from the pHash pass
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for rec in audit_records:
        sha = rec.get("sha256")
        if sha:
            mfn = manifest_by_fn.get(rec["filename"], {})
            enriched = {**rec, "msg_id": mfn.get("msg_id")}
            groups[sha].append(enriched)

    exact_clusters: list[dict] = []
    exact_non_keeps: set[str] = set()
    cluster_id = 0

    for sha, members in groups.items():
        if len(members) < 2:
            continue
        cluster_id += 1
        members_sorted = sorted(members, key=_exact_keep_key)
        keep = members_sorted[0]
        dupes = members_sorted[1:]
        for d in dupes:
            exact_non_keeps.add(d["filename"])
        exact_clusters.append({
            "cluster_id": cluster_id,
            "cluster_type": "exact",
            "keep_filename": keep["filename"],
            "duplicate_filenames": [d["filename"] for d in dupes],
            "hamming_distance": None,
            "reason": f"sha256={sha}",
        })

    return exact_clusters, exact_non_keeps


# ─── Pass 2: perceptual dedup ─────────────────────────────────────────────────


def _pass2_perceptual(
    audit_records: list[dict],
    manifest_by_fn: dict[str, dict],
    exact_non_keeps: set[str],
    export_dir: Path,
    threshold: int,
    batch_size: int = BATCH_SIZE,
    cluster_id_start: int = 1,
) -> list[dict]:
    """
    Compute pHash for non-corrupt, non-exact-dup-non-keep images, find near-duplicate
    pairs via numpy vectorized chunked comparison, form clusters via union-find.
    """
    candidates = [
        r for r in audit_records
        if not r.get("corrupt") and r["filename"] not in exact_non_keeps
    ]

    if not candidates:
        return []

    filenames = [r["filename"] for r in candidates]
    audit_by_fn = {r["filename"]: r for r in candidates}

    log.info("Computing pHash for %d candidate images ...", len(filenames))
    valid_fns, hashes = _compute_phashes(filenames, export_dir)

    if len(valid_fns) < 2:
        return []

    log.info("Comparing %d pHashes (threshold=%d) ...", len(valid_fns), threshold)
    pairs = _find_near_dup_pairs(hashes, threshold, batch_size)
    log.info("Found %d near-duplicate pairs", len(pairs))

    # Log top-10 closest pairs for threshold validation
    if pairs:
        top10 = sorted(pairs, key=lambda p: p[2])[:10]
        log.info("Top-10 closest pairs (hamming distance distribution):")
        for pi, pj, pd in top10:
            log.info("  hamming=%d  %s  <->  %s", pd, valid_fns[pi], valid_fns[pj])

    if not pairs:
        return []

    # Union-find cluster formation
    n = len(valid_fns)
    uf = _UnionFind(n)
    for i, j, _ in pairs:
        uf.union(i, j)

    # Pre-compute minimum hamming distance for each pair (for cluster record)
    pair_min_dist: dict[tuple[int, int], int] = {}
    for pi, pj, pd in pairs:
        key = (min(pi, pj), max(pi, pj))
        if key not in pair_min_dist or pd < pair_min_dist[key]:
            pair_min_dist[key] = pd

    perceptual_clusters: list[dict] = []
    cluster_id = cluster_id_start

    for root, members in uf.components().items():
        if len(members) < 2:
            continue

        member_fns = [valid_fns[m] for m in members]
        member_recs = []
        for fn in member_fns:
            audit_rec = audit_by_fn.get(fn, {})
            mfn = manifest_by_fn.get(fn, {})
            member_recs.append({**audit_rec, "msg_id": mfn.get("msg_id")})

        member_recs_sorted = sorted(member_recs, key=_perceptual_keep_key)
        keep = member_recs_sorted[0]
        dupes = member_recs_sorted[1:]

        # Minimum hamming distance among all intra-cluster pairs
        sorted_members = sorted(members)
        min_dist: int | None = None
        for a_idx in range(len(sorted_members)):
            for b_idx in range(a_idx + 1, len(sorted_members)):
                key = (sorted_members[a_idx], sorted_members[b_idx])
                if key in pair_min_dist:
                    d = pair_min_dist[key]
                    if min_dist is None or d < min_dist:
                        min_dist = d

        perceptual_clusters.append({
            "cluster_id": cluster_id,
            "cluster_type": "perceptual",
            "keep_filename": keep["filename"],
            "duplicate_filenames": [d["filename"] for d in dupes],
            "hamming_distance": min_dist,
            "reason": f"phash_hamming<={threshold}",
        })
        cluster_id += 1

    return perceptual_clusters


# ─── Cross-pass conflict resolution ─────────────────────────────────────────


def _resolve_conflicts(
    exact_clusters: list[dict],
    perceptual_clusters: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Resolve cases where an exact-cluster keep_filename overlaps with a
    perceptual cluster — either as a perceptual duplicate or as the
    perceptual keep itself.

    In both cases the exact cluster is absorbed:
    - The exact cluster's own duplicates are added to the perceptual
      cluster's duplicate list.
    - The exact cluster is removed.

    This preserves the integrity invariant: no keep_filename appears in
    more than one cluster record, and no filename appears as both
    keep_filename and in duplicate_filenames[].
    """
    # Index 1: perceptual dup filename → cluster
    fn_to_pc_dup: dict[str, dict] = {}
    for pc in perceptual_clusters:
        for fn in pc["duplicate_filenames"]:
            fn_to_pc_dup[fn] = pc

    # Index 2: perceptual keep filename → cluster
    fn_to_pc_keep: dict[str, dict] = {}
    for pc in perceptual_clusters:
        fn_to_pc_keep[pc["keep_filename"]] = pc

    resolved_exact: list[dict] = []
    for ec in exact_clusters:
        keep = ec["keep_filename"]
        if keep in fn_to_pc_dup:
            # exact keep is a perceptual dup → absorb into that cluster
            target = fn_to_pc_dup[keep]
            target["duplicate_filenames"].extend(ec["duplicate_filenames"])
            log.info(
                "Conflict resolved (keep-as-dup): exact cluster %d (keep=%s) "
                "absorbed into perceptual cluster %d",
                ec["cluster_id"],
                keep,
                target["cluster_id"],
            )
        elif keep in fn_to_pc_keep:
            # exact keep is also the perceptual keep → absorb dups only
            target = fn_to_pc_keep[keep]
            target["duplicate_filenames"].extend(ec["duplicate_filenames"])
            log.info(
                "Conflict resolved (keep-as-keep): exact cluster %d (keep=%s) "
                "absorbed into perceptual cluster %d",
                ec["cluster_id"],
                keep,
                target["cluster_id"],
            )
        else:
            resolved_exact.append(ec)

    return resolved_exact, perceptual_clusters


# ─── Main pipeline ────────────────────────────────────────────────────────────


def run(
    audit_path: Path,
    manifest_path: Path,
    export_dir: Path,
    output_dir: Path,
    phash_threshold: int = DEFAULT_PHASH_THRESHOLD,
    dry_run: bool = False,
    batch_size: int = BATCH_SIZE,
) -> tuple[int, dict]:
    """
    Main entry point. Returns (clusters_written, summary_dict).
    Writes dedup_clusters.jsonl and dedup_summary.json unless dry_run=True.
    """
    if not audit_path.exists():
        log.error("audit.jsonl not found: %s", audit_path)
        log.error("Run intake_telegram_audit.py first")
        sys.exit(1)

    if not manifest_path.exists():
        log.error("manifest.jsonl not found: %s", manifest_path)
        log.error("Run intake_telegram_manifest.py first")
        sys.exit(1)

    # Load audit.jsonl
    audit_records: list[dict] = []
    with audit_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                audit_records.append(json.loads(line))
    log.info("Loaded %d audit records", len(audit_records))

    # Load manifest.jsonl (msg_id join only — no captions or sender data used)
    manifest_by_fn: dict[str, dict] = {}
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                manifest_by_fn[rec["filename"]] = rec
    log.info("Loaded %d manifest records", len(manifest_by_fn))

    input_records = len(audit_records)

    # Pass 1: exact dedup
    log.info("Pass 1: exact dedup ...")
    exact_clusters, exact_non_keeps = _pass1_exact(audit_records, manifest_by_fn)
    exact_removed = sum(len(c["duplicate_filenames"]) for c in exact_clusters)
    log.info(
        "  exact clusters: %d  exact removed: %d",
        len(exact_clusters),
        exact_removed,
    )

    # Pass 2: perceptual dedup
    log.info("Pass 2: perceptual dedup (threshold=%d) ...", phash_threshold)
    # Upper bound on images entering the pHash pass: total minus exact non-keeps.
    # This is a pre-conflict-resolution snapshot; the post-resolution exact_removed
    # in the summary may differ by the number of absorbed clusters. Corrupt images
    # are excluded inside _pass2_perceptual but not subtracted here, so the name
    # intentionally says "upper_bound" rather than "exact count".
    phash_candidates_upper_bound = input_records - exact_removed
    perceptual_clusters = _pass2_perceptual(
        audit_records,
        manifest_by_fn,
        exact_non_keeps,
        export_dir,
        phash_threshold,
        batch_size,
        cluster_id_start=len(exact_clusters) + 1,
    )
    perceptual_removed = sum(len(c["duplicate_filenames"]) for c in perceptual_clusters)
    log.info(
        "  perceptual clusters: %d  perceptual removed: %d",
        len(perceptual_clusters),
        perceptual_removed,
    )

    # Resolve cross-pass conflicts: exact-cluster keep that is also a perceptual dup
    exact_clusters, perceptual_clusters = _resolve_conflicts(exact_clusters, perceptual_clusters)

    # Recompute counts from resolved clusters
    all_clusters = exact_clusters + perceptual_clusters
    exact_removed = sum(len(c["duplicate_filenames"]) for c in exact_clusters)
    perceptual_removed = sum(len(c["duplicate_filenames"]) for c in perceptual_clusters)
    total_removed = exact_removed + perceptual_removed
    total_unique = input_records - total_removed

    # Clusters whose minimum intra-cluster hamming distance equals the threshold exactly —
    # these are boundary cases with elevated false-positive risk and require spot-check.
    boundary_clusters_at_threshold = sum(
        1 for c in perceptual_clusters if c["hamming_distance"] == phash_threshold
    )

    summary = {
        "input_records": input_records,
        "exact_clusters": len(exact_clusters),
        "exact_removed": exact_removed,
        "phash_candidates_upper_bound": phash_candidates_upper_bound,
        "phash_threshold": phash_threshold,
        "perceptual_clusters": len(perceptual_clusters),
        "perceptual_removed": perceptual_removed,
        "boundary_clusters_at_threshold": boundary_clusters_at_threshold,
        "total_unique_after_dedup": total_unique,
        "provisional": True,
        "manual_review_required": True,
        "manual_review_reason": (
            f"{boundary_clusters_at_threshold} of {len(perceptual_clusters)} perceptual "
            f"clusters have hamming_distance={phash_threshold} (the threshold boundary), "
            "creating false-positive risk. Spot-check boundary clusters before use in "
            "staging or training."
        ),
        "source": SOURCE,
        "license": LICENSE,
    }

    log.info(
        "Summary: input=%d exact_clusters=%d perceptual_clusters=%d unique=%d",
        input_records,
        len(exact_clusters),
        len(perceptual_clusters),
        total_unique,
    )

    if dry_run:
        log.info("[dry-run] No files written.")
        print(json.dumps(summary, indent=2), file=sys.stdout)
        return 0, summary

    output_dir.mkdir(parents=True, exist_ok=True)
    dedup_path = output_dir / "dedup_clusters.jsonl"
    summary_path = output_dir / "dedup_summary.json"

    with dedup_path.open("w", encoding="utf-8") as fh:
        for cluster in all_clusters:
            fh.write(json.dumps(cluster, ensure_ascii=False) + "\n")

    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log.info("dedup_clusters.jsonl → %s (%d clusters)", dedup_path, len(all_clusters))
    log.info("dedup_summary.json   → %s", summary_path)

    return len(all_clusters), summary


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Exact + perceptual dedup for Telegram export intake pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--audit",
        default=str(REPO_ROOT / "data" / "intake_meta" / "tg_2026-04-24" / "audit.jsonl"),
        help="Path to audit.jsonl produced by intake_telegram_audit.py",
    )
    p.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "data" / "intake_meta" / "tg_2026-04-24" / "manifest.jsonl"),
        help="Path to manifest.jsonl produced by intake_telegram_manifest.py",
    )
    p.add_argument(
        "--export-dir",
        default=str(
            Path.home() / "Downloads" / "Telegram Desktop" / "ChatExport_2026-04-24"
        ),
        help="Path to the Telegram export directory (read-only; used for pHash).",
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "intake_meta" / "tg_2026-04-24"),
        help="Directory where dedup_clusters.jsonl will be written.",
    )
    p.add_argument(
        "--phash-threshold",
        type=int,
        default=DEFAULT_PHASH_THRESHOLD,
        help="Maximum hamming distance to consider two images near-duplicates.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute everything, log summary, do not write output files.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    audit_path = Path(args.audit).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    export_dir = Path(args.export_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    n, summary = run(
        audit_path,
        manifest_path,
        export_dir,
        output_dir,
        phash_threshold=args.phash_threshold,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(
            f"[OK] dedup_clusters.jsonl written: {n} clusters → "
            f"{output_dir / 'dedup_clusters.jsonl'}"
        )
        print(f"[OK] dedup_summary.json           → {output_dir / 'dedup_summary.json'}")


if __name__ == "__main__":
    main()
