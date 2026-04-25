---
title: "feat: U3 Perceptual Dedup — Telegram Export Intake Pipeline"
type: feat
status: active
date: 2026-04-25
origin: docs/plans/2026-04-25-001-feat-telegram-export-dataset-intake-plan.md
---

# feat: U3 Perceptual Dedup — Telegram Export Intake Pipeline

## Overview

Implement U3 of the Telegram export intake pipeline: exact-duplicate detection using
pre-computed SHA-256 from `audit.jsonl` (no image re-reads), followed by perceptual
near-duplicate clustering using `imagehash.phash()` with numpy-vectorized chunked
comparison. Outputs `dedup_clusters.jsonl` (one record per cluster, tracked, ~20–50 KB)
and `dedup_summary.json` (tracked aggregate counts).

Phase 1 baseline from `audit_summary.json`: 32,664 records, 0 corrupt, 47 SHA-256
exact-duplicate groups, 47 excess files.

---

## Problem Frame

The Telegram export contains re-posted and slightly re-compressed versions of the same
fish photo sent across multiple chat sessions. Before classification (U4/U5) and staging
(U6), redundant images must be identified and one canonical copy selected per cluster.
U2 already computed SHA-256 for every photo; those hashes drive exact dedup without any
additional file I/O. Perceptual dedup catches near-duplicates that escape SHA-256 (same
photo, different compression settings or minor crops).

(see origin: `docs/plans/2026-04-25-001-feat-telegram-export-dataset-intake-plan.md`)

---

## Requirements Trace

- R3. Detect and cluster exact (SHA-256) and near-duplicate (pHash) photos; select one
  `keep` per cluster.
- R10. JSONL text artifacts in `data/intake_meta/` are committable; no photo files enter git.

---

## Scope Boundaries

- No image file modification, no copies, no training, no GPT calls, no classification.
- Reads: `audit.jsonl`, `manifest.jsonl`, photo files (read-only, pHash only).
- Writes: `dedup_clusters.jsonl`, `dedup_summary.json` only.
- No `.gitignore` modifications required (see Output Policy in Key Technical Decisions).
- No upscaling; no changes to `bot/`, `data/fish_dataset/`, or any training scripts.

---

## Context & Research

### Relevant Code and Patterns

- `scripts/intake_telegram_audit.py` — canonical argparse / logging / JSONL write /
  summary JSON conventions to mirror exactly.
- `scripts/check_duplicates.py` — SHA-256 group aggregation pattern; adapt for Pass 1.
- `scripts/intake_constants.py` — AUDIT_PATH, MANIFEST_PATH, DEDUP_PATH, EXPORT_DIR,
  SOURCE_TAG, LICENSE_TAG.
- Phase 1 outputs: `data/intake_meta/tg_2026-04-24/audit.jsonl` (32,664 records, all
  sha256 present, 0 corrupt); `data/intake_meta/tg_2026-04-24/manifest.jsonl` (32,664
  records, all msg_id present).
- `tests/test_intake_audit.py` — test structure and helper patterns to follow.
- `tests/test_intake_safety.py` — gitignore assertion pattern to extend.

### Institutional Learnings

- PIL two-open pattern from U2: verify() invalidates the handle; re-open to read .size.
  imagehash.phash() accepts a PIL Image object directly — open once, pass to phash(),
  no double-open needed (verify separately if needed, but audit already confirmed 0
  corrupt; skip verify in pHash pass).
- JSONL write pattern: open output file, write one record per line with `json.dumps(...,
  ensure_ascii=False) + "\n"`.
- Summary JSON: written alongside JSONL; contains provenance fields (source, license).
- Progress log: every 1,000 items, as in audit script.

---

## Key Technical Decisions

- **SHA-256 from audit.jsonl, not re-read**: SHA-256 is already computed and stored.
  Pass 1 reads `audit.jsonl` in-memory only — zero image file opens. Rationale: avoids
  redundant I/O for 32K files; hashes are authoritative from U2.

- **Keep selection — exact duplicates**: within each SHA-256 group, keep the record with
  the lowest `msg_id` (earliest message in the chat, preserving original). The sort key
  must handle None msg_id safely: use `(msg_id if msg_id is not None else float('inf'),
  filename)` as the sort tuple. This ensures records with integer msg_id always rank
  above records with None, and a group containing a mix of integer and None msg_ids
  sorts correctly without a TypeError (Python 3 raises TypeError on int < None).
  Requires join of `audit.jsonl` and `manifest.jsonl` on `filename` field.

- **Keep selection — perceptual duplicates**: within each perceptual cluster, keep the
  record with the largest `max_side` (highest resolution). Tie-break: lowest `msg_id`;
  second tie-break: ascending filename sort. `max_side` is available in `audit.jsonl`.

- **pHash library**: `imagehash.phash()` (64-bit DCT-based hash). Not `average_hash`
  or `dhash`. pHash is the most robust against minor JPEG re-compression and resize,
  which are the dominant transformations in re-posted Telegram photos.

- **Images to pHash**: all non-corrupt images that are not exact-dup non-keeps. With
  0 corrupt and 47 exact-dup excess, this is 32,664 - 47 = 32,617 images.

- **Comparison algorithm: numpy vectorized chunked brute force** — NOT the single-bucket
  approach described in the parent plan's U3 section. Rationale: the "top 8 bits" bucket
  strategy is incorrect for hamming threshold ≤ 8. Counter-example: two images differing
  in exactly 8 bits could have all 8 differences in the top 8 bits of their pHash integer,
  landing them in different buckets → missed pair. For 32,617 images, numpy chunked brute
  force is correct (zero false negatives), practical (~15 sec comparison), and
  straightforward to validate:
    1. Convert each pHash to 8 uint8 bytes (64 bits).
    2. Load all n hashes into a numpy array of shape (n, 8).
    3. Process in batches of B ≈ 256 images vs all n. Let `i_offset = batch_index * B`:
       - XOR: hashes[i_offset:i_offset+B] vs all n hashes → (B, n, 8) uint8
       - Popcount via lookup table (precomputed 256-entry LUT): (B, n, 8) → (B, n) hamming distances
       - Collect pairs `(i_offset + b_local, j)` where distance ≤ threshold and
         `(i_offset + b_local) < j`. Both conditions use global indices — `b_local`
         must be promoted to global with `i_offset + b_local` before comparing against
         `j`. Using `b_local < j` instead would accept same-batch pairs (j, k) where
         j < i_offset, producing double-counted or inverted pairs.
    4. Memory per batch: ~67 MB. Total batches: ~128. Expected comparison time: ~15 sec.
  The parent plan's bucket approach is replaced. See Alternatives Considered.

- **Perceptual cluster formation**: greedy union-find. When pair (i, j) is found with
  hamming ≤ threshold, merge their components. One cluster record per component of size > 1.
  Greedy is correct here: we expect O(10–100) perceptual clusters; no pathological chain
  behavior.

- **Threshold default**: hamming ≤ 8, exposed as `--phash-threshold N` CLI arg.
  Rationale: JPEG re-compression of the same image typically differs by 2–4 hamming bits;
  unrelated fish photos typically differ by 20–40 bits. 8 provides a conservative buffer
  with low false-positive risk.

- **Output policy**: `dedup_clusters.jsonl` is **tracked** (committed). Contains only
  filenames, cluster type, and hamming distance — no captions, no sender names, no PII.
  File is ~20–50 KB (O(100) clusters × ~200 bytes/record). The `audit.jsonl` gitignore
  rule was motivated by size (8+ MB); `manifest.jsonl` by privacy (captions, sender
  names). Neither motivation applies here. `dedup_summary.json` is also tracked. No new
  `.gitignore` rules needed.

- **Idempotent re-run**: script overwrites output files on every run. No resume logic.
  Full run time < 3 minutes; re-running is cheap.

- **Cluster ID**: sequential integer, 1-indexed, stable within a single run.

- **`hamming_distance` field**: `null` for exact-dup clusters (SHA-256 match, not pHash).
  Integer 0–8 for perceptual clusters.

---

## Open Questions

### Resolved During Planning

- **Should dedup_clusters.jsonl be gitignored?** No. No PII; ~20–50 KB. Unlike
  audit.jsonl (8+ MB) and manifest.jsonl (captions + sender names), this file is small
  and privacy-safe. Tracked alongside dedup_summary.json.
- **Should we re-hash images for exact dedup?** No. SHA-256 is already in audit.jsonl.
  Pass 1 is a pure in-memory group-by operation.
- **Should exact-dup non-keeps be pHashed?** No. They will be discarded in staging; pHashing
  them adds work with no benefit.
- **Does imagehash.phash() accept a PIL Image?** Yes. Open image once, pass to
  `imagehash.phash(img)` directly. No double-open needed for pHash pass.
- **Single-bucket correctness issue from parent plan**: Confirmed incorrect; replaced with
  numpy chunked brute force (see Key Technical Decisions).

### Deferred to Implementation

- **Actual perceptual cluster count**: Unknown until the script runs. Expected: 10–100
  clusters based on re-post patterns in chat exports. Verify after run.
- **Exact chunk size B**: Start at 256. Increase to 512 if timing allows; decrease if
  memory pressure observed. 512 requires ~135 MB per batch.
- **Optimal progress granularity for pHash phase**: 1,000 images (matching audit script).
  Adjust if pHash compute is faster than expected.
- **Hamming distance distribution shape**: Log top-10 closest pairs and their distances
  after the run as a threshold validation aid.

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not
> implementation specification. The implementing agent should treat it as context, not
> code to reproduce.*

```
audit.jsonl  ──┐
               ├──► [Pass 1: SHA-256 group-by]
manifest.jsonl ┘         │
                         │  47 exact-dup clusters
                         │  32,617 non-redundant filenames
                         ▼
                [Pass 2: imagehash.phash() for 32,617 images]
                  open each image with PIL
                  imagehash.phash(img) → ImageHash object
                  convert to 8-byte uint8 row
                  progress: log every 1,000
                         │
                         ▼
           [numpy chunked comparison, B=256 batch size]
             for i_offset in range(0, n, B):
               batch = hashes[i_offset:i_offset+B]          # (B, 8)
               xor   = batch[:, None, :] ^ hashes[None, :, :]  # (B, n, 8)
               dist  = popcount_lut[xor].sum(axis=-1)          # (B, n)
               # collect only upper-triangle pairs (global i < global j)
               for b_local, j in argwhere(dist <= threshold):
                   i_global = i_offset + b_local
                   if i_global < j:
                       emit_pair(i_global, j, dist[b_local, j])
             → list of (i_global, j, distance) near-duplicate pairs
                         │
                         ▼
           [Union-Find cluster formation]
             merge components for each near-dup pair
             components of size > 1 → perceptual clusters
             keep per cluster = max(max_side); tie-break min(msg_id)
                         │
                         ▼
           [Write outputs]
             dedup_clusters.jsonl  (exact + perceptual clusters)
               {cluster_id, cluster_type, keep_filename,
                duplicate_filenames, hamming_distance, reason}
             dedup_summary.json    (aggregate counts, tracked)
               {input_records, exact_clusters, exact_removed,
                perceptual_candidates, phash_threshold,
                perceptual_clusters, perceptual_removed,
                total_unique_after_dedup, source, license}
```

---

## Implementation Units

- U1. **Dedup Script**

**Goal:** Implement `scripts/intake_telegram_dedup.py`: two-pass dedup (SHA-256 exact
then pHash perceptual), numpy vectorized comparison, cluster output, summary JSON.

**Requirements:** R3, R10

**Dependencies:** audit.jsonl and manifest.jsonl must exist and be complete (U1 + U2
from parent plan).

**Files:**
- Create: `scripts/intake_telegram_dedup.py`
- Creates: `data/intake_meta/tg_2026-04-24/dedup_clusters.jsonl`
- Creates: `data/intake_meta/tg_2026-04-24/dedup_summary.json`
- Test: `tests/test_intake_dedup.py`

**Approach:**
- CLI args via argparse (follow audit script conventions): `--audit`, `--manifest`,
  `--export-dir`, `--output-dir`, `--phash-threshold` (default 8), `--dry-run`
- Load audit.jsonl → dict keyed on `filename`
- Load manifest.jsonl → dict keyed on `filename` (for msg_id join only)
- **Pass 1**: group audit records by sha256; each group of size > 1 is one exact cluster;
  select keep (lowest msg_id; tie-break: filename sort); remainder are duplicates
- **Pass 2**: for each non-corrupt, non-exact-dup-non-keep image, open with PIL and call
  `imagehash.phash(img)` to get 64-bit hash; convert to 8-byte uint8 row; log progress
  every 1,000 images
- **Comparison**: build numpy array of shape (n, 8); process in batches of 256; apply XOR
  + popcount LUT; collect pairs with hamming ≤ threshold; form clusters via union-find;
  select keep (largest max_side; tie-break: lowest msg_id)
- Write `dedup_clusters.jsonl`: one record per cluster, format:
  `{cluster_id, cluster_type, keep_filename, duplicate_filenames[], hamming_distance, reason}`
- Write `dedup_summary.json`: aggregate counts + provenance fields (source, license)
- Log top-10 closest perceptual pairs and their hamming distances for threshold validation
- `--dry-run`: compute everything, log summary, do not write output files

**Patterns to follow:**
- `scripts/intake_telegram_audit.py` — argparse, logging, JSONL write, summary dict,
  progress every 1,000, `run()` function returning `(n_written, summary)`, main()
- `scripts/check_duplicates.py` — hash group aggregation from `defaultdict(list)`
- `scripts/intake_constants.py` — DEDUP_PATH, AUDIT_PATH, MANIFEST_PATH constants

**Test scenarios:**
- Happy path — exact dup pair: two audit records with identical sha256, both with valid
  msg_id → one cluster, `cluster_type=exact`, keep = record with lower msg_id
- Mixed-None msg_id in exact-dup group: one record has `msg_id=None`, other has
  `msg_id=42` → keep = the record with `msg_id=42`; no TypeError raised. Sort key
  must use `(msg_id if msg_id is not None else float('inf'), filename)`, not a bare
  comparison of int against None.
- All-None msg_id in exact-dup group: both records have `msg_id=None` → keep =
  lexicographically first filename; no exception raised
- Happy path — near-dup pair: two images with hamming distance 4 (≤ 8) → one cluster,
  `cluster_type=perceptual`, keep = record with larger `max_side`
- Tie-break max_side: two near-dup images with equal max_side → keep has lower msg_id
- Distinct pair: hamming distance 12 → no cluster, neither filename in output
- All-unique dataset: `dedup_clusters.jsonl` has 0 records; summary clusters == 0
- Custom threshold: `--phash-threshold 12` clusters the hamming-12 pair that default misses
- Exact-dup non-keep excluded from pHash: the discarded exact duplicate must not appear
  in any perceptual cluster's keep_filename
- Corrupt image handling: image with `corrupt=true` in audit skipped in pHash pass;
  its valid near-duplicate partner remains unclustered (not a false cluster)
- Same-batch pair correctness: use a test dataset of 3 near-duplicate images where
  all 3 fit within a single batch (B=256). After running with threshold 8, all 3
  must be merged into one cluster with one `keep_filename` and two `duplicate_filenames`.
  This exercises the global index mapping in the batch loop (b_local must be promoted
  to `i_offset + b_local` before the `< j` comparison).
- Cluster integrity — no filename in both keep and duplicates: across all clusters,
  no filename appears as both `keep_filename` and in any `duplicate_filenames[]`
- Cluster integrity — no filename in multiple clusters: no filename appears in
  `duplicate_filenames[]` of more than one cluster
- Count invariant: `total_unique_after_dedup` in summary equals
  `input_records - sum(len(duplicate_filenames) for all clusters)`
- Summary JSON required keys: `input_records`, `exact_clusters`, `exact_removed`,
  `perceptual_candidates`, `phash_threshold`, `perceptual_clusters`,
  `perceptual_removed`, `total_unique_after_dedup`, `source`, `license`
- `--dry-run`: no files written; summary printed to stderr/stdout
- Missing audit file: `sys.exit(1)` with message "Run intake_telegram_audit.py first"
- Missing manifest file: `sys.exit(1)` with message "Run intake_telegram_manifest.py first"

**Verification:**
- `dedup_summary.json` field `exact_clusters == 47` (matches audit_summary.json baseline)
- No filename appears as both `keep_filename` and in any `duplicate_filenames[]`
- `total_unique_after_dedup + sum(len(duplicate_filenames))` == 32,664
- Script completes without exception on the full 32,664-record dataset
- Total runtime < 5 minutes on Apple Silicon

---

- U2. **Safety Test Extension**

**Goal:** Extend `tests/test_intake_safety.py` with two gitignore assertions confirming
that `dedup_clusters.jsonl` and `dedup_summary.json` are tracked (not gitignored).
Enforces the output policy decision for U3 so it cannot be accidentally broken by a
future `.gitignore` edit.

**Requirements:** R10

**Dependencies:** U1 (path constants must be importable from `intake_constants.py`)

**Files:**
- Modify: `tests/test_intake_safety.py`

**Approach:**
- Add `test_dedup_clusters_not_gitignored()`: assert `git check-ignore -v
  data/intake_meta/tg_2026-04-24/dedup_clusters.jsonl` returns non-zero (not ignored)
- Add `test_dedup_summary_not_gitignored()`: assert same for `dedup_summary.json`
- Mirror the `test_manifest_summary_not_gitignored()` pattern already in the file

**Patterns to follow:**
- `tests/test_intake_safety.py` `test_manifest_summary_not_gitignored()` function

**Test scenarios:**
- `dedup_clusters.jsonl` at `data/intake_meta/tg_2026-04-24/dedup_clusters.jsonl` is
  not matched by any gitignore rule → tracked
- `dedup_summary.json` at same path is not matched by any gitignore rule → tracked

**Verification:**
- Both new tests pass against the current `.gitignore` without any `.gitignore` changes

---

## System-Wide Impact

- **Interaction graph:** Dedup script reads audit.jsonl and manifest.jsonl (written by
  U1+U2 of parent plan). It does not touch `bot/`, `data/bot.db`, `data/fish_dataset/`,
  launchd, or any running process. Downstream U6 (staging) reads `dedup_clusters.jsonl`
  to build its skip set — dedup output is a prerequisite for staging.
- **Error propagation:** Individual image pHash failures log and skip. Missing input
  files cause `sys.exit(1)`. Output files are not written until processing completes
  (write to temp file, rename on success, or write in one pass — implementer choice).
- **State lifecycle risks:** Script is idempotent. Re-running overwrites `dedup_clusters.jsonl`
  and `dedup_summary.json`. No partial-write persistence; if interrupted, re-run from scratch.
- **API surface parity:** No new bot endpoints. Purely offline.
- **Unchanged invariants:** `audit.jsonl` and `manifest.jsonl` are read-only in this pass.
  Existing `.gitignore` unchanged.

---

## Alternative Approaches Considered

- **Multi-index hashing (q=9 tables)**: Correct for threshold ≤ 8 via pigeonhole (q > T
  guarantees ≥1 table has identical substring). Rejected: implementation complexity is
  higher than numpy brute force, and for 32K images the brute-force approach is fast
  enough (~15 sec) with zero false negatives.
- **Single-bucket by top 8 bits (parent plan approach)**: Incorrect — two images differing
  in ≤ 8 bits could have all differences in the same byte, placing them in different
  buckets. Rejected.
- **Sorted-array sliding-window**: Sort pHash integers, compare adjacent entries within
  window W. Misses pairs with large numerical distance but small hamming distance (common
  in pHashes). Rejected due to false negatives.
- **BK-tree / VP-tree**: O(n log n) exact search. Correct and efficient for very large
  datasets (>1M images). Overkill for 32K; adds a dependency. Rejected.

---

## Risks & Dependencies

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| pHash false positives (different species, similar color) | Medium | Medium | Fishing photos share recurrent visual patterns (water, dark silhouettes, bright sky); two taxonomically distinct species in similar lighting could produce hamming < 8. Threshold ≤ 8 is a starting point — empirical validation is required. Log hamming distance distribution; manually inspect all perceptual clusters before staging. Do not treat staging as safe until cluster inspection passes. |
| numpy XOR comparison misses pairs | None | — | Brute force: all (i,j) pairs examined; zero false negatives by design |
| pHash compute time > 5 min | Low | Low | Progress bar; escape hatch: `concurrent.futures.ThreadPoolExecutor` for parallel pHash compute if needed |
| Memory pressure from numpy batches | Low | Low | 67 MB per batch (B=256); reduce B to 128 (33 MB) if needed |
| `dedup_clusters.jsonl` accidentally gitignored in future | Low | Low | U2 safety test enforces tracked status; will fail CI if gitignore adds the rule |
| Union-find chain errors in cluster formation | Low | Medium | Verify with cluster integrity tests (no filename in both keep and duplicates) |
| audit.jsonl sha256 field None for a record | Low | Low | Skip records with sha256==None in Pass 1 (would indicate a corrupt image from U2, but corrupt=0 in current dataset) |

---

## Documentation / Operational Notes

- **Rollback**: U3 writes only `dedup_clusters.jsonl` and `dedup_summary.json`. Delete
  both files to reset the phase entirely. `audit.jsonl` and `manifest.jsonl` are
  unaffected. Re-run the script to regenerate.
- **Threshold validation**: After first run, inspect `dedup_clusters.jsonl` perceptual
  clusters. Open each pair of photos (keep + duplicate) side-by-side to verify visual
  similarity. The logged top-10 closest pairs with their hamming distances provide a
  starting point. If false positives found, decrease threshold (e.g., 6) and re-run.
  If real near-duplicates are visually obvious but unclustered during staging review,
  increase threshold (e.g., 10) and re-run.
- **Commit boundary**: This is commit boundary 3 from the parent plan's Phased Delivery.
  Commit includes: `scripts/intake_telegram_dedup.py`, `tests/test_intake_dedup.py`,
  updated `tests/test_intake_safety.py`, `data/intake_meta/tg_2026-04-24/dedup_clusters.jsonl`,
  `data/intake_meta/tg_2026-04-24/dedup_summary.json`.
- **Pre-commit check**: run `pytest tests/test_intake_dedup.py tests/test_intake_safety.py -v`
  before committing.
- **Provenance staleness on future re-runs**: `dedup_summary.json` contains
  `SOURCE_TAG = "telegram_private_2026-04-24"` from `intake_constants.py`. If this
  pipeline is re-run on a new export (e.g., `tg_2027-01-15`), update `BATCH_ID` in
  `intake_constants.py` before running — otherwise the committed summary will carry the
  wrong batch provenance. The `source` field in `dedup_summary.json` is the canonical
  signal for which export the file belongs to.

---

## Sources & References

- **Origin document:** [docs/plans/2026-04-25-001-feat-telegram-export-dataset-intake-plan.md](docs/plans/2026-04-25-001-feat-telegram-export-dataset-intake-plan.md) (U3, Key Technical Decisions)
- Phase 1 results: [data/intake_meta/tg_2026-04-24/audit_summary.json](data/intake_meta/tg_2026-04-24/audit_summary.json) (47 SHA-256 dup groups, 0 corrupt)
- Related code: [scripts/intake_telegram_audit.py](scripts/intake_telegram_audit.py) — argparse/logging/summary/JSONL output conventions
- Related code: [scripts/check_duplicates.py](scripts/check_duplicates.py) — SHA-256 group aggregation
- Related code: [scripts/intake_constants.py](scripts/intake_constants.py) — DEDUP_PATH, AUDIT_PATH, MANIFEST_PATH
- Related tests: [tests/test_intake_audit.py](tests/test_intake_audit.py) — test structure and helper pattern
- Related tests: [tests/test_intake_safety.py](tests/test_intake_safety.py) — gitignore assertion pattern to extend
- External: `imagehash>=4.3.1` (PyPI, already in `requirements-intake.txt`)
