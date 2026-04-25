---
title: "feat: U3 Manual Inspection Workflow for Perceptual Dedup Boundary Clusters"
type: feat
status: active
date: 2026-04-25
origin: docs/plans/2026-04-25-002-feat-telegram-dedup-u3-perceptual-plan.md
---

# feat: U3 Manual Inspection Workflow for Perceptual Dedup Boundary Clusters

## Overview

`dedup_clusters.jsonl` is provisional: 285 of 512 perceptual clusters formed at exactly
`hamming_distance == 8`, the detection-threshold boundary. These carry the highest false-positive
risk — visually similar but taxonomically distinct fishing photos in similar lighting can produce
hamming == 8. This workflow adds local-only tooling to inspect those 285 clusters side-by-side,
record human review decisions at cluster and per-image granularity, and produce a privacy-safe
tracked summary that either validates the current threshold or recommends a re-run at a lower
value. Finalizing this step is the hard blocker before any downstream classification,
privacy-flagging, or staging work (U4–U8) can proceed.

---

## Problem Frame

`intake_telegram_dedup.py` ran at `--phash-threshold 8` and removed 835 images in 512 clusters.
Of those, 285 clusters formed at exactly `hamming_distance == 8` — the weakest case for
"near-duplicate." The fishing domain exacerbates false-positive risk: water backgrounds, dark
fish silhouettes, and similar lighting conditions are common across taxonomically distinct species.
The existing `dedup_summary.json` explicitly sets `provisional: true` and
`manual_review_required: true` to block downstream use until this review completes.

The full perceptual cluster population by hamming distance is: `{0: 150, 2: 25, 4: 14, 6: 38,
8: 285}`. The 150 clusters at hamming=0 are pHash-identical images that differ only in byte
content (JPEG re-compression artifacts) — these are functionally equivalent to SHA-256 exact
duplicates and are not in scope for visual review (see Key Technical Decisions). The 77 clusters
at hamming 2–6 are low false-positive risk. Only the 285 hamming=8 boundary clusters require
mandatory human inspection. The 42 SHA-256 exact clusters need no visual review.

Reference: `docs/plans/2026-04-25-002-feat-telegram-dedup-u3-perceptual-plan.md` (Operational
Notes and Risk table, threshold adjustment procedure).

---

## Requirements Trace

- R1. Generate a local-only, browser-openable review artifact showing each boundary cluster as
  side-by-side images without copying, publishing, or embedding photo data in tracked files.
- R2. Support per-cluster AND per-image decision recording for multi-member clusters; no PII in
  any output.
- R3. Produce a privacy-safe tracked review summary (`dedup_review_summary.json`) containing only
  aggregate counts — no captions, sender names, filenames, or per-cluster decisions.
- R4. Support `--sample N` as a calibration preview tool; the sample never gates finalization —
  full 285-cluster review is always required before `provisional` can be cleared.
- R5. Determine whether the threshold should be lowered based on observed false-positive rate;
  emit a threshold recommendation and the exact re-run command when lowering is indicated.
- R6. All review artifacts containing image references or per-cluster decisions must be gitignored;
  the scripts that produce them must be tracked.
- R7. Scripts must be deterministic and reproducible (fixed random seed for sampling, atomic
  writes for all output files).
- R8. No modification to raw photos; all photo access is strictly read-only.
- R9. The `--unsure-from` filter allows targeted re-review of only UNSURE clusters from a prior
  decisions file, avoiding a full-pass reload to resolve a small number of uncertain entries.

---

## Scope Boundaries

- Only `cluster_type == "perceptual"` records with `hamming_distance == 8` are in scope for
  mandatory visual review. Exact-dedup (SHA-256), hamming 2–6, and hamming=0 perceptual clusters
  are explicitly excluded (rationale in Key Technical Decisions).
- A `--hamming-min`/`--hamming-max` flag allows optional spot-checking of other distance ranges
  (e.g., hamming=0 for pHash collision auditing) but this is not part of the mandatory workflow.
- Captions, sender names, and Telegram message IDs must never appear in any output artifact.
- U4–U8 (classification, privacy flagging, staging, training assembly) remain deferred until
  `provisional: false` is written.
- No bot runtime files, model weights, launchd service files, training scripts, or secrets
  are touched.
- The review HTML uses `file://` links to photos in the original Telegram export directory;
  it never creates symlinks, copies, or thumbnails outside that directory.

### Deferred to Follow-Up Work

- Full threshold sweep (4, 5, 6, 7, 8) with cluster-count comparison table — separate iteration
  if the initial review reveals high ambiguity across the board.
- Systematic hamming < 8 spot-check beyond optional one-off audits with `--hamming-max 6`.

---

## Context & Research

### Relevant Code and Patterns

- `scripts/intake_telegram_dedup.py` — produces `dedup_clusters.jsonl`; field schema:
  `cluster_id` (int), `cluster_type` (str), `keep_filename` (str), `duplicate_filenames`
  (list[str]), `hamming_distance` (int|null), `reason` (str). Paths are export-relative:
  `photos/photo_XXX@date.jpg`. Multi-dup clusters form via union-find on pairwise comparisons —
  members may have pairwise distances exceeding the threshold when comparing non-adjacent nodes.
- `data/intake_meta/tg_2026-04-24/dedup_clusters.jsonl` — 554 records; 285 boundary perceptual
  clusters (285 cluster records, 835 total duplicates across all 512 perceptual clusters).
- `data/intake_meta/tg_2026-04-24/dedup_summary.json` — `provisional: true`,
  `manual_review_required: true`, `phash_threshold: 8`, `boundary_clusters_at_threshold: 285`.
- `tests/test_intake_safety.py` — existing gitignore-assertion tests via
  `subprocess.run(['git', 'check-ignore', '-v', path])`; new assertions follow same pattern.
- `scripts/intake_telegram_audit.py` — established argparse style, stderr logging conventions.
- `.gitignore` — existing rules: `data/intake_meta/**/manifest.jsonl`,
  `data/intake_meta/**/audit.jsonl`. Review directory needs a new rule.

### Institutional Learnings

- Origin plan (`2026-04-25-002`) defers threshold decision to empirical inspection: "If false
  positives found, decrease threshold (e.g., 6) and re-run." This plan operationalizes that
  procedure.
- `dedup_clusters.jsonl` is tracked; this must not change — only the new `review/` subdirectory
  contents are gitignored.
- Privacy boundary from parent intake plan: any file that could identify a sender or reproduce
  message text must be gitignored. Photo `file://` links and per-cluster reviewer decisions cross
  this boundary. Aggregate counts alone do not.
- U6 staging organizer (parent plan `2026-04-25-001`) builds its skip-set from
  `duplicate_filenames[]`. A false-positive propagates directly: a valid fish photo is excluded
  from training candidates silently. The review is a high-stakes data integrity gate.

---

## Key Technical Decisions

- **hamming=0 perceptual clusters excluded from mandatory review**: A pHash distance of 0 means
  the two images are perceptually identical (same 64-bit hash). The only reason they escaped
  SHA-256 exact-dedup is JPEG re-compression changing byte content while preserving visual
  content. At hamming=0, pHash collision probability across 32k natural images is negligible (the
  64-bit hash space has 2^64 values; the dataset has ~32k images, giving a birthday-paradox
  collision probability of roughly 10^-10). These clusters are functionally confirmed duplicates
  and require no visual review. An optional `--hamming-min 0 --hamming-max 0` audit pass can be
  run if any hamming=0 cluster looks suspicious during the hamming=8 review.
- **Sampling (`--sample N`) is a calibration preview, not a decision gate**: At n=30, the 95%
  confidence interval around any observed FP rate spans ±14 percentage points — wide enough to
  contain the entire decision range. The sample gives the reviewer a feel for the dataset before
  committing to all 285, but it does not support an automated abort. Full 285-cluster review is
  always required for finalization. The `--partial` flag in `intake_review_finalize.py` exists
  only for iterative progress-saving, not for conditional finalization.
- **MIXED decision option for multi-member clusters**: Union-find clustering is transitive — a
  cluster with N duplicates may contain images whose pairwise distance exceeds the threshold if
  they were connected via intermediate nodes. For clusters with `len(duplicate_filenames) > 1`,
  the HTML exposes a MIXED option in addition to KEEP_DEDUP / FALSE_POSITIVE / UNSURE. MIXED
  signals "some members look like duplicates, some do not." The finalize script treats MIXED
  conservatively: the cluster is not deduped (same outcome as FALSE_POSITIVE), and the cluster_id
  is written to a MIXED log for optional manual re-inspection.
- **HTML contact sheet with `file://` image references, no copied thumbnails**: Keeps raw photos
  exclusively in the original Telegram export directory. `file://` paths work on macOS Safari for
  absolute paths. Path spaces (e.g., `"Telegram Desktop"`) must be percent-encoded via
  `urllib.parse.quote()` — unencoded spaces cause silent image load failures in the browser.
- **JavaScript decision export, no local server needed**: The reviewer marks radio buttons and
  clicks "Export Decisions"; the browser downloads `review_decisions.json`. No Flask/FastAPI.
- **`review_decisions.json` gitignored; `dedup_review_summary.json` tracked**: Per-cluster
  decisions may contain reviewer notes that could be identifying; only the aggregate summary is
  safe for git.
- **`provisional` flag cleared only by the finalize script, only after full review**: The
  finalize script writes `provisional: false` in `dedup_summary.json` only when all 285 boundary
  clusters have a non-UNSURE decision and the FP rate (FP + MIXED / total) is below the
  threshold. `--partial` runs never clear the flag.
- **Threshold recommendation is directional, not prescriptive**: When FP rate exceeds the limit,
  the summary records `threshold_recommendation: "lower_threshold"` and the script prints a
  suggested re-run command using `max(1, 8 - round(fp_rate * 8))` as the starting suggested
  value. This is a heuristic, not a calibrated formula — the human confirms the target threshold
  before re-running dedup.

---

## Open Questions

### Resolved During Planning

- **Review artifacts location**: `data/intake_meta/tg_2026-04-24/review/` for co-location with
  cluster data. New gitignore rule: `data/intake_meta/**/review/`.
- **Absolute or relative `file://` paths**: Absolute paths resolved at generation time from
  `--export-dir`, percent-encoded. Avoids broken links if the HTML is moved within the same
  machine.
- **FP threshold for lowering recommendation**: 0.15 (15%) default, overridable via
  `--fp-threshold`. Chosen as the practical break-even point where 1 in 7 deduped cluster
  decisions is wrong — conservative enough to warrant a lower threshold but not so strict that
  high-hamming noise triggers re-runs unnecessarily. The sample does not enforce this gate.
- **UNSURE handling**: UNSURE is non-FP in the rate calculation, but presence of any UNSURE entry
  prevents finalization. The `--unsure-from` filter generates a focused re-review HTML for only
  UNSURE entries from a prior decisions file, providing a clear resolution path.
- **Sample decisions and full-pass decisions**: Sample decisions are a separate `review_decisions.json`
  file and are discarded before the full-pass review. The operational notes make this explicit.
  There is no auto-merge — the full pass re-reviews all 285 from scratch.

### Deferred to Implementation

- Exact JavaScript blob/download API implementation and localStorage fallback.
- Whether a CLI stdin-based review mode (Y/N per cluster) is worth adding alongside the HTML
  approach — decide during U2 based on ergonomics.

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not
> implementation specification. The implementing agent should treat it as context, not code to
> reproduce.*

```
                ┌──────────────────────────────────────────────────────────┐
                │   LOCAL MACHINE ONLY — nothing in this box is committed   │
                └──────────────────────────────────────────────────────────┘

dedup_clusters.jsonl ──► intake_review_boundary.py ──► boundary_review.html
(tracked, no PII)         --clusters PATH               (gitignored, file:// refs)
                          --export-dir PATH              percent-encoded paths
                          [--sample N]                   MIXED option for multi-dup
                          [--unsure-from DECISIONS]             │
                          [--hamming-min 8]                     │ open -a Safari
                                                                ▼
                                                  Human marks per cluster:
                                                  KEEP_DEDUP / FALSE_POSITIVE
                                                  UNSURE / MIXED (multi-dup only)
                                                  Clicks "Export Decisions"
                                                                │
                                                                ▼
                                              review_decisions.json  (gitignored)
                                                                │
  dedup_clusters.jsonl ──┐                                     │
  dedup_summary.json ────┴─► intake_review_finalize.py ◄───────┘
                               --fp-threshold 0.15
                               [--partial]
                               [--dry-run]
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             │                          │                          │
       incomplete                  fp_rate ≥ 0.15          fp_rate < 0.15
       decisions                         │                  no UNSURE remaining
             │                    Print suggested re-run         │
       exit non-zero               threshold command      dedup_review_summary.json
       (list missing IDs)          threshold_rec:           (tracked, counts only)
                                   "lower_threshold"       dedup_summary.json:
                                   do NOT touch              provisional → false
                                   dedup_summary.json        review_completed_at set

                          [if UNSURE remain: prompt user to run --unsure-from]
```

---

## Implementation Units

- U1. **Gitignore extension and safety test**

**Goal:** Add a `.gitignore` rule protecting all files under `data/intake_meta/**/review/`, and
add gitignore-assertion tests covering both new and existing tracked files.

**Requirements:** R6

**Dependencies:** None

**Files:**
- Modify: `.gitignore`
- Modify: `tests/test_intake_safety.py`

**Approach:**
- Add rule `data/intake_meta/**/review/` immediately after the `data/intake_meta/**/audit.jsonl`
  entry in `.gitignore`, with a comment noting the content type (HTML with `file://` image
  references, per-cluster review decisions).
- In `tests/test_intake_safety.py`, add four assertions following the existing
  `git check-ignore -v` pattern:
  - Positive: `data/intake_meta/tg_2026-04-24/review/boundary_review.html` → excluded.
  - Positive: `data/intake_meta/tg_2026-04-24/review/review_decisions.json` → excluded.
  - Negative (regression guard): `data/intake_meta/tg_2026-04-24/dedup_clusters.jsonl` → NOT
    excluded (exit code 1).
  - Negative: `data/intake_meta/tg_2026-04-24/dedup_summary.json` → NOT excluded.

**Patterns to follow:**
- Existing `.gitignore` block structure (grouped by concern, with inline comments).
- `tests/test_intake_safety.py` `subprocess.run(['git', 'check-ignore', '-v', path])` pattern.

**Test scenarios:**
- Happy path: `git check-ignore -v data/intake_meta/tg_2026-04-24/review/boundary_review.html`
  exits 0.
- Happy path: `git check-ignore -v data/intake_meta/tg_2026-04-24/review/review_decisions.json`
  exits 0.
- Regression: `dedup_clusters.jsonl` still exits 1 (not ignored) after the new rule is added.
- Regression: `dedup_summary.json` still exits 1 (not ignored).

**Verification:**
- `pytest tests/test_intake_safety.py` passes with all four new assertions.
- Creating `data/intake_meta/tg_2026-04-24/review/probe.html` and running `git status` confirms
  it is untracked/ignored.

---

- U2. **Boundary cluster HTML contact sheet generator**

**Goal:** Generate a local-only HTML file showing each boundary cluster as side-by-side images
with decision controls, supporting multi-dup granularity (MIXED option), UNSURE re-review
filtering, and correct `file://` URI encoding. No photos are copied.

**Requirements:** R1, R2, R4, R7, R8, R9

**Dependencies:** U1 (review directory rule in `.gitignore` before output is written)

**Files:**
- Create: `scripts/intake_review_boundary.py`
- Create: `tests/test_intake_review_boundary.py`

**Approach:**
- CLI: `python3 scripts/intake_review_boundary.py --clusters <path> --export-dir <path>
  [--output <path>] [--sample N] [--hamming-min 8] [--hamming-max 8]
  [--unsure-from DECISIONS_JSON]`
- Default output: `data/intake_meta/tg_2026-04-24/review/boundary_review.html` (derived from
  `--clusters` parent if `--output` is omitted).
- Before filtering, print a hamming-distance distribution table to stderr for all perceptual
  clusters (counts at each distance level) so the reviewer sees the full population shape.
- Filter records: `cluster_type == "perceptual"` and `hamming_distance` in
  `[hamming-min, hamming-max]`.
- If `--unsure-from DECISIONS_JSON`: load that file, extract `cluster_id` values where
  `decision == "UNSURE"`, and filter the cluster list to only those IDs. This mode is for
  resolving prior UNSURE entries without re-reviewing the whole set.
- If `--sample N` (no `--unsure-from`): shuffle with seed 42 and take first N. Print a clear
  notice: "SAMPLE MODE: This is a calibration preview. Full 285-cluster review required before
  finalization."
- For each cluster, build absolute `file://` image paths by joining `--export-dir` with
  `keep_filename` and each `duplicate_filenames[]` entry. Apply `urllib.parse.quote()` to each
  path component that may contain spaces (using `safe='/'` to preserve directory separators).
- HTML structure per cluster row: `cluster_id`, `hamming_distance`, `cluster_type` tag, one
  `<img loading="lazy">` per keep + each duplicate. For clusters with
  `len(duplicate_filenames) > 1`, add a `<small>multi-member: N+1 images</small>` label.
- Decision controls per cluster: radio buttons KEEP_DEDUP / FALSE_POSITIVE / UNSURE, plus MIXED
  when `len(duplicate_filenames) > 1`. Text input for optional reviewer note.
- If an image path does not exist, render `<span class="img-missing">[missing: {filename}]</span>`
  instead of `<img>` — do not abort.
- JavaScript `exportDecisions()` collects all cluster decisions + notes into the
  `review_decisions.json` schema and triggers a browser file download. The export button is
  disabled until all clusters have a non-null decision (prevents partial export).
- On success, print: output path, cluster count, `open -a Safari '{output_path}'`.
- Exit non-zero with stderr message if `--clusters` file does not exist or is empty.

**`review_decisions.json` schema:**
```
{
  "schema_version": 1,
  "reviewed_at": "<ISO date from browser>",
  "export_dir": "<absolute path>",
  "threshold_reviewed": 8,
  "sample_size": N or null,
  "unsure_repass": true or false,
  "decisions": [
    {
      "cluster_id": <int>,
      "hamming_distance": <int>,
      "is_multi_member": <bool>,
      "decision": "KEEP_DEDUP" | "FALSE_POSITIVE" | "UNSURE" | "MIXED",
      "note": "<str, may be empty>"
    }
  ]
}
```
This schema is directional — the finalize script must tolerate additional optional fields.

**Patterns to follow:**
- `scripts/intake_telegram_audit.py` argparse structure, stderr logging, exit-code conventions.
- `scripts/intake_telegram_manifest.py` JSONL line-by-line reading pattern.

**Test scenarios:**
- Happy path: synthetic 3-record JSONL (all hamming=8 perceptual, each with 1 duplicate);
  generated HTML contains `KEEP_DEDUP`, `FALSE_POSITIVE`, `UNSURE` strings, each `keep_filename`,
  and `file://` in at least one `img src`.
- Happy path multi-member: a cluster with 3 duplicates; HTML contains `MIXED` string and
  `multi-member` label; single-duplicate clusters in same HTML do not show `MIXED`.
- Happy path `--sample 1`: 3 input records → HTML contains exactly one cluster row; sample
  notice present.
- Happy path `--unsure-from`: decisions file marks cluster_id=2 as UNSURE; script generates HTML
  containing only cluster_id=2.
- Edge case — `file://` path with spaces: `--export-dir` contains a space ("Telegram Desktop");
  generated `img src` attribute contains `%20` not a raw space.
- Edge case — missing image path: a `keep_filename` points to non-existent file; HTML contains
  `class="img-missing"` for that entry; exits 0.
- Edge case — non-boundary records filtered: input has hamming=5 and hamming=8 clusters; only
  hamming=8 appears in output with default `--hamming-min 8`.
- Edge case — empty input after filter (no hamming=8 records): HTML generated with a "No clusters
  to review" message; exits 0.
- Edge case — sampling reproducibility: `--sample 10` twice produces the same 10 clusters in
  the same order.
- Error path — missing `--clusters` file: exits non-zero, message on stderr.
- Edge case — exact clusters excluded: `cluster_type == "exact"` records never appear regardless
  of `--hamming-min/max` values.

**Verification:**
- `pytest tests/test_intake_review_boundary.py` passes.
- Manual smoke-check: running with `--sample 5` against the real `dedup_clusters.jsonl` with
  `--export-dir "/Users/imac/Downloads/Telegram Desktop/ChatExport_2026-04-24"` produces an HTML
  that opens in Safari and displays 5 image pairs with visible images (not broken), radio buttons
  per cluster, and a functional Export Decisions button.

---

- U3. **Decision ingestion and dedup finalization**

**Goal:** Read `review_decisions.json`, validate completeness, compute false-positive rate (FP +
MIXED), produce tracked `dedup_review_summary.json`, and either clear `provisional` in
`dedup_summary.json` or print a threshold re-run recommendation with a suggested target.

**Requirements:** R3, R5, R7

**Dependencies:** U2 (human has completed the full 285-cluster review and exported decisions)

**Files:**
- Create: `scripts/intake_review_finalize.py`
- Create: `tests/test_intake_review_finalize.py`
- Create (runtime, tracked): `data/intake_meta/tg_2026-04-24/dedup_review_summary.json`
- Modify (runtime, conditional): `data/intake_meta/tg_2026-04-24/dedup_summary.json`

**Approach:**
- CLI: `python3 scripts/intake_review_finalize.py --decisions <path> --clusters <path>
  --summary <path> --output <path> [--fp-threshold 0.15] [--partial] [--dry-run]`
- Load boundary clusters: `cluster_type == "perceptual"` and `hamming_distance == 8`.
- Load `decisions[]` from `--decisions`.
- Completeness check: every boundary `cluster_id` must appear in decisions. If any missing:
  - Without `--partial`: print missing IDs to stderr, exit non-zero.
  - With `--partial`: proceed with reviewed subset; set `review_complete: false`.
- Compute: `keep_count`, `fp_count` (FP + MIXED combined), `unsure_count`, `mixed_count`.
  - `fp_rate = (fp_count) / (keep_count + fp_count + unsure_count)`
  - UNSURE entries: if any remain, `review_complete = false` regardless of fp_rate; print
    message suggesting `--unsure-from` filter in U2.
- MIXED log: if any MIXED decisions present, write their `cluster_id` list to stderr for
  optional follow-up inspection.
- Compute `suggested_lower_threshold = max(1, 8 - round(fp_rate * 8))` when fp_rate ≥ threshold
  (directional heuristic — human confirms before use).
- Write `dedup_review_summary.json` via atomic temp-file + `os.replace()`:
  `reviewed_at`, `threshold_reviewed`, `boundary_clusters_total`, `boundary_clusters_reviewed`,
  `keep_dedup_count`, `false_positive_count`, `mixed_count`, `unsure_count`,
  `false_positive_rate` (fp + mixed / total), `threshold_recommendation` (`"validated"` or
  `"lower_threshold"`), `suggested_lower_threshold` (int or null), `review_complete`,
  `source`, `license`. No filenames, no per-cluster decisions, no captions.
- If `fp_rate >= fp_threshold` or UNSURE remain:
  - `threshold_recommendation = "lower_threshold"`.
  - If fp_rate too high: print suggested re-run command to stdout.
  - Do NOT modify `dedup_summary.json`.
- If `fp_rate < fp_threshold` and no UNSURE remain and not `--partial`:
  - `threshold_recommendation = "validated"`, `review_complete = true`.
  - If not `--dry-run`: update `dedup_summary.json` atomically: `provisional = false`,
    `manual_review_required = false`, add `review_completed_at` (ISO timestamp). All other
    fields unchanged.
- Guard: if `dedup_summary.json` already has `provisional: false`, warn and skip the update;
  still write `dedup_review_summary.json`; exit 0.
- Print human-readable summary to stdout regardless of outcome.

**Technical design (directional guidance, not implementation specification):**
```
boundary = [r for r in read_jsonl(clusters) if perceptual and hamming==8]
reviewed = {d["cluster_id"]: d for d in load_json(decisions)["decisions"]}
missing = [c["cluster_id"] for c in boundary if c["cluster_id"] not in reviewed]
if missing and not partial: fail(missing)

counts = Counter(d["decision"] for d in reviewed.values())
fp_count = counts["FALSE_POSITIVE"] + counts["MIXED"]
fp_rate = fp_count / sum(counts.values())

write_review_summary_atomic(counts, fp_rate, ...)

if fp_rate < fp_threshold and counts["UNSURE"] == 0 and not partial:
    if not dry_run:
        update_dedup_summary_atomic(provisional=False, ...)
else:
    print_recommendation_and_command(fp_rate, ...)
```

**Patterns to follow:**
- `scripts/intake_telegram_audit.py` JSON read/write and stderr logging.
- Atomic write via `tempfile.NamedTemporaryFile` + `os.replace()` pattern.

**Test scenarios:**
- Happy path — all KEEP_DEDUP: 5 boundary clusters all KEEP_DEDUP; `fp_rate = 0.0`;
  `dedup_summary.provisional` updated to `false`; `threshold_recommendation = "validated"`.
- Happy path — mixed below threshold: 20 clusters, 2 FP (fp_rate = 0.10 < 0.15); provisional
  cleared; summary counts correct.
- Threshold exceeded: 20 clusters, 4 FP (fp_rate = 0.20 ≥ 0.15); `dedup_summary.json` NOT
  modified; re-run command + `suggested_lower_threshold` printed; `threshold_recommendation =
  "lower_threshold"`.
- MIXED counts as FP: 20 clusters, 2 FALSE_POSITIVE + 2 MIXED (fp_rate = 0.20 ≥ 0.15);
  same outcome as above; MIXED cluster IDs printed to stderr.
- Incomplete without `--partial`: 2 of 5 cluster IDs missing; exits non-zero; missing IDs on
  stderr.
- Incomplete with `--partial`: same input; proceeds with 3 reviewed; `review_complete = false`;
  `provisional` NOT cleared.
- `--dry-run` with fp_rate < threshold: `dedup_summary.json` NOT modified; summary still written.
- UNSURE present: fp_rate = 0 but unsure_count > 0; `review_complete = false`; provisional NOT
  cleared; message suggests `--unsure-from` flag.
- Already finalized: `dedup_summary.json` has `provisional: false`; script warns, skips update,
  still writes summary, exits 0.
- Integration end-to-end: generate synthetic 5-cluster JSONL, build matching
  `review_decisions.json` with mixed decisions, run finalize; assert `dedup_review_summary.json`
  has correct counts, no filenames, no captions, no sender names; assert `dedup_summary.json`
  updated when finalization conditions are met.

**Verification:**
- `pytest tests/test_intake_review_finalize.py` passes all scenarios.
- Inspecting `dedup_review_summary.json` in a text editor confirms no filenames, captions, or
  sender names — only aggregate counts and metadata.
- After a non-dry-run finalization, `dedup_summary.json` no longer contains `"provisional": true`
  and `review_completed_at` is present.

---

## System-Wide Impact

- **Interaction graph:** `dedup_clusters.jsonl` and `dedup_summary.json` are consumed by the
  planned U6 staging organizer (parent plan `2026-04-25-001`). The `provisional` flag is the
  machine-readable gate — U6 must not run while it is `true`. This review workflow is the sole
  step that clears it.
- **Error propagation:** If `intake_review_finalize.py` exits non-zero (incomplete decisions, FP
  rate too high, UNSURE remaining), `dedup_summary.json` remains in provisional state. The
  non-zero exit code makes the block machine-checkable.
- **State lifecycle risks:** Both output writes (`dedup_review_summary.json` and `dedup_summary.json`)
  use atomic temp-file + `os.replace()`. An interrupted finalization leaves `dedup_summary.json`
  in its prior state; re-running the script with `--dry-run` is safe before a retry. The
  `provisional: false` guard prevents double-finalization.
- **Multi-member cluster risk:** Union-find clustering is transitive. Large clusters (up to 12
  duplicates in the real data) may contain images connected only through intermediate nodes.
  The MIXED decision option provides conservative handling — the cluster is not deduped if the
  reviewer cannot confirm all members are duplicates. `dedup_clusters.jsonl` is never modified
  by this workflow; a false-positive determination means the image simply remains in the unique
  set, not that the cluster record is patched.
- **API surface parity:** None — offline pipeline scripts; the live bot never reads intake
  metadata.
- **Integration coverage:** The U2 → U3 file hand-off (`review_decisions.json`) is exercised
  end-to-end in the U3 integration test scenario using synthetic data.
- **Unchanged invariants:** `dedup_clusters.jsonl` is read-only in this workflow. If the review
  reveals the threshold should be lowered, the correct path is to re-run
  `intake_telegram_dedup.py --phash-threshold N`, which regenerates both output files from
  scratch. Cluster records must never be patched manually.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| FP rate ≥ 15%: threshold must be lowered, review restarts | Finalize script prints suggested target threshold and exact re-run command; re-running dedup takes ~15 sec; review tooling is reusable at any threshold |
| Safari blocks cross-directory `file://` image loading | Absolute paths work in Safari on macOS for local HTML; percent-encode spaces in paths; HTML header includes `open -a Safari` note |
| HTML slow to render with 285 clusters × multiple images | `loading="lazy"` on all `<img>` tags; `--sample 30` recommended for calibration preview before full pass |
| Large multi-member clusters (up to 12 images) hard to evaluate | MIXED option provides conservative handling without requiring per-image decisions; note in HTML labels multi-member clusters explicitly |
| Reviewer is stuck on UNSURE clusters | `--unsure-from` flag generates a targeted re-review HTML for only UNSURE entries |
| `dedup_decisions.json` accidentally committed | Covered by U1 gitignore rule + safety test; `git status` will not surface the file |
| Review HTML accidentally shared (exposes `file://` photo paths) | HTML header includes a visible LOCAL-ONLY notice; operational notes reiterate this |
| Finalization run interrupted between two atomic writes | Both writes are atomic (temp + replace); re-running with `--dry-run` is safe; `provisional: false` guard prevents double-write |

---

## Documentation / Operational Notes

**Recommended workflow — calibration preview then full review:**

```bash
# Step 1 — calibration preview (30 randomly sampled boundary clusters)
python3 scripts/intake_review_boundary.py \
  --clusters data/intake_meta/tg_2026-04-24/dedup_clusters.jsonl \
  --export-dir "/Users/imac/Downloads/Telegram Desktop/ChatExport_2026-04-24" \
  --sample 30 \
  --output data/intake_meta/tg_2026-04-24/review/boundary_sample.html

open -a Safari data/intake_meta/tg_2026-04-24/review/boundary_sample.html
# → mark 30 clusters, click Export  →  downloads review_decisions_sample.json
# Review this to understand the dataset before committing to all 285.
# Sample decisions are DISCARDED — do not pass them to finalize.

# Step 2 — full review (all 285 boundary clusters)
python3 scripts/intake_review_boundary.py \
  --clusters data/intake_meta/tg_2026-04-24/dedup_clusters.jsonl \
  --export-dir "/Users/imac/Downloads/Telegram Desktop/ChatExport_2026-04-24" \
  --output data/intake_meta/tg_2026-04-24/review/boundary_review.html

open -a Safari data/intake_meta/tg_2026-04-24/review/boundary_review.html
# → mark all 285 clusters, click Export  →  downloads review_decisions.json

# Step 3 — resolve any UNSURE entries (run after step 2 if needed)
python3 scripts/intake_review_boundary.py \
  --clusters data/intake_meta/tg_2026-04-24/dedup_clusters.jsonl \
  --export-dir "/Users/imac/Downloads/Telegram Desktop/ChatExport_2026-04-24" \
  --unsure-from ~/Downloads/review_decisions.json \
  --output data/intake_meta/tg_2026-04-24/review/boundary_unsure.html
# → re-mark only UNSURE clusters, export review_decisions_resolved.json

# Step 4 — finalize (use the complete decisions file)
python3 scripts/intake_review_finalize.py \
  --decisions ~/Downloads/review_decisions.json \
  --clusters data/intake_meta/tg_2026-04-24/dedup_clusters.jsonl \
  --summary data/intake_meta/tg_2026-04-24/dedup_summary.json \
  --output data/intake_meta/tg_2026-04-24/dedup_review_summary.json

# Step 5a — if fp_rate < 15% and no UNSURE: dedup_summary.json updated, provisional cleared.
#           Commit dedup_review_summary.json. U4–U8 are now unblocked.

# Step 5b — if fp_rate ≥ 15%: re-run dedup at the suggested threshold, then restart from step 1.
#   python3 scripts/intake_telegram_dedup.py \
#     --export-dir "/Users/imac/Downloads/Telegram Desktop/ChatExport_2026-04-24" \
#     --phash-threshold <suggested_value>
#   # Commit updated dedup_clusters.jsonl and dedup_summary.json, then restart review.
```

**What gets committed from this step:** Only `data/intake_meta/tg_2026-04-24/dedup_review_summary.json`
(aggregate counts, no PII). Updated `dedup_summary.json` (existing tracked file, provisional flag
cleared). Do not commit any file under `data/intake_meta/tg_2026-04-24/review/`.

**Downstream unblock:** Once `dedup_summary.json` has `provisional: false`, U4 (fish/no-fish/lure
classification filtering) and U5 (caption class-gap analysis) can begin. U5 cross-references the
finalized unique set against `manifest.jsonl` (gitignored, PII-containing) to extract species
labels from captions — that manifest must never leave the local machine.

---

## Sources & References

- **Origin document:** [docs/plans/2026-04-25-002-feat-telegram-dedup-u3-perceptual-plan.md](docs/plans/2026-04-25-002-feat-telegram-dedup-u3-perceptual-plan.md)
  (Operational Notes, Risk table, threshold re-run procedure)
- **Parent intake plan:** [docs/plans/2026-04-25-001-feat-telegram-export-dataset-intake-plan.md](docs/plans/2026-04-25-001-feat-telegram-export-dataset-intake-plan.md)
  (U6 staging dependency, privacy output policy)
- Related scripts: `scripts/intake_telegram_dedup.py`, `scripts/intake_telegram_audit.py`
- Related data: `data/intake_meta/tg_2026-04-24/dedup_clusters.jsonl`,
  `data/intake_meta/tg_2026-04-24/dedup_summary.json`
- Related tests: `tests/test_intake_safety.py`, `tests/test_intake_dedup.py`
