---
title: "feat: U4 Coarse Category Filtering for Deduplicated Telegram Photo Set"
type: feat
status: active
date: 2026-04-26
origin: docs/plans/2026-04-25-003-feat-dedup-boundary-review-plan.md
---

# feat: U4 Coarse Category Filtering for Deduplicated Telegram Photo Set

## Overview

U3 produced 32,420 unique images after full dedup finalization (`provisional: false`,
`manual_review_required: false`). U4 builds the next pipeline stage: a **local-only,
privacy-safe coarse category filter** that classifies each unique image into one of
seven coarse categories. No cloud APIs, no external image services, no photo copying.
All classification logic runs locally against audit metadata, caption-derived keyword
signals, and image geometry. An HTML review pack enables targeted manual validation
by category sample. The finalized category manifest gates all downstream staging,
class-gap analysis, and dataset expansion work.

---

## Problem Frame

32,420 images from a private fishing Telegram channel are a mixed bag: recognizable
fish catch photos, lure/gear shots, club announcements (posters, screenshots of text),
fish parts (filleting shots), juvenile fish (fry), and noise. Feeding this raw set
directly into species classification training would poison Stage B with non-fish, lure,
or poster content.

U4 must sort these images into coarse holding categories so that:
- Only `fish` images enter species candidate labeling
- `lure_fishing_gear`, `fish_part`, `fry_juvenile` are staged separately for Stage A
  detector training or extension
- `poster_screenshot` and `no_fish` are quarantined from species training
- `unknown_needs_review` receives targeted human validation before assignment

All classification runs on metadata already available locally (`audit.jsonl`,
`manifest.jsonl`, `dedup_clusters.jsonl`) without reading raw photo bytes beyond
what PIL already computed (width, height, file_size). Caption-derived keyword signals
supplement geometry signals using the existing `manifest.jsonl` caption field.
An optional ML-assisted pass (deferred to implementation) could run `detector_v1.pt`
over the `unknown_needs_review` residual.

---

## Requirements Trace

- R1. Derive the unique post-dedup image set from existing artifacts without copying photos.
- R2. Extract per-image coarse category signals from audit metadata and caption keyword
  patterns only — no external API calls, no OCR libraries not already in requirements-intake.txt.
- R3. Assign each image a coarse category and confidence level via deterministic priority-ordered
  rules; uncertain cases route to `unknown_needs_review`.
- R3a. Caption keywords are weak candidate hints only; no image may be assigned a specific
  category based solely on caption text. Any conflict between a caption signal and visual
  evidence (geometry, resolution, file size) must route to `unknown_needs_review`. No image
  may enter fish/species-training staging based on caption text alone.
- R4. Generate local-only HTML review contact sheets for each category to enable targeted
  manual validation (same pattern as `intake_review_boundary.py`).
- R5. Ingest manual review decisions and produce a tracked privacy-safe `filter_review_summary.json`
  (aggregate counts only — no filenames, captions, sender names, or per-image decisions).
- R6. All new artifacts containing filenames, caption-derived signals, or photo references must
  be gitignored; the scripts that produce them must be tracked.
- R7. No new ML dependencies may be added to `requirements-intake.txt` (existing safety test
  enforces this; torch/ultralytics remain forbidden).
- R8. All photo access is strictly read-only; no photo bytes are copied, embedded, or written
  to any tracked output.
- R9. The category assignment must be conservative: when signals conflict (including
  caption-to-visual conflicts), are absent, or when only caption evidence supports a category,
  route to `unknown_needs_review` rather than asserting a category.
- R10. The finalized filter set must gate downstream species labeling, class-gap reporting, and
  Stage B dataset expansion — nothing proceeds on an `unknown_needs_review` residual > 5%.

---

## Scope Boundaries

- Only images whose `filename` appears in the unique post-dedup set (U4.1) are in scope.
  Dedup non-keeps, corrupt images, and audit-missing filenames are explicitly excluded.
- No external vision APIs, cloud OCR, GPT image upload, or any network call during filtering.
- No copying of raw photos into any tracked location.
- No modification to bot runtime, model weights, launchd service files, training scripts,
  deployment files, `.env`, or secrets.
- The optional local ML-assisted pass (`detector_v1.pt` via YOLO) is deferred to implementation;
  it must not be wired into `requirements-intake.txt` (ML deps stay in `requirements-ml.txt`).
- Caption signals use Python `str.lower()` + simple substring matching — no NLP library, no
  regex corpus, no stemmer; maintainability matters more than recall here.
- Stage B species classification labeling (assigning specific species to `fish` images) is out
  of scope for U4 and belongs to U5.
- Class-gap report from captions is out of scope for U4 and belongs to U6.
- Training dataset staging (copying candidates into `data/fish_dataset/`) is out of scope for U4.

### Deferred to Follow-Up Work

- Species candidate labeling for images in the `fish` category — separate future PR.
- Class-gap report from captions cross-referenced against Stage B species list — separate future PR.
- Training dataset staging from finalized filter candidates — separate future PR.
- Optional ML pass: run `detector_v1.pt` over `unknown_needs_review` residual to auto-refine
  assignment — implement as standalone `intake_filter_ml.py`, separate from intake requirements.

  Note: the labels "U5" and "U6" above this section refer to implementation units **within
  this plan** (`intake_filter_review.py` and `intake_filter_finalize.py`), not to future
  follow-up PRs.

---

## Context & Research

### Relevant Code and Patterns

- `scripts/intake_constants.py` — canonical path and class constants; U4 must extend this file.
- `scripts/intake_telegram_dedup.py` — produces `dedup_clusters.jsonl` consumed by U4.1.
- `scripts/intake_telegram_audit.py` — produces `audit.jsonl`; field schema: `filename, sha256,
  width, height, max_side, file_size, low_res, corrupt`.
- `scripts/intake_telegram_manifest.py` — produces `manifest.jsonl`; caption field is the
  primary text signal source.
- `scripts/intake_review_boundary.py` — HTML contact sheet generator with `file://` URI pattern,
  path-traversal guard (`_safe_photo_path`), and cluster radio-button decision controls.
  U4's review script follows this pattern exactly.
- `scripts/intake_review_finalize.py` — decision ingestion + tracked summary writer pattern;
  atomic JSON writes via `tempfile + os.replace()`; exit codes 0/1/2. U4's finalize script
  follows this pattern.
- `tests/test_intake_safety.py` — gitignore assertion pattern using `git check-ignore -v` +
  `subprocess.run()`. U4 safety tests extend this file.
- `data/fish_models/detector_v1.pt`, `data/fish_models/class_names_a.json` — local YOLO
  Stage A weights (gitignored); STAGE_A_CLASSES = `["whole_fish", "lure", "fish_part", "fry",
  "no_fish"]`. Usable only in the optional ML pass outside intake requirements.
- `requirements-intake.txt` — PIL (`Pillow`), `imagehash`, `beautifulsoup4` are present;
  torch/ultralytics forbidden by safety test.

### Institutional Learnings

- Privacy boundary: tracked outputs may contain aggregate counts and schema definitions only.
  Filenames are sensitive enough to warrant gitignoring unless the file is small and matches
  the dedup_clusters.jsonl precedent (< 100 KB, cluster-indexed, not full-manifest-indexed).
  At 32 K records × ~150 bytes the filter JSONL files exceed this threshold — all are local-only.
- The `data/intake_meta/**/review/` directory is already gitignored; new per-category review
  HTML goes there without additional gitignore rules.
- `dedup_clusters.jsonl` is tracked (223-record cluster summary, not full image list).
  The filter pipeline produces full-image-list JSONL files — a qualitatively different artifact
  class that requires explicit local-only rules.
- Atomic writes (`tempfile + os.replace()`) are the established pattern for all summary files.
- `--dry-run` and `--partial` flags are expected on pipeline scripts (established by finalize).

### External References

No external research needed — the local codebase has all necessary patterns.

---

## Key Technical Decisions

- **Signal source priority**: manual review decisions (authoritative, applied in U6) >
  local visual heuristics (geometry: aspect ratio, file_size, resolution — primary
  classification signals) > caption keywords (weak candidate hints only, supplemental).
  Geometry is the primary classification engine. Caption keywords may increase or decrease
  confidence in a geometry-derived assignment but may not independently assign a category.
  Any conflict between a caption signal and visual evidence routes to `unknown_needs_review`.
- **Unique set derivation**: build by subtracting duplicate non-keeps from the full audit set
  rather than re-reading dedup_clusters.jsonl's keep list. This is more robust because it
  avoids reprocessing cluster ordering logic and simply mirrors what was already decided.
- **Caption keyword matching**: `str.lower()` + `in` substring matching on Russian and
  transliterated terms. No NLP library, no stemmer. Recall will be imperfect; that is
  acceptable because uncertain cases fall to `unknown_needs_review` not a wrong category.
- **Category assignment as a two-phase waterfall**: Phase 1 fires geometry/visual rules to
  establish a candidate (`poster_screenshot`, `fish_candidate`, or `unresolved`). Phase 2
  applies caption hints to refine a geometry-established candidate — e.g., a `fish_candidate`
  with a `caption_lure_keyword` hint is reclassified to `lure_fishing_gear` if no visual
  contradiction exists. Caption hints that contradict the geometry candidate, or that fire
  when no geometry rule established a candidate, route to `unknown_needs_review`. Staging
  exclusion: `poster_screenshot`, `no_fish`, `fry_juvenile`, `fish_part`, and
  `lure_fishing_gear` are excluded from species training; `fish` is the only
  staging-eligible category.
- **Confidence levels** (high / medium / low): high = strong unambiguous visual evidence
  with no caption contradiction (e.g., extreme aspect + tiny size → `poster_screenshot`);
  medium = geometry-consistent candidate with corroborating or neutral caption;
  low = weak geometry signal only, or caption hint without visual corroboration.
  Caption-only matches are capped at low confidence and must not bypass review.
  Only `high` confidence assignments bypass review; `medium` and `low` go into their
  category's review sample.
- **Corrupt image handling**: excluded from the universe in U4.1 with a count in the summary.
  They are not assigned a coarse category and not offered for manual review.
- **No new requirements-intake.txt dependencies**: `Pillow` and standard library string
  operations cover all heuristic needs. `imagehash` (already present) is not needed for U4.
- **HTML review pack sampling**: default 50 images per category, configurable. For
  `unknown_needs_review` default is 100 (higher variance expected). Fixed random seed (42)
  for reproducibility.
- **Tracker field**: `filter_candidates.jsonl` records carry a `review_status` field
  (`auto_assigned | reviewed_confirmed | reviewed_reassigned`) to support merge logic in U6.
- **Tracked summary schemas** (canonical; implementer must match exactly):
  - `filter_universe_summary.json`: `{total_unique, non_keep_excluded, corrupt_excluded,
    low_res_count, source, license}`
  - `filter_signals_summary.json`: `{total_images, no_manifest_record_count, low_res_count,
    extreme_aspect_count, tiny_file_count, caption_lure_keyword_count,
    caption_fish_part_keyword_count, caption_fry_keyword_count, caption_no_fish_keyword_count,
    caption_text_heavy_count, source, license}`
  - `filter_candidates_summary.json`: `{total, per_category: {<cat>: {count, high, medium, low}},
    source, license}`
  - `filter_review_summary.json`: `{total_unique, per_category: {<cat>: {auto_count,
    reviewed_count, confirmed_count, reassigned_count, unknown_count, override_rate,
    final_count}}, unknown_needs_review_pct, review_complete, source, license}`

---

## Open Questions

### Resolved During Planning

- **Should the unique universe be derived from the keep list or by subtracting non-keeps?**
  Resolved: subtract non-keeps from the full audit set. More robust, avoids re-implementing
  cluster ordering. (see Key Technical Decisions)
- **Can we use the local YOLO detector in requirements-intake.txt?**
  Resolved: No — `ultralytics` is forbidden in requirements-intake.txt. The optional ML pass
  is deferred to a separate script with its own requirements and is not part of the U4 acceptance
  criteria.
- **Should filter_candidates.jsonl be tracked (like dedup_clusters.jsonl) or local-only?**
  Resolved: local-only. At ~32 K records × 150 bytes ≈ 4.8 MB it exceeds the cluster-summary
  precedent; and caption-derived signal flags (even as booleans) indirectly encode caption content.
- **Should the HTML review pack use file:// photo refs (like U3) or thumbnails?**
  Resolved: file:// refs, same pattern as `intake_review_boundary.py`. No thumbnail
  generation; no photo copying.

### Deferred to Implementation

- **Exact Russian keyword lists for caption signals**: determined during implementation by
  inspecting a sample of manifest.jsonl captions. The plan defines signal categories; exact
  keyword sets are an implementation detail.
- **Geometry thresholds for poster/screenshot detection**: aspect ratio and file_size bucket
  boundaries. Calibrate from a sample during implementation; include in script docstring.
- **File_size bucket boundaries**: implementation must sample the audit.jsonl distribution
  to set `tiny/small/medium/large` cut points that separate genuine fish photos from
  screenshots.
- **Per-category `--sample` defaults**: confirm during implementation whether 50/100 are
  sensible given actual category distribution; adjust if any category has < 50 members.
- **Whether `unknown_needs_review` residual meets the < 5% gate (R10)** depends on
  caption signal quality observed during implementation.

---

## Output Structure

    data/intake_meta/tg_2026-04-24/
    ├── filter_universe.jsonl          ← local-only (filenames + audit fields)
    ├── filter_universe_summary.json   ← tracked (aggregate counts)
    ├── filter_signals.jsonl           ← local-only (filenames + signal flags)
    ├── filter_signals_summary.json    ← tracked (per-signal bucket counts)
    ├── filter_candidates.jsonl        ← local-only (filenames + category + confidence)
    ├── filter_candidates_summary.json ← tracked (per-category counts + confidence dist.)
    ├── filter_review_summary.json     ← tracked (post-review aggregate counts)
    └── review/                        ← already gitignored
        ├── filter_review_fish.html
        ├── filter_review_no_fish.html
        ├── filter_review_lure_fishing_gear.html
        ├── filter_review_fish_part.html
        ├── filter_review_fry_juvenile.html
        ├── filter_review_poster_screenshot.html
        ├── filter_review_unknown_needs_review.html
        └── filter_decisions_<category>.json

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not
> implementation specification. The implementing agent should treat it as context, not
> code to reproduce.*

```
Data flow — U4 pipeline

dedup_clusters.jsonl ──┐
audit.jsonl ───────────┤──► intake_filter_universe ──► filter_universe.jsonl (local)
                        │                            └─► filter_universe_summary.json (tracked)
                        │
manifest.jsonl ────────┐│
filter_universe.jsonl ─┘┴──► intake_filter_heuristic ──► filter_signals.jsonl (local)
                                                       └─► filter_signals_summary.json (tracked)

filter_signals.jsonl ───────► intake_filter_classify ──► filter_candidates.jsonl (local)
                                                      └─► filter_candidates_summary.json (tracked)

filter_candidates.jsonl ────► intake_filter_review ──► review/filter_review_<cat>.html (gitignored)

                           Manual reviewer
                                │
filter_decisions_<cat>.json ───┤
filter_candidates.jsonl ───────┴──► intake_filter_finalize ──► filter_review_summary.json (tracked)
```

Two-phase classification (geometry primary, caption supplemental):

```
Phase 1 — visual heuristics establish candidate (geometry is the primary signal):
  1. extreme_aspect AND (tiny_size OR text_heavy) → poster_screenshot (high)
  2. low_res AND tiny_size                        → poster_screenshot (medium)
  3. NOT low_res AND size∈{medium,large}
     AND aspect∈{portrait,square,landscape}       → fish_candidate (medium)
  4. size=small AND aspect NOT extreme            → fish_candidate (low)
  5. no rule fired                                → unresolved

Phase 2 — caption hints refine geometry-established candidate (caption cannot independently assign):
  fish_candidate + fry caption hint               → fry_juvenile (carry phase-1 confidence)
  fish_candidate + fish_part hint (not fry)       → fish_part (carry phase-1 confidence)
  fish_candidate + lure hint (not fish_part)      → lure_fishing_gear (carry phase-1 confidence)
  fish_candidate + no_fish caption hint           → unknown_needs_review (low) [conflict]
  fish_candidate + no negative caption hint       → fish (carry phase-1 confidence)
  poster_screenshot candidate + fish/species hint → unknown_needs_review (low) [conflict]

Conflict / caption-only:
  unresolved AND any caption keyword fires        → unknown_needs_review (low) [no visual anchor]
  any phase-1 candidate + contradicting caption   → unknown_needs_review (low) [conflict]

catch-all:
  unresolved AND no caption hints                 → unknown_needs_review (low)
```

---

## Implementation Units

- U1. **Update `intake_constants.py` with U4 paths and category constants**

**Goal:** Add a `COARSE_CATEGORIES` canonical list and `FILTER_*` path constants so all
U4 scripts share a single source of truth for paths and category names.

**Requirements:** R1, R6

**Dependencies:** None

**Files:**
- Modify: `scripts/intake_constants.py`

**Approach:**
- Add `COARSE_CATEGORIES: list[str]` mirroring the seven target categories, in priority order.
- Add `FILTER_UNIVERSE_PATH`, `FILTER_UNIVERSE_SUMMARY_PATH`, `FILTER_SIGNALS_PATH`,
  `FILTER_SIGNALS_SUMMARY_PATH`, `FILTER_CANDIDATES_PATH`, `FILTER_CANDIDATES_SUMMARY_PATH`,
  `FILTER_REVIEW_SUMMARY_PATH` — all as `INTAKE_META_ROOT / "<filename>"`.
- `FILTER_REVIEW_DIR = INTAKE_META_ROOT / "review"` (already gitignored by existing rule).
- No imports added; all paths are `pathlib.Path` compositions.

**Patterns to follow:**
- Existing `MANIFEST_PATH`, `AUDIT_PATH`, `DEDUP_PATH` constant pattern in `intake_constants.py`.
- `STAGE_A_CLASSES` / `STAGE_A_CLASSES_SET` pattern for the canonical category list.

**Test scenarios:**
- Happy path: `import intake_constants` succeeds and `COARSE_CATEGORIES` has exactly 7 entries.
- Happy path: all `FILTER_*_PATH` constants resolve under `INTAKE_META_ROOT`.
- Edge case: `COARSE_CATEGORIES` contains no duplicates.
- Edge case: `"unknown_needs_review"` is the last entry in `COARSE_CATEGORIES` (priority order
  invariant — it must be the catch-all).

**Verification:**
- `python3 -c "from intake_constants import COARSE_CATEGORIES, FILTER_CANDIDATES_PATH; print(len(COARSE_CATEGORIES))"` prints `7`.
- Safety test (extended): `import intake_constants` exits 0 in subprocess.

---

- U2. **`intake_filter_universe.py` — Build deduped image universe**

**Goal:** Derive the unique post-dedup image list from `dedup_clusters.jsonl` + `audit.jsonl`,
excluding corrupt images and dedup non-keeps. Write a local-only `filter_universe.jsonl` and a
tracked `filter_universe_summary.json`.

**Requirements:** R1, R6, R8

**Dependencies:** None (reads already-finalized artifacts)

**Files:**
- Create: `scripts/intake_filter_universe.py`
- Create: `tests/test_intake_filter_universe.py` (or extend `tests/test_intake_filter.py`)

**Approach:**
- Load `dedup_clusters.jsonl`; collect all `duplicate_filenames` entries across all clusters
  into a `non_keeps` set. The `keep_filename` of each cluster is kept; singletons (filenames
  not in any cluster) are also kept.
- Load `audit.jsonl`; for each record not in `non_keeps`, include it in the universe.
- Exclude `corrupt: true` records; count separately.
- Write `filter_universe.jsonl` (local-only): one record per unique image with fields
  `filename, sha256, width, height, max_side, file_size, low_res` (no captions, no sender names).
- Write `filter_universe_summary.json` (tracked): `total_unique, corrupt_excluded,
  low_res_count, non_keep_excluded, source, license`.
- CLI: `--clusters`, `--audit`, `--output-dir`, `--dry-run`.

**Patterns to follow:**
- `intake_telegram_dedup.py` JSONL loader pattern (`_read_jsonl`, set-based exclusion).
- `intake_review_finalize.py` atomic write pattern (`_write_json_atomic`).
- `intake_constants.py` path constants for default argument values.

**Test scenarios:**
- Happy path: all audit records not in `non_keeps` and not corrupt appear in universe.
- Happy path: total = audit_count - non_keeps_count - corrupt_count.
- Edge case: audit record appears in `keep_filename` position → included.
- Edge case: audit record appears in `duplicate_filenames` → excluded.
- Edge case: corrupt record that is a cluster keep → excluded from universe (corrupt wins).
- Edge case: empty clusters file → all non-corrupt audit records in universe.
- Edge case: `duplicate_filenames` list is empty → no exclusions from that cluster.
- Error path: `audit.jsonl` missing → exits non-zero with clear message.
- Error path: `dedup_clusters.jsonl` missing → exits non-zero with clear message.
- Integration: summary `total_unique + non_keep_excluded + corrupt_excluded == audit_total`.

**Verification:**
- `filter_universe_summary.json` total matches expected arithmetic identity.
- `filter_universe.jsonl` is present and gitignored; `filter_universe_summary.json` is
  tracked (verified by U8 safety tests).

---

- U3. **`intake_filter_heuristic.py` — Extract per-image coarse signals**

**Goal:** For each image in `filter_universe.jsonl`, join with the `manifest.jsonl` caption
and compute a deterministic signal vector. Write a local-only `filter_signals.jsonl` and a
tracked `filter_signals_summary.json`.

**Requirements:** R2, R6, R8, R9

**Dependencies:** U1 (filter_universe.jsonl must exist)

**Files:**
- Create: `scripts/intake_filter_heuristic.py`
- Test: `tests/test_intake_filter_heuristic.py`

**Approach:**
- Load `filter_universe.jsonl` (primary); load `manifest.jsonl` keyed by `filename` for captions.
  Manifest records with missing `filename` in universe are skipped (logged as warning).
- For each universe record, compute and record:
  - `low_res`: carry from audit (`low_res` field, threshold: `max_side < 800`).
  - `aspect_ratio`: `width / height` (float, 2 decimal places). `None` if dimensions null.
  - `aspect_class`: `"portrait"` (0.5–0.8), `"square"` (0.8–1.25), `"landscape"` (1.25–2.0),
    `"extreme_portrait"` (< 0.5), `"extreme_landscape"` (> 2.0), `"unknown"` if no dimensions.
  - `file_size_bucket`: `"tiny"` / `"small"` / `"medium"` / `"large"` (thresholds calibrated
    at implementation time from distribution; defer exact values).
  - `caption_length`: `len(caption)` or `0` if no manifest record.
  - `caption_empty`: bool.
  - `caption_text_heavy`: `caption_length > 200` (adjust at implementation).
  - `caption_lure_keyword`: bool — Russian/transliterated terms for lures, tackle, gear.
  - `caption_fish_part_keyword`: bool — terms for fish parts (head, spine, roe, guts, scale).
  - `caption_fry_keyword`: bool — terms for juvenile fish (малёк, мальки, молодь, сеголеток).
  - `caption_no_fish_keyword`: bool — terms for ads, announcements, river conditions without fish.
  - `has_manifest_record`: bool — whether a manifest record was found for this filename.
- Write `filter_signals.jsonl` (local-only): one record per image, all signal fields.
- Write `filter_signals_summary.json` (tracked): counts per signal bucket (how many images have
  each signal true), plus `total_images`, `no_manifest_record_count`, `source`, `license`.

**Patterns to follow:**
- JSONL key-by-field pattern from `intake_telegram_dedup.py` (`manifest_by_fn` dict).
- `str.lower() + keyword in text` matching (no regex dependency).
- `intake_constants.py` for keyword list constants (add keyword lists there).

**Test scenarios:**
- Happy path: image with lure caption → `caption_lure_keyword=True`.
- Happy path: image with no manifest record → `has_manifest_record=False`, all caption signals
  `False`, `caption_length=0`.
- Happy path: normal portrait fish photo → aspect_class="portrait", no keyword signals.
- Edge case: caption contains both a lure keyword AND a fish-part keyword → both signals `True`
  (classify step resolves conflict).
- Edge case: `width=0` or `height=0` → `aspect_ratio=None`, `aspect_class="unknown"`.
- Edge case: `width=None` (corrupt image in universe — should not happen after U4.1 filtering).
- Edge case: very long caption (> 500 chars) → `caption_text_heavy=True`.
- Error path: `filter_universe.jsonl` missing → exits non-zero.
- Error path: `manifest.jsonl` missing → warn and continue with `has_manifest_record=False`
  for all records (graceful degradation).

**Verification:**
- `filter_signals_summary.json` field counts sum correctly: each count ≤ `total_images`.
- `filter_signals.jsonl` gitignored; `filter_signals_summary.json` tracked (U8 safety tests).

---

- U4. **`intake_filter_classify.py` — Assign coarse categories via priority waterfall**

**Goal:** Apply a deterministic priority-ordered rule set to `filter_signals.jsonl` to assign
each image a `coarse_category` and `confidence`. Write local-only `filter_candidates.jsonl`
and tracked `filter_candidates_summary.json`.

**Requirements:** R3, R3a, R6, R9

**Dependencies:** U3 (filter_signals.jsonl must exist)

**Files:**
- Create: `scripts/intake_filter_classify.py`
- Test: `tests/test_intake_filter_classify.py`

**Approach:**
- Load `filter_signals.jsonl`.
- Classify in two phases (geometry first, caption refinement second):
  **Phase 1 — visual heuristics (geometry is the primary signal):**
  1. `poster_screenshot`, confidence=high: `aspect_class IN ["extreme_landscape",
     "extreme_portrait"]` AND (`file_size_bucket="tiny"` OR `caption_text_heavy=True`).
  2. `poster_screenshot`, confidence=medium: `low_res=True` AND `file_size_bucket="tiny"`.
  3. `fish_candidate`, confidence=medium: NOT `low_res` AND
     `file_size_bucket IN ["medium", "large"]` AND
     `aspect_class IN ["portrait", "square", "landscape"]`.
  4. `fish_candidate`, confidence=low: `file_size_bucket="small"` AND aspect_class not extreme.
  5. `unresolved`: no geometry rule matched.
  **Phase 2 — caption hints refine a geometry-established candidate (caption cannot
  independently assign a category; conflicts route to `unknown_needs_review`):**
  6. `fish_candidate` AND `caption_fry_keyword=True`
     → `fry_juvenile` (carry phase-1 confidence).
  7. `fish_candidate` AND `caption_fish_part_keyword=True` AND NOT `caption_fry_keyword`
     → `fish_part` (carry phase-1 confidence).
  8. `fish_candidate` AND `caption_lure_keyword=True` AND NOT `caption_fish_part_keyword`
     → `lure_fishing_gear` (carry phase-1 confidence).
  9. `fish_candidate` AND `caption_no_fish_keyword=True`
     → `unknown_needs_review`, confidence=low [caption contradicts visual fish signal].
  10. `fish_candidate` AND no negative caption signal → `fish` (carry phase-1 confidence).
  11. `poster_screenshot` candidate AND any fish/species caption keyword
      → `unknown_needs_review`, confidence=low [caption contradicts visual poster signal].
  12. `unresolved` AND any caption keyword fires
      → `unknown_needs_review`, confidence=low [caption hint with no visual anchor].
  13. catch-all → `unknown_needs_review`, confidence=low.
- Record `coarse_category`, `confidence`, and `matched_rule_index` (int 1–13) for
  debuggability.
- Record `review_status: "auto_assigned"` for all records at this stage.
- Write `filter_candidates.jsonl` (local-only).
- Write `filter_candidates_summary.json` (tracked): per-category counts, per-confidence
  counts, per-category confidence breakdown, total, `source`, `license`.
- CLI: `--signals`, `--output-dir`, `--dry-run`.

**Patterns to follow:**
- `intake_constants.py` COARSE_CATEGORIES for iteration and validation.
- Waterfall as explicit `if/elif` chain (not a dict dispatch) for readability.

**Test scenarios:**
- Happy path: medium file, portrait aspect, no caption signals → `fish`, medium (geometry only).
- Happy path: medium file, portrait aspect + `caption_fry_keyword=True` → `fry_juvenile`, medium
  (caption refines geometry candidate).
- Happy path: medium file, portrait aspect + `caption_lure_keyword=True` → `lure_fishing_gear`,
  medium (caption refines geometry candidate).
- Happy path: medium file, portrait aspect + `caption_fish_part_keyword=True` → `fish_part`,
  medium (caption refines geometry candidate).
- Happy path: extreme landscape aspect + tiny file size → `poster_screenshot`, high (visual only).
- Happy path: all signals False, no clear geometry signal → `unknown_needs_review`, low.
- Edge case: `caption_lure_keyword=True` alone, no geometry rule fires → `unknown_needs_review`,
  low (caption with no visual anchor — rule 12).
- Edge case: `caption_fry_keyword=True` alone, no geometry rule fires → `unknown_needs_review`,
  low (caption with no visual anchor — rule 12).
- Edge case: `fish_candidate` established + `caption_no_fish_keyword=True` →
  `unknown_needs_review`, low (caption-visual conflict — rule 9).
- Edge case: `poster_screenshot` candidate + fish species caption keyword →
  `unknown_needs_review`, low (caption-visual conflict — rule 11).
- Edge case: `caption_lure_keyword=True` AND `caption_fish_part_keyword=True`, fish_candidate
  established → rule 8 blocked by `caption_fish_part_keyword`; falls to rule 7 → `fish_part`.
- Edge case: `caption_text_heavy=True` but normal aspect, normal size → rule 1 blocked; phase-1
  falls to rule 3/4 if geometry qualifies; phase-2 checks no conflict; result is `fish` (medium).
- Edge case: `low_res=True` but large file size → rule 2 blocked; continues through phase-1.
- Edge case: every category appears at least once in a mixed batch → summary counts sum to total.
- Integration: every output record's `coarse_category` is in `COARSE_CATEGORIES`.
- Integration: no output record has `coarse_category != "unknown_needs_review"` where only a
  caption signal fired and no geometry rule was satisfied.

**Verification:**
- `filter_candidates_summary.json` category counts sum to `filter_universe_summary.total_unique`
  (minus corrupt_excluded, which were excluded in U4.1).
- `unknown_needs_review` category count < 20% of total is a soft warning (not a gate here;
  the 5% gate is post-review).

---

- U5. **`intake_filter_review.py` — Generate per-category HTML contact sheets**

**Goal:** For each coarse category, generate a local-only HTML review pack sampling N images,
with file:// photo references and per-image confirm/reassign/unknown radio buttons. The reviewer
uses the browser to inspect samples and exports decisions as `filter_decisions_<category>.json`.

**Requirements:** R4, R6, R8

**Dependencies:** U4 (filter_candidates.jsonl must exist)

**Files:**
- Create: `scripts/intake_filter_review.py`
- Test: `tests/test_intake_filter_review.py`

**Approach:**
- Load `filter_candidates.jsonl`.
- By default generate packs for all 7 categories; `--category` flag restricts to one.
- For each category, sample up to `--sample N` records (default 50; `unknown_needs_review`
  default 100). Seed: 42.
- HTML structure per category: title, category summary stats, scrollable grid of image cards.
  Each card: the photo (via `file://` URI to the original export dir), filename (no caption
  displayed), auto-assigned category badge, confidence badge, and three radio buttons:
  `CONFIRM` / `REASSIGN: <dropdown of other categories>` / `UNKNOWN`.
- "Export Decisions" button generates `filter_decisions_<category>.json` via browser JS download.
- Decision record schema: `{ "filename": "...", "auto_category": "...", "decision": "CONFIRM|REASSIGN|UNKNOWN", "reassign_to": "<category>|null" }`.
- Path traversal guard: same `_safe_photo_path()` pattern as `intake_review_boundary.py`.
- No captions, sender names, or Telegram metadata displayed in HTML.
- CLI: `--candidates`, `--export-dir`, `--output-dir`, `--category`, `--sample`, `--seed`.

**Patterns to follow:**
- `intake_review_boundary.py` for `_file_uri()`, `_safe_photo_path()`, HTML structure, radio
  buttons, JS export.
- `intake_constants.py` COARSE_CATEGORIES for category dropdown population.

**Test scenarios:**
- Happy path: 7 categories in candidates → 7 HTML files generated.
- Happy path: category with 200 images sampled at default 50 → HTML has 50 image cards.
- Happy path: `--category fish` → only fish HTML generated.
- Edge case: category with 0 images → HTML generated with empty-state message (no error).
- Edge case: category with fewer images than `--sample` → all images included (no truncation error).
- Edge case: filename with spaces or special characters → `_file_uri()` percent-encodes correctly.
- Error path: path traversal attempt in filename (`../../../etc/passwd`) → `ValueError` raised
  before URI generation.
- Error path: export_dir missing → exits non-zero with message.
- Error path: `filter_candidates.jsonl` missing → exits non-zero.

**Verification:**
- All HTML files land in `review/` and are gitignored (U8 safety test).
- No captions or sender names appear in the HTML source.

---

- U6. **`intake_filter_finalize.py` — Ingest decisions and produce tracked summary**

**Goal:** Read `filter_decisions_<category>.json` files produced by the HTML review packs,
merge with `filter_candidates.jsonl`, compute per-category override rates, and write a tracked
privacy-safe `filter_review_summary.json`. Optionally write a finalized local-only
`filter_candidates_final.jsonl` with updated category assignments.

**Requirements:** R5, R6, R9, R10

**Dependencies:** U4 (filter_candidates.jsonl), U5 (review decisions)

**Files:**
- Create: `scripts/intake_filter_finalize.py`
- Test: `tests/test_intake_filter_finalize.py`

**Approach:**
- Load `filter_candidates.jsonl` (full universe).
- Load all `filter_decisions_<category>.json` files from `review/` dir (or explicit `--decisions`
  paths).
- For each reviewed image: update `review_status` to `reviewed_confirmed` or `reviewed_reassigned`;
  update `coarse_category` if REASSIGN decision.
- For each category: compute `override_rate = reassigned_count / reviewed_count`.
- Gate check: if any category's override_rate > `--override-threshold` (default 0.20) → print
  recommendation to re-inspect that category; exit code 2.
- Completeness check: validate that reviewed images cover at least `min_coverage_pct` (default
  80%) of each category's sampled images.
- Write `filter_review_summary.json` (tracked): per-category `{auto_count, reviewed_count,
  confirmed_count, reassigned_count, unknown_count, override_rate, final_count}`, plus
  `total_unique`, `review_complete` flag, `unknown_needs_review_pct` (must be < 5% for R10),
  `source`, `license`. No filenames, captions, or per-image decisions.
- Write local-only `filter_candidates_final.jsonl` with updated categories (gitignored).
- CLI: `--candidates`, `--decisions-dir`, `--output-dir`, `--override-threshold`,
  `--min-coverage-pct`, `--dry-run`, `--force`.
- Exit codes: 0 = complete and accepted; 1 = hard error; 2 = valid but non-final.

**Patterns to follow:**
- `intake_review_finalize.py` exit code 0/1/2 pattern, `_write_json_atomic`, `--force` flag,
  `--dry-run`.
- `intake_review_finalize.py` idempotent re-run detection (hash of input file).

**Test scenarios:**
- Happy path: all decisions are CONFIRM → override_rate=0, review_complete=True, exit 0.
- Happy path: 15% REASSIGN → below threshold, review_complete=True, exit 0.
- Error path: override_rate > 0.20 for a category → print recommendation, exit 2.
- Error path: decisions file references filename not in filter_candidates.jsonl → exit 1.
- Edge case: no decisions files found and `--partial` not set → exit 1.
- Edge case: `--dry-run` → prints summary, does not write files, exits 2.
- Edge case: decisions file has `UNKNOWN` decisions → counted separately, not reassigned.
- Edge case: `unknown_needs_review_pct >= 5%` after finalization → R10 warning emitted.
- Integration: categories in `filter_review_summary.json` all appear in `COARSE_CATEGORIES`.
- Integration: `total_final = sum(category final_count values)` == `filter_universe_summary.total_unique`.

**Verification:**
- `filter_review_summary.json` is tracked and present; `filter_candidates_final.jsonl` is gitignored.
- `review_complete: true` written only when all gates pass.

---

- U7. **`.gitignore` updates and `test_intake_filter_safety.py` — Safety gates**

**Goal:** Add gitignore rules for the three new local-only JSONL files, verify all new tracked
summaries are not accidentally gitignored, and add a regression test that requirements-intake.txt
still does not gain ML deps.

**Requirements:** R6, R7

**Dependencies:** U1–U6 (must know all artifact paths before writing rules)

**Files:**
- Modify: `.gitignore`
- Create: `tests/test_intake_filter_safety.py`

**Approach:**
- `.gitignore` additions (append under the existing intake privacy section):
  ```
  data/intake_meta/**/filter_universe.jsonl
  data/intake_meta/**/filter_signals.jsonl
  data/intake_meta/**/filter_candidates.jsonl
  data/intake_meta/**/filter_candidates_final.jsonl
  ```
- `review/filter_review_*.html` and `review/filter_decisions_*.json` are already covered
  by the existing `data/intake_meta/**/review/` rule.
- `tests/test_intake_filter_safety.py` safety checks:
  - `filter_universe.jsonl` IS gitignored.
  - `filter_signals.jsonl` IS gitignored.
  - `filter_candidates.jsonl` IS gitignored.
  - `filter_candidates_final.jsonl` IS gitignored.
  - `filter_review_<category>.html` IS gitignored (covered by review/ rule).
  - `filter_decisions_<category>.json` IS gitignored (covered by review/ rule).
  - `filter_universe_summary.json` is NOT gitignored (tracked).
  - `filter_signals_summary.json` is NOT gitignored (tracked).
  - `filter_candidates_summary.json` is NOT gitignored (tracked).
  - `filter_review_summary.json` is NOT gitignored (tracked).
  - Regression: `dedup_clusters.jsonl` and `dedup_summary.json` still NOT gitignored.
  - Regression: `audit.jsonl` still IS gitignored.
  - Regression: requirements-intake.txt still has no torch/ultralytics entries.

**Patterns to follow:**
- `tests/test_intake_safety.py` `git check-ignore -v` + `subprocess.run()` pattern verbatim.

**Test scenarios:**
- Happy path: all 4 new local-only JSONL paths return `returncode == 0` from `git check-ignore`.
- Happy path: all 4 new summary JSON paths return `returncode != 0` from `git check-ignore`.
- Regression: existing artifacts maintain correct gitignore status after new rules added.
- Regression: new `.gitignore` rules do not accidentally glob-match tracked summary files.

**Verification:**
- `pytest tests/test_intake_filter_safety.py -v` exits 0.
- No existing safety tests in `tests/test_intake_safety.py` regress.

---

## System-Wide Impact

- **Interaction graph:** U4 scripts read `manifest.jsonl`, `audit.jsonl`, `dedup_clusters.jsonl`
  (all existing local-only or tracked artifacts). They do not modify any existing file. The bot
  runtime, training scripts, launchd service, and model weights are untouched.
- **Error propagation:** Each script fails fast with a clear message and non-zero exit if its
  required input files are missing. No script writes partial output on failure (atomic writes).
- **State lifecycle risks:** `filter_candidates_final.jsonl` must not be consumed by downstream
  staging (U7) until `filter_review_summary.json` has `review_complete: true`. Scripts should
  check this gate flag before reading final assignments.
- **Staging eligibility:** only images with `coarse_category = "fish"` are eligible for species
  candidate labeling (U5+) and Stage B dataset staging. `poster_screenshot`, `no_fish`,
  `fry_juvenile`, `fish_part`, and `lure_fishing_gear` are excluded from normal species training
  unless explicitly re-routed by a future plan. Any downstream staging script must assert this
  gate before copying candidates.
- **API surface parity:** `intake_constants.py` is imported by all intake scripts; adding
  constants there is safe provided existing constant names are not changed.
- **Integration coverage:** The `filter_universe_summary.total_unique` value must equal
  `dedup_summary.total_unique_after_dedup - corrupt_excluded`. If this invariant fails, a data
  pipeline bug exists between U3 dedup and U4.1 universe derivation.
- **Unchanged invariants:** `manifest.jsonl`, `audit.jsonl`, `dedup_clusters.jsonl`,
  `dedup_summary.json`, and `dedup_review_summary.json` are read-only throughout U4. No existing
  tracked file is modified.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Caption quality is low (short, empty, or inconsistent Russian) — caption signals cover < 30% of images | Geometry signals provide a fallback path; high `unknown_needs_review` fraction triggers R10 warning; manual review packs handle residual |
| `filter_universe.jsonl` accidentally gitignored (negation rule fails) | U7 safety test explicitly verifies all four new gitignore rules with `git check-ignore` |
| `filter_signals_summary.json` accidentally gitignored (new rule too broad) | U7 safety test regression: verifies tracked summaries return `returncode != 0` |
| Caption keyword list misses key Russian terms (false negatives) → under-classification | `unknown_needs_review` is the conservative fallback; keyword list refined during implementation from caption sample |
| Geometry thresholds calibrated on wrong distribution → over-classification of fish as poster | Calibrate from actual audit.jsonl distribution before finalizing thresholds; `--dry-run` mode lets implementer inspect distribution before writing |
| `filter_candidates_final.jsonl` consumed by U7 staging before review_complete=True | Stage U7+ must check gate flag; note this dependency in U6 finalize output and U7 staging plan |
| `detector_v1.pt` model format or class order drifts before optional ML pass is implemented | ML pass is deferred and isolated from requirements-intake.txt; no risk to U4 acceptance criteria |
| requirements-intake.txt gains an ML dep by accident during U4 implementation | U7 safety test explicitly asserts no torch/ultralytics in requirements-intake.txt |

---

## Documentation / Operational Notes

- **Running order:** `intake_filter_universe.py` → `intake_filter_heuristic.py` →
  `intake_filter_classify.py` → `intake_filter_review.py` → *(manual review)* →
  `intake_filter_finalize.py`.
- **Re-run safety:** all scripts accept `--dry-run`; all writes are atomic. Scripts can be
  re-run safely without corrupting partial outputs.
- **Manual review scope:** reviewer only needs to inspect the sampled HTML packs, not all 32K
  images. If a category's override rate exceeds 20% the finalize script will recommend a
  re-classification pass with adjusted thresholds. Manual review decisions produced by U6
  are authoritative — they override all automatic classification regardless of confidence level.
- **Caption signals as candidate hints only:** caption keywords inform confidence but do not
  determine category. An image assigned `unknown_needs_review` because its caption fired with
  no visual anchor is a normal expected outcome, not a signal-extraction failure. The keyword
  lists should be calibrated for recall (catch candidates), not precision (decide outcomes).
- **Optional ML pass:** after U4 is complete, run `intake_filter_ml.py` (separate PR) to
  pass `unknown_needs_review` images through `detector_v1.pt` (requires `requirements-ml.txt`
  environment on Python 3.12). The ML pass refines assignments but is not required for U4
  acceptance.

---

## Sources & References

- **Origin document:** [docs/plans/2026-04-25-003-feat-dedup-boundary-review-plan.md](docs/plans/2026-04-25-003-feat-dedup-boundary-review-plan.md)
- Related code: `scripts/intake_constants.py`, `scripts/intake_review_boundary.py`, `scripts/intake_review_finalize.py`
- Related data: `data/intake_meta/tg_2026-04-24/dedup_summary.json`, `data/intake_meta/tg_2026-04-24/dedup_review_summary.json`
- Related tests: `tests/test_intake_safety.py`, `tests/test_intake_review_finalize.py`
