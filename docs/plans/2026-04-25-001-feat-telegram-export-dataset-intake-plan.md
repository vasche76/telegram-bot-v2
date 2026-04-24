---
title: feat: Telegram Export Dataset Intake Pipeline
type: feat
status: active
date: 2026-04-25
---

# feat: Telegram Export Dataset Intake Pipeline

## Overview

Build a multi-phase offline pipeline that safely turns the raw private Telegram chat export (65,423 JPGs + `messages*.html`) into curated, labeled candidate images ready to promote into the existing `data/fish_dataset/` structure. No photos are committed to git. No training is triggered. The export is treated as read-only throughout.

---

## Problem Frame

The bot's fish-recognition classifier is bottlenecked by class imbalance: `taimen` (1 image), `zander` (0), `unknown_fish` (0), `whitefish` (46), `crucian_carp`/`wels_catfish` (70 each). Any of these classes below 15 images triggers the `InactiveClassFallbackError` guard in `bot/fish_vision/local_classifier.py` and forces a GPT fallback. A private Telegram fishing chat has accumulated 65,423 in-domain photos — the richest available source for rare classes. This pipeline extracts structured metadata and candidate images from that export without modifying any source files, without committing photos to git, and without training.

---

## Requirements Trace

- R1. Parse all `messages*.html` files to build a photo→message manifest (caption, timestamp, message_id, sender)
- R2. Audit all 65,423 JPGs: dimensions, file size, SHA-256 hash; flag `low_res` when max-side < 800 px
- R3. Detect and cluster exact (SHA-256) and near-duplicate (pHash) photos; select one `keep` per cluster
- R4. Extract fish category (whole_fish / lure / fish_part / fry / no_fish / unknown) from captions and/or GPT
- R5. Extract species label from captions using Russian name mapping from `data/fish_dataset/class_map.json`
- R6. Flag images containing prominent human faces for privacy review; never auto-promote flagged images
- R7. Organize candidate images into staged intake directories (gitignored) by category and species
- R8. Produce a stats report comparing candidate counts against current dataset gaps
- R9. Provide a gated promotion script that copies approved candidates to `data/fish_dataset/` with provenance records
- R10. JSONL manifest/audit text artifacts in `data/intake_meta/` are committable; no photo files ever enter git
- R11. Mass upscaling is rejected; all images proceed at native resolution; low-res images tagged for future optional evaluation
- R12. License/provenance: all intake records tagged `source=telegram_private_2026-04-24`, `license=private_training_only`
- R13. New dependencies isolated in `requirements-intake.txt`; intake scripts must not import `torch` or `ultralytics`

---

## Scope Boundaries

- No training (`train_stage_b.py` / `train_stage_a.py`) during this pipeline
- No modification of source export files (read-only access)
- No upscaling — rejected for mass intake; may be re-evaluated for selected rare-class candidates only in a future iteration
- No public sharing of intaked images; all remain private local training data
- No automatic promotion — U8 promotion requires explicit `--approve` flag and per-class targeting

### Deferred to Follow-Up Work

- Manual YOLO bounding-box annotation for Stage A whole_fish promotions: separate Roboflow workflow (see `data/fish_dataset/LABELING_GUIDE.md`)
- Zander training: requires intake → manual species confirmation → count reaches 15+ → retrain trigger
- Taimen training: same gate — intake first, evaluate count before scheduling retrain
- Optional upscaling for rare-class low-res candidates: separate decision, requires `low_res=true` query on manifest

---

## Context & Research

### Relevant Code and Patterns

- `scripts/check_duplicates.py` — MD5 hash dedup pattern; new pipeline extends to SHA-256 + pHash
- `scripts/ingest_external_dataset.py` — provenance tracking (`PROVENANCE_external.json`), `_validate_image()`, `_safe_filename()` helpers to re-use
- `data/fish_dataset/class_map.json` — canonical Russian names and aliases for all 16 species; primary reference for caption→species mapping
- `bot/services/ai.py` — OpenAI vision + retry pattern; reference for GPT batch classifier
- `scripts/validate_dataset.py` — `STAGE_A_CLASSES`, `STAGE_B_SPECIES` canonical class lists
- `scripts/build_dataset.py` — argparse/logging/progress conventions for new scripts to follow
- `data/fish_dataset/LABELING_GUIDE.md` — defines Stage A and Stage B labeling semantics; intake output must conform
- `data/fish_models/metadata.json` — current per-class image counts; reference for gap analysis in U7

### Institutional Learnings

- Images down to 400 px max-side are usable for EfficientNet-B0 (224 px input); no upscaling needed for Stage B
- `data/intake/` and `ChatExport_*/` are already gitignored in `.gitignore` (lines 49–52); photo files will never be tracked
- `data/intake/` is blanket-ignored as a directory — `.jsonl` text manifests need a gitignore exception (`!data/intake/**/*.jsonl`) to become committable; see U9
- PIL/Pillow is the only image library used project-wide; no OpenCV dependency
- `beauitfulsoup4` is already in `requirements.txt`; no new HTML-parsing dependency needed

### External References

No external research needed — all required patterns are established in the existing codebase.

---

## Key Technical Decisions

- **JSONL over CSV for manifests**: Records have variable optional fields (species, confidence, privacy_flag). JSONL handles sparse fields cleanly; one file per pipeline phase.
- **SHA-256 over MD5 for new pipeline**: Stronger hash; MD5 remains in `check_duplicates.py` for existing dataset and can cross-reference when needed.
- **pHash (imagehash) for near-duplicate clustering with bucket pre-filtering**: Fast, offline, no API cost. Naive O(n²) comparison of 65K images is ~2.1 billion pairs — unusable. Strategy: bucket images by the first 8 bits of the 64-bit pHash (256 buckets); run hamming ≤ 8 comparison only within each bucket. This reduces comparisons to O(n) with bounded bucket sizes while preserving correctness for tight thresholds. Threshold exposed as `--phash-threshold` CLI arg (default 8).
- **Caption-first, GPT-second for classification**: Rule-based Russian keyword matching from `class_map.json` covers labeled photos cheaply. GPT-4o-mini batch pass (default 20 images/call) handles unlabeled/ambiguous residuals. Minimizes API cost for 65K photos.
- **Privacy flagging integrated into GPT pass**: GPT returns `{category, species, confidence, face_visible}` in one call per batch. Face-visible images are flagged, not deleted.
- **Staged intake directory, not dataset directory**: All candidates land in `data/intake/tg_2026-04-24/candidates/` (gitignored). Promotion to `data/fish_dataset/` is a separate, explicitly-gated operation.
- **JSONL manifests stored in `data/intake_meta/` (not `data/intake/`)**: Git does not recurse into an already-ignored directory — a `!data/intake/**/*.jsonl` negation after the `data/intake/` blanket rule is silently ineffective (verified with `git check-ignore`). Solution: store all text manifests in `data/intake_meta/tg_2026-04-24/` which is NOT gitignored, and keep photos-only under `data/intake/tg_2026-04-24/candidates/`. This requires no `.gitignore` modification and keeps the safety invariant clean.
- **No mediapipe dependency**: Privacy flagging delegated to the existing GPT-4o-mini vision pass. Avoids a heavy native library; keeps pipeline pure-Python installable under Python 3.13.
- **Intake scripts run on Python 3.13**: Unlike training scripts (require 3.12 for PyTorch), intake scripts use only PIL + imagehash + BeautifulSoup — all available on 3.13. `requirements-intake.txt` must not include `torch`.
- **Upscaling explicitly rejected**: All images processed at native resolution. Low-res images (max-side < 800 px) are tagged `low_res=true` in audit.jsonl for future optional evaluation per R11.

---

## Open Questions

### Resolved During Planning

- **Should manifests be committed?** Optional — text JSONL files can be committed by adding `!data/intake/**/*.jsonl` gitignore exception (recommended for reproducibility); U9 provides both options.
- **SHA-256 or MD5?** SHA-256 for new pipeline; existing `check_duplicates.py` MD5 unchanged.
- **pHash threshold?** Hamming ≤ 8 default; configurable via `--phash-threshold`.
- **Privacy approach?** Integrate `face_visible` detection into the GPT batch classification call — no separate script or native library.
- **Where to put new scripts?** `scripts/` directory, following existing conventions.

### Deferred to Implementation

- **Exact HTML structure of `messages*.html`**: BeautifulSoup selector paths depend on the actual Telegram export format. Implementer must inspect 2–3 actual HTML files before writing the parser.
- **Whether photo hrefs in HTML use relative or absolute paths**: Inspect before implementing path resolution.
- **Optimal GPT batch size**: 20 images/call is the starting estimate; adjust if rate limits are hit.
- **pHash performance on 65K images**: Expected ~5 min on Apple Silicon; may need chunked processing with `concurrent.futures` if slower.
- **Actual pending-classification count**: Run U4 caption pass first; the GPT cost estimate for U5 depends on how many records remain `pending` after caption matching.

---

## Output Structure

    data/intake_meta/               # NOT gitignored — text manifests, committable
      tg_2026-04-24/
        manifest.jsonl            # one record per photo (HTML metadata)
        audit.jsonl               # one record per photo (dimensions, SHA-256, low_res)
        dedup_clusters.jsonl      # one record per near-duplicate cluster
        classification.jsonl      # category + species labels per photo
        privacy_flags.jsonl       # images flagged face_visible=true
        stats_report.txt          # gap analysis vs current dataset

    data/intake/                  # gitignored — photos only, never committed
      tg_2026-04-24/
        candidates/
          stage_a/
            whole_fish/
            lure/
            fish_part/
            fry/
            no_fish/
            needs_review/
          stage_b/
            pike/
            taimen/
            grayling/
            whitefish/
            perch/
            brown_trout/
            rainbow_trout/
            atlantic_salmon/
            common_carp/
            crucian_carp/
            bream/
            roach/
            ide/
            wels_catfish/
            zander/
            unknown_fish/
            needs_review/
          MANIFEST.json           # copy counts per class

    scripts/
      intake_constants.py         # U9: shared paths + class lists
      intake_telegram_manifest.py # U1: HTML parser → manifest.jsonl
      intake_telegram_audit.py    # U2: image auditor → audit.jsonl
      intake_telegram_dedup.py    # U3: pHash clustering → dedup_clusters.jsonl
      intake_telegram_classify.py # U4+U5: caption rules + GPT → classification.jsonl
      intake_telegram_stage.py    # U6: staging organizer → candidates/
      intake_telegram_stats.py    # U7: gap analysis report
      intake_telegram_promote.py  # U8: gated promotion → data/fish_dataset/

    requirements-intake.txt       # imagehash>=4.3.1; beautifulsoup4 reference

    tests/
      test_intake_manifest.py
      test_intake_audit.py
      test_intake_dedup.py
      test_intake_classify.py
      test_intake_stage.py
      test_intake_promote.py
      test_intake_safety.py

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
[Telegram Export — READ ONLY]
~/Downloads/Telegram Desktop/ChatExport_2026-04-24/
  photos/  (65,423 JPGs)
  messages.html, messages2.html, ...
         │
         ▼
  U1: intake_telegram_manifest.py
  ───────────────────────────────
  BeautifulSoup parse messages*.html
  → data/intake_meta/tg_2026-04-24/manifest.jsonl
    {filename, msg_id, ts, sender_id, sender_name, caption, source, license}
         │
         ▼
  U2: intake_telegram_audit.py
  ────────────────────────────
  Two-open pattern: open→verify() then re-open→read dimensions
  SHA-256 hash in 64 KB chunks
  → data/intake_meta/tg_2026-04-24/audit.jsonl
    {filename, sha256, width, height, max_side, file_size, low_res, corrupt}
         │
         ▼
  U3: intake_telegram_dedup.py
  ────────────────────────────
  Pass 1: group by SHA-256 → exact_duplicate clusters
  Pass 2: imagehash.phash() + bucket pre-filter → hamming ≤ 8 → perceptual clusters
  → data/intake_meta/tg_2026-04-24/dedup_clusters.jsonl
    {cluster_id, keep_filename, duplicates[], cluster_type, reason}
         │
         ▼
  U4+U5: intake_telegram_classify.py
  ────────────────────────────────────
  Pass 1: caption text → CAPTION_RULES from class_map.json
           Russian keywords → category + species  (label_source=caption_rule)
  Pass 2: pending records → GPT-4o-mini batches of 20
           returns {category, species, confidence, face_visible}
           face_visible=true → privacy_flags.jsonl + privacy_review_required=true
           parse_error → needs_review/ with privacy_check_incomplete=true
  → data/intake_meta/tg_2026-04-24/classification.jsonl
    {filename, category, species, confidence, label_source,
     privacy_review_required, privacy_check_incomplete, needs_manual_review}
         │
         ▼
  U6: intake_telegram_stage.py
  ────────────────────────────
  Skip: is_duplicate=true, privacy_review_required=true
  needs_review: parse_error (privacy_check_incomplete=true), needs_manual_review
  Copy → data/intake/tg_2026-04-24/candidates/stage_a/{category}/
  Copy → data/intake/tg_2026-04-24/candidates/stage_b/{species}/
  → data/intake/tg_2026-04-24/candidates/MANIFEST.json {counts + flags per class}
         │
         ▼
  U7: intake_telegram_stats.py
  ────────────────────────────
  Read metadata.json (current counts) + candidates/MANIFEST.json
  → data/intake_meta/tg_2026-04-24/stats_report.txt
    class | current | candidates | projected | gap_to_30 | status
    Highlight: taimen (CRITICAL), zander (CRITICAL), unknown_fish (CRITICAL)
         │
         ▼ (manual review by operator)
  U8: intake_telegram_promote.py
  ───────────────────────────────
  Default: dry-run (print what would be promoted)
  --approve: confirm → shutil.copy2 to data/fish_dataset/stage_a/raw/{class}/
                                    or data/fish_dataset/stage_b/{species}/
  SHA-256 cross-check against existing dataset (no re-duplication)
  Write PROVENANCE_telegram.json per destination class
  Remind operator to run validate_dataset.py before training
```

---

## Implementation Units

- U1. **HTML Manifest Parser**

**Goal:** Parse all `messages*.html` files in the Telegram export; produce `data/intake/tg_2026-04-24/manifest.jsonl` with one record per photo.

**Requirements:** R1, R10, R12

**Dependencies:** None

**Files:**
- Create: `scripts/intake_telegram_manifest.py`
- Create: `data/intake/tg_2026-04-24/manifest.jsonl`
- Test: `tests/test_intake_manifest.py`

**Approach:**
- Accept `--export-dir` CLI arg (default: `~/Downloads/Telegram Desktop/ChatExport_2026-04-24`)
- Accept `--output-dir` CLI arg (default: `data/intake/tg_2026-04-24`)
- Discover all `messages*.html` files in export-dir; sort numerically (messages.html, messages2.html, …)
- Use BeautifulSoup with `html.parser` (already in `requirements.txt`) to extract per-message blocks
- Per message: extract message_id, timestamp (parse Telegram date string → Unix epoch + ISO-8601 UTC), sender_id, sender_name, text/caption, and photo filename (href pointing into `photos/` subdir)
- Skip messages with no photo attachment; skip messages with no parseable photo href
- Tag every record with `source=telegram_private_2026-04-24`, `license=private_training_only`
- Write one JSONL record per unique photo filename; skip duplicates if the same filename appears in multiple HTML files
- Log: total messages scanned, total photo records written, total skipped

**Patterns to follow:**
- `scripts/build_dataset.py` argparse + logging conventions
- `scripts/ingest_external_dataset.py` JSONL write pattern

**Test scenarios:**
- Happy path: synthetic HTML with 10 photo messages → 10 JSONL records, all required fields present
- Multi-file: `messages.html` + `messages2.html` with non-overlapping photos → combined, no duplicates
- No-caption message: photo with no text → record created, `caption` field is empty string
- Text-only message: no photo href → skipped, not in output
- Duplicate filename: same photo href in two HTML files → only one record written
- Malformed date string: log warning, set `timestamp=null`, continue
- Missing `photos/` subdir: href points to non-existent file → record created with `parse_error=true`

**Verification:**
- Output record count matches the number of unique JPG filenames discoverable in `photos/` that appear in the HTML
- All records contain: `filename`, `msg_id`, `timestamp`, `source`, `license`
- Script completes without exception on the full 65,423-photo export

---

- U2. **Image Auditor**

**Goal:** For every photo in `manifest.jsonl`, open with PIL, record dimensions, file size, SHA-256, and `low_res` flag; write `data/intake/tg_2026-04-24/audit.jsonl`.

**Requirements:** R2, R10, R11

**Dependencies:** U1 (manifest.jsonl must exist)

**Files:**
- Create: `scripts/intake_telegram_audit.py`
- Create: `data/intake/tg_2026-04-24/audit.jsonl`
- Test: `tests/test_intake_audit.py`

**Approach:**
- Read `manifest.jsonl` to get filename list and export-dir
- For each photo: **two-open pattern is mandatory** — (1) `PIL.Image.open(path).verify()` for corruption check; (2) re-open with a fresh `PIL.Image.open(path)` to read `width`, `height`, `mode`. **`verify()` invalidates the file handle and leaves `img.size` stale or raises on subsequent access — never read dimensions from the same handle that called `verify()`.** The `_validate_image()` helper in `ingest_external_dataset.py` only uses `verify()` as a pass/fail guard and does not read dimensions — do not use it as a template for dimension extraction
- Compute SHA-256 in 64 KB chunks (`hashlib.sha256`); same pattern as `check_duplicates.py` MD5 loop
- Record `file_size` from `Path.stat().st_size`
- Flag `low_res=true` when `max_side < 800` (based on audit sample: 1,016 images in 400–799 range)
- Flag `corrupt=true` on any PIL exception; log and continue without aborting
- Write one JSONL record per photo, keyed on `filename`; include all audit fields

**Patterns to follow:**
- `scripts/check_duplicates.py` hash chunk loop
- `scripts/ingest_external_dataset.py` `_validate_image()` guard

**Test scenarios:**
- Happy path: valid 1280×960 JPEG → correct width/height/sha256/file_size, `low_res=false`, `corrupt=false`
- Low-res flag: 600×400 image → `low_res=true`; 800×600 → `low_res=false` (boundary: max_side=800 is OK)
- Corrupt file: truncated JPEG bytes → `corrupt=true`, no exception propagation, processing continues
- All records in manifest processed: audit.jsonl count == manifest.jsonl count
- Progress reporting: script logs or prints progress every 1,000 images

**Verification:**
- `audit.jsonl` record count matches `manifest.jsonl`
- SHA-256 values are unique across all non-duplicate images (spot-check: no hash collision in first 5,000)
- `corrupt` count == 0 consistent with initial audit sample (0 bad files in 2,000-image sample)

---

- U3. **Perceptual Dedup**

**Goal:** Cluster exact (SHA-256) and near-duplicate (pHash) photos. Output `dedup_clusters.jsonl` describing each cluster: which filename to `keep` and which are `duplicates`. Non-clustered images are not written to this file.

**Requirements:** R3, R10

**Dependencies:** U2 (audit.jsonl must have sha256 values)

**Files:**
- Create: `scripts/intake_telegram_dedup.py`
- Create: `data/intake/tg_2026-04-24/dedup_clusters.jsonl`
- Modify: `requirements-intake.txt` (add `imagehash>=4.3.1`)
- Test: `tests/test_intake_dedup.py`

**Approach:**
- Pass 1 — exact dedup: group audit records by SHA-256; groups of size > 1 are exact duplicates. Within each group, keep the record with the lowest `msg_id` (earliest message); mark rest as `exact_duplicate`.
- Pass 2 — perceptual dedup: compute `imagehash.phash()` for all non-exact-duplicate images (re-open with PIL). **Do not use naive O(n²) all-pairs comparison** — 65K images → ~2.1B pairs ≈ 30+ min. Use bucket pre-filtering: dict keyed on the top 8 bits of the 64-bit pHash integer (256 buckets, ~255 images/bucket average); run hamming comparison only within each bucket. Correct for thresholds ≤ 8 (images with hamming ≤ 8 differ in at most 8 of 64 bits; same-bucket guarantee holds). Expected runtime: < 2 min on Apple Silicon. For each perceptual cluster > 1: keep the image with the largest `max_side`; mark others as `perceptual_duplicate`.
- Write one JSONL record per cluster (exact or perceptual): `{cluster_id, keep_filename, duplicate_filenames[], cluster_type, reason}`
- For images with `corrupt=true`: skip pHash computation; flag as `corrupt_skipped`
- Support `--phash-threshold` CLI arg

**Patterns to follow:**
- `scripts/check_duplicates.py` exact-hash grouping pattern

**Test scenarios:**
- Exact duplicate pair: same SHA-256 → one cluster, keep lower msg_id, other marked `exact_duplicate`
- Near-duplicate pair: different SHA-256, hamming ≤ 8 → one cluster, keep highest `max_side`, other `perceptual_duplicate`
- Distinct images: hamming > 8 → not clustered, not in output
- All-unique dataset: dedup_clusters.jsonl is empty (zero records)
- Custom threshold: `--phash-threshold 12` clusters more aggressively than default 8
- Corrupt image in near-duplicate pair: corrupt image is not pHashed; only the valid image proceeds

**Verification:**
- No filename appears as `keep` in more than one cluster
- No filename appears as both `keep` and in any `duplicate_filenames[]`
- Total unique (non-duplicate) images = total manifest records − sum of all `duplicate_filenames[]` lengths

---

- U4. **Caption Analyzer + Russian Species Mapper**

**Goal:** First classification pass using caption text. Match Russian fish names and category keywords from `class_map.json` to assign `category` (Stage A class) and `species` (Stage B class) to as many records as possible. Write `classification.jsonl` with all manifest records; unresolved records carry `category=pending`.

**Requirements:** R4, R5, R10

**Dependencies:** U1 (manifest.jsonl)

**Files:**
- Create: `scripts/intake_telegram_classify.py` (handles both U4 caption pass and U5 GPT pass as subcommands)
- Create: `data/intake/tg_2026-04-24/classification.jsonl`
- Test: `tests/test_intake_classify.py`

**Approach:**
- Build `CAPTION_RULES` from `data/fish_dataset/class_map.json`:
  - For each species entry: collect `common_name_ru` + `aliases` (Russian variants and transliterations)
  - Stage A category keywords: приманка/воблер/блесна/мушка → `lure`; малёк/малёк → `fry`; etc.
  - Normalize caption to lowercase, strip punctuation before matching
- For each manifest record with non-empty caption:
  - Check Stage A category keywords first (lure, fry, no_fish patterns)
  - Check species keywords; if matched → `species={matched}`, `category=whole_fish` (default for species photos)
  - If species AND a non-whole_fish category both match: take the more specific; flag `ambiguous=true`
  - If no match: `category=pending`, `species=null`
- For records with empty caption: `category=pending`, `species=null`
- Tag `label_source=caption_rule` for resolved; `label_source=pending` for unresolved
- This script's `--caption-only` flag runs U4 and stops (no GPT); default run chains into U5

**Patterns to follow:**
- `data/fish_dataset/class_map.json` `common_name_ru` and `aliases` fields
- `scripts/validate_dataset.py` `STAGE_A_CLASSES` and `STAGE_B_SPECIES` as canonical class name authority

**Test scenarios:**
- Russian species name: caption "поймали щуку" → `species=pike`, `category=whole_fish`, `label_source=caption_rule`
- Lure keyword: caption "блесна уловистая" → `category=lure`, `species=null`
- No caption: empty string → `category=pending`, `species=null`, `label_source=pending`
- Ambiguous caption ("сегодня хорошая рыбалка" — no species/category keyword) → `category=pending`
- Case insensitivity: "Щука" and "щука" both match `pike`
- Multiple species keywords in one caption (unusual): first match wins, `ambiguous=true`
- All 16 species Russian names resolve to their correct class label

**Verification:**
- `classification.jsonl` record count == `manifest.jsonl` record count
- Spot-check: 50 manually selected captions with known species names are all resolved correctly
- Pending count logged at script exit

---

- U5. **GPT Batch Category Classifier**

**Goal:** For all `category=pending` records in `classification.jsonl`, send batches of images to GPT-4o-mini vision to classify category, species, confidence, and face_visible. Merge results back into `classification.jsonl`. Write `privacy_flags.jsonl` for face_visible=true records.

**Requirements:** R4, R5, R6, R10

**Dependencies:** U4 (classification.jsonl with pending records), U2 (for photo file paths)

**Files:**
- Create: integrated into `scripts/intake_telegram_classify.py` as `--gpt-pass` flag / second phase
- Create: `data/intake/tg_2026-04-24/privacy_flags.jsonl`
- Test: `tests/test_intake_classify.py` (extend with GPT mock tests)

**Approach:**
- Load `pending` records from `classification.jsonl`
- Build batches of up to `--batch-size` images (default 20)
- GPT prompt: structured JSON response per image — `{category: str, species: str|null, confidence: float 0–1, face_visible: bool}`. Categories constrained to Stage A class list. Species constrained to Stage B list + `"unknown"`.
- Use `OPENAI_API_KEY` from `.env` (via `python-dotenv`); replicate retry/backoff pattern from `bot/services/ai.py`
- `face_visible=true` records: append to `privacy_flags.jsonl`; in `classification.jsonl` set `privacy_review_required=true`
- `confidence < 0.5`: set `needs_manual_review=true`
- Support `--dry-run` (no API calls; all pending → `category=unknown, confidence=0.0, label_source=dry_run`)
- Support resume: if `classification.jsonl` already has resolved records, skip already-classified filenames
- Emit per-batch progress and running API cost estimate

**Patterns to follow:**
- `bot/services/ai.py` OpenAI retry/backoff and `.env` loading
- `scripts/build_dataset.py` batch progress logging

**Test scenarios:**
- Valid GPT JSON response: category/species correctly merged into `classification.jsonl`
- `face_visible=true` GPT response: record in `privacy_flags.jsonl`, `privacy_review_required=true` in classification
- Low confidence (0.3): `needs_manual_review=true` set in record
- GPT 429 rate limit: retry with backoff (test with mocked 429 → eventual success)
- Malformed GPT JSON response: logged as `parse_error`, record stays `category=pending`, script continues
- `--dry-run`: zero API calls, all pending records get mock label
- Resume: pre-populated `classification.jsonl` with some resolved records → only pending records re-queried
- Empty pending list: clean exit, log "no pending records to classify"

**Verification:**
- After full run: no records with `category=pending` (or only `parse_error` residuals, logged)
- `privacy_flags.jsonl` exists (may be empty if no faces detected)
- GPT cost estimate printed at completion

---

- U6. **Intake Staging Organizer**

**Goal:** Read merged `classification.jsonl` and `dedup_clusters.jsonl`; copy each non-duplicate, non-privacy-flagged, resolved-category image into the appropriate `candidates/` subdirectory; write `candidates/MANIFEST.json` with copy counts per class.

**Requirements:** R7, R10

**Dependencies:** U3, U4, U5 (classification.jsonl and dedup_clusters.jsonl must be final)

**Files:**
- Create: `scripts/intake_telegram_stage.py`
- Creates directories under: `data/intake/tg_2026-04-24/candidates/` (gitignored)
- Creates: `data/intake/tg_2026-04-24/candidates/MANIFEST.json`
- Test: `tests/test_intake_stage.py`

**Approach:**
- Build skip sets: filenames in any `duplicate_filenames[]` from `dedup_clusters.jsonl` (not keep); filenames with `privacy_review_required=true`
- **`parse_error` records from U5 (GPT could not classify)**: these were never privacy-checked; copy to `needs_review/` but tag with `privacy_check_incomplete=true` in `MANIFEST.json`. U8 must also block promotion of any `privacy_check_incomplete=true` image without `--force-privacy-override`.
- For each remaining resolved record:
  - Stage A: copy to `candidates/stage_a/{category}/tg_{msg_id}_{filename}`
  - Stage B: if `species` is non-null and non-pending, also copy to `candidates/stage_b/{species}/tg_{msg_id}_{filename}`
  - `needs_manual_review=true` or `category=pending` residuals: copy to `candidates/stage_a/needs_review/` and `candidates/stage_b/needs_review/`
- Filename convention: `tg_{msg_id}_{original_filename}` ensures traceability and prevents collisions
- Support `--dry-run` (log intended copies without writing)
- Write/update `candidates/MANIFEST.json`: `{class_name: count, ...}` for every destination directory

**Patterns to follow:**
- `scripts/ingest_external_dataset.py` copy + provenance pattern

**Test scenarios:**
- Normal whole_fish record with known species: appears in both `stage_a/whole_fish/` and `stage_b/{species}/`
- Lure record: appears only in `stage_a/lure/` (not in stage_b)
- Duplicate image (in dedup cluster): not copied, counted in skip log
- Privacy-flagged image: not copied, counted in privacy-skip log
- Pending/unresolved category: goes to `needs_review/`
- `--dry-run`: no filesystem changes, only log lines
- MANIFEST.json: after staging, counts per class match actual file counts in candidates/ directories

**Verification:**
- `candidates/MANIFEST.json` totals sum to total-manifest minus duplicates minus privacy-flagged minus already-staged
- No filename appears more than once per destination subdirectory

---

- U7. **Stats Report & Gap Analysis**

**Goal:** Compare candidate counts per class against current dataset image counts from `data/fish_models/metadata.json`; produce `data/intake/tg_2026-04-24/stats_report.txt` highlighting critical classes.

**Requirements:** R8

**Dependencies:** U6 (candidates/ directories populated)

**Files:**
- Create: `scripts/intake_telegram_stats.py`
- Create: `data/intake/tg_2026-04-24/stats_report.txt`

**Approach:**
- Load `data/fish_models/metadata.json` for current per-class training image counts
- Read `candidates/MANIFEST.json` for new candidate counts per class
- Print formatted table per Stage B species: `class | current | new_candidates | projected | gap_to_30 | status`
- Status thresholds: `CRITICAL` (projected < 5), `WARN` (projected < 30), `OK` (projected ≥ 30)
- Print Stage A summary similarly
- Header summary: total photos processed, duplicates removed, privacy-flagged, needs_review count
- Emphasize taimen/zander/unknown_fish as CRITICAL

**Test scenarios:**
Test expectation: none — pure reporting script, no behavioral logic to test.

**Verification:**
- Report runs without exception
- Projected counts = current + candidates (per class, manually verified for 3 spot-check classes)

---

- U8. **Promotion Gatekeeper**

**Goal:** Gated script to copy approved candidates from `data/intake/tg_2026-04-24/candidates/` to `data/fish_dataset/stage_a/raw/` or `data/fish_dataset/stage_b/{species}/`, with SHA-256 cross-dedup against existing dataset and `PROVENANCE_telegram.json` per destination class.

**Requirements:** R9, R12

**Dependencies:** U6 (candidates/ populated), U7 (stats report for decision guidance)

**Files:**
- Create: `scripts/intake_telegram_promote.py`
- Modifies (on explicit run): `data/fish_dataset/stage_a/raw/{class}/` or `data/fish_dataset/stage_b/{species}/`
- Creates: `data/fish_dataset/stage_b/{species}/PROVENANCE_telegram.json` (per promoted class)
- Test: `tests/test_intake_promote.py`

**Approach:**
- Without `--approve`: dry-run by default — print what would be promoted, exit 0
- With `--approve --class {class_name}`: print promotion plan; require interactive `y/N` confirmation before proceeding. **`--class all` is explicitly not supported** — the script must exit with an error if `all` is passed as the class name, requiring the operator to target one class at a time. This prevents a single command from mass-promoting all classes simultaneously.
- The interactive prompt is readable by a human but can be bypassed by piping input (`echo y | ...`). This is documented as an accepted operational risk for a local single-operator tool; no additional technical mitigation is required.
- Also block promotion of any `privacy_check_incomplete=true` image (from U6) without `--force-privacy-override`
- For each candidate image: re-verify SHA-256 against `audit.jsonl`; skip with error log on mismatch
- Cross-dedup: compute SHA-256 of candidate; if hash already exists in destination directory (scan existing files), skip as "already in dataset"
- Block promotion of any `privacy_review_required=true` image; log error; require explicit `--force-privacy-override` flag to override
- Copy using `shutil.copy2` (preserves timestamps)
- Append to `data/fish_dataset/stage_b/{species}/PROVENANCE_telegram.json`: `{filename, sha256, source_msg_id, source_timestamp, source, license, promoted_at}`
- Print per-class promotion summary; remind operator: "Run `python3 scripts/validate_dataset.py` before training"

**Patterns to follow:**
- `scripts/ingest_external_dataset.py` PROVENANCE_external.json write pattern
- `scripts/check_duplicates.py` hash cross-check

**Test scenarios:**
- No `--approve`: dry-run output only, no files copied, exit 0
- With `--approve --class pike` + `y` confirmation: images copied, PROVENANCE_telegram.json written
- With `--approve --class pike` + `n` confirmation: aborted, no files copied
- SHA-256 mismatch on candidate file: skipped with error, other files in batch continue
- Cross-dedup: SHA-256 of candidate matches an existing file → logged "already in dataset", not re-copied
- Privacy-flagged image in candidates/: blocked unless `--force-privacy-override` is also passed
- PROVENANCE_telegram.json structure: valid JSON with all required fields per promoted image

**Verification:**
- Destination class directory gains expected number of new `.jpg` files
- `PROVENANCE_telegram.json` is valid JSON and contains one entry per promoted image
- `python3 scripts/validate_dataset.py` exits with `Result: PASS` after promoting a class to ≥ 15 images

---

- U9. **Safety Infrastructure & Requirements**

**Goal:** Create `requirements-intake.txt`, `scripts/intake_constants.py` (shared path + class constants). Manifests go to `data/intake_meta/` (tracked); photos go to `data/intake/` (gitignored). No `.gitignore` modification needed.

**Requirements:** R10, R13

**Dependencies:** None (implement first or in parallel with U1)

**Files:**
- Create: `requirements-intake.txt`
- Create: `scripts/intake_constants.py`
- Test: `tests/test_intake_safety.py`

**Approach:**
- **Directory split**: `data/intake_meta/tg_2026-04-24/` for all JSONL/TXT text manifests (not in `.gitignore`, committable); `data/intake/tg_2026-04-24/candidates/` for photo copies (under `data/intake/` which is already gitignored). This avoids the git directory-ignore limitation — a `!negation` rule inside an already-ignored directory is silently ineffective in all versions of git.
- `requirements-intake.txt`: `imagehash>=4.3.1` (new); reference `beautifulsoup4>=4.12.0` for completeness (already in main requirements.txt); add a comment explicitly excluding `torch`/`torchvision`/`ultralytics` to prevent accidental ML dep creep
- `intake_constants.py`: `EXPORT_DIR`, `INTAKE_META_ROOT`, `INTAKE_CANDIDATES_ROOT`, `BATCH_ID`, `MANIFEST_PATH`, `AUDIT_PATH`, `DEDUP_PATH`, `CLASSIFICATION_PATH`, `PRIVACY_FLAGS_PATH`, `CANDIDATES_ROOT`; import `STAGE_A_CLASSES` and `STAGE_B_SPECIES` from `scripts/validate_dataset.py` to stay in sync with canonical class lists
- **Provenance requirement**: every new script must include a module-level docstring stating `source=telegram_private_2026-04-24, license=private_training_only`

**Patterns to follow:**
- `requirements-ml.txt` structure and version-pin style

**Test scenarios:**
- gitignore photos: `git check-ignore -v data/intake/tg_2026-04-24/candidates/stage_b/pike/test.jpg` → ignored
- manifests tracked: `git check-ignore -v data/intake_meta/tg_2026-04-24/manifest.jsonl` → NOT ignored (no rule matches)
- `intake_constants.py` imports without error from any `scripts/` module
- `requirements-intake.txt` does not contain `torch`, `torchvision`, or `ultralytics`

**Verification:**
- No `.jpg` or `.jpeg` files anywhere in `data/intake/` appear in `git status` after a staging run
- `data/intake_meta/tg_2026-04-24/manifest.jsonl` DOES appear in `git status` as a new tracked file

---

## Phased Delivery

### Phase 1 — Audit & Manifest (U9 → U1 → U2 → U3)
Establish safety infrastructure first, then build the audit chain. After this phase: full manifest + audit JSONL exists; exact and perceptual dedup clusters identified. No classification yet.

**Commit boundary 1:** U9 (requirements + .gitignore + constants) alone  
**Commit boundary 2:** U1 + U2 + their tests  
**Commit boundary 3:** U3 + its tests

### Phase 2 — Classification (U4 → U5)
Caption-based pass first (free). Estimate pending count before running GPT pass. Approve GPT cost before running U5.

**Commit boundary 4:** U4 + U5 + `tests/test_intake_classify.py`

### Phase 3 — Staging & Reporting (U6 → U7)
Organize candidates; generate gap report. No promotions yet.

**Commit boundary 5:** U6 + U7 + `tests/test_intake_stage.py`

### Phase 4 — Promotion (U8)
Implement and test gatekeeper. Do not run with `--approve` until stats report confirms which classes to promote first.

**Commit boundary 6:** U8 + `tests/test_intake_promote.py` — requires extra review before merge

---

## System-Wide Impact

- **Interaction graph:** All intake scripts are offline tools; they do not touch `bot/`, `data/bot.db`, the launchd service, or any running bot process. Promotion (U8) writes to `data/fish_dataset/` which `build_dataset.py` and training scripts read — but training is never triggered automatically.
- **Error propagation:** Each script writes to its own output file. Record-level failures log and skip; the run continues. Scripts are designed to be re-runnable (idempotent: skip already-processed records).
- **State lifecycle risks:** `classification.jsonl` is written by two scripts (U4 caption pass, U5 GPT pass) — U5 must update existing records in-place, not overwrite. Partial GPT runs must be resumable by re-running U5 (skips already-classified records).
- **API surface parity:** No new bot API endpoints; intake is entirely offline.
- **Integration coverage:** Promoted images must pass `validate_dataset.py` gate before any training run. `check_duplicates.py` should be run after promotion to catch any SHA-256 overlap with pre-existing dataset images.
- **Unchanged invariants:** `bot/`, `data/bot.db`, `data/fish_models/*.pt`, launchd plist, and the existing `data/fish_dataset/` structure are all untouched until U8 promotion is explicitly run with `--approve`.

---

## Risk Analysis & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GPT cost for 65K images ($6.50 is a floor, not a ceiling — colloquial Russian captions may not match class_map.json aliases, leaving many records `pending`) | Medium | Low | Run U4 caption pass first; print pending count before starting U5; use `--dry-run` cost estimate; budget for up to 2× the floor estimate |
| pHash false-positive: different species, similar color/shape clustered | Medium | Medium | Threshold ≤ 8 is conservative; `dedup_clusters.jsonl` stores all decisions for manual inspection before staging |
| Telegram HTML structure differs from expected | Medium | Medium | Deferred to implementation; inspect 2–3 actual files before writing selectors |
| Privacy-flagged images accidentally promoted | Low | High | U8 hard-blocks any `privacy_review_required=true` image without explicit `--force-privacy-override` |
| Large pHash compute time for 65K images | Low | Low | Progress bar; ~5 min on Apple Silicon; `concurrent.futures` escape hatch if needed |
| Intake candidates contain images already in existing dataset | Medium | Low | U8 SHA-256 cross-check against existing `stage_a/raw/` and `stage_b/` before copying |
| `data/intake_meta/` accidentally added to .gitignore in future | Low | Low | `data/intake_meta/` path is intentionally absent from .gitignore; document this as a committed-data path in U9 |
| Intake scripts accidentally import torch (Python 3.13 incompatibility) | Low | Medium | `requirements-intake.txt` explicitly lists forbidden deps; U9 test validates the file |

---

## Documentation / Operational Notes

- **After promotion:** Run `python3 scripts/validate_dataset.py` to verify class minimums before scheduling any training run with `build_dataset.py --train`.
- **Rollback:** Intake scripts write only to `data/intake/` (gitignored) and — after U8 promotion — to `data/fish_dataset/{class}/`. Promotion rollback: `rm data/fish_dataset/stage_b/{class}/tg_*.jpg` and remove corresponding entries from `PROVENANCE_telegram.json`. Intake manifests are unaffected.
- **License note:** All images from this Telegram export carry `license=private_training_only`. Do not share, publish, or upload to any external service without explicit consent from the group members.
- **Upscaling note (R11):** Mass upscaling is rejected. Images at 400–799 px max-side are sufficient for EfficientNet-B0 (224 px input) and usable for YOLO with letterboxing. If a specific rare-class species (taimen, zander) yields only low-res candidates, optional per-image upscaling may be evaluated in a future separate decision — never applied in bulk.
- **Training trigger:** After promotion, the standard training command is `python3 scripts/build_dataset.py --skip-download --train-b-only --device mps`. The `InactiveClassFallbackError` guard in `local_classifier.py` will automatically re-enable promoted classes once they exceed 15 training images.

---

## Sources & References

- Related code: `scripts/ingest_external_dataset.py` — provenance tracking pattern
- Related code: `scripts/check_duplicates.py` — SHA-256/MD5 dedup pattern
- Related code: `bot/services/ai.py` — GPT retry/backoff pattern
- Related code: `data/fish_dataset/class_map.json` — Russian species name authority
- Related code: `scripts/validate_dataset.py` — canonical class lists
- Related data: `data/fish_models/metadata.json` — current training image counts per class
- External: `imagehash>=4.3.1` (PyPI) — perceptual hash library
