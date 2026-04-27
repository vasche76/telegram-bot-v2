---
title: "feat: U4 Phase B — Per-Image Weak-Signal Heuristic Extraction"
type: feat
status: active
date: 2026-04-26
origin: docs/plans/2026-04-26-001-feat-intake-coarse-filter-u4-plan.md
---

# feat: U4 Phase B — Per-Image Weak-Signal Heuristic Extraction

## Overview

U4 Phase A produced `filter_universe.jsonl` (32,420 unique images) and confirmed
`filter_universe_summary.json`. Phase B extracts a deterministic per-image signal
vector — geometry signals from existing metadata and caption keyword hints from
`manifest.jsonl` — and writes a local-only `filter_signals.jsonl` plus a
tracked privacy-safe `filter_signals_summary.json`. The signal vector feeds the
Phase C classifier (`intake_filter_classify.py`), which assigns coarse categories
via a priority waterfall without touching this script.

**Caption is NOT truth.** Caption keywords are weak candidate hints recorded as
boolean flags alongside geometry signals. They cannot independently assign a
category. All conflict resolution is the classifier's responsibility.

---

## Problem Frame

32,420 images need coarse-category signals so the Phase C classifier can assign
fish / no_fish / lure_fishing_gear / fish_part / fry_juvenile / poster_screenshot /
unknown_needs_review. The only local, privacy-safe sources are:

- `filter_universe.jsonl` — pre-audited geometry (width, height, file_size, low_res, dedup_role).
- `manifest.jsonl` — captions (Russian Cyrillic, 24.5% of images have non-empty captions).

No raw image bytes need to be opened for the core signal set. PIL is available but
out-of-scope for Phase B to keep runtime fast (< 2 min for 32K records). Image
statistics are deferred to an optional `--with-image-stats` flag that is explicitly
off by default.

Geometry is the primary signal source. Caption keywords are supplemental weak hints.

---

## Requirements Trace

- R1. Join `filter_universe.jsonl` with `manifest.jsonl` captions; compute a
  deterministic signal vector for each of the 32,420 universe records.
- R2. All signals must be local, deterministic, and require no cloud API, OCR library,
  or NLP dependency beyond `str.lower() + keyword in text` substring matching.
- R3. Caption signals must be stored as boolean hint flags, structurally separated from
  geometry signals in the schema; each field name must include "caption_" prefix.
- R4. Conflicts (e.g., multiple caption signals True simultaneously) are recorded as-is;
  conflict resolution is the classifier's responsibility, not the heuristic script's.
- R5. `filter_signals.jsonl` must be local-only (gitignored); `filter_signals_summary.json`
  must be tracked and contain aggregate counts only — no filenames, no caption text, no
  sender names, no per-image decisions.
- R6. No new entries may be added to `requirements-intake.txt`; `Pillow` and stdlib cover
  all geometry and optional image stats needs.
- R7. The script must be re-runnable without error (idempotent writes); `--dry-run` must
  write no files.
- R8. Manifest missing → graceful degradation: warn, continue with `has_manifest_record=False`
  for all records. Universe missing → exit 1.

---

## Scope Boundaries

- This plan covers only `intake_filter_heuristic.py` and its constants extension in
  `intake_constants.py`. Phase C classifier, review pack, finalize, and staging scripts
  are out of scope.
- No raw photo bytes are read in the default (non-`--with-image-stats`) code path.
- Caption keyword matching uses `str.lower()` + substring; no regex corpus, no
  stemmer, no NLP library.
- `--with-image-stats` code path is scaffolded (flag accepted, output fields nulled) but
  image-stat computation is not implemented in Phase B. The flag exists so downstream
  tests can confirm the field schema without requiring full implementation.
- No modification to bot runtime, launchd service, model weights, training scripts,
  `.env`, or secrets.

### Deferred to Follow-Up Work

- `--with-image-stats` full implementation (mean_luminance, is_grayscale_like, edge_density)
  — separate PR or Phase B+ patch once Phase C baseline is established.
- Phase C: `intake_filter_classify.py` and `filter_candidates.jsonl` — separate plan.
- Keyword list expansion after Phase C override-rate analysis reveals which categories are
  being mis-classified at high rates — iterative improvement, separate PR.

---

## Context & Research

### Relevant Code and Patterns

- `scripts/intake_constants.py` — already defines `FILTER_SIGNALS_PATH`,
  `FILTER_SIGNALS_SUMMARY_PATH`, `COARSE_CATEGORIES`. U1 extends it with keyword frozensets
  and bucket-threshold constants.
- `scripts/intake_filter_universe.py` — canonical pattern for `_read_jsonl`, `_write_jsonl`,
  `_write_json_atomic`, `--dry-run` flag, `PROGRESS_EVERY = 1000` logging, `sys.exit(1)` on
  missing inputs, `SCHEMA_VERSION = 1`.
- `scripts/intake_telegram_dedup.py` — `manifest_by_fn` dict-keying join pattern.
- `tests/test_intake_filter_universe.py` — test structure, `_make_*` fixture helpers,
  `tmp_path` isolation, `@pytest.mark.skipif` for real-data tests.
- `tests/test_intake_filter_safety.py` — already includes `test_filter_signals_jsonl_is_gitignored`
  and `test_filter_signals_summary_not_gitignored`; Phase B must not break either.

### Corpus Data (Calibration Source — 2026-04-26 run)

Computed from live `filter_universe.jsonl` (32,420 records):

| Metric | Value |
|---|---|
| Total unique images | 32,420 |
| low_res (max_side < 800) | 371 (1.1%) |
| extreme_portrait (ar < 0.5) | 60 (0.2%) |
| extreme_landscape (ar ≥ 2.0) | 221 (0.7%) |
| extreme_aspect total | 281 (0.9%) |
| landscape (1.25–2.0) | 28,407 (87.6%) — dominant 4:3 JPEG format |
| portrait (0.5–0.8) | 2,595 (8.0%) |
| File size p25 / p50 / p75 / p95 | 161 / 214 / 278 / 393 KB |
| File size min / max | 6 KB / 983 KB |
| file_size < 50 KB ("tiny") | 137 (0.4%) |
| file_size 50–150 KB ("small") | ~8,100 (25.0%) |
| file_size 150–400 KB ("medium") | ~22,900 (70.6%) |
| file_size ≥ 400 KB ("large") | ~1,300 (4.0%) |
| Non-empty captions | 7,984 (24.5%) |
| Caption length p50 / p90 | 285 / 602 chars |
| Captions > 200 chars | ~4,944 (15.2% of all images) |

From `manifest_summary.json`: the channel is a private Russian fishing club
("Золотой Сазан" / Polivanovo / Buzlanovo / Varvarino locations). Captions are
primarily field reports by club admins describing which fish are biting on which
lures — not photo captions.

### Caption Keyword Corpus Analysis

**Critical finding:** This corpus is a fishing club's *reporting channel*, not a gear
review channel. Captions that mention lure names (воблер, блесна, джиг, балансир) are
overwhelmingly field reports of *fish caught on* that lure — not photos of the lure
itself. Similarly, "чистку улова" (clean the catch) appears as a prize description,
not as a filleting-photo caption. "Приглашаем" (we invite) appears in fish-stock-release
announcements that may include fish photos.

**Consequence:** All five caption keyword signals will have non-trivial false positive
rates for this corpus. The conservative waterfall in Phase C (caption refines geometry,
never assigns independently) is essential. The review phase (U5) will catch systematic
misclassification; the 20% override-rate gate in the finalize step (U6) will alert.

### External References

No external research needed — all necessary patterns are established locally.

---

## Key Technical Decisions

- **Geometry signals from JSONL metadata, not pixel reads (default path):** `filter_universe.jsonl`
  already carries `width`, `height`, `max_side`, `file_size`, `low_res`. Pixel reading is
  deferred to `--with-image-stats` (off by default). Rationale: avoids 30+ min runtime for
  32K JPEG reads; metadata is sufficient for Phase C's classifier.

- **File-size bucket constants in `intake_constants.py`, not inline:** four threshold constants
  (`BUCKET_TINY_MAX`, `BUCKET_SMALL_MAX`, `BUCKET_MEDIUM_MAX`) define three boundaries; all scripts
  use the same constants. Calibrated from actual corpus distribution:
  `tiny < 50 KB, small 50–150 KB, medium 150–400 KB, large ≥ 400 KB`.

- **Caption keyword lists as `frozenset` constants in `intake_constants.py`:** consistent
  with `STAGE_A_CLASSES_SET` pattern; easily extensible between Phase B and Phase C without
  touching the heuristic script.

- **Conservative keyword lists — exclude high-FPR terms confirmed from corpus sample:**
  "резина" (rubber lure — appears in catch reports), "снасть"/"снасти" (tackle — in field
  reports), "резину", "микроколебалки", "приглашаем" (in fish-release announcements) are
  excluded. Only terms specific enough to a non-fish photo context are included.

- **`has_manifest_record` as a first-class signal field:** tells the Phase C classifier
  whether caption signals are meaningful (`False` = no data) vs. computed from actual text.
  Not included in the public summary (privacy neutral, but count derivable from
  `no_manifest_record_count`).

- **`caption_text_heavy` threshold = 200 chars:** keeps ~15% of all images flagged;
  useful as a weak corroborating signal for poster detection only when combined with
  `extreme_aspect`. At p50 = 285 chars for captioned images, the threshold is generous
  (high recall, low precision), consistent with the conservative approach.

- **Conflict recording strategy — multiple signals can be simultaneously True:** the
  heuristic script records raw boolean signals independently. If both
  `caption_lure_keyword=True` and `caption_fish_part_keyword=True`, both are written as
  True. Phase C resolves this via the waterfall priority (fry beats fish_part beats lure).

- **`image_stats_computed=False` field in schema (scaffolded but not computed):** downstream
  scripts can inspect this flag to determine whether optional stats are available.
  `mean_luminance`, `is_grayscale_like`, `edge_density` are present as `null` placeholders.
  This keeps the schema stable when `--with-image-stats` is eventually implemented.

- **No numpy dependency:** numpy is not in `requirements-intake.txt`. Any optional image
  stat computation must use pure PIL (`Image.getdata()`, PIL `ImageFilter`). This constraint
  is factored into the `--with-image-stats` deferral above.

- **`generated_at` and `schema_version` added to summary** for consistency with
  `filter_universe_summary.json` — they do not appear in the canonical schema from the
  origin plan but are strictly additive and contain no PII.

---

## Open Questions

### Resolved During Planning

- **What file-size bucket thresholds to use?**
  Resolved: `tiny < 50 KB, small 50–150 KB, medium 150–400 KB, large ≥ 400 KB`.
  Calibrated from actual corpus: p25=161KB, p50=214KB, p95=393KB. Only 1 image exceeds
  983KB (max). The "large" bucket captures ~4% of images that are higher-quality or
  minimally Telegram-compressed.

- **Should image statistics be computed in Phase B?**
  Resolved: No. numpy is not available; PIL-only stats are slow (30+ min) for 32K images.
  The `--with-image-stats` flag is scaffolded in the CLI but computation is deferred.
  The schema includes null placeholder fields so the downstream schema remains stable.

- **Should any caption keyword signal use regex (e.g., word-boundary matching)?**
  Resolved: No. `str.lower() + keyword in text` substring matching only.
  Word-boundary edge cases (e.g., "малёк" vs "малёкам") are handled by including
  inflected forms explicitly in the frozenset.

- **Should a `caption_fish_species_keyword` signal be added?**
  Resolved: No. A positive fish-species mention in a caption does not change Phase C
  classification (fish_candidate + species_keyword = still fish, same result as no keyword).
  Adding a signal with zero classification impact would inflate the schema and summary.

- **aspect_class boundary between portrait / square / landscape?**
  Resolved: carry existing plan thresholds verbatim — established in the origin U4 plan and
  confirmed consistent with corpus distribution (p5 of aspect ratios = 0.75, p50 = 1.333).

### Deferred to Implementation

- **Exact keyword list expansion:** the starter sets below should be validated by inspecting
  a random 200-record sample of non-empty captions from `manifest.jsonl`. Add terms found
  in that sample that are specific enough to avoid high FPR.

- **`caption_text_heavy` threshold adjustment (currently 200 chars):** if Phase C override
  analysis shows `poster_screenshot` has a high false-positive rate, raising this threshold
  to 400 chars reduces the `caption_text_heavy=True` population from 15% to 8% of all images.

- **Progress interval:** 1,000 records (per existing scripts) or 5,000 for a 32K corpus.
  Implementer should choose based on observed performance.

---

## Output Structure

    data/intake_meta/tg_2026-04-24/
    ├── filter_universe.jsonl          ← Phase A output (input to this script)
    ├── filter_universe_summary.json   ← Phase A output (input reference)
    ├── filter_signals.jsonl           ← Phase B output — local-only, gitignored
    └── filter_signals_summary.json    ← Phase B output — tracked, aggregate counts only

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not
> implementation specification. The implementing agent should treat it as context, not
> code to reproduce.*

```
Inputs:
  filter_universe.jsonl ─────────────────────────────────┐
  manifest.jsonl  ────► dict[filename → {caption, ...}]  │
                                                          ▼
                                    intake_filter_heuristic.py
                                               │
                        ┌──────────────────────┴──────────────────────┐
                        │  For each universe record:                   │
                        │  1. Compute geometry signals                 │
                        │     aspect_ratio = width / height            │
                        │     aspect_class = classify(aspect_ratio)    │
                        │     file_size_bucket = classify(file_size)   │
                        │     low_res = carry from universe            │
                        │  2. Look up manifest by filename             │
                        │     has_manifest_record = filename in dict   │
                        │     caption = dict[filename].caption or ""   │
                        │     caption_length = len(caption)            │
                        │     caption_text_heavy = length > 200        │
                        │     caption_*_keyword = any(kw in lower)     │
                        │  3. Scaffold optional image stats (null)     │
                        └──────────────────────┬──────────────────────┘
                                               │
                         ┌─────────────────────┴────────────────────┐
                         ▼                                           ▼
               filter_signals.jsonl                  filter_signals_summary.json
               (local-only, 32420 records)           (tracked, counts only)
               one record per universe image         no filenames, no captions
```

Signal priority (enforced in Phase C, not here):

```
Manual review decisions      ← authoritative (Phase D/E)
        ↑
Local geometry signals       ← primary classification evidence
   (aspect_class, file_size_bucket, low_res)
        ↑
Caption keyword hints        ← supplemental weak hints only
   (caption_lure_keyword, caption_fry_keyword, etc.)
```

---

## Implementation Units

- U1. **Extend `intake_constants.py` with heuristic constants**

**Goal:** Add file-size bucket thresholds, `caption_text_heavy` threshold, aspect-class
boundaries, and four caption keyword hint frozensets to `intake_constants.py`. All
`intake_filter_heuristic.py` classification logic imports from here.

**Requirements:** R2, R6

**Dependencies:** None (extends already-updated constants file from Phase A)

**Files:**
- Modify: `scripts/intake_constants.py`

**Approach:**

*File-size bucket thresholds (calibrated from actual corpus):*
```
FILE_SIZE_BUCKET_TINY_MAX  =  50_000   # bytes; < 50 KB  → "tiny"  (0.4% of corpus)
FILE_SIZE_BUCKET_SMALL_MAX = 150_000   # bytes; < 150 KB → "small" (~25% of corpus)
FILE_SIZE_BUCKET_MEDIUM_MAX = 400_000  # bytes; < 400 KB → "medium" (~71% of corpus)
                                       # ≥ 400 KB → "large" (~4% of corpus)
```

*Caption text-heavy threshold:*
```
CAPTION_TEXT_HEAVY_THRESHOLD = 200  # chars; > 200 → caption_text_heavy=True
```

*Caption keyword hint frozensets (starter lists — implementer must expand from corpus):*

```python
# Lure/gear-specific product names only.
# EXCLUDE: резина, силикон, снасть, микроколебалки, наживка
# — they appear in fish-catch field reports, not gear-photo captions.
CAPTION_LURE_KEYWORD_HINTS: frozenset[str] = frozenset({
    "воблер",    # wobbler / crankbait
    "воблеры",
    "блесна",    # spoon lure
    "блёсны",
    "мормышка",  # jig / ice fishing lure
    "балансир",  # balance jig (ice fishing)
    "спиннер",   # spinner
    "раттлин",   # rattlin lure
    "раклин",    # rattlin variant
    "твистер",   # soft twister body (less risky than "резина")
    "топвотер",  # topwater
})

# Fish-processing / filleting terms.
# EXCLUDE: икра (roe — appears in lure descriptions too),
#          чистк (appears in prize captions: "сертификат на чистку улова")
CAPTION_FISH_PART_KEYWORD_HINTS: frozenset[str] = frozenset({
    "разделк",    # filleting/processing (substring: разделка, разделки)
    "потрош",     # gutting (substring: потрошить, потрошение)
    "хребет",     # backbone/spine
    "жабры",      # gills
    "филе",       # fillet
    "плавник",    # fin (use cautiously — may appear in species descriptions)
})

# Juvenile / fry fish terms.
# These are specific enough to have low FPR.
CAPTION_FRY_KEYWORD_HINTS: frozenset[str] = frozenset({
    "малёк",      # fry (singular)
    "мальки",     # fry (plural nominative)
    "мальков",    # fry (genitive)
    "малькам",    # fry (dative)
    "мальками",   # fry (instrumental)
    "молодь",     # juvenile (collective)
    "сеголеток",  # fish of the year / fingerling
    "молодняк",   # young stock
})

# Announcement / administrative content terms.
# EXCLUDE: конкурс, турнир, соревнование, чемпионат, кубок
# — these appear in fishing-competition captions WITH fish photos.
# EXCLUDE: приглашаем — appears in fish-stock-release announcements WITH fish photos.
CAPTION_NO_FISH_KEYWORD_HINTS: frozenset[str] = frozenset({
    "объявление",     # announcement / notice
    "расписание",     # schedule / timetable
    "записывайтесь",  # sign up / register
    "регистрация",    # registration
    "афиша",          # event poster / program
    "анонс",          # promo / announcement
})
```

**Patterns to follow:**
- `STAGE_A_CLASSES_SET: set[str] = set(STAGE_A_CLASSES)` constant pattern.
- All constants in the `# ─── U4 coarse-filter` block already present.

**Test scenarios:**
- Happy path: `from intake_constants import CAPTION_FRY_KEYWORD_HINTS` succeeds; result is
  a `frozenset`.
- Happy path: `FILE_SIZE_BUCKET_TINY_MAX < FILE_SIZE_BUCKET_SMALL_MAX < FILE_SIZE_BUCKET_MEDIUM_MAX`
  ordering invariant holds.
- Happy path: each keyword frozenset is non-empty.
- Edge case: no keyword appears in more than one frozenset (disjointness not strictly
  required but check for obviously problematic overlaps like "малёк" in lure set).
- Edge case: `CAPTION_TEXT_HEAVY_THRESHOLD` is a positive integer.

**Verification:**
- `python3 -c "from intake_constants import CAPTION_FRY_KEYWORD_HINTS, FILE_SIZE_BUCKET_TINY_MAX; print('ok')"` exits 0.

---

- U2. **`scripts/intake_filter_heuristic.py` + `tests/test_intake_filter_heuristic.py`**

**Goal:** For each of the 32,420 universe records, compute the full signal vector by
joining geometry fields (from `filter_universe.jsonl`) with caption keyword flags (from
`manifest.jsonl`). Write `filter_signals.jsonl` (local-only) and
`filter_signals_summary.json` (tracked aggregate counts only).

**Requirements:** R1, R2, R3, R4, R5, R6, R7, R8

**Dependencies:** U1 (constants must exist); Phase A (`filter_universe.jsonl` must exist)

**Files:**
- Create: `scripts/intake_filter_heuristic.py`
- Create: `tests/test_intake_filter_heuristic.py`

**Approach:**

*Data loading:*
- Load `filter_universe.jsonl` via `_read_jsonl` (copy verbatim from `intake_filter_universe.py`).
  Missing → `log.error` + `sys.exit(1)`.
- Load `manifest.jsonl` into `manifest_by_fn: dict[str, dict]` keyed by `filename`.
  Missing → `log.warning("manifest.jsonl not found — proceeding with has_manifest_record=False for all records")`.
  Do not exit. Do not abort.

*Per-record signal computation:*

```
# Geometry signals
width  = rec.get("width")   # already int from audit; None only if corrupted (shouldn't exist after Phase A)
height = rec.get("height")
if width and height and width > 0 and height > 0:
    aspect_ratio = round(width / height, 2)
    aspect_class = _classify_aspect(aspect_ratio)   # from C.ASPECT class boundaries (below)
else:
    aspect_ratio = None
    aspect_class = "unknown"

file_size_bucket = _classify_file_size(rec["file_size"])
low_res          = rec["low_res"]           # carry through
dedup_role       = rec["dedup_role"]        # carry through

# Caption signals
mfst = manifest_by_fn.get(rec["filename"])
has_manifest_record = mfst is not None
caption         = (mfst.get("caption") or "") if mfst else ""
caption_length  = len(caption)
caption_empty   = (caption_length == 0)
text_lower      = caption.lower()
caption_text_heavy         = caption_length > C.CAPTION_TEXT_HEAVY_THRESHOLD
caption_lure_keyword       = any(kw in text_lower for kw in C.CAPTION_LURE_KEYWORD_HINTS)
caption_fish_part_keyword  = any(kw in text_lower for kw in C.CAPTION_FISH_PART_KEYWORD_HINTS)
caption_fry_keyword        = any(kw in text_lower for kw in C.CAPTION_FRY_KEYWORD_HINTS)
caption_no_fish_keyword    = any(kw in text_lower for kw in C.CAPTION_NO_FISH_KEYWORD_HINTS)

# Optional image stats (scaffolded, not computed in Phase B)
image_stats_computed = False
mean_luminance   = None
is_grayscale_like = None
edge_density     = None
```

*Aspect classification helper:*
```
def _classify_aspect(ar: float) -> str:
    if ar < 0.5:   return "extreme_portrait"
    if ar < 0.8:   return "portrait"
    if ar < 1.25:  return "square"
    if ar < 2.0:   return "landscape"
    return "extreme_landscape"
```

*File-size classification helper:*
```
def _classify_file_size(file_size: int) -> str:
    if file_size < C.FILE_SIZE_BUCKET_TINY_MAX:   return "tiny"
    if file_size < C.FILE_SIZE_BUCKET_SMALL_MAX:  return "small"
    if file_size < C.FILE_SIZE_BUCKET_MEDIUM_MAX: return "medium"
    return "large"
```

*`filter_signals.jsonl` record schema (one record per universe image):*

```json
{
  "filename":                 "photos/photo_1@25-12-2017_19-47-37.jpg",
  "sha256":                   "04a78d21...",
  "width":                    1280,
  "height":                   960,
  "aspect_ratio":             1.33,
  "aspect_class":             "landscape",
  "file_size":                312310,
  "file_size_bucket":         "medium",
  "low_res":                  false,
  "dedup_role":               "unique",
  "has_manifest_record":      true,
  "caption_length":           45,
  "caption_empty":            false,
  "caption_text_heavy":       false,
  "caption_lure_keyword":     false,
  "caption_fish_part_keyword": false,
  "caption_fry_keyword":      false,
  "caption_no_fish_keyword":  false,
  "image_stats_computed":     false,
  "mean_luminance":           null,
  "is_grayscale_like":        null,
  "edge_density":             null,
  "source":                   "telegram_private_2026-04-24",
  "schema_version":           1
}
```

*`filter_signals_summary.json` schema (canonical — each count field is derived from the
corresponding boolean signal in `filter_signals.jsonl`; `license` is collection-level
and belongs in the summary only, consistent with `filter_universe_summary.json`):*

```json
{
  "total_images":                    32420,
  "no_manifest_record_count":        0,
  "low_res_count":                   371,
  "extreme_aspect_count":            281,
  "tiny_file_count":                 137,
  "caption_lure_keyword_count":      0,
  "caption_fish_part_keyword_count": 0,
  "caption_fry_keyword_count":       0,
  "caption_no_fish_keyword_count":   0,
  "caption_text_heavy_count":        0,
  "source":                          "telegram_private_2026-04-24",
  "license":                         "private_training_only",
  "generated_at":                    "2026-04-26T...",
  "schema_version":                  1
}
```

`extreme_aspect_count` = count where `aspect_class in {"extreme_portrait", "extreme_landscape"}`.
`tiny_file_count` = count where `file_size_bucket == "tiny"`.
All counts must be ≤ `total_images`. No filenames, no caption text, no sender names.

*CLI:*
- `--universe`: path to `filter_universe.jsonl` (default: `C.FILTER_UNIVERSE_PATH`)
- `--manifest`: path to `manifest.jsonl` (default: `C.MANIFEST_PATH`)
- `--output-dir`: output directory (default: `C.INTAKE_META_ROOT`)
- `--dry-run`: compute but do not write any files; print summary to stderr
- `--with-image-stats`: accept flag, print deprecation-style note "image stats not yet
  implemented", set `image_stats_computed=False` for all records (no-op in Phase B)

*Writes:*
- `filter_signals.jsonl`: `_write_jsonl(output_dir / "filter_signals.jsonl", signals)`
- `filter_signals_summary.json`: `_write_json_atomic(output_dir / "filter_signals_summary.json", summary)`

**Patterns to follow:**
- `intake_filter_universe.py` verbatim for: `_read_jsonl`, `_write_jsonl`, `_write_json_atomic`,
  `SCHEMA_VERSION = 1`, `PROGRESS_EVERY = 1000` logging, module docstring provenance tag,
  `from __future__ import annotations`, `sys.path.insert` + `import intake_constants as C`.
- `intake_telegram_dedup.py` for the `manifest_by_fn = {r["filename"]: r for r in manifest}` join.
- `_parse_args(argv)` / `main(argv=None)` / `if __name__ == "__main__": main()` entry pattern.

**Test scenarios:**

*Happy path — geometry signals:*
- `width=1280, height=960` → `aspect_ratio=1.33, aspect_class="landscape"`
- `width=960, height=1280` → `aspect_ratio=0.75, aspect_class="portrait"`
- `width=1000, height=200` → `aspect_ratio=5.0, aspect_class="extreme_landscape"`
- `width=200, height=1000` → `aspect_ratio=0.20, aspect_class="extreme_portrait"`
- `width=1000, height=1000` → `aspect_ratio=1.0, aspect_class="square"`

*Happy path — file_size_bucket:*
- `file_size=20_000` → `"tiny"` (< 50 KB)
- `file_size=80_000` → `"small"` (50–150 KB)
- `file_size=200_000` → `"medium"` (150–400 KB)
- `file_size=500_000` → `"large"` (≥ 400 KB)
- Boundary: `file_size=50_000` → `"small"` (not "tiny")
- Boundary: `file_size=150_000` → `"medium"` (not "small")
- Boundary: `file_size=400_000` → `"large"` (not "medium")

*Happy path — caption keyword signals:*
- Caption `"балансир со дна"` → `caption_lure_keyword=True`, all other signals False
- Caption `"БАЛАНСИР СО ДНА"` (uppercase) → `caption_lure_keyword=True` (case-insensitive)
- Caption `"мальки у берега"` → `caption_fry_keyword=True`
- Caption `"разделка улова"` → `caption_fish_part_keyword=True`
- Caption `"расписание на сезон"` → `caption_no_fish_keyword=True`
- Caption `""` (empty) → all keyword signals False, `caption_empty=True`
- Caption with 201 chars → `caption_text_heavy=True`; caption with 200 chars → `caption_text_heavy=False`

*Edge cases:*
- `width=0` or `height=0` → `aspect_ratio=None, aspect_class="unknown"`
- `width=None` or `height=None` → `aspect_ratio=None, aspect_class="unknown"`
- No manifest record for filename → `has_manifest_record=False`, `caption_length=0`,
  `caption_empty=True`, all four keyword signals `False`
- Caption contains both "балансир" and "мальки" → `caption_lure_keyword=True`
  AND `caption_fry_keyword=True` (both True simultaneously — conflict left for Phase C)
- Caption contains both "балансир" and "разделка" → both `caption_lure_keyword=True`
  AND `caption_fish_part_keyword=True`
- Universe record with `dedup_role="cluster_keep"` → carried through to signal record

*Error paths:*
- `filter_universe.jsonl` missing → `sys.exit(1)` with descriptive log message
- Corrupt JSONL line in universe → `ValueError` raised with `file:line` context
- `manifest.jsonl` missing → `log.warning`, continue; all records get `has_manifest_record=False`
- `--dry-run` → no files written to output directory; summary printed to stderr

*Privacy / summary integrity:*
- `filter_signals_summary.json` must not contain any filename string from the universe
- `filter_signals_summary.json` must not contain any caption text, sender names, or `file://` paths
- `filter_signals_summary.json` must not contain "caption", "sender", "from_id" as field values
- `filter_signals_summary.json` counts: `low_res_count == 371` (regression from known corpus)
- `filter_signals_summary.json` counts: `extreme_aspect_count == 281` (60 + 221, regression)
- `filter_signals_summary.json` counts: `tiny_file_count == 137` (regression)
- All summary counts ≤ `total_images`

*Integration / pipeline invariants:*
- Every filename in `filter_universe.jsonl` appears exactly once in `filter_signals.jsonl` (bijective)
- `len(filter_signals.jsonl records) == filter_signals_summary.total_images`
- `sum(1 for r in signals if r["low_res"]) == summary["low_res_count"]`
- `sum(1 for r in signals if r["aspect_class"] in {"extreme_portrait", "extreme_landscape"}) == summary["extreme_aspect_count"]`
- `sum(1 for r in signals if r["file_size_bucket"] == "tiny") == summary["tiny_file_count"]`
- `--dry-run` writes zero files to the output directory

*Real-data integration (skipif `filter_universe.jsonl` absent):*
- `summary["total_images"] == 32420`
- `summary["low_res_count"] == 371`
- `summary["extreme_aspect_count"] == 281`
- `summary["tiny_file_count"] == 137`
- `summary["consistency_ok"]` is not present (not a signal field — belongs in universe summary)

**Verification:**
- `python3 scripts/intake_filter_heuristic.py --dry-run` exits 0, prints summary.
- `python3 scripts/intake_filter_heuristic.py` exits 0; `filter_signals.jsonl` and
  `filter_signals_summary.json` are written.
- `pytest tests/test_intake_filter_heuristic.py -v` exits 0.
- `pytest tests/test_intake_filter_safety.py -v` exits 0 (no regression on gitignore tests).
- `filter_signals_summary.json` `total_images == 32420` when run against real data.

---

## System-Wide Impact

- **Interaction graph:** the script reads `filter_universe.jsonl` and `manifest.jsonl`
  (both local-only, neither modified). It writes two new files only. No existing tracked
  file is modified. No bot runtime, model, or service file is touched.
- **Error propagation:** missing universe → immediate exit 1; missing manifest → warning,
  continues. Corrupt JSONL → `ValueError` propagates to `main`, logged, exit 1. Partial
  writes cannot occur (atomic writes via `_write_json_atomic`).
- **State lifecycle risks:** `filter_signals.jsonl` does not carry a completion gate flag;
  Phase C (`intake_filter_classify.py`) must read from `filter_signals.jsonl` only after
  this script exits 0. A partial run that exits non-zero should leave no output files (dry
  run) or leave the previous complete run's files unchanged (re-run overwrites atomically).
- **API surface parity:** `filter_signals_summary.json` schema is consumed by Phase C's
  plausibility checks and by tracked git history. Schema must not change between Phase B
  runs without a `schema_version` bump.
- **Integration coverage:** `test_intake_filter_safety.py::test_filter_signals_jsonl_is_gitignored`
  and `test_filter_signals_summary_not_gitignored` already exist and will pass once the
  files exist on disk. No new gitignore rules are needed (already added in Phase A).
- **Unchanged invariants:** `filter_universe.jsonl`, `manifest.jsonl`, `audit.jsonl`,
  and all dedup artifacts are read-only throughout Phase B.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Caption keyword signals have high FPR for this corpus (fish caught ON a lure → caption names lure → classified as lure photo) | Signals are weak hints only; classify waterfall requires visual anchor; review phase (U5) and override-rate gate (U6) catch systematic misclassification |
| `"твистер"` or `"балансир"` appears in a fish-catch caption (confirmed in corpus sample) | Accepted — these signal True even in catch reports; Phase C classifier routes caption-refined categories to review rather than auto-staging |
| `caption_text_heavy` threshold = 200 chars flags 15% of all images (many legitimate field reports) | Combined with extreme_aspect in Phase C; text_heavy alone never assigns poster_screenshot |
| `filter_signals.jsonl` accidentally tracked (gitignore rule missing or too narrow) | `test_intake_filter_safety.py::test_filter_signals_jsonl_is_gitignored` asserts this; rule already in `.gitignore` from Phase A |
| numpy absent from requirements-intake.txt blocks image stats implementation | Deferred to `--with-image-stats` flag; null placeholders in schema keep downstream stable |
| Manifest record count (32,664) exceeds universe count (32,420); some manifest filenames not in universe | Correct — manifest includes dedup non-keeps; heuristic script iterates universe, looks up manifest. Extra manifest records are silently ignored |
| Re-run after partial failure leaves stale `filter_signals.jsonl` | Atomic writes via `_write_json_atomic` ensure the complete file or the previous file; incomplete writes cannot persist |

---

## Documentation / Operational Notes

- **Running order:** `intake_filter_heuristic.py` must run after `intake_filter_universe.py`
  (Phase A) and before `intake_filter_classify.py` (Phase C).
- **Runtime estimate:** metadata-only path (default): < 2 min for 32K records.
  `--with-image-stats` path (not yet implemented): ~30 min estimated.
- **Re-run safety:** idempotent; any re-run overwrites outputs atomically. Safe to re-run
  if keyword lists are updated between Phase B and Phase C.
- **Keyword list calibration note:** implementer must inspect a random 200-record sample
  of non-empty captions from `manifest.jsonl` before finalizing the starter keyword lists
  in U1. Any term that appears in > 50% of matching captions in a fish-photo context
  should be removed from the hint set.
- **Phase C interface contract:** `intake_filter_classify.py` will read `filter_signals.jsonl`
  and expect all fields listed in the record schema above. If any field is renamed or
  dropped, the classify script must be updated in the same commit.

---

## Acceptance Criteria

1. `python3 scripts/intake_filter_heuristic.py --dry-run` exits 0; no files written.
2. `python3 scripts/intake_filter_heuristic.py` exits 0 against real data.
3. `filter_signals.jsonl` exists locally with exactly 32,420 records.
4. `filter_signals_summary.json` exists and is tracked (git-add ready):
   - `total_images == 32420`
   - `low_res_count == 371`
   - `extreme_aspect_count == 281`
   - `tiny_file_count == 137`
   - No field contains a filename, caption text, or sender name.
5. `pytest tests/test_intake_filter_heuristic.py -v` exits 0 (all unit + integration tests pass).
6. `pytest tests/test_intake_filter_safety.py -v` exits 0 (no regression on gitignore assertions).
7. `git check-ignore -v data/intake_meta/tg_2026-04-24/filter_signals.jsonl` returns 0.
8. `git check-ignore -v data/intake_meta/tg_2026-04-24/filter_signals_summary.json` returns 1.

---

## Implementation Command Sequence

```bash
# 1. Extend intake_constants.py with U1 constants
#    (no separate command — edit the file)

# 2. Implement the script
#    (no separate command — write the file)

# 3. Smoke test with dry-run (verify no crashes, print summary)
python3 scripts/intake_filter_heuristic.py --dry-run

# 4. Run for real
python3 scripts/intake_filter_heuristic.py

# 5. Inspect output summary
python3 -m json.tool data/intake_meta/tg_2026-04-24/filter_signals_summary.json

# 6. Verify record count
python3 -c "
import json
from pathlib import Path
records = [json.loads(l) for l in Path('data/intake_meta/tg_2026-04-24/filter_signals.jsonl').read_text().splitlines() if l.strip()]
print(f'Records: {len(records)}  (expected 32420)')
"

# 7. Run unit tests
pytest tests/test_intake_filter_heuristic.py -v

# 8. Run safety tests (no regression)
pytest tests/test_intake_filter_safety.py -v

# 9. Stage only tracked outputs + scripts
git add \
  scripts/intake_constants.py \
  scripts/intake_filter_heuristic.py \
  tests/test_intake_filter_heuristic.py \
  data/intake_meta/tg_2026-04-24/filter_signals_summary.json

# 10. Commit (after code review)
```

---

## Sources & References

- **Origin document:** [docs/plans/2026-04-26-001-feat-intake-coarse-filter-u4-plan.md](docs/plans/2026-04-26-001-feat-intake-coarse-filter-u4-plan.md) (U3 of that plan)
- Related code: `scripts/intake_constants.py`, `scripts/intake_filter_universe.py`
- Related tests: `tests/test_intake_filter_safety.py`, `tests/test_intake_filter_universe.py`
- Corpus calibration data computed: 2026-04-26 from `data/intake_meta/tg_2026-04-24/filter_universe.jsonl` (32,420 records)
