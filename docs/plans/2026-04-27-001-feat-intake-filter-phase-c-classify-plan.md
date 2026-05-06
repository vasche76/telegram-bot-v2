---
title: "feat: Intake filter Phase C — deterministic candidate classification"
type: feat
status: completed
date: 2026-04-27
---

# feat: Intake filter Phase C — deterministic candidate classification

## Overview

Phase C consumes the per-image signal vectors produced by Phase B (`filter_signals.jsonl`) and assigns each of the 32,420 unique images a **coarse candidate category** using a conservative, deterministic rule waterfall. No visual model inference is performed. No image is promoted to training staging. All output is candidate-only; Phase D manual review is authoritative.

---

## Problem Frame

U4 Phase B produced a 20-field signal vector per image but made no category assignments. Phase C must turn those signals into structured candidate labels that Phase D reviewers can act on efficiently. The corpus is a private Russian-language fishing-club Telegram channel. The base rate is overwhelmingly fish-photo content (~95%+ of posts are catch reports), which means:

- Caption lure keywords are high-FPR because fishers routinely describe the lure used in a successful catch.
- Most images with no disqualifying signal are *probably* fish photos — but "probably" is not enough for Phase C to assert `fish` without visual confirmation.
- The minority signal categories (poster/screenshot, gear, fish parts, juveniles) are small but detectable through specific keyword combinations.

Phase C deliberately errs toward `unknown_needs_review`. Mislabelling a catch-report photo as `lure_gear` is a worse mistake than leaving it for Phase D review.

---

## Requirements Trace

- R1. Every image in `filter_signals.jsonl` receives exactly one `candidate_category` from the canonical `COARSE_CATEGORIES` list.
- R2. `candidate_category=fish` is **not** assigned in Phase C (no visual/manual signal exists yet).
- R3. Caption-only evidence produces at most `confidence=low`; it must never produce `confidence=high`.
- R4. Any caption–geometry conflict must route to `unknown_needs_review` regardless of other signals.
- R5. `review_required=True` must be set for every record with any conflict flag, quality flag, or low/medium confidence (i.e., for every record in Phase C).
- R6. `filter_candidates_summary.json` must contain only aggregate counts — no filenames, captions, sender metadata, or per-image details.
- R7. `filter_candidates.jsonl` (filename-level) remains local-only and gitignored.
- R8. The script must be deterministic (same input → identical output) and side-effect-free on raw photos.
- R9. A `--dry-run` flag must produce logs and a summary without writing output files.

---

## Scope Boundaries

- Assigning `fish` as a candidate category is explicitly out of scope — deferred to Phase D visual/manual review.
- No visual model inference (no PIL pixel analysis beyond what Phase B already computed, no cloud APIs, no OCR).
- No modification of raw photo files.
- No changes to `filter_signals.jsonl`, `filter_universe.jsonl`, or any earlier pipeline artifact.
- `lure_gear` candidate is not assigned from caption alone due to corpus-specific high FPR; see Key Technical Decisions.
- `no_fish` candidate is not assigned from caption alone in Phase C; the `caption_no_fish_keyword` signal routes to `poster_screenshot` only when combined with `caption_text_heavy`.

### Deferred to Follow-Up Work

- Phase D: Manual reviewer tooling consuming `filter_candidates.jsonl` and producing `filter_review_summary.json`.
- Reassignment of `fish` and `no_fish` as confirmed categories (Phase D output, not Phase C).
- Any image-stats-based rules (`mean_luminance`, `edge_density`, `is_grayscale_like`) — these fields are null for all Phase B records and could enable richer lure_gear/poster detection when populated.

---

## Context & Research

### Relevant Code and Patterns

- `scripts/intake_filter_heuristic.py` — canonical Phase B pattern: read JSONL, compute per-record signals, write atomically via temp file, produce a privacy-safe summary. Follow exactly.
- `scripts/intake_constants.py` — centralized paths (`FILTER_CANDIDATES_PATH`, `FILTER_CANDIDATES_SUMMARY_PATH`), `COARSE_CATEGORIES`, keyword frozensets, file-size bucket constants. All Phase C rules must reference these constants, not magic literals.
- `scripts/intake_filter_universe.py` — `_read_jsonl`, `_write_jsonl`, `_write_json_atomic` helpers (also in `intake_filter_heuristic.py`) — copy verbatim.
- `scripts/intake_review_finalize.py` — example of a pipeline termination script with acceptance-criteria checks; useful reference for Phase C's acceptance gate.

### Corpus-grounding facts (from actual data)

| Signal | Count | % of 32,420 |
|---|---|---|
| `low_res=True` | 371 | 1.1% |
| `aspect_class` extreme | 281 | 0.9% |
| `file_size_bucket=tiny` | 137 | 0.4% |
| `caption_lure_keyword=True` | 846 | 2.6% |
| `caption_text_heavy=True` | 4,913 | 15.2% |
| `caption_text_heavy + caption_lure_keyword` | 778 | 2.4% |
| `caption_fish_part_keyword=True` | 19 | 0.06% |
| `caption_fry_keyword=True` | 5 | 0.015% |
| `caption_no_fish_keyword=True` | 3 | 0.009% |
| `caption_no_fish_keyword + caption_text_heavy` | 3 | 0.009% (all 3 overlap) |

The three `caption_no_fish_keyword` records all also have `caption_text_heavy=True`, making the combined rule safe. The 778 text-heavy+lure records have normal geometry (landscape/square) — consistent with detailed catch reports, not with gear-only posts.

### Institutional Learnings

- Phase B keyword frozensets were deliberately conservative: high-FPR terms (`резина`, `силикон`, `конкурс`) were excluded. Phase C rules inherit this conservatism.
- Phase B image stats (`mean_luminance`, `edge_density`, `is_grayscale_like`) are scaffolded but null for all records. Phase C rules must not reference them.
- `dedup_role` in the universe is always `"unique"` for all 32,420 records (duplicates were removed in Phase A/B). This field is passed through to candidates for provenance but plays no classification role.

---

## Key Technical Decisions

- **`fish` not assigned in Phase C**: The corpus base rate is high (~95%), but "probably fish" is not a classification. Without visual model inference or manual confirmation, assigning `fish` would create a false sense of certainty and risk polluting the training staging gate. Phase D assigns `fish` after visual review.

- **`lure_gear` not assigned from caption alone**: `caption_lure_keyword` has ~2.6% occurrence in a channel where fishers routinely mention the lure used in a catch (`"поймал щуку на воблер"`). False positive rate is unacceptably high for a label that means "this image shows gear, not fish." All lure-keyword records receive a `caption_lure_hint` reason tag in `unknown_needs_review`, surfacing them for Phase D without asserting a label.

- **`poster_screenshot` assigned at medium confidence (not high)**: `caption_no_fish_keyword + caption_text_heavy` is the strongest available non-visual signal (3 records, all no-fish keyword + heavy caption). Medium confidence is appropriate — the image could theoretically be a fish photo with a long event-announcement caption accidentally triggering both signals. `review_required=True` is mandatory.

- **Conflict detection precedes category assignment**: Conflicts are computed before the category waterfall runs. A conflicted record routes directly to `unknown_needs_review` regardless of which other rules might match.

- **`review_required=True` for every Phase C record**: Phase C has no visual model and no manual confirmation. Every record needs Phase D review; making this universal prevents accidental staging of Phase C output.

- **`confidence=high` is never assigned in Phase C**: Highest reachable confidence is `medium` (poster_screenshot with two strong converging signals). All other categories are `low`.

- **Atomic writes via temp file + `os.replace()`**: Follows the established pattern in `intake_filter_heuristic.py`. Prevents partial JSONL output on crash.

- **Schema version 1** for `filter_candidates.jsonl`: New output artifact type; starts at 1 independent of other schema versions.

---

## Open Questions

### Resolved During Planning

- *Should `no_fish` be assigned when `caption_no_fish_keyword=True`?*
  No — the `no_fish_keyword` signal (only 3 records) routes to `poster_screenshot` when combined with `caption_text_heavy`. Without `caption_text_heavy`, a single no-fish keyword in an otherwise normal photo is ambiguous (e.g., the word "расписание" could appear in a caption that also describes a fish). Routes to `unknown_needs_review`.

- *Should extreme aspect alone be sufficient for any category?*
  No. `extreme_landscape` or `extreme_portrait` alone adds a reason tag but does not determine category. The combination of `extreme_landscape + caption_text_heavy + caption_no_fish_keyword` would route to `poster_screenshot` via Rule 1 (since no_fish_keyword implies text_heavy already). Geometry alone is insufficient.

- *How many reasons can a record have?*
  Multiple — reasons is a list. A record can simultaneously be low_res, extreme_aspect, AND have a lure_hint. All reasons are surfaced.

### Deferred to Implementation

- Whether the `conflicts` field should be an empty list `[]` vs. omitted when no conflicts exist — implement either, keep it consistent.
- Exact progress logging cadence (likely `PROGRESS_EVERY = 1000` matching Phase B).
- Whether to add a `--signals` CLI argument override or always use `C.FILTER_SIGNALS_PATH` as default (follow Phase B's `--universe` pattern).

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

**Rule waterfall (priority-ordered, first match wins):**

```
INPUT: one filter_signals record

STEP 1 — Collect active signals
  active_caption_keywords = count of {lure_keyword, fish_part_keyword, fry_keyword, no_fish_keyword} that are True

STEP 2 — Detect conflicts
  CONFLICT_A: caption_text_heavy=True AND caption_lure_keyword=True AND aspect_class NOT extreme
              → "ambiguous_text_heavy_lure" (catch report vs. gear post unresolvable)
  CONFLICT_B: active_caption_keywords >= 2
              → "competing_caption_keywords"
  IF any conflict → category=unknown_needs_review, confidence=low, conflicts=[...], goto STEP 5

STEP 3 — Category waterfall (no conflict)
  RULE 1: caption_no_fish_keyword=True AND caption_text_heavy=True
          → poster_screenshot, confidence=medium
          reasons=[caption_no_fish_keyword, caption_text_heavy]

  RULE 2: caption_fish_part_keyword=True
          → fish_part, confidence=low
          reasons=[caption_fish_part_keyword]

  RULE 3: caption_fry_keyword=True
          → fry_juvenile, confidence=low
          reasons=[caption_fry_keyword]

  RULE 4 (default): → unknown_needs_review, confidence=low
          reasons=[no_strong_signal]  (or other reason tags from STEP 4)

  NOTE: caption_lure_keyword=True without conflict adds "caption_lure_hint" to reasons
        but does NOT assign lure_gear. Stays in unknown_needs_review.

STEP 4 — Append quality/format reason tags (to any category)
  IF low_res=True           → append "low_res" to reasons
  IF file_size_bucket=tiny  → append "tiny_file" to reasons
  IF aspect_class IN (extreme_portrait, extreme_landscape)
                            → append "extreme_aspect" to reasons

STEP 5 — Finalize
  review_required = True  (always)
  emit candidate record
```

**Expected distribution (approximate, data-grounded):**

| candidate_category | confidence | approx. count |
|---|---|---|
| `poster_screenshot` | medium | ~3 |
| `fish_part` | low | ~17–19 |
| `fry_juvenile` | low | ~4–5 |
| `unknown_needs_review` | low | ~32,390+ |
| `fish` | — | 0 (not assigned) |
| `lure_gear` | — | 0 (not assigned) |
| `no_fish` | — | 0 (not assigned) |

All `review_required=True`. `conflict_flag_count` expected ~778+ (text_heavy+lure ambiguity).

---

## Implementation Units

- U1. **Extend `intake_constants.py` with Phase C rule constants**

**Goal:** Add the confidence level constants, reason code constants, and conflict code constants that the Phase C classify script will reference. Paths (`FILTER_CANDIDATES_PATH`, `FILTER_CANDIDATES_SUMMARY_PATH`) are already present.

**Requirements:** R1, R3, R5, R8

**Dependencies:** None

**Files:**
- Modify: `scripts/intake_constants.py`

**Approach:**
- Add a `# ─── U4 Phase C candidate classification constants ─────────────────────────────` section
- Confidence levels as string constants: `CONFIDENCE_HIGH`, `CONFIDENCE_MEDIUM`, `CONFIDENCE_LOW`
- Reason codes as string constants (one per named reason): `REASON_CAPTION_NO_FISH_KW`, `REASON_CAPTION_TEXT_HEAVY`, `REASON_CAPTION_LURE_HINT`, `REASON_CAPTION_FISH_PART_KW`, `REASON_CAPTION_FRY_KW`, `REASON_LOW_RES`, `REASON_TINY_FILE`, `REASON_EXTREME_ASPECT`, `REASON_NO_STRONG_SIGNAL`
- Conflict codes as string constants: `CONFLICT_AMBIGUOUS_TEXT_HEAVY_LURE`, `CONFLICT_COMPETING_CAPTION_KEYWORDS`
- Do not add logic — constants only

**Patterns to follow:**
- Existing constant blocks in `scripts/intake_constants.py` (frozenset definitions, file-size thresholds, `COARSE_CATEGORIES` list)

**Test scenarios:**
- Test expectation: none — constants only, no logic to test. Integration coverage provided by U2 tests that import and use them.

**Verification:**
- `python3 -c "import scripts.intake_constants as C; print(C.CONFIDENCE_MEDIUM, C.REASON_LOW_RES)"` executes without error.
- All new constant names are referenced in the Phase C classify script (U2) with no string literals for reason/conflict codes.

---

- U2. **Implement `scripts/intake_filter_classify.py`**

**Goal:** The Phase C classify script. Reads `filter_signals.jsonl`, applies the deterministic rule waterfall, writes `filter_candidates.jsonl` (local-only) and `filter_candidates_summary.json` (tracked). Follows the Phase B script structure exactly.

**Requirements:** R1, R2, R3, R4, R5, R6, R7, R8, R9

**Dependencies:** U1

**Files:**
- Create: `scripts/intake_filter_classify.py`
- Create: `data/intake_meta/tg_2026-04-24/filter_candidates.jsonl` (local-only, gitignored — written at runtime)
- Create: `data/intake_meta/tg_2026-04-24/filter_candidates_summary.json` (tracked — written at runtime)

**Approach:**

*Script structure (mirror `intake_filter_heuristic.py`):*
- Module docstring explaining Phase C purpose, CANDIDATE-ONLY warning, inputs/outputs
- Same `_read_jsonl`, `_write_jsonl`, `_write_json_atomic` helpers (copy verbatim)
- One `classify_record(rec: dict) -> dict` pure function that implements the waterfall and returns a candidate dict — this is the unit-testable core
- One `classify_all(signals_path, output_dir, dry_run)` orchestration function
- `_parse_args` and `main` following Phase B CLI pattern

*`classify_record` logic (see High-Level Technical Design):*
1. Collect which caption keywords are active
2. Detect conflicts (CONFLICT_A: text_heavy + lure + non-extreme aspect; CONFLICT_B: 2+ caption keywords)
3. If conflict → `unknown_needs_review`, append conflict codes
4. If no conflict → apply RULE 1 (poster_screenshot), RULE 2 (fish_part), RULE 3 (fry_juvenile), RULE 4 (default unknown)
5. Append quality/format reason tags for `low_res`, `tiny_file`, `extreme_aspect` to any record
6. Append `REASON_CAPTION_LURE_HINT` to reasons when `caption_lure_keyword=True` and no conflict already named it
7. Set `review_required=True`
8. Return candidate dict (see schema below)

*Candidate JSONL record schema:*
```
{
  "filename": str,           # passes through from signals record
  "sha256": str,             # passes through from signals record
  "candidate_category": str, # one of COARSE_CATEGORIES (never "fish", "lure_gear", "no_fish" in Phase C)
  "confidence": str,         # "medium" | "low" (never "high" in Phase C)
  "review_required": bool,   # always True in Phase C
  "reasons": [str],          # ordered list of REASON_* constants; non-empty
  "conflicts": [str],        # ordered list of CONFLICT_* constants; [] when no conflict
  "source": str,             # C.SOURCE_TAG pass-through
  "schema_version": int      # 1
}
```

*Summary JSON schema (tracked):*
```
{
  "total_images": int,
  "by_candidate_category": {
    "fish": int,
    "no_fish": int,
    "lure_fishing_gear": int,
    "fish_part": int,
    "fry_juvenile": int,
    "poster_screenshot": int,
    "unknown_needs_review": int
  },
  "by_confidence": {
    "high": int,
    "medium": int,
    "low": int
  },
  "review_required_count": int,
  "conflict_flag_count": int,
  "by_reason": {                        # count of records carrying each reason tag
    "<reason_code>": int,
    ...
  },
  "source": str,
  "license": str,
  "generated_at": str,                  # ISO 8601 UTC
  "schema_version": int                 # 1
}
```

*CLI flags (mirror Phase B):*
- `--signals PATH` (default `C.FILTER_SIGNALS_PATH`)
- `--output-dir PATH` (default `C.INTAKE_META_ROOT`)
- `--dry-run` (compute but do not write files; print summary to stderr)

**Patterns to follow:**
- `scripts/intake_filter_heuristic.py` — script structure, I/O helpers, atomic write, progress logging, CLI shape
- `scripts/intake_constants.py` — all constant references, no magic string literals for category names or reason codes

**Test scenarios:**
- Happy path — poster_screenshot: record with `caption_no_fish_keyword=True, caption_text_heavy=True`, normal geometry → `candidate_category=poster_screenshot`, `confidence=medium`, `review_required=True`, reasons contains both signal tags, conflicts is `[]`
- Happy path — fish_part: record with `caption_fish_part_keyword=True`, no other caption keywords → `candidate_category=fish_part`, `confidence=low`, reasons contains `REASON_CAPTION_FISH_PART_KW`
- Happy path — fry_juvenile: record with `caption_fry_keyword=True`, no other caption keywords → `candidate_category=fry_juvenile`, `confidence=low`, reasons contains `REASON_CAPTION_FRY_KW`
- Happy path — default unknown (empty caption): record with all caption booleans False, normal geometry, not low_res, not tiny → `candidate_category=unknown_needs_review`, reasons contains `REASON_NO_STRONG_SIGNAL`
- Conflict A — text_heavy + lure + normal geometry: `caption_text_heavy=True, caption_lure_keyword=True, aspect_class=landscape` → `candidate_category=unknown_needs_review`, conflicts contains `CONFLICT_AMBIGUOUS_TEXT_HEAVY_LURE`
- Conflict A does NOT fire for extreme aspect: `caption_text_heavy=True, caption_lure_keyword=True, aspect_class=extreme_landscape` → no CONFLICT_A (extreme aspect exemption); routes via waterfall; reasons include `extreme_aspect`
- Conflict B — two caption keywords: `caption_fish_part_keyword=True, caption_fry_keyword=True` → `candidate_category=unknown_needs_review`, conflicts contains `CONFLICT_COMPETING_CAPTION_KEYWORDS`
- Conflict B — lure + fish_part: `caption_lure_keyword=True, caption_fish_part_keyword=True` → conflict, not fish_part
- Quality flags: `low_res=True, file_size_bucket=tiny` → reasons contains `REASON_LOW_RES` AND `REASON_TINY_FILE` regardless of primary category
- Quality flags don't override category: a fish_part record that is also low_res → `candidate_category=fish_part` with `REASON_CAPTION_FISH_PART_KW, REASON_LOW_RES` in reasons
- Lure hint (no conflict): `caption_lure_keyword=True, caption_text_heavy=False, aspect_class=landscape` → `candidate_category=unknown_needs_review`, reasons contains `REASON_CAPTION_LURE_HINT` (NOT `lure_gear`)
- Extreme aspect alone: `aspect_class=extreme_portrait`, no caption signals → `unknown_needs_review`, reasons contains `REASON_EXTREME_ASPECT`
- Poster + quality: `caption_no_fish_keyword=True, caption_text_heavy=True, low_res=True` → `poster_screenshot/medium` with reasons containing poster tags AND `REASON_LOW_RES`
- review_required: every record must have `review_required=True` (parametric test over all above scenarios)
- fish not assigned: no combination of inputs produces `candidate_category=fish` (invariant test)
- high confidence not assigned: no combination of inputs produces `confidence=high` (invariant test)
- Summary total: `sum(by_candidate_category.values()) == total_images` for any valid input
- Summary review_required_count: equals `total_images` (since always True in Phase C)
- Dry-run: no files written to disk when `--dry-run` flag is set

**Verification:**
- `python3 scripts/intake_filter_classify.py --dry-run` completes with `total_images=32420` in summary output
- `by_candidate_category.unknown_needs_review` is the dominant count (>32,300)
- `by_candidate_category.fish == 0`, `by_candidate_category.lure_fishing_gear == 0`, `by_candidate_category.no_fish == 0`
- `review_required_count == 32420`
- `conflict_flag_count >= 778` (text_heavy+lure ambiguity records)
- `by_confidence.high == 0`
- `filter_candidates.jsonl` has exactly 32,420 lines
- `filter_candidates_summary.json` contains no filenames, captions, or sender metadata

---

- U3. **Tests for `intake_filter_classify.py`**

**Goal:** Unit and integration tests covering the rule waterfall, conflict detection, quality flag accumulation, invariants (fish never assigned, high confidence never assigned), and summary accuracy.

**Requirements:** R1, R2, R3, R4, R5, R8

**Dependencies:** U1, U2

**Files:**
- Create: `tests/test_intake_filter_classify.py`

**Approach:**
- Import `classify_record` directly from `scripts.intake_filter_classify` (the function is the only testable unit; I/O functions are covered by integration)
- Use `pytest` parametrize for invariant tests (fish not assigned, high confidence not assigned) across a representative set of signal combinations
- Build minimal fixture records as dicts (only the fields consumed by `classify_record`: `aspect_class`, `file_size_bucket`, `low_res`, `caption_lure_keyword`, `caption_fish_part_keyword`, `caption_fry_keyword`, `caption_no_fish_keyword`, `caption_text_heavy`, `filename`, `sha256`, `source`)
- Follow the existing test file pattern in `tests/` if one exists; otherwise use flat pytest functions

**Patterns to follow:**
- Other `tests/test_intake_*.py` files if present; otherwise mirror `intake_filter_heuristic.py` test coverage style

**Test scenarios:**
*(All scenarios listed in U2 are the implementation targets; the test file implements them as pytest functions/parametrized cases)*
- One test function per rule (poster_screenshot, fish_part, fry_juvenile, default_unknown, conflict_A, conflict_B variants)
- One parametrized invariant test: `@pytest.mark.parametrize` over 20 representative signal combos asserting `candidate_category != "fish"` and `confidence != "high"` and `review_required == True`
- One test for summary accuracy: feed a small synthetic list through classify_all with a temp output dir, assert `sum(by_candidate_category.values()) == len(input)` and `review_required_count == len(input)`

**Verification:**
- `python3 -m pytest tests/test_intake_filter_classify.py -v` passes with 0 failures
- Coverage includes at least one test for each of the 7 named conflict/category rules

---

## System-Wide Impact

- **Interaction graph:** Phase C reads Phase B output only. It does not modify `filter_signals.jsonl`, `filter_universe.jsonl`, manifest, dedup artifacts, or any bot runtime file.
- **Error propagation:** Missing `filter_signals.jsonl` should cause `sys.exit(1)` with a clear error (mirrors Phase B behavior). Corrupt JSONL lines raise `ValueError` with line number context.
- **State lifecycle risks:** Partial-write risk is eliminated by atomic temp-file + `os.replace()` pattern. Re-running the script overwrites previous output safely.
- **API surface parity:** `filter_candidates_summary.json` must be tracked. `filter_candidates.jsonl` must NOT be tracked. Verify `.gitignore` covers `filter_candidates.jsonl` before running (it should already be covered by the blanket `data/intake/` rule — but confirm `data/intake_meta/*/filter_candidates.jsonl` is also gitignored).
- **Integration coverage:** Run `--dry-run` against the actual 32,420-record corpus before committing the script. The summary distribution must match the expected ranges in U2 Verification. A dry-run pass with implausible numbers (e.g., `poster_screenshot > 100`) indicates a rule logic error.
- **Unchanged invariants:** Phase A dedup results, Phase B signal vectors, manifest, and all bot runtime files are untouched by Phase C.

---

## Risks & Dependencies

| Risk | Mitigation |
|---|---|
| Lure keyword FPR causes unexpected category assignment | Rule waterfall explicitly routes `caption_lure_keyword` to `REASON_CAPTION_LURE_HINT` in unknown, never to `lure_gear`. Invariant test asserts `lure_gear==0`. |
| `poster_screenshot` rule fires too broadly (high false positive) | Rule requires BOTH `caption_no_fish_keyword=True` AND `caption_text_heavy=True`. Only 3 records in corpus. Review those 3 manually as part of acceptance. |
| `fish` accidentally assigned (e.g., future rule addition) | Invariant test in U3 parametrizes over many signal combos asserting `fish` is never returned. Must remain in test suite permanently. |
| Summary leaks PII | Summary schema reviewed in plan; implementation must not include filename, caption text, or any per-record field. Code review should grep for `filename` in summary construction. |
| `.gitignore` gap — `filter_candidates.jsonl` accidentally committed | Verify gitignore before `git add` at commit time. `filter_candidates_summary.json` must be committed; `filter_candidates.jsonl` must not. |
| Script run on wrong batch/directory | `--signals` and `--output-dir` CLI flags allow explicit path override. Default paths from `intake_constants.py` are batch-specific (`tg_2026-04-24`). |
| `image_stats_computed=False` for all records limits rule richness | Accepted and documented. Phase C is explicitly limited to geometry + caption signals. Image-stats rules deferred to a future phase if stats are populated. |

---

## Documentation / Operational Notes

- `filter_candidates_summary.json` should be committed with the batch, documenting the Phase C candidate distribution for audit.
- Do NOT commit `filter_candidates.jsonl` — it contains filenames which may be correlatable with Telegram metadata.
- Privacy check before commit: `grep -r "caption\|sender\|filename" data/intake_meta/tg_2026-04-24/filter_candidates_summary.json` must return empty.
- After Phase C runs, the gitignore for `filter_candidates.jsonl` should be confirmed: `git status` must not show it as a staged or untracked tracked file.

---

## Acceptance Criteria (Phase C Complete)

1. `python3 -m pytest tests/test_intake_filter_classify.py -v` → all pass
2. `python3 scripts/intake_filter_classify.py --dry-run` → exits 0, summary shows `total_images=32420`
3. Full run: `filter_candidates.jsonl` has exactly 32,420 lines, each valid JSON with all required fields
4. `by_candidate_category.fish == 0`, `.lure_fishing_gear == 0`, `.no_fish == 0`
5. `by_confidence.high == 0`
6. `review_required_count == 32420`
7. `conflict_flag_count >= 778` (text_heavy+lure baseline)
8. `by_candidate_category.unknown_needs_review > 32300` (dominant category)
9. `by_candidate_category.poster_screenshot <= 10` (strict — if higher, investigate rule logic)
10. `filter_candidates_summary.json` is committed; `filter_candidates.jsonl` is not

---

## Future Implementation Command Sequence

```
# 1. Extend constants (U1)
#    Edit scripts/intake_constants.py — add Phase C section

# 2. Implement classify script (U2)
#    Create scripts/intake_filter_classify.py

# 3. Write tests (U3)
#    Create tests/test_intake_filter_classify.py

# 4. Run tests
python3 -m pytest tests/test_intake_filter_classify.py -v

# 5. Dry-run against real corpus
python3 scripts/intake_filter_classify.py --dry-run

# 6. Full run
python3 scripts/intake_filter_classify.py

# 7. Verify output
wc -l data/intake_meta/tg_2026-04-24/filter_candidates.jsonl
cat data/intake_meta/tg_2026-04-24/filter_candidates_summary.json

# 8. Privacy check
grep -i "caption\|sender\|filename" data/intake_meta/tg_2026-04-24/filter_candidates_summary.json

# 9. Git status check — confirm filter_candidates.jsonl is NOT listed
git status

# 10. Stage and commit summary only
git add data/intake_meta/tg_2026-04-24/filter_candidates_summary.json
git add scripts/intake_constants.py scripts/intake_filter_classify.py tests/test_intake_filter_classify.py
```

---

## Sources & References

- Related code: `scripts/intake_filter_heuristic.py` (Phase B canonical pattern)
- Related code: `scripts/intake_constants.py` (constants, paths, keyword frozensets)
- Related plans: `docs/plans/2026-04-26-002-feat-intake-filter-phase-b-heuristic-plan.md`
- Input data: `data/intake_meta/tg_2026-04-24/filter_signals.jsonl` (32,420 records, schema_version=1)
- Input data: `data/intake_meta/tg_2026-04-24/filter_signals_summary.json`
