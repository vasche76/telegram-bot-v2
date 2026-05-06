---
title: "feat: Second active-learning negative-focused candidate selection (pass 2)"
type: feat
status: active
date: 2026-05-05
---

# Second Active-Learning Negative-Focused Candidate Selection (Pass 2)

## Overview

AL run 1 (`alvrun_20260501T194300Z`) reviewed 497 records but yielded only 110 new
Telegram-domain negatives (22.1% yield) because 75.3% of selected candidates turned
out to be fish. This pass redesigns the selection bucket strategy for a second,
more negative-focused batch (target 250 records) to close the remaining gap of 58
negatives before structural v2 retraining.

---

## Problem Frame

The MVP structural model is blocked: Telegram-domain not_fish precision = 14% at
threshold 0.5. Root cause: training negatives were iNaturalist + Wikimedia, not
real Telegram photos. Gate: ≥ 200 confirmed Telegram-domain negatives. Current
state: 142. Gap: 58.

AL run 1 was too fish-heavy because:
- Bucket A (Phase C negatives) had only 22 records (fish_part/fry_juvenile/poster_screenshot)
- Bucket B sorted descending fish_conf — took records where model thought "fish" but
  had a lure caption hint. In a fishing reporting channel, these turned out to be
  real fish 92%+ of the time.
- Bucket E (50 random controls) further diluted the negative yield.

---

## Requirements Trace

- R1. New batch excludes all records reviewed in rvrun_20260427T184629Z (500 records).
- R2. New batch excludes all records reviewed in alvrun_20260501T194300Z (497 records).
- R3. Batch size 200–250 records (hard max 250).
- R4. Batch is more negative-focused than AL run 1 (expected yield > 22.1%).
- R5. Model predictions used as triage only; never written as truth labels.
- R6. Phase C categories used as triage only; final_category not pre-filled.
- R7. Annotation guide embedded in every HTML batch; model predictions hidden from reviewer.
- R8. Privacy-safe counts-only summary is tracked; detailed outputs are gitignored.
- R9. Existing reviewed decision files are never overwritten.
- R10. Tests cover new pass-2 requirements.

---

## Scope Boundaries

- Does NOT train, retrain, or fine-tune any model.
- Does NOT auto-label or pseudo-label.
- Does NOT mutate Phase C outputs or existing reviewed aggregates.
- Does NOT use internet, GPT Vision, or cloud APIs.
- Does NOT commit, stage, or push any files.

---

## Context & Research

### Key Data Facts

- Total Phase C candidates: 32,420
- Phase C negatives: 22 total (fish_part:16, poster_screenshot:3, fry_juvenile:3)
  — ALL were in AL run 1 Bucket A → **Bucket A is exhausted for pass 2**
- Caption signals remaining: ~641 caption_lure_hint + few others (~850 total - 200 in pass 1)
- Quality signals remaining: ~700 (low_res:371 + extreme_aspect:281 + tiny_file:137, with overlaps; pass 1 took 100)
- No-signal records: ~31,557 (virtually the entire corpus)
- Reviewed excluded: 997 (500 rvrun + 497 alvrun)
- Remaining unreviewed pool: ~31,423

### Why Bucket B Yielded Mostly Fish

Pass 1 Bucket B sort was DESC fish_conf — this prioritized records where the model
thought "fish" but had a caption_lure_hint. In a fishing catch-reporting channel,
fishermen mention lure names while reporting catches → those records are fish.
Pass 2 must sort ASCENDING fish_conf: take records where the model thinks "not_fish"
AND there's a negative caption hint → much stronger negative evidence.

### Exclusion Logic Already Correct

The existing `load_reviewed_ids_from_decisions_dir` scans `filter_decisions_alvrun*.json`
and `filter_decisions_rvrun_20260427T184629Z*.json` — this already correctly excludes
both rvrun and alvrun reviewed records for any new run.

### Existing Script Structure

The script in `scripts/select_telegram_negative_review_candidates.py` uses a 5-bucket
design (A–E) with hardcoded caps. Adding a `--mode` parameter is the minimal, safe
change: default mode is backward-compatible; `negative-focused-v2` mode applies new caps
and sort logic.

---

## Key Technical Decisions

- **Mode flag, not default change**: Existing behavior unchanged. New
  `--mode negative-focused-v2` activates pass-2 caps. Tests for existing behavior
  remain green.
- **Bucket B sort reversal**: Ascending fish_conf in v2 (takes records where model
  predicts not_fish + has caption hint first). Descending in v1 (potential FP detection).
- **Bucket D redesign in v2**: Prioritize strong not_fish (fish_conf < 0.25) before
  medium uncertainty (0.25–0.65). Increases genuine negative capture.
- **Bucket E cap reduction**: 50→15. We have enough fish; controls just dilute yield.
- **No new bucket letter**: All changes are within A–E to keep the schema stable.
- **Generic unknown cap via Bucket E**: Records with no signal and no model negative
  evidence go to Bucket E only. With cap=15 for a 250-record batch = 6%, well under
  the 10% requirement.

---

## Implementation Units

- U1. **Add v2 mode constants and update `select_candidates()` signature**

**Goal:** Add `MODE_DEFAULT`/`MODE_NEGATIVE_FOCUSED_V2` constants and v2 bucket caps.
Update `select_candidates()` to accept `mode` parameter and apply v2 logic when set.

**Requirements:** R3, R4, R5, R6

**Dependencies:** None

**Files:**
- Modify: `scripts/select_telegram_negative_review_candidates.py`

**Approach:**
- Add string constants for modes
- Add v2 caps: A=5, B=70, C=60, D=90, E=15; strong_not_fish_conf_max=0.25
- `select_candidates(unreviewed, predictions, target_size, rng, mode=MODE_DEFAULT)`
- In v2 mode: Bucket B sorts ascending fish_conf; Bucket D fills strong not_fish
  (fish_conf < 0.25) before medium uncertainty (0.25–0.80); use v2 caps throughout

**Test scenarios:**
- Happy path: default mode → same bucket caps and sort as before (backward compat)
- Happy path v2: fish controls ≤ V2_BUCKET_E_CAP in v2 mode
- Happy path v2: Bucket B records in v2 have ascending fish_conf ordering
- Edge case: Bucket A empty (no Phase C negatives left) → selection fills from B onward
- Edge case: fewer candidates than cap → fills what's available without error

**Verification:** All 18 existing tests still pass; new v2 tests pass.

---

- U2. **Add `--mode` and update `run()` and `_build_tracked_summary()`**

**Goal:** Wire mode through CLI → run() → select_candidates(); track mode in summary.

**Requirements:** R4, R8

**Dependencies:** U1

**Files:**
- Modify: `scripts/select_telegram_negative_review_candidates.py`

**Approach:**
- Add `--mode` argparse argument with choices=[MODE_DEFAULT, MODE_NEGATIVE_FOCUSED_V2]
- Pass mode through `run()` to `select_candidates()`
- Add `selection_mode` and `second_pass` fields to `_build_tracked_summary()` output
- Update final print report to show mode name and expected negative yield note

**Test scenarios:**
- Happy path: `--mode negative-focused-v2` runs without error
- Happy path: tracked summary contains `selection_mode` field
- Edge case: default (no flag) still uses original caps

**Verification:** `--dry-run` with mode flag exits cleanly and logs correctly.

---

- U3. **Add pass-2 specific tests**

**Goal:** Cover new requirements: alvrun exclusion, unknown cap, v2 mode negative focus.

**Requirements:** R1, R2, R4, R10

**Dependencies:** U1, U2

**Files:**
- Modify: `tests/test_select_telegram_negative_review_candidates.py`

**Approach:**
Add:
- `test_excludes_alvrun_reviewed_ids`: `load_reviewed_ids_from_decisions_dir` correctly
  picks up ids from files matching `filter_decisions_alvrun*.json`
- `test_v2_mode_fish_controls_capped`: In v2 mode, Bucket E ≤ V2_BUCKET_E_CAP (≤ 10% of 250)
- `test_v2_mode_bucket_b_ascending_sort`: In v2, Bucket B records ordered ascending fish_conf
- `test_v2_mode_strong_not_fish_before_uncertainty`: In v2 Bucket D, fish_conf < 0.25
  records come before 0.25–0.65 records
- `test_selection_does_not_exceed_target_size`: total selected ≤ target_size for any mode

**Test scenarios:**
- `test_excludes_alvrun_reviewed_ids`: write a temp decisions file named
  `filter_decisions_alvrun_TEST_0001.json` with final_category set → IDs must appear
  in the exclusion set
- `test_v2_mode_fish_controls_capped`: build pool with 100 no-signal records, run
  v2 selection, assert Bucket E ≤ 15
- `test_selection_does_not_exceed_target_size`: target=50, 200 candidates → ≤ 50 selected

**Verification:** `venv_ml/bin/python3 -m pytest tests/test_select_telegram_negative_review_candidates.py -v` all green.

---

- U4. **Execute script and generate HTML batch**

**Goal:** Run the script with `--mode negative-focused-v2 --target-size 250` to produce
a single 250-record HTML batch for human review.

**Requirements:** R3, R4, R7, R8, R9

**Dependencies:** U1, U2, U3

**Files:**
- Create: `data/intake_meta/tg_2026-04-24/review/filter_review_batch_<RUN_ID>_0001.html`
- Create: `data/intake_meta/tg_2026-04-24/review/filter_review_manifest_<RUN_ID>.json`
- Create: `data/intake_meta/tg_2026-04-24/review/filter_decisions_<RUN_ID>_0001.json` (blank template)
- Create: `data/intake_meta/tg_2026-04-24/review/assets/<RUN_ID>/` (thumbnails)
- Create: `private/active_learning/telegram_negative_candidates/<RUN_ID>/` (gitignored details)
- Create: `data/fish_models/telegram_negative_candidate_selection_summary_<RUN_ID>.json`

**Approach:**
1. Dry-run first to validate inputs
2. Full run with MPS device for inference
3. Verify output files exist and are well-formed

**Test scenarios:**
- HTML batch loads in browser without JS errors
- Decision template has all final_category=null
- Manifest counts match selected records

**Verification:** Batch HTML opens in browser; model predictions not visible in UI.

---

- U5. **Privacy and gitignore verification**

**Goal:** Confirm all detailed outputs are gitignored; tracked summary is counts-only.

**Requirements:** R8

**Dependencies:** U4

**Files:** No file changes; verification only.

**Approach:**
- `git check-ignore` for HTML batch, manifest, assets, private/
- `grep` tracked summary for paths/filenames/captions

**Verification:** `git status --short` shows no untracked files in sensitive paths.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Bucket A completely empty (all 22 Phase C negatives reviewed) | Script handles 0 gracefully; fills from B onward |
| caption_lure_hint ascending-sort still yields fish | Bucket B capped at 70; Bucket D strong not_fish provides alternative negatives |
| Not enough candidates in any single bucket | Script fills what's available and proceeds; minimum 150 records acceptable |
| Private dir missing → shadow predictions not written | private/ is created by the script on demand |
