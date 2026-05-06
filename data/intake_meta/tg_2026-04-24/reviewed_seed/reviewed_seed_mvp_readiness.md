# Reviewed Seed MVP Readiness Report

Generated: 2026-05-01T14:12:53.924016+00:00

Source: telegram_private_2026-04-24

## Summary

| Metric | Value |
| --- | --- |
| Reviewed records in seed | 258 |
| Excluded (needs_human_review) | 242 |
| Unreviewed (not eligible) | 31920 |
| Training-usable total | 258 |
| MVP class: fish | 245 |
| MVP class: not_fish_or_other | 13 |
| Fish:non-fish ratio | 18.8:1 |

## Structural MVP Status: **BLOCKED**

### Why blocked
- Not enough non-fish examples: 13 < 20 minimum
- Class imbalance too severe: 18.8:1 fish-to-non-fish (limit 10:1)

### Next action to unblock
1. Review more batches with non-fish content (batches 0003–0130 contain ~31,920 unreviewed records).
2. OR ingest external public negatives (DeepFish no_fish class, GBIF background images).
3. Minimum target: 100+ non-fish at confidence >= 3 before re-running this script.

## Class Mapping Used

| final_category | mvp_class | min_confidence |
| --- | --- | --- |
| bad_quality | needs_human_review | 3 |
| duplicate_suspect | needs_human_review | 3 |
| fish | fish | 3 |
| fish_part | fish | 3 |
| fry_juvenile | needs_human_review | 3 |
| lure_gear | not_fish_or_other | 3 |
| no_fish | not_fish_or_other | 3 |
| out_of_scope | needs_human_review | 3 |
| poster_screenshot | not_fish_or_other | 3 |
| unsure | needs_human_review | 3 |

## Recommended Actions
1. Review 10+ more batches to increase non-fish coverage.
2. Run `intake_external_lite.py` to record license-safe external sources.
3. Re-run `intake_phase_e_lite.py` after each batch review session.
4. Only run `build_mvp_training_dataset.py` when status is READY or CAUTION.
