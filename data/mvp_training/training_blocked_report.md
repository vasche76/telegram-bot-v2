# MVP Structural Training — Blocked Report

**Date:** 2026-05-01  
**Status:** BLOCKED  
**Gate failed:** GATE 3 — Class imbalance  

---

## Why Training is Blocked

Structural MVP training requires a balanced fish vs. non-fish dataset.  
The current data is **severely imbalanced (19.4:1)** — too many fish examples, too few non-fish.

| Class | Count | Required |
| --- | --- | --- |
| fish | 1224 | ≥ 1 |
| not_fish_or_other | 63 | ≥ 122 (≤ 10:1 ratio) |

A classifier trained on this data would learn to always predict "fish" and achieve ~95% accuracy while being useless.

---

## What We Have

| Source | Fish | Non-fish |
| --- | --- | --- |
| Telegram reviewed seed (conf ≥ 3) | 245 | 13 |
| External GBIF / iNaturalist (stage_b) | 979 | 0 |
| External stage_a no_fish | 0 | 50 |
| **Total** | **1224** | **63** |

---

## What We Need

| Option | Action | Estimated effort |
| --- | --- | --- |
| **Option A** | Review 10–15 more Telegram batches that contain lure, no_fish, out_of_scope records | 2–4 hours |
| **Option B** | Download DeepFish no_fish frames (CC-BY-4.0) | 1–2 hours manual download |
| **Option C** | Both A + B | Best result |

**Minimum to unblock:** `not_fish_or_other` ≥ 122 samples (at current fish count of 1224).  
**Better target:** 200+ non-fish for a 6:1 ratio or better.

---

## Next Commands After Collecting Non-Fish Data

```bash
# Step 1: Re-run Phase E after reviewing more batches
python3 scripts/intake_phase_e_lite.py

# Step 2: Rebuild dataset manifest (will check gates again)
python3 scripts/build_mvp_training_dataset.py

# Step 3: If gates pass, run structural training (Python 3.12 required for ultralytics)
~/.pyenv/versions/3.12.10/bin/python3 scripts/train_stage_a.py --epochs 30 --batch 16 --device cpu
```

---

## Data Integrity Guarantee

No unreviewed Telegram records will be included regardless of imbalance.  
Phase C labels are not truth and will never be used for training.  
Training will remain blocked until imbalance is resolved with verified data.
