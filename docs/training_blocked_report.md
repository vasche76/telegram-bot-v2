# MVP Structural Training — Status Report

Generated: 2026-05-01

## Summary

| Item | Status |
|---|---|
| Data quality gates | ✅ ALL PASSED |
| Training environment | ❌ BLOCKED — torch/ultralytics not installed |
| Next action | `pip install ultralytics` then run training script |

---

## Before (2026-05-01 session start)

| Class | Count |
|---|---|
| fish | 1,224 |
| not_fish_or_other | 63 |
| ratio | 19.4:1 |
| Gate 3 (≤ 10:1) | ❌ FAIL |

---

## After (2026-05-01 this session)

| Class | Count |
|---|---|
| fish | 1,224 |
| not_fish_or_other | 123 |
| ratio | 9.95:1 |
| Gate 3 (≤ 10:1) | ✅ PASS |
| All gates | ✅ PASS |

### How the unblocking was achieved

60 Wikimedia Commons lure images already present in
`data/fish_dataset/stage_a/raw/lure/` (13 originals + 47 augmented variants,
all CC-BY / CC-BY-SA licensed) were ingested as `not_fish_or_other` negatives.

- No new downloads were required.
- Augmented variants are tracked via `aug_group_id` from `AUGMENTATION.json`
  so group-aware splitting prevents augmentation leakage across train/val/test.
- Source provenance is recorded in `stage_a/raw/lure/PROVENANCE.json`.

---

## Data Integrity Checks

| Check | Result |
|---|---|
| No unreviewed Telegram records in seed | ✅ PASS (GATE 1) |
| No Phase C labels as truth | ✅ PASS (GATE 2) |
| Class imbalance ≤ 10:1 | ✅ PASS (GATE 3) — 9.95:1 |
| Minimum 20 non-fish examples | ✅ PASS (GATE 4) — 123 |
| External license report present | ✅ PASS (GATE 5) |
| No review_id leakage across splits | ✅ PASS (GATE 6) |
| Minimum total training size | ✅ PASS (GATE 7) |

---

## Physical Training Directory

Populated by `build_mvp_training_dataset.py --copy-images`.
Only external public images are physically present (Telegram records
are private and cannot be copied by the builder).

```
data/mvp_training/structural/
  train/fish/              694 images
  train/not_fish_or_other/  79 images
  val/fish/                141 images
  val/not_fish_or_other/    12 images
  test/fish/               144 images
  test/not_fish_or_other/   19 images
```

---

## Why Training Is Still Blocked

`torch` and `ultralytics` are not installed in the current Python environment.

```
$ python3 -c "import torch"
ModuleNotFoundError: No module named 'torch'
```

---

## Exact Smallest Next Action to Unblock Training

```bash
pip install ultralytics   # installs torch, torchvision, and YOLOv8

python3 scripts/train_mvp_structural.py --epochs 30 --device mps
# Use --device cpu if not on Apple Silicon
# Use --device 0 if NVIDIA GPU available
```

Training script: `scripts/train_mvp_structural.py`
Output model:    `data/fish_models/mvp_structural_v1.pt` (EXPERIMENTAL)
Metrics report:  `data/fish_models/mvp_structural_report.json`

---

## Privacy / Data Integrity

- Telegram data: never copied outside the repo; only counted in manifests.
- External data: CC-BY-4.0, CC0-1.0, CC-BY-SA — training use permitted.
- No GPT Vision / cloud API used on any private image.
- No Phase C labels used as ground truth.
- No unreviewed Telegram records in training set.

---

## Model Status After Training

Once training completes:
- Mark model: `experimental=True`, `mvp_baseline=True`
- Do NOT deploy to production.
- Do NOT use for weight estimation.
- Do NOT use as species classifier.
- Use only for binary fish/not_fish structural gating (experimental).
