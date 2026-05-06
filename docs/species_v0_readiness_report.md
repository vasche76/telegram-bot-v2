# Species Classifier v0 Readiness Report

**Date:** 2026-05-01  
**Status:** PARTIALLY READY (external data available; Telegram species labels not yet collected)

---

## Summary

A species classifier (Stage B) can be trained on existing external data, but results will be limited by:
1. Domain mismatch: external data is mostly specimen/museum photos, not fishing photos
2. Missing classes: taimen = 1 image, zander = 0 images
3. No Telegram-sourced species labels yet (would require future review batches with species selection)

---

## Available Data for Species Training

External data in `data/fish_dataset/stage_b/` (GBIF + iNaturalist, CC-licensed):

| Species | Images | Training quality |
| --- | --- | --- |
| pike | 81 | ✅ Sufficient for baseline |
| perch | 81 | ✅ Sufficient for baseline |
| grayling | 81 | ✅ Sufficient for baseline |
| whitefish | 46 | ⚠️ Marginal |
| brown_trout | 81 | ✅ Sufficient for baseline |
| rainbow_trout | 81 | ✅ Sufficient for baseline |
| atlantic_salmon | 81 | ✅ Sufficient for baseline |
| common_carp | 81 | ✅ Sufficient for baseline |
| crucian_carp | 70 | ✅ Sufficient for baseline |
| bream | 81 | ✅ Sufficient for baseline |
| roach | 81 | ✅ Sufficient for baseline |
| ide | 77 | ✅ Sufficient for baseline |
| wels_catfish | 70 | ✅ Sufficient for baseline |
| taimen | 1 | ❌ Not trainable |
| unknown_fish | 0 | — (fallback class, no training examples) |

**Training-eligible species (≥ 40 images):** 13 out of 15

---

## Domain Mismatch Warning

External images are:
- Mostly underwater / museum specimen / field guide photos
- Very different from Telegram fishing group photos (holding fish, dock, water surface)

A species classifier trained on external data **will have degraded performance on fishing photos**.

**Recommended mitigation:**
1. Use cautious confidence thresholds (≥ 0.75 for any species claim)
2. Always append "possible species" wording, never "definitely"
3. Collect species labels from future Telegram review batches to fine-tune

---

## Target Species Mapping for Bot

```python
SPECIES_DISPLAY = {
    "pike": {"ru": "Щука", "en": "Pike"},
    "perch": {"ru": "Окунь", "en": "Perch"},
    "grayling": {"ru": "Хариус", "en": "Grayling"},
    "whitefish": {"ru": "Сиг", "en": "Whitefish"},
    "brown_trout": {"ru": "Кумжа/ручьевая форель", "en": "Brown Trout"},
    "rainbow_trout": {"ru": "Радужная форель", "en": "Rainbow Trout"},
    "atlantic_salmon": {"ru": "Лосось атлантический", "en": "Atlantic Salmon"},
    "common_carp": {"ru": "Карп / сазан", "en": "Common Carp"},
    "crucian_carp": {"ru": "Карась", "en": "Crucian Carp"},
    "bream": {"ru": "Лещ", "en": "Bream"},
    "roach": {"ru": "Плотва", "en": "Roach"},
    "ide": {"ru": "Язь", "en": "Ide"},
    "wels_catfish": {"ru": "Сом", "en": "Wels Catfish"},
    "taimen": {"ru": "Таймень", "en": "Taimen"},
    "unknown_fish": {"ru": "Неизвестная рыба", "en": "Unknown fish"},
}
```

---

## Training Requirements for v0 Baseline

If you want to train now on external-only data:

```bash
# Requires Python 3.12 (PyTorch not available on Python 3.13)
~/.pyenv/versions/3.12.10/bin/python3 scripts/prepare_stage_b.py
~/.pyenv/versions/3.12.10/bin/python3 scripts/train_stage_b.py --epochs 30
```

**Expected outcome:** ~50–65% accuracy on external test set.  
**Expected accuracy on fishing photos:** significantly lower (domain mismatch).

---

## Status Decision

| Gate | Status |
| --- | --- |
| External data has license/provenance | ✅ PASS |
| Target species mapping explicit | ✅ PASS |
| Enough examples per class (≥ 40) | ✅ 13/15 species |
| taimen class | ❌ 1 image — cannot train |
| No weak captions as truth | ✅ PASS (no Telegram labels yet) |
| Domain mismatch acknowledged | ✅ PASS |

**Decision: CONDITIONAL READY**  
Species v0 can be trained with external data and deployed cautiously.  
taimen and unknown_fish should be merged into `unknown_fish` fallback in v0.

---

## Exact Next Steps

1. Run `prepare_stage_b.py` to build train/val/test split from stage_b images
2. Train with `train_stage_b.py` using Python 3.12
3. Set bot species confidence threshold ≥ 0.75 for any claim
4. Evaluate on 20–30 manually labeled fishing photos from the group
5. Document accuracy on real fishing photos vs. external test set
