# Fish Dataset Inventory — 2026-04-15

## Stage A (Detector)

| Class | Raw Images | Status |
|-------|-----------|--------|
| whole_fish | 286 | OK |
| lure | 60 | OK |
| fish_part | 0 | EMPTY |
| fry | 0 | EMPTY |
| no_fish | 50 | OK |

Note: Bootstrap YOLO labels are synthetic full-image bounding boxes, not real hand-drawn bounding boxes. Labels were generated automatically by assigning a single full-image box per class folder image.

Labeled splits: train=341, val=103, test=109

CRITICAL: class_id 2 (fish_part) and class_id 3 (fry) have 0 annotations in all label files.

## Stage B (Species Classifier)

| Species | Class Index | Images | Status |
|---------|------------|--------|--------|
| pike | 0 | 80 | OK |
| taimen | 1 | 0 | EMPTY |
| grayling | 2 | 80 | OK |
| whitefish | 3 | 46 | WARN |
| perch | 4 | 80 | OK |
| brown_trout | 5 | 0 | EMPTY |
| rainbow_trout | 6 | 0 | EMPTY |
| atlantic_salmon | 7 | 0 | EMPTY |
| common_carp | 8 | 0 | EMPTY |
| crucian_carp | 9 | 0 | EMPTY |
| bream | 10 | 0 | EMPTY |
| roach | 11 | 0 | EMPTY |
| ide | 12 | 0 | EMPTY |
| wels_catfish | 13 | 0 | EMPTY |
| unknown_fish | 14 | 0 | EMPTY |

## Data Sources

- iNaturalist: pike(80), grayling(80), perch(80), whitefish(46) — research-grade, CC-licensed
- GBIF: 0 images acquired yet
- DeepFish: skipped (requires login)
- FishNet: skipped (requires login)

## Training Readiness

- Stage A: PARTIALLY READY — detector can train on 2/5 classes only; fish_part and fry have zero data
- Stage B: NOT READY — only 4/15 species have sufficient data; 11 species completely empty

## Next Actions (Priority Order)

1. Acquire images for taimen (manual photography required — no open dataset source available)
2. Collect Stage B images for remaining 10 empty species (brown_trout, rainbow_trout, atlantic_salmon, common_carp, crucian_carp, bream, roach, ide, wels_catfish, unknown_fish) via iNaturalist or GBIF
3. Hand-label fish_part and fry images for Stage A to fix zero-annotation classes (class_id 2 and 3)
4. Top up whitefish from 46 to 80+ images to match other species counts
5. Re-run validate_dataset.py after each data acquisition batch to track progress
