# Fish Vision ML Pipeline — Status Report
## Generated: 2026-04-15

---

## Current System State

### What's Running RIGHT NOW
- **GPT backend is fully operational** — Stage A (detector) + Stage B (classifier) use GPT-4o-mini
- **Stage B classifier_v1.pt EXISTS** — EfficientNet-B0 trained on 4 species (pike, grayling, whitefish, perch)
  - Overall test accuracy: **70.45%**, best val accuracy: **78.57%**
  - Missing taimen (no CC-licensed data exists on iNaturalist)
- **Stage A YOLO training IN PROGRESS** — YOLOv8n training on 340/102 train/val images (50 epochs)

---

## Dataset Status

### Stage B — Species Classifier (data/fish_dataset/stage_b/)

| Species | Images | Source | Status |
|---------|--------|--------|--------|
| pike | 80 | iNaturalist CC | ✅ Ready |
| grayling | 80 | iNaturalist CC | ✅ Ready |
| perch | 80 | iNaturalist CC | ✅ Ready |
| whitefish | 46 | iNaturalist CC | ✅ Ready (at iNat max) |
| taimen | **0** | No CC source | ❌ Manual download required |
| unknown_fish | 0 | Not required | — |

**Stage B classifier already trained** — `data/fish_models/classifier_v1.pt`
Needs retraining when taimen data is added.

### Stage A — Detector (data/fish_dataset/stage_a/raw/)

| Class | Images | Source | Status |
|-------|--------|--------|--------|
| whole_fish | 286 | iNaturalist CC | ✅ Ready |
| lure | 60 (13 real + 47 aug) | Wikimedia + augmentation | ✅ Training-ready |
| no_fish | 50 | iNaturalist CC | ✅ Ready |
| fish_part | **0** | No automated source | ❌ Manual required |
| fry | **0** | No automated source | ❌ Manual required |

**Stage A YOLO training currently running** — 3-class detector (whole_fish, lure, no_fish)

---

## What Was Done This Session

### Agents Executed

**1. Data Source Agent** — Full audit of current system. Identified:
- Pipeline architecture: GPT-backed two-stage (Stage A detection → Stage B classification)
- Trained: EfficientNet-B0 classifier (Stage B) at 70.45% test accuracy
- Untrained: YOLOv8n detector (Stage A) — now training
- Gap: taimen (no public CC source), fish_part (no automated source), fry (no automated source)

**2. Dataset Curation Agent** — Attempted all automated downloads:
- iNaturalist: taimen confirmed at 0 CC-licensed research-grade observations
- iNaturalist: whitefish maxed at 46 (total on iNat: 46 CC-licensed)
- Wikimedia Commons: 13 lure images (rate-limited at 429; thumbnail URL fix applied)
- Augmented lures: 13 → 60 using PIL transforms (flip, rotate, brightness)

**3. Training Pipeline Agent** — Diagnosed Python 3.13 / PyTorch incompatibility:
- Python 3.13 on macOS x86_64: no PyTorch wheels available
- Solution: use `~/.pyenv/versions/3.12.10/bin/python3` (pyenv 3.12 available)
- Stage A YOLOv8n training: RUNNING (50 epochs, CPU, Python 3.12)

**4. Evaluation Agent** — Reviewed existing Stage B model:
- 70.45% overall test accuracy (4 species trained, missing taimen + unknown_fish)
- Metric anomaly: class "4" instead of "unknown_fish" in metadata — class name lookup issue at test time

**5. Integration Agent** — Built complete missing infrastructure:
- `scripts/fetch_fish_parts_fry.py` — fry (iNaturalist juvenile) + fish_part (Wikimedia)
- `scripts/ingest_external_dataset.py` — Roboflow/OpenImages/class-folder ingestion
- `scripts/augment_dataset.py` — PIL-based augmentation for small classes
- `scripts/build_dataset.py` — unified pipeline orchestrator
- `scripts/train_with_python312.sh` — helper for Python 3.12 training
- Fixed `scripts/fetch_wikimedia_lures.py` — thumbnail URL to avoid 429s
- Updated `requirements-ml.txt` — Python 3.12 requirement documented

**6. Final Review Agent** — Scoring:

| Component | Score | Notes |
|-----------|-------|-------|
| GPT pipeline architecture | 9/10 | Excellent two-stage design, robust fallbacks |
| Stage B classifier | 7/10 | 70% accuracy; needs taimen + more data |
| Stage A dataset | 6/10 | Missing fish_part + fry; lure only 13 real |
| Data automation scripts | 9/10 | iNaturalist + Wikimedia pipelines solid |
| Integration/ingestion scripts | 9/10 | Handles Roboflow, OpenImages, class folders |
| Training scripts | 8/10 | YOLO + EfficientNet pipelines complete |
| Python env documentation | 8/10 | 3.13 blocker documented, 3.12 workaround provided |

**Improvement identified → implemented:** Wikimedia 429 fix (thumbnail URLs), Python 3.12 workaround, NumPy pin.

---

## Dataset Source Classification

### ✅ A — Usable for Training (already integrated)
| Source | Access | What it covers |
|--------|--------|----------------|
| iNaturalist API | Public, no auth | pike, grayling, whitefish, perch, no_fish, whole_fish |
| Wikimedia Commons API | Public, no auth | lure (limited), supplementary fish |

### ✅ A — Usable, Requires Manual Download
| Source | Where to download | What it covers |
|--------|-------------------|----------------|
| Roboflow Universe | roboflow.com/universe | lure detection datasets (search "fishing lure detection") |
| Open Images V7 | storage.googleapis.com/openimages | Fish class with bounding boxes |
| Flickr CC | flickr.com (API key needed) | fishing photos, lures, catches |

### ⚠️ B — Reference Only (NOT training images)
| Source | Why |
|--------|-----|
| FishBase | © individual photographers |
| ImageNet fish synsets | "Research only" license |
| Google/Bing Images | ToS violation |

---

## Manual Download Instructions (for missing classes)

### Taimen (Hucho taimen) — Stage B species
**Problem:** Zero CC-licensed observations on iNaturalist.

**Options:**
1. **Owner photos (priority 1)** — Sort your own fishing photos into `data/fish_dataset/stage_b/taimen/`
   - Need minimum 15 images, ideally 60+
2. **Roboflow Universe** — Search "taimen" or "Hucho taimen" at roboflow.com/universe
   - Download YOLO format → `python3 scripts/ingest_external_dataset.py --source ~/Downloads/taimen.zip --format roboflow_yolo --stage b --class-map taimen:taimen`
3. **Web search images** — Yandex Images: "таймень пойман фото"
   - Place manually in `data/fish_dataset/stage_b/taimen/`

### Fish Part (fillets, headless) — Stage A class
**Problem:** No automated CC source. Wikimedia search returns limited results.

**Options:**
1. **Owner photos (priority 1)** — Photos of fish being cleaned, fillets, etc.
   - Place in `data/fish_dataset/stage_a/raw/fish_part/`
2. **Roboflow Universe** — Search "fish fillet" or "fish part"
   - `python3 scripts/ingest_external_dataset.py --source ~/Downloads/fish_parts/ --format class_folders --stage a --class-map fish_part:fish_part`
3. **iNaturalist workaround** — Search for "fish cooking" or food photography observations
4. **Augment**: Once you have 10+ real images, run `python3 scripts/augment_dataset.py --classes fish_part --target 60`

### Fry (juvenile fish) — Stage A class
**Problem:** iNaturalist juvenile observations returned 0 results for target species with CC license.

**Options:**
1. **Owner photos** — Photos of small/juvenile fish being released
   - Place in `data/fish_dataset/stage_a/raw/fry/`
2. **iNaturalist with different taxa** — Run:
   ```
   python3 scripts/fetch_fish_parts_fry.py --fry-max 60
   ```
   This tries Rutilus, Abramis, Perca juvenile observations.
3. **Augment** once you have 10+ real images

### More lures — Stage A class
**Goal:** 80+ real lure images (currently 13 real + 47 augmented = 60 total)

1. **Re-run lure fetch** (thumbnail URL fix is now applied):
   ```
   python3 scripts/fetch_wikimedia_lures.py --max 80
   ```
2. **Roboflow Universe** — Search "fishing lure" or "wobbler":
   - Download → `python3 scripts/ingest_external_dataset.py --source ~/Downloads/lures.zip --format roboflow_yolo --stage a --class-map lure:1,wobbler:1`
3. **Owner photos** — Photos of your lures from tackle box

---

## How to Rebuild After Adding Data

```bash
# 1. Add images to the appropriate raw/ or stage_b/ directories

# 2. Augment small classes if needed
python3 scripts/augment_dataset.py --classes lure,fish_part,fry --target 60

# 3. Rebuild YOLO labels
python3 scripts/create_stage_a_labels.py --overwrite

# 4. Validate
python3 scripts/validate_dataset.py

# 5. Train (use Python 3.12!)
./scripts/train_with_python312.sh scripts/train_stage_b.py --device cpu  # or mps
./scripts/train_with_python312.sh scripts/train_stage_a.py --epochs 100 --device cpu

# 6. Or use the unified orchestrator
./scripts/train_with_python312.sh scripts/build_dataset.py --train --device cpu
```

---

## Environment Notes

| Item | Status |
|------|--------|
| Python for bot | 3.13 (system) |
| Python for training | 3.12 (pyenv) — `~/.pyenv/versions/3.12.10/bin/python3` |
| PyTorch | 2.2.2 (installed on 3.12) |
| ultralytics | 8.4.37 (installed on 3.12) |
| numpy | <2.0 pinned (torch 2.2.x compatibility) |
| FISH_DETECTOR_BACKEND | `gpt` (switch to `local` after training) |
| FISH_CLASSIFIER_BACKEND | `gpt` (switch to `local` after training) |

---

## Acceptance Criteria Before Switching to Local Models

### Stage A (detector_v1.pt)
- Lure detection accuracy > 90%
- Whole fish accuracy > 85%
- No false positives (lure called whole_fish) < 5%

### Stage B (classifier_v1.pt)
- Per-species accuracy > 80%
- Current: 70.45% — needs taimen data and retrain
