# Fish Vision Dataset Specification

## Purpose

This document specifies exactly what images are needed to train a reliable
local fish-recognition model for the Telegram bot.

Currently the bot uses GPT-4o-mini vision with a structured two-stage pipeline.
Once you collect enough labeled photos, the pipeline can be upgraded to run
a local YOLO + classification model — faster, cheaper, and more accurate
for your specific fish species.

---

## What Classes Are Needed

The pipeline has two stages, each needing separate training data.

### Stage A — Object Detection (what is in the photo)

| Class | Russian name | Purpose |
|-------|-------------|---------|
| `whole_fish` | целая рыба | Main positive class — must be detected |
| `lure` | приманка | Critical negative — vobler, spinner, spoon, rubber bait |
| `fish_part` | часть рыбы | Negative — fillets, headless, tailless pieces |
| `fry` | малёк | Negative — juvenile fish under ~15cm |
| `no_fish` | не рыба | Negative — landscapes, gear, boats, people |

### Stage B — Species Classification (which fish)

| Class | Russian name | Visual key features |
|-------|-------------|-------------------|
| `pike` | щука | Duck-bill snout, green spots, rear dorsal fin |
| `taimen` | таймень | Huge salmon-like, red/orange fins, X-spots |
| `grayling` | хариус | Tall sail-like dorsal fin, small mouth |
| `whitefish` | сиг | Silver compressed body, small adipose fin |
| `perch` | окунь | Vertical stripes, spiny dorsal, orange pectoral fins |
| `unknown_fish` | неопр. рыба | Any fish you cannot confidently ID above |

---

## How Many Photos Per Class

### Stage A (object detection)

| Class | Minimum | Target | Priority |
|-------|---------|--------|----------|
| whole_fish | 200 | 500+ | High |
| lure | 150 | 400+ | **Critical** — most errors come from lures |
| fish_part | 100 | 250 | High |
| fry | 80 | 200 | Medium |
| no_fish | 100 | 250 | Medium |

### Stage B (species classification)

| Species | Minimum | Target | Priority |
|---------|---------|--------|----------|
| pike | 100 | 300+ | High — most common catch |
| perch | 100 | 300+ | High — very common |
| grayling | 80 | 200 | High — distinctive but less common |
| taimen | 60 | 150 | Medium — rare trophy |
| whitefish | 80 | 200 | Medium |
| unknown_fish | 50 | 150 | Low — any unidentified fish |

**Important:** lures need the most hard examples because:
- Pike lures (воблеры) are specifically designed to look like pike
- Soft plastic baits look like small fish
- These cause the most real-world errors

---

## What Images to Collect

### For whole_fish / species classes

Good photos to collect:
- ✅ Fish held in hands (most common fishing photo)
- ✅ Fish lying on ground/snow/boat/grass
- ✅ Fish in shallow water being released
- ✅ Fish hanging on stringer
- ✅ Multiple fish in frame (still count as valid)
- ✅ Wet fish, bloody fish, dirty fish
- ✅ Different lighting: sunny, cloudy, dawn, dusk
- ✅ Different seasons: summer, winter (ice fishing), autumn

Bad/borderline photos to include as hard examples:
- ⚠️ Partially occluded fish (hand covering part of body)
- ⚠️ Fish at unusual angle (from above, from below)
- ⚠️ Small fish (but not fry — adult but small individuals)
- ⚠️ Fish with lures visible nearby in frame
- ⚠️ Low-light/blurry photos (these are common in practice)

### For lure class (critical!)

Collect photos of:
- ✅ Воблеры (crankbaits/plugs) — especially pike-shaped ones
- ✅ Колеблющиеся блёсны (spoons)
- ✅ Вертушки (spinners)
- ✅ Джиг-головки с резиной (jigheads + soft plastic)
- ✅ Твистеры, виброхвосты (rubber twister/shad baits)
- ✅ Попперы, уоки (surface lures)
- ✅ Fly fishing lures / мушки
- ✅ Lure in hand (being held)
- ✅ Lure lying on ground
- ✅ Lure in water/on hook
- ✅ **The most important: pike-shaped wobblers** (e.g., Rapala, Strike Pro)

### For fish_part class

- ✅ Fish fillet (филе)
- ✅ Headless fish
- ✅ Fish steak / cross-section
- ✅ Fish on cutting board being processed
- ✅ Fish head only

### For fry class

- ✅ Small juvenile perch, roach, etc. in hand
- ✅ Fry in net or bucket
- ✅ Very small fish (<10cm) being released

---

## Labeling Instructions

### For Stage A (detection) — use bounding boxes

Label each image with a rectangular bounding box around the main subject.
For whole_fish: box should tightly enclose the entire fish.
For lure: box around the main lure body (can ignore hooks if tiny).
For fish_part: box around the visible piece.

**Recommended tool:** Label Studio (free), Roboflow (free tier), CVAT

**Format:** YOLO format (class_id x_center y_center width height, normalized 0-1)

### For Stage B (classification) — image-level labels

For species classification, you only need one label per image (the species).
Each image should contain primarily one fish species.
Use the canonical class names: pike, taimen, grayling, whitefish, perch, unknown_fish

---

## Train/Validation/Test Split

- Training set: 70% of images per class
- Validation set: 15% (used during training to monitor accuracy)
- Test set: 15% (never seen during training — final accuracy measurement)

**Critical rule:** If you take multiple photos of the same fish (same catch session),
put ALL of those photos in the SAME split (all in train OR all in test).
Never split the same fish across train and test — this causes inflated test accuracy.

---

## Acceptance Criteria for the Trained Model

Before deploying the local model to production, it must pass:

### Stage A (detector)
- Lure detection accuracy: > 90% (must catch 9 out of 10 lures)
- Whole fish detection accuracy: > 85%
- Fish part detection accuracy: > 80%
- False positive rate for lures (lure called whole_fish): < 5%

### Stage B (classifier)
- Per-species accuracy when confident: > 80%
- Overall rejection rate for low-confidence cases: > 20%
  (it should say "unknown" rather than guess wrongly)
- Species confusion between pike and taimen: < 5%
  (they look quite different, should be easy to separate)

---

## Practical Collection Strategy

### Phase 1 — Collect your own photos (Week 1-2)
Go through your existing fishing photos:
- Sort them into folders by class
- Aim for 50-100 per class to start

### Phase 2 — Supplement with internet images (Week 2-3)
Search for photos on:
- Google Images / Yandex Images: "щука фото", "хариус в руках", "воблер на щуку"
- Fishing forums: fishingsib.ru, rybalka.guru, fishernet.ru
- YouTube screenshots from fishing videos

### Phase 3 — Label and train (Week 3-4)
Use Roboflow.com (free for small datasets):
1. Create project → Object Detection (for Stage A)
2. Upload images → Draw boxes → Export in YOLO format
3. Train YOLOv8 (Roboflow can train directly, or use the export + ultralytics locally)

### Phase 4 — Evaluate and deploy
Run: `python3 -m bot.fish_vision.evaluation.eval_runner eval_cases.json`
If metrics pass → set FISH_DETECTOR_BACKEND=local in .env

---

## Where to Put Model Files

Trained model files go in:
```
data/fish_models/
    detector_v1.pt         # YOLOv8 model for Stage A
    classifier_v1.pt       # EfficientNet for Stage B
    class_names.json       # {"0": "pike", "1": "taimen", ...}
    model_card.md          # Training date, dataset size, accuracy metrics
```

The bot will automatically use them when `FISH_DETECTOR_BACKEND=local` is set in `.env`.

---

## What You Can Do Right Now (Before Training Data)

The bot already uses the two-stage GPT pipeline which is significantly better
than the old single-call approach. The improvements are:

1. Dedicated lure detection logic with specific visual cues
2. Two-step reasoning (describe first, then classify)
3. Confidence thresholds — uncertain results become "unknown" instead of wrong guesses
4. Bad detections still stored for audit but excluded from statistics
5. Species features are explicitly described to the model

Start collecting photos now. Even 30-50 photos per class gives you enough
to test whether local training is feasible.
