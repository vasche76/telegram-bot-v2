# Bot MVP Integration Todo

**Date:** 2026-05-01  
**Status:** PLANNED — integration requires careful risk assessment before implementation

---

## Current Bot State

The bot has:
- `FishVisionPipeline` in `bot/fish_vision/pipeline.py` with `weight_kg_estimate: Optional[float]`
- Photo handler: `bot/handlers/vision.py` → `handle_photo_message()`
- `FishAnalysisResult.weight_kg_estimate` field (currently `None`)

The weight estimator module is now at `bot/fish_vision/weight_estimator.py`.

---

## Integration Checklist

### 1. Wire Weight Estimator into Pipeline

**File:** `bot/fish_vision/pipeline.py`  
**Change:** After species classification, call `estimate_fish_weight()` with available data.

```python
# In analyze() method, after ClassificationResult:
from bot.fish_vision.weight_estimator import estimate_fish_weight, VisualSizeBucket

weight_result = estimate_fish_weight(
    is_fish=is_fish,
    species=classification.species_key,
    length_cm=None,      # not available from photo alone
    girth_cm=None,       # not available from photo alone
    image_quality=None,  # derive from detection confidence
)
# Store in FishAnalysisResult or return separately
```

**Risk:** LOW — weight estimator is pure function, no side effects.

---

### 2. Update FishAnalysisResult to Include Weight Range

**File:** `bot/fish_vision/pipeline.py`  
**Change:** Replace `weight_kg_estimate: Optional[float]` with weight range fields.

```python
@dataclass
class FishAnalysisResult:
    # ... existing fields ...
    weight_kg_estimate: Optional[float] = None          # DEPRECATED — keep for compat
    weight_min_kg: Optional[float] = None
    weight_max_kg: Optional[float] = None
    weight_confidence: Optional[str] = None             # none/low/medium/high
    weight_needs_length: bool = True                    # True if we need user input
    weight_prompt_ru: Optional[str] = None              # prompt to ask user
```

**Risk:** MEDIUM — may break existing callers of `weight_kg_estimate`.

---

### 3. Update Vision Handler Response

**File:** `bot/handlers/vision.py`  
**Change:** Format the weight range in the reply message.

```python
from bot.fish_vision.weight_estimator import format_weight_reply

# After fish analysis:
if result.weight_needs_length:
    weight_msg = "⚖️ Пришли длину рыбы в см для оценки веса."
else:
    weight_msg = format_weight_reply(weight_result, lang="ru")

reply = f"{fish_classification_text}\n\n{weight_msg}"
```

**Risk:** LOW — output only, no data mutation.

---

### 4. Handle Length/Girth Feedback

**File:** `bot/handlers/` (new or existing text handler)  
**Change:** Parse user text messages for length/girth input and update the weight estimate.

```python
# Detect patterns like:
# "58 см", "58cm", "длина 58", "length 58"
# "обхват 30", "girth 30cm"

import re
LENGTH_PATTERN = re.compile(r'(?:длина|length|длин)?\s*(\d+(?:[.,]\d+)?)\s*(?:см|cm)', re.IGNORECASE)
GIRTH_PATTERN = re.compile(r'(?:обхват|girth)\s*(\d+(?:[.,]\d+)?)\s*(?:см|cm)', re.IGNORECASE)
```

**Risk:** MEDIUM — requires conversation state tracking (last fish photo context).

---

### 5. Local Feedback Logging

**File:** `bot/storage/` (existing storage layer)  
**Change:** Save user corrections to `bot.db` catch records.

Fields to save:
- `species_corrected: bool`
- `corrected_species: str`
- `length_cm: float`
- `girth_cm: float`
- `actual_weight_kg: float`

**Risk:** LOW — append-only, existing storage infrastructure.

---

## Structural Classifier Integration

**Status:** BLOCKED pending dataset

Once structural training passes gates:

```python
# Replace GPT-only path with local model path where available
# bot/fish_vision/local_classifier.py already exists as stub
# bot/fish_vision/local_detector.py already exists as stub
```

**File targets:**
- `bot/fish_vision/local_detector.py` — Stage A structural detector
- `bot/fish_vision/local_classifier.py` — Stage B species classifier
- `bot/fish_vision/pipeline.py` — route to local model if available, GPT fallback otherwise

---

## Do NOT Change

- Do NOT remove the GPT fallback (project safety rule)
- Do NOT modify `bot/services/vision.py` (existing GPT integration)
- Do NOT commit bot.db or any private data
- Do NOT auto-label unreviewed photos using the bot

---

## Recommended Integration Order

1. **Week 1:** Wire weight estimator → pipeline (lowest risk)
2. **Week 2:** Update vision handler response format
3. **Week 3:** Add length/girth parsing in text handler
4. **Week 4:** Local feedback logging to DB
5. **After training:** Local model integration (when structural training passes gates)

---

## Test Commands After Integration

```bash
# Run existing test suite
python3 -m pytest tests/ -x -q

# Manual bot test (with real Telegram bot token)
python3 main.py  # then send a fish photo via Telegram
```
