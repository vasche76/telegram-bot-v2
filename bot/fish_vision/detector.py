"""
Stage A — Fish Object Detector / Filter.

Determines what is actually in the image BEFORE any species classification.
This stage prevents lures, fish parts, fry, and non-fish objects from
reaching the species classifier and polluting catch statistics.

Current backend: GPT-4o-mini vision with two-step chain-of-thought reasoning.
Future backend: local YOLO/EfficientDet model (infrastructure is ready — see models/config.py).

Output classes:
    whole_fish  — complete fish suitable for species classification
    fish_part   — fragment/piece of fish, no head or no tail
    lure        — artificial fishing lure (wobbler, spinner, spoon, soft bait)
    fry         — juvenile fish, too small to be a meaningful catch
    no_fish     — no fish or fish-like object present

Design notes:
    - Two-step reasoning: first describe what you see, then classify.
    - Extremely specific lure detection cues (hooks, hardware, artificial materials).
    - Confidence thresholding: if Stage A confidence < FILTER_CONFIDENCE_THRESHOLD,
      the result is treated as uncertain and the catch is not recorded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from bot.services.ai import chat_completion
from bot.utils.logging import get_logger

log = get_logger("fish_vision.detector")

# Object types returned by Stage A
OBJECT_TYPES = {
    "whole_fish",
    "fish_part",
    "lure",
    "fry",
    "no_fish",
}

# Only these types should proceed to Stage B
PROCEED_TO_CLASSIFIER = {"whole_fish"}

# Minimum confidence to trust the Stage A decision.
# Below this, treat as uncertain (we don't want false positives in stats).
FILTER_CONFIDENCE_THRESHOLD = 0.60

# Size threshold for fry detection (cm): fish smaller than this are flagged as fry
FRY_LENGTH_THRESHOLD_CM = 15.0


@dataclass
class DetectionResult:
    object_type: str          # one of OBJECT_TYPES
    confidence: float         # 0.0 – 1.0
    fish_count: int           # number of fish-like objects seen (0 for no_fish/lure/fry)
    estimated_length_cm: Optional[float]  # rough scale estimate if visible
    reasoning: str            # step-by-step reasoning for audit trail
    raw_description: str      # objective description before classification

    @property
    def should_classify(self) -> bool:
        """True only if the image contains a valid whole fish at sufficient confidence."""
        return (
            self.object_type in PROCEED_TO_CLASSIFIER
            and self.confidence >= FILTER_CONFIDENCE_THRESHOLD
        )

    @property
    def rejection_reason(self) -> Optional[str]:
        if self.object_type == "lure":
            return "На фото рыболовная приманка (воблер/блесна/джиг), а не настоящая рыба"
        if self.object_type == "fish_part":
            return "На фото часть рыбы (нет головы или хвоста), а не целая рыба"
        if self.object_type == "fry":
            return "На фото малёк — слишком маленькая рыба, не считается уловом"
        if self.object_type == "no_fish":
            return "На фото нет рыбы или рыболовных объектов"
        if self.confidence < FILTER_CONFIDENCE_THRESHOLD:
            return (
                f"Низкая уверенность определения ({self.confidence:.0%}) — "
                "результат не записывается, чтобы не испортить статистику"
            )
        return None


# ─── GPT-based detector ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
Ты — специализированный детектор объектов на рыболовных фотографиях.
Твоя единственная задача — определить ЧТО именно изображено на фото:
целая рыба, часть рыбы, рыболовная приманка, малёк или ничего похожего на рыбу.
Ты НЕ определяешь вид рыбы на этом этапе.
Отвечай только JSON.\
"""

_DETECTION_PROMPT = """\
Тебе нужно последовательно ответить на два вопроса.

═══ ШАГ 1: ОБЪЕКТИВНОЕ ОПИСАНИЕ ═══
Опиши что ты видишь на фото без интерпретации.
Обрати особое внимание на:
- Текстуру объекта: чешуя, кожа, пластик, металл, резина?
- Крючки: есть ли тройные крючки, карабины, заводные кольца?
- Форма: целое тело с головой и хвостом? Или только часть?
- Размер: если есть ориентир (рука, линейка, спиннинг) — оцени размер
- Состояние: живая рыба, снулая, замороженная, кусок, искусственный объект?

═══ ШАГ 2: КЛАССИФИКАЦИЯ ═══
На основе описания выбери ОДИН тип объекта:

• whole_fish   — ЦЕЛАЯ настоящая рыба. Должна быть видна и голова, и хвостовой плавник.
                 Рыба может быть в руках, на земле/снегу, в воде. Может быть мокрой, в крови.
                 ВАЖНО: если есть чешуя или естественная кожа — это рыба.

• lure          — РЫБОЛОВНАЯ ПРИМАНКА. Признаки:
                  ✗ пластиковое/металлическое тело
                  ✗ видны тройные крючки (тройники) или одинарный крючок с поддевом
                  ✗ заводные кольца, карабины на теле
                  ✗ искусственные глаза (нарисованные, круглые вставки)
                  ✗ блесна-вращалка (металлическая лопасть на оси)
                  ✗ ярко-кислотный цвет (зелёный, оранжевый, белый) без разнообразия
                  ✗ отверстие/петля/ушко на носу и хвосте
                  ✗ воблер: пластиковый «лоб» (лопасть) под носом
                  ДАЖЕ ЕСЛИ ПРИМАНКА ПОХОЖА НА ЩУКУ — это приманка.

• fish_part     — ЧАСТЬ рыбы. Признаки:
                  ✗ нет головы (отрезана) или нет хвоста
                  ✗ виден срез мяса, кости, кожа
                  ✗ стейк, филе, тушка без головы/хвоста

• fry           — МАЛЁК. Признаки:
                  ✗ очень маленькая рыба (менее 15 см по оценке)
                  ✗ прозрачное тело, молодая рыбка
                  ✗ стайка мальков

• no_fish       — Ни рыбы, ни приманки. Пейзаж, снаряжение, человек, лодка и т.д.

═══ ВЫВОД ═══
Подпись к фото от пользователя: "{caption}"

Ответь JSON:
{{
  "raw_description": "объективное описание из Шага 1 (2-4 предложения)",
  "reasoning": "краткое объяснение выбора из Шага 2",
  "object_type": "whole_fish | lure | fish_part | fry | no_fish",
  "confidence": 0.0 до 1.0,
  "fish_count": число целых рыб (0 если не whole_fish),
  "estimated_length_cm": число или null
}}\
"""


async def _detect_fish_object_gpt(
    image_url: str,
    caption: str = "",
) -> DetectionResult:
    """
    Stage A: Detect and filter the image content (GPT backend).

    Returns a DetectionResult. Check result.should_classify before
    proceeding to Stage B species classification.
    """
    prompt = _DETECTION_PROMPT.replace("{caption}", caption or "(нет подписи)")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            ],
        },
    ]

    try:
        raw = await chat_completion(
            messages=messages,
            model=None,  # uses OPENAI_VISION_MODEL from config
            temperature=0.1,   # low temperature = more deterministic
            max_tokens=800,
            json_mode=True,
        )
        data = json.loads(raw)
    except Exception as e:
        log.error(f"Stage A detection failed: {e}", exc_info=True)
        # Safe fallback: treat as uncertain
        return DetectionResult(
            object_type="no_fish",
            confidence=0.0,
            fish_count=0,
            estimated_length_cm=None,
            reasoning=f"Ошибка детектора: {e}",
            raw_description="",
        )

    obj_type = data.get("object_type", "no_fish")
    if obj_type not in OBJECT_TYPES:
        obj_type = "no_fish"

    confidence = float(data.get("confidence", 0.0))
    confidence = max(0.0, min(1.0, confidence))

    result = DetectionResult(
        object_type=obj_type,
        confidence=confidence,
        fish_count=int(data.get("fish_count", 0) or 0),
        estimated_length_cm=data.get("estimated_length_cm"),
        reasoning=data.get("reasoning", ""),
        raw_description=data.get("raw_description", ""),
    )

    log.info(
        f"Stage A: {obj_type} (conf={confidence:.2f}, count={result.fish_count}) "
        f"→ proceed={result.should_classify}"
    )
    return result


# ─── Backend dispatcher ─────────────────────────────────────────────────────

from bot.fish_vision.models.config import DETECTOR_BACKEND  # noqa: E402


async def detect_fish_object(
    image_url: str,
    caption: str = "",
) -> DetectionResult:
    """
    Stage A dispatcher. Routes to local YOLO or GPT based on FISH_DETECTOR_BACKEND.

    Set FISH_DETECTOR_BACKEND=local in .env (and ensure detector_v1.pt exists in
    data/fish_models/) to use the local YOLO model. Falls back to GPT automatically
    if the local model is unavailable.
    """
    if DETECTOR_BACKEND == "local":
        log.info("Stage A: using local YOLO backend")
        try:
            from bot.fish_vision.local_detector import LocalYOLODetector
            detector = LocalYOLODetector.get_instance()
            return await detector.detect(image_url, caption)
        except Exception as e:
            log.warning(f"Local detector failed ({e}), falling back to GPT")

    log.debug("Stage A: using GPT backend")
    return await _detect_fish_object_gpt(image_url, caption)
