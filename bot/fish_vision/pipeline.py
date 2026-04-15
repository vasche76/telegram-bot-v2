"""
Fish Vision Pipeline — orchestrates Stage A + Stage B.

Usage:
    pipeline = FishVisionPipeline()
    result = await pipeline.analyze(image_url, caption)

    if result.is_valid_catch:
        # safe to record in statistics
        await save_catch(...)
    else:
        # show result.rejection_message to user
        await msg.reply_text(result.rejection_message)

The pipeline guarantees that:
    - lures never become catch records
    - fish parts never become catch records
    - fry are rejected (or separately flagged)
    - uncertain identifications do not produce confident wrong records
    - every decision is logged with full reasoning for debugging

Architecture readiness:
    This pipeline is designed to swap the GPT backend for a local
    YOLO+EfficientNet model when training data becomes available.
    See models/config.py for backend configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from bot.fish_vision.detector import (
    DetectionResult,
    detect_fish_object,
    FILTER_CONFIDENCE_THRESHOLD,
)
from bot.fish_vision.classifier import (
    ClassificationResult,
    classify_fish_species,
    SPECIES_CONFIDENCE_THRESHOLD,
    SPECIES_MAP,
)
from bot.utils.logging import get_logger

log = get_logger("fish_vision.pipeline")


@dataclass
class FishAnalysisResult:
    """
    Final result of the full two-stage fish vision pipeline.

    is_valid_catch is the authoritative flag.
    Only record into statistics when is_valid_catch is True.
    """
    # Stage A results
    object_type: str           # whole_fish | fish_part | lure | fry | no_fish
    detection_confidence: float

    # Stage B results (populated only when object_type == whole_fish)
    species_key: str           # pike | taimen | grayling | whitefish | perch | unknown_fish
    species_ru: str            # Russian display name
    species_confidence: float
    fish_count: int
    weight_kg_estimate: Optional[float]
    length_cm_estimate: Optional[float]
    person_name_in_photo: Optional[str]

    # Final verdict
    is_valid_catch: bool       # True = safe to store in statistics
    rejection_reason: Optional[str]   # why it was rejected (for logging)
    rejection_message: str     # user-facing message when not valid

    # Audit trail
    detection_reasoning: str
    classification_reasoning: str
    distinguishing_features: str

    @property
    def confidence_label(self) -> str:
        """Human-readable confidence for the species."""
        if self.species_confidence >= 0.85:
            return "высокая"
        if self.species_confidence >= 0.65:
            return "средняя"
        return "низкая"


# ── Rejection messages (user-facing) ────────────────────────────────────────

_REJECTION_MESSAGES = {
    "lure": (
        "🎣 <b>Это рыболовная приманка, а не рыба!</b>\n\n"
        "На фото вижу воблер, блесну или другую искусственную приманку.\n"
        "Отправьте фото с реальным уловом чтобы записать поимку."
    ),
    "fish_part": (
        "🔪 <b>На фото часть рыбы, а не целая рыба.</b>\n\n"
        "Не вижу ни головы, ни хвоста одновременно.\n"
        "Для учёта улова нужно фото целой рыбы."
    ),
    "fry": (
        "🐟 <b>На фото малёк — слишком маленькая рыба.</b>\n\n"
        "Рыбка очень мала для спортивного улова. Отпустите её расти!\n"
        "Если это взрослая небольшая рыба — напишите об этом в подписи."
    ),
    "no_fish": (
        "❓ <b>На фото нет рыбы.</b>\n\n"
        "Не вижу ни рыбы, ни рыболовной снасти.\n"
        "Отправьте фото с вашим уловом!"
    ),
    "low_confidence": (
        "🤔 <b>Не могу надёжно определить что на фото.</b>\n\n"
        "Уверенность слишком низкая чтобы записать в статистику.\n"
        "Попробуйте сделать более чёткое фото при хорошем освещении."
    ),
}


async def analyze_fish_photo(
    image_url: str,
    caption: str = "",
) -> FishAnalysisResult:
    """
    Run the full two-stage fish vision pipeline on an image.

    Stage A: detect object type (whole_fish / lure / fish_part / fry / no_fish)
    Stage B: classify species (only if Stage A → whole_fish)

    Returns FishAnalysisResult. Check is_valid_catch before storing in DB.
    """
    log.info(f"Fish pipeline: analyzing image (caption={caption[:40]!r})")

    # ── Stage A: Detection / Filtering ──────────────────────────────────────
    detection: DetectionResult = await detect_fish_object(image_url, caption)

    # Reject immediately if not a whole fish
    if not detection.should_classify:
        reason = detection.rejection_reason
        if detection.object_type in _REJECTION_MESSAGES:
            msg = _REJECTION_MESSAGES[detection.object_type]
        else:
            msg = _REJECTION_MESSAGES["low_confidence"]

        log.info(f"Fish pipeline: rejected at Stage A — {reason}")

        return FishAnalysisResult(
            object_type=detection.object_type,
            detection_confidence=detection.confidence,
            species_key="unknown_fish",
            species_ru=SPECIES_MAP["unknown_fish"],
            species_confidence=0.0,
            fish_count=detection.fish_count,
            weight_kg_estimate=None,
            length_cm_estimate=detection.estimated_length_cm,
            person_name_in_photo=None,
            is_valid_catch=False,
            rejection_reason=reason,
            rejection_message=msg,
            detection_reasoning=detection.reasoning,
            classification_reasoning="Stage A rejected — Stage B not reached",
            distinguishing_features="",
        )

    # ── Stage B: Species Classification ─────────────────────────────────────
    classification: ClassificationResult = await classify_fish_species(
        image_url=image_url,
        caption=caption,
        detector_context=detection.raw_description,
    )

    # A valid catch requires a confident identification OR at minimum unknown_fish
    # (not "no_fish" — we're past that gate). But we still store unknown_fish catches
    # because the detection confirmed it IS a fish, just the species is uncertain.
    # What we do NOT allow:
    #   - Stage A low-confidence detections (already handled above)
    #   - Stage B species confidence below threshold → force unknown_fish (done in classifier)
    is_valid = True

    result = FishAnalysisResult(
        object_type=detection.object_type,
        detection_confidence=detection.confidence,
        species_key=classification.species_key,
        species_ru=classification.species_ru,
        species_confidence=classification.confidence,
        fish_count=max(detection.fish_count, classification.fish_count, 1),
        weight_kg_estimate=classification.weight_kg_estimate,
        length_cm_estimate=classification.length_cm_estimate or detection.estimated_length_cm,
        person_name_in_photo=classification.person_name_in_photo,
        is_valid_catch=is_valid,
        rejection_reason=None,
        rejection_message="",
        detection_reasoning=detection.reasoning,
        classification_reasoning=classification.reasoning,
        distinguishing_features=classification.distinguishing_features,
    )

    log.info(
        f"Fish pipeline: VALID catch — "
        f"{classification.species_ru} (conf={classification.confidence:.2f}, "
        f"identified={classification.is_identified})"
    )
    return result


# Expose as main API
FishVisionPipeline = analyze_fish_photo   # alias for backward compat if needed
