"""
weight_estimator.py — Conservative fish weight estimation.

Produces weight ranges, not exact values. Never claims scientific precision.
Uses formula-based estimation from species + length + girth where available,
with graceful degradation when data is missing.

Design principles:
  - Always return a range, never a single weight
  - Lower confidence = wider range
  - Unknown species → lower confidence
  - No measurements → ask user for length/girth
  - Visual size bucket → very rough range only (low confidence)
  - Never claim image-only weight accuracy

Formulas used (all approximate, marked as such):
  - Length-girth cylinder: W ≈ (L × G²) / 800  (kg, cm)
    A conservative approximation for moderately fusiform fish.
  - Length-only generic: W ≈ a × (L/100)^3  where a is species condition factor
    Condition factors (k) from FishBase ranges, used as approximate only.
  - Visual bucket: preset conservative ranges by perceived size

All coefficients are stored in SPECIES_CONFIG and can be refined with better data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─── Enums ────────────────────────────────────────────────────────────────────


class VisualSizeBucket(str, Enum):
    """Perceived size of fish from photo, without scale reference."""
    TINY = "tiny"         # < ~20 cm / < 0.1 kg (fry, small juveniles)
    SMALL = "small"       # ~20–35 cm / 0.05–0.5 kg
    MEDIUM = "medium"     # ~35–55 cm / 0.3–2 kg
    LARGE = "large"       # ~55–75 cm / 1.5–5 kg
    VERY_LARGE = "very_large"  # > ~75 cm / 4+ kg


class ImageQuality(str, Enum):
    GOOD = "good"
    PARTIAL = "partial"    # partial fish visible
    BAD = "bad"            # blurry, out of frame, very dark


class WeightConfidence(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EstimationMethod(str, Enum):
    NO_ESTIMATE = "no_estimate"
    INSUFFICIENT_DATA = "insufficient_data"
    ROUGH_VISUAL = "rough_visual"
    LENGTH_SPECIES_FORMULA = "length_species_formula"
    LENGTH_GIRTH_FORMULA = "length_girth_formula"


# ─── Species config ───────────────────────────────────────────────────────────


@dataclass
class SpeciesWeightConfig:
    """Approximate weight parameters for a species. All values are ranges."""
    # Condition factor range for W = k * L^3 formula (L in cm, W in grams; k * 1e-5 roughly)
    # Using a simplified: W_kg ≈ condition_factor * (L_cm / 100)^3
    # condition_factor is approximate kg/m^3 equivalent
    condition_factor_min: float  # lower bound
    condition_factor_max: float  # upper bound
    # Typical length ranges (cm) for adults
    typical_length_min_cm: float
    typical_length_max_cm: float
    # Display name
    name_en: str
    name_ru: str


# Approximate condition factors derived from published FishBase weight-length data.
# All values are approximate and should only be used to produce conservative ranges.
# Source: FishBase weight-length relationships (multiple studies, approximate).
_SPECIES_CONFIG: dict[str, SpeciesWeightConfig] = {
    "pike": SpeciesWeightConfig(
        condition_factor_min=3500, condition_factor_max=6000,
        typical_length_min_cm=40, typical_length_max_cm=110,
        name_en="Pike", name_ru="Щука",
    ),
    "perch": SpeciesWeightConfig(
        condition_factor_min=8000, condition_factor_max=14000,
        typical_length_min_cm=15, typical_length_max_cm=50,
        name_en="Perch", name_ru="Окунь",
    ),
    "grayling": SpeciesWeightConfig(
        condition_factor_min=4000, condition_factor_max=7000,
        typical_length_min_cm=25, typical_length_max_cm=55,
        name_en="Grayling", name_ru="Хариус",
    ),
    "whitefish": SpeciesWeightConfig(
        condition_factor_min=4000, condition_factor_max=7500,
        typical_length_min_cm=30, typical_length_max_cm=65,
        name_en="Whitefish", name_ru="Сиг",
    ),
    "brown_trout": SpeciesWeightConfig(
        condition_factor_min=5000, condition_factor_max=9000,
        typical_length_min_cm=25, typical_length_max_cm=80,
        name_en="Brown Trout", name_ru="Кумжа / ручьевая форель",
    ),
    "rainbow_trout": SpeciesWeightConfig(
        condition_factor_min=5500, condition_factor_max=9500,
        typical_length_min_cm=25, typical_length_max_cm=75,
        name_en="Rainbow Trout", name_ru="Радужная форель",
    ),
    "atlantic_salmon": SpeciesWeightConfig(
        condition_factor_min=6000, condition_factor_max=10000,
        typical_length_min_cm=50, typical_length_max_cm=120,
        name_en="Atlantic Salmon", name_ru="Атлантический лосось",
    ),
    "common_carp": SpeciesWeightConfig(
        condition_factor_min=14000, condition_factor_max=22000,
        typical_length_min_cm=30, typical_length_max_cm=80,
        name_en="Common Carp", name_ru="Карп / сазан",
    ),
    "crucian_carp": SpeciesWeightConfig(
        condition_factor_min=10000, condition_factor_max=18000,
        typical_length_min_cm=15, typical_length_max_cm=40,
        name_en="Crucian Carp", name_ru="Карась",
    ),
    "bream": SpeciesWeightConfig(
        condition_factor_min=9000, condition_factor_max=15000,
        typical_length_min_cm=20, typical_length_max_cm=55,
        name_en="Bream", name_ru="Лещ",
    ),
    "roach": SpeciesWeightConfig(
        condition_factor_min=7000, condition_factor_max=12000,
        typical_length_min_cm=15, typical_length_max_cm=40,
        name_en="Roach", name_ru="Плотва",
    ),
    "ide": SpeciesWeightConfig(
        condition_factor_min=7000, condition_factor_max=12000,
        typical_length_min_cm=25, typical_length_max_cm=60,
        name_en="Ide", name_ru="Язь",
    ),
    "wels_catfish": SpeciesWeightConfig(
        condition_factor_min=4000, condition_factor_max=8000,
        typical_length_min_cm=60, typical_length_max_cm=200,
        name_en="Wels Catfish", name_ru="Сом",
    ),
    "taimen": SpeciesWeightConfig(
        condition_factor_min=5000, condition_factor_max=9000,
        typical_length_min_cm=60, typical_length_max_cm=150,
        name_en="Taimen", name_ru="Таймень",
    ),
}

# Rough visual size ranges (conservative: min well below typical, max well above)
_VISUAL_BUCKET_RANGES: dict[VisualSizeBucket, tuple[float, float]] = {
    VisualSizeBucket.TINY:       (0.01, 0.15),
    VisualSizeBucket.SMALL:      (0.05, 0.60),
    VisualSizeBucket.MEDIUM:     (0.20, 2.50),
    VisualSizeBucket.LARGE:      (1.00, 6.00),
    VisualSizeBucket.VERY_LARGE: (3.00, 30.0),
}


# ─── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class WeightEstimateResult:
    estimate_available: bool
    weight_min_kg: Optional[float]
    weight_max_kg: Optional[float]
    confidence: WeightConfidence
    method: EstimationMethod
    explanation: str
    explanation_ru: str
    user_prompt_for_more_data: Optional[str]
    user_prompt_for_more_data_ru: Optional[str]

    def as_dict(self) -> dict:
        return {
            "estimate_available": self.estimate_available,
            "weight_min_kg": self.weight_min_kg,
            "weight_max_kg": self.weight_max_kg,
            "confidence": self.confidence.value if self.confidence else None,
            "method": self.method.value if self.method else None,
            "explanation": self.explanation,
            "explanation_ru": self.explanation_ru,
            "user_prompt_for_more_data": self.user_prompt_for_more_data,
            "user_prompt_for_more_data_ru": self.user_prompt_for_more_data_ru,
        }


# ─── Pre-built no-estimate responses ─────────────────────────────────────────


def _no_estimate(reason: str, reason_ru: str) -> WeightEstimateResult:
    return WeightEstimateResult(
        estimate_available=False,
        weight_min_kg=None,
        weight_max_kg=None,
        confidence=WeightConfidence.NONE,
        method=EstimationMethod.NO_ESTIMATE,
        explanation=reason,
        explanation_ru=reason_ru,
        user_prompt_for_more_data=None,
        user_prompt_for_more_data_ru=None,
    )


def _ask_for_measurements() -> WeightEstimateResult:
    return WeightEstimateResult(
        estimate_available=False,
        weight_min_kg=None,
        weight_max_kg=None,
        confidence=WeightConfidence.NONE,
        method=EstimationMethod.INSUFFICIENT_DATA,
        explanation=(
            "Weight from a photo is only a rough estimate. "
            "For a better estimate, send the fish length in cm and, "
            "if possible, the girth at the thickest point."
        ),
        explanation_ru=(
            "Вес по фото можно оценить только грубо. "
            "Для более точной оценки пришли длину рыбы в см и, "
            "если можешь, обхват в самой толстой части."
        ),
        user_prompt_for_more_data="Please send the fish length in cm (e.g. 45 cm).",
        user_prompt_for_more_data_ru="Пришли длину рыбы в сантиметрах (например: 45 см).",
    )


# ─── Estimation helpers ───────────────────────────────────────────────────────


def _round_range(lo: float, hi: float, sig: int = 2) -> tuple[float, float]:
    """Round range endpoints to sig significant figures."""
    def _round_sig(x: float) -> float:
        if x <= 0:
            return x
        magnitude = math.floor(math.log10(x))
        factor = 10 ** (sig - 1 - magnitude)
        return round(x * factor) / factor

    return _round_sig(lo), _round_sig(hi)


def _length_species_range(length_cm: float, species: str) -> tuple[float, float]:
    """
    W_kg ≈ k * (L_cm/100)^3 where k is species condition factor (in grams/m^3).
    condition_factor units: grams; divide by 1000 for kg result.
    """
    cfg = _SPECIES_CONFIG.get(species)
    if cfg is None:
        k_min, k_max = 4000.0, 12000.0  # generic freshwater fallback
    else:
        k_min, k_max = cfg.condition_factor_min, cfg.condition_factor_max
    l_m = length_cm / 100.0
    lo_g = k_min * (l_m ** 3)
    hi_g = k_max * (l_m ** 3)
    # Convert grams → kg, then widen by ±25% for conservatism
    return _round_range((lo_g / 1000.0) * 0.75, (hi_g / 1000.0) * 1.25)


def _length_girth_range(length_cm: float, girth_cm: float) -> tuple[float, float]:
    """
    Metric length-girth formula: W_kg ≈ (L_cm × G_cm²) / 29000
    Derived from the standard angling formula W_lbs = (L_in × G_in²) / 800
    converted to metric (÷ 2.54³ for cm, ÷ 2.2046 for kg → denominator ≈ 28900).
    Apply ±20% conservative bounds.
    """
    nominal_kg = (length_cm * (girth_cm ** 2)) / 29000.0
    return _round_range(nominal_kg * 0.80, nominal_kg * 1.20)


# ─── Main estimator ───────────────────────────────────────────────────────────


def estimate_fish_weight(
    *,
    is_fish: bool,
    species: Optional[str] = None,
    length_cm: Optional[float] = None,
    girth_cm: Optional[float] = None,
    fish_visible_whole: Optional[bool] = None,
    has_scale_reference: Optional[bool] = None,
    visual_size_bucket: Optional[VisualSizeBucket] = None,
    image_quality: Optional[ImageQuality] = None,
) -> WeightEstimateResult:
    """
    Estimate fish weight conservatively.

    Returns a range with explicit uncertainty — never a single exact value.
    Asks for more data when estimates would be unreliable.
    """
    # Not a fish → no estimate
    if not is_fish:
        return _no_estimate(
            "Not a fish — no weight estimate.",
            "Это не рыба — оценка веса недоступна.",
        )

    # Partial or bad image → no reliable estimate
    if image_quality in (ImageQuality.PARTIAL, ImageQuality.BAD):
        return _no_estimate(
            "Fish is not fully visible or image quality is too low for any reliable estimate.",
            "Рыба видна частично или качество фото слишком низкое — надёжная оценка невозможна.",
        )

    if fish_visible_whole is False:
        return _no_estimate(
            "Fish is not fully visible — weight estimate not available.",
            "Рыба видна не полностью — оценка веса недоступна.",
        )

    # Length + girth → best formula estimate
    if length_cm is not None and girth_cm is not None:
        lo, hi = _length_girth_range(length_cm, girth_cm)
        species_note = (
            f" (species: {_SPECIES_CONFIG[species].name_en})"
            if species in _SPECIES_CONFIG
            else ""
        )
        species_note_ru = (
            f" (вид: {_SPECIES_CONFIG[species].name_ru})"
            if species in _SPECIES_CONFIG
            else ""
        )
        return WeightEstimateResult(
            estimate_available=True,
            weight_min_kg=lo,
            weight_max_kg=hi,
            confidence=WeightConfidence.MEDIUM,
            method=EstimationMethod.LENGTH_GIRTH_FORMULA,
            explanation=(
                f"Length-girth estimate{species_note}: approximately {lo}–{hi} kg. "
                "This is an approximate range (±20%). Actual weight may differ."
            ),
            explanation_ru=(
                f"Оценка по длине и обхвату{species_note_ru}: примерно {lo}–{hi} кг. "
                "Это приблизительный диапазон (±20%). Реальный вес может отличаться."
            ),
            user_prompt_for_more_data=None,
            user_prompt_for_more_data_ru=None,
        )

    # Length + species → formula estimate
    if length_cm is not None:
        lo, hi = _length_species_range(length_cm, species or "unknown")
        known_species = species in _SPECIES_CONFIG
        confidence = WeightConfidence.LOW if not known_species else WeightConfidence.MEDIUM
        sp_label = (
            _SPECIES_CONFIG[species].name_en if known_species else "unknown species"
        )
        sp_label_ru = (
            _SPECIES_CONFIG[species].name_ru if known_species else "неизвестный вид"
        )
        return WeightEstimateResult(
            estimate_available=True,
            weight_min_kg=lo,
            weight_max_kg=hi,
            confidence=confidence,
            method=EstimationMethod.LENGTH_SPECIES_FORMULA,
            explanation=(
                f"Length-based estimate for {sp_label} at {length_cm} cm: "
                f"approximately {lo}–{hi} kg. "
                "Wide range due to individual variation. "
                "For a better estimate, also send the girth at the thickest point."
            ),
            explanation_ru=(
                f"Оценка по длине для {sp_label_ru} ({length_cm} см): "
                f"примерно {lo}–{hi} кг. "
                "Диапазон широкий из-за индивидуальной изменчивости. "
                "Для уточнения пришли обхват в самой толстой части."
            ),
            user_prompt_for_more_data=(
                "For a tighter estimate, send the girth at the thickest point in cm."
            ),
            user_prompt_for_more_data_ru=(
                "Для уточнения пришли обхват рыбы в самой широкой части (в сантиметрах)."
            ),
        )

    # Visual size bucket (no measurements) → very rough range, low confidence
    if visual_size_bucket is not None:
        lo, hi = _VISUAL_BUCKET_RANGES[visual_size_bucket]
        return WeightEstimateResult(
            estimate_available=True,
            weight_min_kg=lo,
            weight_max_kg=hi,
            confidence=WeightConfidence.LOW,
            method=EstimationMethod.ROUGH_VISUAL,
            explanation=(
                f"Very rough visual estimate ({visual_size_bucket.value} fish): "
                f"approximately {lo}–{hi} kg. "
                "Highly uncertain — photo-based size is unreliable without a scale reference."
            ),
            explanation_ru=(
                f"Очень грубая оценка по размеру на фото ({visual_size_bucket.value}): "
                f"примерно {lo}–{hi} кг. "
                "Большая погрешность — без масштаба оценить вес по фото сложно."
            ),
            user_prompt_for_more_data=(
                "For a better estimate, send the fish length in cm "
                "and the girth at the thickest point."
            ),
            user_prompt_for_more_data_ru=(
                "Пришли длину рыбы в сантиметрах и обхват в самой широкой части "
                "— это поможет дать более точную оценку."
            ),
        )

    # No usable data → ask for measurements
    return _ask_for_measurements()


# ─── Convenience formatter ────────────────────────────────────────────────────


def format_weight_reply(result: WeightEstimateResult, lang: str = "ru") -> str:
    """Format a weight estimate result as a bot-ready message."""
    if not result.estimate_available:
        if lang == "ru":
            return result.explanation_ru or result.explanation
        return result.explanation

    if lang == "ru":
        prompt = (
            f"\n\n{result.user_prompt_for_more_data_ru}"
            if result.user_prompt_for_more_data_ru
            else ""
        )
        return (
            f"⚖️ {result.explanation_ru}"
            f"{prompt}"
        )
    else:
        prompt = (
            f"\n\n{result.user_prompt_for_more_data}"
            if result.user_prompt_for_more_data
            else ""
        )
        return (
            f"⚖️ {result.explanation}"
            f"{prompt}"
        )
