"""
Tests for bot/fish_vision/weight_estimator.py

Verifies:
  - Not a fish → no estimate
  - Partial/bad photo → no estimate
  - No measurements → ask for length/girth
  - Length only → range estimate
  - Length + girth → range estimate
  - Never returns a single exact weight (always a range)
  - No estimate for unknown species at low confidence
  - All estimates have non-None min and max
  - min < max for all range estimates
  - Russian and English prompts exist
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bot.fish_vision.weight_estimator import (
    estimate_fish_weight,
    format_weight_reply,
    VisualSizeBucket,
    ImageQuality,
    WeightConfidence,
    EstimationMethod,
)


# ─── Not a fish ───────────────────────────────────────────────────────────────


class TestNotFish:
    def test_not_fish_no_estimate(self):
        r = estimate_fish_weight(is_fish=False)
        assert not r.estimate_available
        assert r.weight_min_kg is None
        assert r.weight_max_kg is None
        assert r.method == EstimationMethod.NO_ESTIMATE
        assert r.confidence == WeightConfidence.NONE

    def test_not_fish_with_length_still_no_estimate(self):
        r = estimate_fish_weight(is_fish=False, length_cm=60, species="pike")
        assert not r.estimate_available

    def test_not_fish_explanation_present(self):
        r = estimate_fish_weight(is_fish=False)
        assert r.explanation
        assert r.explanation_ru


# ─── Partial/bad image ────────────────────────────────────────────────────────


class TestBadImage:
    def test_partial_image_no_estimate(self):
        r = estimate_fish_weight(is_fish=True, image_quality=ImageQuality.PARTIAL)
        assert not r.estimate_available
        assert r.method == EstimationMethod.NO_ESTIMATE

    def test_bad_image_no_estimate(self):
        r = estimate_fish_weight(is_fish=True, image_quality=ImageQuality.BAD)
        assert not r.estimate_available

    def test_fish_not_fully_visible_no_estimate(self):
        r = estimate_fish_weight(is_fish=True, fish_visible_whole=False)
        assert not r.estimate_available


# ─── No measurements → ask for data ──────────────────────────────────────────


class TestNoMeasurements:
    def test_fish_no_measurements_asks_for_length(self):
        r = estimate_fish_weight(is_fish=True)
        assert not r.estimate_available
        assert r.method == EstimationMethod.INSUFFICIENT_DATA
        assert r.user_prompt_for_more_data is not None
        assert r.user_prompt_for_more_data_ru is not None
        assert "length" in r.user_prompt_for_more_data.lower() or "см" in r.user_prompt_for_more_data_ru.lower()

    def test_fish_with_good_quality_no_measurements_still_asks(self):
        r = estimate_fish_weight(is_fish=True, image_quality=ImageQuality.GOOD)
        assert not r.estimate_available
        assert r.method == EstimationMethod.INSUFFICIENT_DATA


# ─── Visual size bucket ───────────────────────────────────────────────────────


class TestVisualSizeBucket:
    @pytest.mark.parametrize("bucket", list(VisualSizeBucket))
    def test_visual_bucket_returns_range(self, bucket):
        r = estimate_fish_weight(is_fish=True, visual_size_bucket=bucket)
        assert r.estimate_available
        assert r.weight_min_kg is not None
        assert r.weight_max_kg is not None
        assert r.weight_min_kg < r.weight_max_kg
        assert r.method == EstimationMethod.ROUGH_VISUAL
        assert r.confidence == WeightConfidence.LOW

    def test_visual_bucket_prompts_for_measurements(self):
        r = estimate_fish_weight(is_fish=True, visual_size_bucket=VisualSizeBucket.MEDIUM)
        assert r.user_prompt_for_more_data is not None
        assert r.user_prompt_for_more_data_ru is not None

    def test_visual_bucket_range_ordering(self):
        """Larger buckets should have higher weight ranges."""
        small = estimate_fish_weight(is_fish=True, visual_size_bucket=VisualSizeBucket.SMALL)
        large = estimate_fish_weight(is_fish=True, visual_size_bucket=VisualSizeBucket.LARGE)
        assert large.weight_max_kg > small.weight_max_kg


# ─── Length only ─────────────────────────────────────────────────────────────


class TestLengthOnly:
    def test_length_returns_range(self):
        r = estimate_fish_weight(is_fish=True, length_cm=50)
        assert r.estimate_available
        assert r.weight_min_kg is not None and r.weight_max_kg is not None
        assert r.weight_min_kg < r.weight_max_kg
        assert r.method == EstimationMethod.LENGTH_SPECIES_FORMULA

    def test_length_with_known_species_medium_confidence(self):
        r = estimate_fish_weight(is_fish=True, length_cm=60, species="pike")
        assert r.confidence == WeightConfidence.MEDIUM
        # Pike at 60cm: roughly 0.5-2 kg expected
        assert r.weight_min_kg < 2.0
        assert r.weight_max_kg > 0.3

    def test_length_with_unknown_species_low_confidence(self):
        r = estimate_fish_weight(is_fish=True, length_cm=60, species="unknown_fish")
        assert r.confidence == WeightConfidence.LOW

    def test_length_no_species_low_confidence(self):
        r = estimate_fish_weight(is_fish=True, length_cm=40)
        assert r.confidence == WeightConfidence.LOW

    def test_longer_fish_heavier_range(self):
        r30 = estimate_fish_weight(is_fish=True, length_cm=30, species="pike")
        r80 = estimate_fish_weight(is_fish=True, length_cm=80, species="pike")
        assert r80.weight_max_kg > r30.weight_max_kg

    def test_length_prompts_for_girth(self):
        r = estimate_fish_weight(is_fish=True, length_cm=50, species="perch")
        assert r.user_prompt_for_more_data is not None
        assert "girth" in r.user_prompt_for_more_data.lower() or "обхват" in (r.user_prompt_for_more_data_ru or "").lower()


# ─── Length + girth ───────────────────────────────────────────────────────────


class TestLengthGirth:
    def test_length_girth_returns_range(self):
        r = estimate_fish_weight(is_fish=True, length_cm=60, girth_cm=28)
        assert r.estimate_available
        assert r.weight_min_kg is not None and r.weight_max_kg is not None
        assert r.weight_min_kg < r.weight_max_kg
        assert r.method == EstimationMethod.LENGTH_GIRTH_FORMULA

    def test_length_girth_medium_confidence(self):
        r = estimate_fish_weight(is_fish=True, length_cm=60, girth_cm=28)
        assert r.confidence == WeightConfidence.MEDIUM

    def test_length_girth_sensible_values(self):
        # 60cm pike with 28cm girth: expect roughly 1-2 kg
        r = estimate_fish_weight(is_fish=True, length_cm=60, girth_cm=28, species="pike")
        assert 0.5 <= r.weight_min_kg <= 2.0
        assert 1.0 <= r.weight_max_kg <= 3.0

    def test_length_girth_no_prompt_needed(self):
        # Already have all data — no further prompt needed
        r = estimate_fish_weight(is_fish=True, length_cm=60, girth_cm=30)
        assert r.user_prompt_for_more_data is None or r.user_prompt_for_more_data == ""

    def test_girth_zero_or_negative_handled(self):
        # Should not crash with unusual inputs
        r = estimate_fish_weight(is_fish=True, length_cm=50, girth_cm=0)
        assert r.weight_min_kg is not None
        assert r.weight_max_kg is not None
        assert r.weight_min_kg <= r.weight_max_kg


# ─── No exact single weight ───────────────────────────────────────────────────


class TestNoExactWeight:
    """Verify we never return min == max (which would be an exact weight)."""

    def test_length_only_not_exact(self):
        r = estimate_fish_weight(is_fish=True, length_cm=50, species="pike")
        assert r.weight_min_kg != r.weight_max_kg

    def test_length_girth_not_exact(self):
        r = estimate_fish_weight(is_fish=True, length_cm=60, girth_cm=25)
        assert r.weight_min_kg != r.weight_max_kg

    def test_visual_bucket_not_exact(self):
        r = estimate_fish_weight(is_fish=True, visual_size_bucket=VisualSizeBucket.MEDIUM)
        assert r.weight_min_kg != r.weight_max_kg


# ─── Bilingual prompts ────────────────────────────────────────────────────────


class TestBilingualOutput:
    def test_insufficient_data_has_russian_prompt(self):
        r = estimate_fish_weight(is_fish=True)
        assert r.explanation_ru
        assert "длин" in r.explanation_ru.lower() or "вес" in r.explanation_ru.lower()

    def test_length_result_has_russian_explanation(self):
        r = estimate_fish_weight(is_fish=True, length_cm=50, species="pike")
        assert r.explanation_ru
        assert r.explanation

    def test_format_weight_reply_russian(self):
        r = estimate_fish_weight(is_fish=True, length_cm=50, species="pike")
        reply_ru = format_weight_reply(r, lang="ru")
        assert reply_ru
        assert len(reply_ru) > 10

    def test_format_weight_reply_english(self):
        r = estimate_fish_weight(is_fish=True, length_cm=50, species="pike")
        reply_en = format_weight_reply(r, lang="en")
        assert reply_en
        assert len(reply_en) > 10

    def test_no_estimate_reply_is_explanation(self):
        r = estimate_fish_weight(is_fish=False)
        reply = format_weight_reply(r, lang="ru")
        assert reply
        # Should return the explanation, not a range
        assert "–" not in reply or "kg" not in reply.lower()


# ─── All known species have sensible estimates ────────────────────────────────


class TestSpeciesCoefficients:
    SPECIES_LENGTH_PAIRS = [
        ("pike", 60),
        ("perch", 25),
        ("grayling", 40),
        ("whitefish", 45),
        ("brown_trout", 50),
        ("rainbow_trout", 45),
        ("atlantic_salmon", 80),
        ("common_carp", 55),
        ("bream", 35),
        ("roach", 25),
        ("ide", 40),
        ("wels_catfish", 100),
    ]

    @pytest.mark.parametrize("species,length", SPECIES_LENGTH_PAIRS)
    def test_species_estimate_positive_range(self, species, length):
        r = estimate_fish_weight(is_fish=True, length_cm=length, species=species)
        assert r.estimate_available
        assert r.weight_min_kg > 0
        assert r.weight_max_kg > 0
        assert r.weight_min_kg < r.weight_max_kg

    @pytest.mark.parametrize("species,length", SPECIES_LENGTH_PAIRS)
    def test_species_estimate_below_100kg(self, species, length):
        r = estimate_fish_weight(is_fish=True, length_cm=length, species=species)
        # Sanity: no freshwater fish weighs > 100 kg at these lengths
        assert r.weight_max_kg < 100.0
