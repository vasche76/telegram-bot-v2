"""
intake_constants.py — Shared path and class constants for the Telegram export intake pipeline.

source=telegram_private_2026-04-24, license=private_training_only
"""

from __future__ import annotations

import sys
from pathlib import Path

# ─── Repo layout ─────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
SCRIPTS_DIR = REPO_ROOT / "scripts"

# ─── Export source ────────────────────────────────────────────────────────────

EXPORT_DIR = Path.home() / "Downloads" / "Telegram Desktop" / "ChatExport_2026-04-24"
BATCH_ID = "tg_2026-04-24"

# ─── Output roots ─────────────────────────────────────────────────────────────
#
# INTAKE_META_ROOT: partially tracked — privacy-safe summary files (e.g.
#   manifest_summary.json) are committed; full manifests (manifest.jsonl)
#   are gitignored because they contain captions and sender names.
#   Git does not recurse into an already-ignored directory, so a !negation
#   rule inside data/intake/ would be silently ineffective. Storing metadata
#   in data/intake_meta/ (a separate directory) is the clean solution.
#
# INTAKE_CANDIDATES_ROOT: gitignored via data/intake/ blanket rule — photo
#   copies only, never committed.

INTAKE_META_ROOT = DATA_ROOT / "intake_meta" / BATCH_ID
INTAKE_CANDIDATES_ROOT = DATA_ROOT / "intake" / BATCH_ID / "candidates"

# ─── Manifest paths ───────────────────────────────────────────────────────────

MANIFEST_PATH = INTAKE_META_ROOT / "manifest.jsonl"          # local-only (contains captions)
MANIFEST_SUMMARY_PATH = INTAKE_META_ROOT / "manifest_summary.json"  # tracked (aggregate only)
AUDIT_PATH = INTAKE_META_ROOT / "audit.jsonl"
DEDUP_PATH = INTAKE_META_ROOT / "dedup_clusters.jsonl"
CLASSIFICATION_PATH = INTAKE_META_ROOT / "classification.jsonl"
PRIVACY_FLAGS_PATH = INTAKE_META_ROOT / "privacy_flags.jsonl"
STATS_REPORT_PATH = INTAKE_META_ROOT / "stats_report.txt"
CANDIDATES_MANIFEST_PATH = INTAKE_CANDIDATES_ROOT / "MANIFEST.json"

# ─── Provenance tags ──────────────────────────────────────────────────────────

SOURCE_TAG = "telegram_private_2026-04-24"
LICENSE_TAG = "private_training_only"

# ─── Canonical class lists ────────────────────────────────────────────────────
# Import from validate_dataset.py to stay in sync with the authoritative lists.

sys.path.insert(0, str(SCRIPTS_DIR))
try:
    from validate_dataset import STAGE_A_CLASSES, STAGE_B_SPECIES  # type: ignore
except ImportError:
    # Fallback in case the import fails in isolated test environments.
    STAGE_A_CLASSES = ["whole_fish", "lure", "fish_part", "fry", "no_fish"]
    STAGE_B_SPECIES = [
        "pike", "taimen", "grayling", "whitefish", "perch",
        "brown_trout", "rainbow_trout", "atlantic_salmon",
        "common_carp", "crucian_carp", "bream", "roach", "ide",
        "wels_catfish", "unknown_fish",
    ]

STAGE_A_CLASSES_SET: set[str] = set(STAGE_A_CLASSES)
STAGE_B_SPECIES_SET: set[str] = set(STAGE_B_SPECIES)

# ─── U4 coarse-filter paths ───────────────────────────────────────────────────

FILTER_UNIVERSE_PATH = INTAKE_META_ROOT / "filter_universe.jsonl"           # local-only
FILTER_UNIVERSE_SUMMARY_PATH = INTAKE_META_ROOT / "filter_universe_summary.json"  # tracked
FILTER_SIGNALS_PATH = INTAKE_META_ROOT / "filter_signals.jsonl"             # local-only
FILTER_SIGNALS_SUMMARY_PATH = INTAKE_META_ROOT / "filter_signals_summary.json"    # tracked
FILTER_CANDIDATES_PATH = INTAKE_META_ROOT / "filter_candidates.jsonl"       # local-only
FILTER_CANDIDATES_SUMMARY_PATH = INTAKE_META_ROOT / "filter_candidates_summary.json"  # tracked
FILTER_REVIEW_SUMMARY_PATH = INTAKE_META_ROOT / "filter_review_summary.json"       # tracked
FILTER_REVIEW_PARTIAL_SUMMARY_PATH = INTAKE_META_ROOT / "filter_review_partial_summary.json"  # tracked
FILTER_REVIEW_DIR = INTAKE_META_ROOT / "review"                              # already gitignored

# ─── U4 Phase E reviewed-seed paths ──────────────────────────────────────────
# Privacy-safe summary and manifest are tracked (counts only, no private fields).
# Record-level JSONL files are gitignored (contain review IDs).

REVIEWED_SEED_DIR = INTAKE_META_ROOT / "reviewed_seed"
REVIEWED_SEED_MANIFEST_PATH = REVIEWED_SEED_DIR / "reviewed_seed_manifest.json"       # tracked
REVIEWED_SEED_SUMMARY_PATH = REVIEWED_SEED_DIR / "reviewed_seed_summary.json"         # tracked
REVIEWED_SEED_READINESS_PATH = REVIEWED_SEED_DIR / "reviewed_seed_mvp_readiness.md"   # tracked
REVIEWED_SEED_RECORDS_PATH = REVIEWED_SEED_DIR / "reviewed_seed_records.jsonl"        # gitignored
REVIEWED_SEED_EXCLUDED_PATH = REVIEWED_SEED_DIR / "reviewed_seed_excluded.jsonl"      # gitignored

# ─── U4 Phase D review paths (all local-only / gitignored) ───────────────────

REVIEW_SECRET_PATH = FILTER_REVIEW_DIR / ".review_secret"
REVIEW_ASSETS_BASE = FILTER_REVIEW_DIR / "assets"

# ─── U4 Phase D decision schema version ──────────────────────────────────────

REVIEW_SCHEMA_VERSION: str = "u4_phase_d_decisions_v1"
REVIEW_PHASE: str = "U4_PHASE_D_MANUAL_REVIEW"
REVIEW_PHASE_AL: str = "AL_NEGATIVE_REVIEW"
REVIEW_PHASES_ALLOWED: frozenset[str] = frozenset({REVIEW_PHASE, REVIEW_PHASE_AL})
REVIEW_CREATED_BY: str = "local_manual_review"

# ─── U4 Phase D allowed decision types ───────────────────────────────────────

DECISION_TYPE_KEEP: str = "KEEP"
DECISION_TYPE_REMOVE: str = "REMOVE"
DECISION_TYPE_RELABEL: str = "RELABEL"
DECISION_TYPE_UNSURE: str = "UNSURE"
DECISION_TYPES: list[str] = [
    DECISION_TYPE_KEEP,
    DECISION_TYPE_REMOVE,
    DECISION_TYPE_RELABEL,
    DECISION_TYPE_UNSURE,
]
DECISION_TYPES_SET: frozenset[str] = frozenset(DECISION_TYPES)

# ─── U4 Phase D final category enum ──────────────────────────────────────────

FINAL_CATEGORY_FISH: str = "fish"
FINAL_CATEGORY_FISH_PART: str = "fish_part"
FINAL_CATEGORY_FRY_JUVENILE: str = "fry_juvenile"
FINAL_CATEGORY_LURE_GEAR: str = "lure_gear"
FINAL_CATEGORY_NO_FISH: str = "no_fish"
FINAL_CATEGORY_POSTER_SCREENSHOT: str = "poster_screenshot"
FINAL_CATEGORY_BAD_QUALITY: str = "bad_quality"
FINAL_CATEGORY_OUT_OF_SCOPE: str = "out_of_scope"
FINAL_CATEGORY_DUPLICATE_SUSPECT: str = "duplicate_suspect"
FINAL_CATEGORY_UNSURE: str = "unsure"

FINAL_CATEGORIES: list[str] = [
    FINAL_CATEGORY_FISH,
    FINAL_CATEGORY_FISH_PART,
    FINAL_CATEGORY_FRY_JUVENILE,
    FINAL_CATEGORY_LURE_GEAR,
    FINAL_CATEGORY_NO_FISH,
    FINAL_CATEGORY_POSTER_SCREENSHOT,
    FINAL_CATEGORY_BAD_QUALITY,
    FINAL_CATEGORY_OUT_OF_SCOPE,
    FINAL_CATEGORY_DUPLICATE_SUSPECT,
    FINAL_CATEGORY_UNSURE,
]
FINAL_CATEGORIES_SET: frozenset[str] = frozenset(FINAL_CATEGORIES)

REMOVE_ALLOWED_CATEGORIES: frozenset[str] = frozenset({
    FINAL_CATEGORY_NO_FISH,
    FINAL_CATEGORY_LURE_GEAR,
    FINAL_CATEGORY_POSTER_SCREENSHOT,
    FINAL_CATEGORY_BAD_QUALITY,
    FINAL_CATEGORY_OUT_OF_SCOPE,
    FINAL_CATEGORY_DUPLICATE_SUSPECT,
})

# Categories that make a reviewed record ineligible for training even at high confidence.
TRAINING_INELIGIBLE_CATEGORIES: frozenset[str] = frozenset({
    FINAL_CATEGORY_UNSURE,
    FINAL_CATEGORY_BAD_QUALITY,
    FINAL_CATEGORY_DUPLICATE_SUSPECT,
    FINAL_CATEGORY_OUT_OF_SCOPE,
})

# Fish-class categories (whole fish, parts, juveniles) — training-positive examples.
TRAINING_FISH_CATEGORIES: frozenset[str] = frozenset({
    FINAL_CATEGORY_FISH,
    FINAL_CATEGORY_FISH_PART,
    FINAL_CATEGORY_FRY_JUVENILE,
})

# ─── U4 Phase D human confidence scale ───────────────────────────────────────

CONFIDENCE_MIN: int = 1
CONFIDENCE_MAX: int = 5
CONFIDENCE_KEEP_MIN: int = 3
CONFIDENCE_UNSURE_MAX: int = 3

# ─── U4 Phase D batch defaults ───────────────────────────────────────────────

REVIEW_BATCH_SIZE_DEFAULT: int = 250

# ─── U4 Phase D review summary schema ────────────────────────────────────────

REVIEW_SUMMARY_SCHEMA_VERSION: str = "u4_phase_d_summary_v1"

# ─── Phase C → Phase D category name normalization ────────────────────────────
# Phase C used longer names that were renamed for Phase D. Map old → canonical.

PHASE_C_TO_FINAL_CATEGORY_MAP: dict[str, str] = {
    "lure_fishing_gear": FINAL_CATEGORY_LURE_GEAR,
}


def normalize_phase_c_category(cat: str) -> str:
    """Return the canonical Phase D category name for a Phase C category string."""
    return PHASE_C_TO_FINAL_CATEGORY_MAP.get(cat, cat)


# ─── U4 coarse category constants ─────────────────────────────────────────────
# Priority-ordered. unknown_needs_review is the catch-all and MUST remain last.

COARSE_CATEGORIES: list[str] = [
    "fish",
    "no_fish",
    "lure_fishing_gear",
    "fish_part",
    "fry_juvenile",
    "poster_screenshot",
    "unknown_needs_review",
]
COARSE_CATEGORIES_SET: set[str] = set(COARSE_CATEGORIES)

# ─── U4 Phase C candidate classification constants ────────────────────────────

# Confidence levels
CONFIDENCE_HIGH: str = "high"
CONFIDENCE_MEDIUM: str = "medium"
CONFIDENCE_LOW: str = "low"

# Reason codes (surfaced in candidate records — no PII)
REASON_CAPTION_NO_FISH_KW: str = "caption_no_fish_keyword"
REASON_CAPTION_TEXT_HEAVY: str = "caption_text_heavy"
REASON_CAPTION_LURE_HINT: str = "caption_lure_hint"
REASON_CAPTION_FISH_PART_KW: str = "caption_fish_part_keyword"
REASON_CAPTION_FRY_KW: str = "caption_fry_keyword"
REASON_LOW_RES: str = "low_res"
REASON_TINY_FILE: str = "tiny_file"
REASON_EXTREME_ASPECT: str = "extreme_aspect"
REASON_NO_STRONG_SIGNAL: str = "no_strong_signal"

# Conflict codes
CONFLICT_AMBIGUOUS_TEXT_HEAVY_LURE: str = "ambiguous_text_heavy_lure"
CONFLICT_COMPETING_CAPTION_KEYWORDS: str = "competing_caption_keywords"

# ─── U4 Phase B heuristic constants ──────────────────────────────────────────

# File-size bucket thresholds (calibrated from actual corpus):
# p25=161KB, p50=214KB, p75=278KB, p95=393KB
FILE_SIZE_BUCKET_TINY_MAX: int = 50_000     # bytes; < 50 KB  → "tiny"  (0.4% of corpus)
FILE_SIZE_BUCKET_SMALL_MAX: int = 150_000   # bytes; < 150 KB → "small" (~25% of corpus)
FILE_SIZE_BUCKET_MEDIUM_MAX: int = 400_000  # bytes; < 400 KB → "medium" (~71% of corpus)
                                             # ≥ 400 KB → "large" (~4% of corpus)

# Caption text-heavy threshold
CAPTION_TEXT_HEAVY_THRESHOLD: int = 200  # chars; > 200 → caption_text_heavy=True

# Caption keyword hint frozensets — weak signals only, NOT truth labels.
# Conservative lists: high-FPR terms (резина, силикон, снасть, приглашаем) excluded.
# This corpus is a fishing-club *reporting* channel — lure names appear in catch reports.

# Lure/gear-specific product names only.
CAPTION_LURE_KEYWORD_HINTS: frozenset[str] = frozenset({
    "воблер",    # wobbler / crankbait
    "воблеры",
    "блесна",    # spoon lure
    "блёсны",
    "мормышка",  # jig / ice fishing lure
    "балансир",  # balance jig (ice fishing)
    "спиннер",   # spinner
    "раттлин",   # rattlin lure
    "раклин",    # rattlin variant
    "твистер",   # soft twister body (less risky than "резина")
    "топвотер",  # topwater
})

# Fish-processing / filleting terms.
# EXCLUDE: икра (roe — appears in lure descriptions too),
#          чистк (appears in prize captions: "сертификат на чистку улова")
CAPTION_FISH_PART_KEYWORD_HINTS: frozenset[str] = frozenset({
    "разделк",   # filleting/processing (substring: разделка, разделки)
    "потрош",    # gutting (substring: потрошить, потрошение)
    "хребет",    # backbone/spine
    "жабры",     # gills
    "филе",      # fillet
    "плавник",   # fin
})

# Juvenile / fry fish terms — specific enough to have low FPR.
CAPTION_FRY_KEYWORD_HINTS: frozenset[str] = frozenset({
    "малёк",     # fry (singular)
    "мальки",    # fry (plural nominative)
    "мальков",   # fry (genitive)
    "малькам",   # fry (dative)
    "мальками",  # fry (instrumental)
    "молодь",    # juvenile (collective)
    "сеголеток", # fish of the year / fingerling
    "молодняк",  # young stock
})

# Announcement / administrative content terms.
# EXCLUDE: конкурс, турнир, соревнование, приглашаем — appear with fish photos.
CAPTION_NO_FISH_KEYWORD_HINTS: frozenset[str] = frozenset({
    "объявление",    # announcement / notice
    "расписание",    # schedule / timetable
    "записывайтесь", # sign up / register
    "регистрация",   # registration
    "афиша",         # event poster / program
    "анонс",         # promo / announcement
})
