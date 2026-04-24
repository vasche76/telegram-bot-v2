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
