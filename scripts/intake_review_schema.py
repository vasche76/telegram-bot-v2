"""
intake_review_schema.py — Schema validation for U4 Phase D manual review decisions.

Enforces:
- Required field presence
- Enum validity (decision_type, final_category)
- Confidence range
- Decision consistency rules (KEEP / REMOVE / RELABEL / UNSURE)
- Privacy scan (notes field must not contain forbidden metadata patterns)

source=telegram_private_2026-04-24, license=private_training_only
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import intake_constants as C

# ─── Required record fields ───────────────────────────────────────────────────

REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "review_id",
    "decision_type",
    "phase_c_category",
    "final_category",
    "human_confidence",
    "refinement",
    "notes",
    "reviewed_at",
)

REQUIRED_FILE_FIELDS: tuple[str, ...] = (
    "schema_version",
    "source",
    "phase",
    "run_id",
    "batch_id",
    "created_by",
    "records",
)

# ─── Privacy: patterns that must never appear in notes ───────────────────────

# Raw filename paths, Telegram message IDs, sender tokens, etc.
# We use conservative substring patterns that should not appear in legitimate review notes.
_PRIVACY_FORBIDDEN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"photos/photo_", re.IGNORECASE),          # raw Telegram filename prefix
    re.compile(r"photo_\d+@\d{2}-\d{2}-\d{4}", re.IGNORECASE),  # Telegram filename pattern
    re.compile(r"\bsha256\b", re.IGNORECASE),             # raw hash reference
    re.compile(r"\b[0-9a-f]{64}\b"),                      # 64-hex SHA256 value
    re.compile(r"\bmsg_id\b|\bmessage_id\b", re.IGNORECASE),
    re.compile(r"\bsender_id\b|\buser_id\b|\bchat_id\b", re.IGNORECASE),
    re.compile(r"\bfrom_id\b|\bpeer_id\b", re.IGNORECASE),
    re.compile(r"ChatExport_", re.IGNORECASE),
    re.compile(r"telegram desktop", re.IGNORECASE),
]


def scan_notes_for_privacy_leak(notes: str | None) -> list[str]:
    """Return list of violation descriptions if notes contains forbidden patterns."""
    if not notes:
        return []
    violations: list[str] = []
    for pattern in _PRIVACY_FORBIDDEN_PATTERNS:
        if pattern.search(notes):
            violations.append(f"forbidden pattern '{pattern.pattern}' in notes")
    return violations


# ─── File-level validation ────────────────────────────────────────────────────


def validate_decision_file(data: dict[str, Any]) -> list[str]:
    """
    Validate the top-level decision file structure.
    Returns a list of error strings (empty = valid).
    """
    errors: list[str] = []

    for field in REQUIRED_FILE_FIELDS:
        if field not in data:
            errors.append(f"missing required file field: '{field}'")

    if data.get("schema_version") != C.REVIEW_SCHEMA_VERSION:
        errors.append(
            f"schema_version mismatch: expected '{C.REVIEW_SCHEMA_VERSION}', "
            f"got '{data.get('schema_version')}'"
        )
    if data.get("phase") not in C.REVIEW_PHASES_ALLOWED:
        errors.append(
            f"phase mismatch: expected one of {sorted(C.REVIEW_PHASES_ALLOWED)}, "
            f"got '{data.get('phase')}'"
        )
    if not isinstance(data.get("records"), list):
        errors.append("'records' must be a list")

    return errors


# ─── Record-level validation ──────────────────────────────────────────────────


def validate_record_schema(rec: dict[str, Any]) -> list[str]:
    """
    Validate a single decision record against the schema.
    Returns list of error strings (empty = valid).
    Does NOT check decision consistency — use validate_record_consistency for that.
    """
    errors: list[str] = []

    for field in REQUIRED_RECORD_FIELDS:
        if field not in rec:
            errors.append(f"missing required field: '{field}'")

    review_id = rec.get("review_id", "")
    if not isinstance(review_id, str) or not review_id.startswith("rv_"):
        errors.append(f"review_id must be a string starting with 'rv_', got: {review_id!r}")

    dt = rec.get("decision_type")
    if dt not in C.DECISION_TYPES_SET:
        errors.append(f"invalid decision_type: {dt!r} (allowed: {sorted(C.DECISION_TYPES_SET)})")

    fc = rec.get("final_category")
    if fc not in C.FINAL_CATEGORIES_SET:
        errors.append(f"invalid final_category: {fc!r} (allowed: {sorted(C.FINAL_CATEGORIES_SET)})")

    conf = rec.get("human_confidence")
    if not isinstance(conf, int) or not (C.CONFIDENCE_MIN <= conf <= C.CONFIDENCE_MAX):
        errors.append(
            f"human_confidence must be int {C.CONFIDENCE_MIN}..{C.CONFIDENCE_MAX}, got: {conf!r}"
        )

    if "refinement" in rec and rec["refinement"] is not None:
        if not isinstance(rec["refinement"], dict):
            errors.append("refinement must be null or an object")

    notes = rec.get("notes")
    if notes is not None and not isinstance(notes, str):
        errors.append("notes must be null or a string")
    elif isinstance(notes, str):
        errors.extend(scan_notes_for_privacy_leak(notes))

    if "reviewed_at" in rec:
        if not isinstance(rec.get("reviewed_at"), str) or not rec["reviewed_at"]:
            errors.append("reviewed_at must be a non-empty ISO 8601 timestamp string")

    return errors


def validate_record_consistency(rec: dict[str, Any]) -> list[str]:
    """
    Validate decision consistency rules for a single record.
    Returns list of error strings (empty = consistent).
    Assumes schema is already valid (call validate_record_schema first).
    """
    errors: list[str] = []
    dt = rec.get("decision_type")
    fc = rec.get("final_category")
    pc = rec.get("phase_c_category")
    conf = rec.get("human_confidence")

    if dt == C.DECISION_TYPE_KEEP:
        if pc == "unknown_needs_review":
            errors.append("KEEP decision: phase_c_category must not be 'unknown_needs_review'")
        if fc != pc:
            errors.append(
                f"KEEP decision: final_category must equal phase_c_category "
                f"(got final={fc!r}, phase_c={pc!r})"
            )
        if isinstance(conf, int) and conf < C.CONFIDENCE_KEEP_MIN:
            errors.append(
                f"KEEP decision: human_confidence must be >= {C.CONFIDENCE_KEEP_MIN}, got {conf}"
            )

    elif dt == C.DECISION_TYPE_REMOVE:
        if fc not in C.REMOVE_ALLOWED_CATEGORIES:
            errors.append(
                f"REMOVE decision: final_category must be one of "
                f"{sorted(C.REMOVE_ALLOWED_CATEGORIES)}, got {fc!r}"
            )

    elif dt == C.DECISION_TYPE_RELABEL:
        if fc == pc:
            errors.append(
                f"RELABEL decision: final_category must differ from phase_c_category "
                f"(both are {fc!r})"
            )
        if fc == C.FINAL_CATEGORY_UNSURE:
            errors.append("RELABEL decision: final_category must not be 'unsure'")

    elif dt == C.DECISION_TYPE_UNSURE:
        if fc != C.FINAL_CATEGORY_UNSURE:
            errors.append(
                f"UNSURE decision: final_category must be 'unsure', got {fc!r}"
            )
        # Note: no upper confidence bound — a reviewer may be highly confident that an image
        # is genuinely ambiguous (e.g. confidence=5 means "certain this is unclassifiable").

    return errors


def validate_record_full(rec: dict[str, Any]) -> list[str]:
    """Run both schema and consistency validation on a record."""
    errors = validate_record_schema(rec)
    if not errors:
        errors.extend(validate_record_consistency(rec))
    return errors


# ─── File-level aggregate validation ─────────────────────────────────────────


def validate_decision_file_records(data: dict[str, Any]) -> tuple[list[str], int, int]:
    """
    Validate all records in a decision file.
    Returns (errors, invalid_count, valid_count).
    """
    errors: list[str] = []
    invalid_count = 0
    valid_count = 0

    records = data.get("records", [])
    seen_ids: set[str] = set()

    for i, rec in enumerate(records):
        rid = rec.get("review_id", f"<unknown:{i}>")
        if rid in seen_ids:
            errors.append(f"duplicate review_id within file: {rid!r}")
            invalid_count += 1
            continue
        seen_ids.add(rid)

        rec_errors = validate_record_full(rec)
        if rec_errors:
            for e in rec_errors:
                errors.append(f"[{rid}] {e}")
            invalid_count += 1
        else:
            valid_count += 1

    return errors, invalid_count, valid_count
