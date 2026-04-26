"""
test_intake_filter_safety.py — Safety invariant tests for the U4 filter pipeline.

Verifies:
- New local-only JSONL files ARE gitignored (filter_universe, filter_signals,
  filter_candidates, filter_candidates_final)
- New tracked summary JSON files are NOT gitignored
- Review HTML and decision JSON files are gitignored (existing review/ rule)
- Regression: dedup_clusters.jsonl and dedup_summary.json still NOT gitignored
- Regression: audit.jsonl still IS gitignored
- Regression: requirements-intake.txt still has no torch/ultralytics entries
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BATCH = "tg_2026-04-24"


def _check_ignored(path: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "check-ignore", "-v", path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


# ─── New local-only filter JSONL must be gitignored ───────────────────────────


def test_filter_universe_jsonl_is_gitignored() -> None:
    """filter_universe.jsonl (contains filenames) MUST be gitignored."""
    result = _check_ignored(f"data/intake_meta/{BATCH}/filter_universe.jsonl")
    assert result.returncode == 0, (
        "filter_universe.jsonl is NOT gitignored — privacy risk. "
        f"git check-ignore output: {result.stdout!r} {result.stderr!r}"
    )


def test_filter_signals_jsonl_is_gitignored() -> None:
    """filter_signals.jsonl (contains filenames + signal flags) MUST be gitignored."""
    result = _check_ignored(f"data/intake_meta/{BATCH}/filter_signals.jsonl")
    assert result.returncode == 0, (
        "filter_signals.jsonl is NOT gitignored — privacy risk. "
        f"git check-ignore output: {result.stdout!r} {result.stderr!r}"
    )


def test_filter_candidates_jsonl_is_gitignored() -> None:
    """filter_candidates.jsonl (contains filenames + categories) MUST be gitignored."""
    result = _check_ignored(f"data/intake_meta/{BATCH}/filter_candidates.jsonl")
    assert result.returncode == 0, (
        "filter_candidates.jsonl is NOT gitignored — privacy risk. "
        f"git check-ignore output: {result.stdout!r} {result.stderr!r}"
    )


def test_filter_candidates_final_jsonl_is_gitignored() -> None:
    """filter_candidates_final.jsonl (post-review filenames) MUST be gitignored."""
    result = _check_ignored(f"data/intake_meta/{BATCH}/filter_candidates_final.jsonl")
    assert result.returncode == 0, (
        "filter_candidates_final.jsonl is NOT gitignored — privacy risk. "
        f"git check-ignore output: {result.stdout!r} {result.stderr!r}"
    )


# ─── Review artifacts (HTML + decisions) must be gitignored ──────────────────


def test_filter_review_html_is_gitignored() -> None:
    """filter_review_<category>.html files MUST be gitignored (under review/ rule)."""
    for category in ["fish", "no_fish", "lure_fishing_gear", "unknown_needs_review"]:
        result = _check_ignored(
            f"data/intake_meta/{BATCH}/review/filter_review_{category}.html"
        )
        assert result.returncode == 0, (
            f"filter_review_{category}.html is NOT gitignored. "
            f"git check-ignore output: {result.stdout!r} {result.stderr!r}"
        )


def test_filter_decisions_json_is_gitignored() -> None:
    """filter_decisions_<category>.json files MUST be gitignored (under review/ rule)."""
    result = _check_ignored(
        f"data/intake_meta/{BATCH}/review/filter_decisions_fish.json"
    )
    assert result.returncode == 0, (
        "filter_decisions_fish.json is NOT gitignored. "
        f"git check-ignore output: {result.stdout!r} {result.stderr!r}"
    )


# ─── Tracked summary JSON files must NOT be gitignored ────────────────────────


def test_filter_universe_summary_not_gitignored() -> None:
    """filter_universe_summary.json (aggregate counts only) must NOT be gitignored."""
    result = _check_ignored(f"data/intake_meta/{BATCH}/filter_universe_summary.json")
    assert result.returncode != 0, (
        "filter_universe_summary.json IS gitignored but should be tracked. "
        f"Matching rule: {result.stdout!r}"
    )


def test_filter_signals_summary_not_gitignored() -> None:
    """filter_signals_summary.json (aggregate counts only) must NOT be gitignored."""
    result = _check_ignored(f"data/intake_meta/{BATCH}/filter_signals_summary.json")
    assert result.returncode != 0, (
        "filter_signals_summary.json IS gitignored but should be tracked. "
        f"Matching rule: {result.stdout!r}"
    )


def test_filter_candidates_summary_not_gitignored() -> None:
    """filter_candidates_summary.json (aggregate counts only) must NOT be gitignored."""
    result = _check_ignored(f"data/intake_meta/{BATCH}/filter_candidates_summary.json")
    assert result.returncode != 0, (
        "filter_candidates_summary.json IS gitignored but should be tracked. "
        f"Matching rule: {result.stdout!r}"
    )


def test_filter_review_summary_not_gitignored() -> None:
    """filter_review_summary.json (post-review aggregate counts) must NOT be gitignored."""
    result = _check_ignored(f"data/intake_meta/{BATCH}/filter_review_summary.json")
    assert result.returncode != 0, (
        "filter_review_summary.json IS gitignored but should be tracked. "
        f"Matching rule: {result.stdout!r}"
    )


# ─── Regression: existing artifacts must retain correct gitignore status ──────


def test_dedup_clusters_still_not_gitignored() -> None:
    """Regression: new filter rules must not accidentally gitignore dedup_clusters.jsonl."""
    result = _check_ignored(f"data/intake_meta/{BATCH}/dedup_clusters.jsonl")
    assert result.returncode != 0, (
        "dedup_clusters.jsonl IS gitignored — new filter rule too broad. "
        f"Matching rule: {result.stdout!r}"
    )


def test_dedup_summary_still_not_gitignored() -> None:
    """Regression: new filter rules must not accidentally gitignore dedup_summary.json."""
    result = _check_ignored(f"data/intake_meta/{BATCH}/dedup_summary.json")
    assert result.returncode != 0, (
        "dedup_summary.json IS gitignored — new filter rule too broad. "
        f"Matching rule: {result.stdout!r}"
    )


def test_audit_jsonl_still_gitignored() -> None:
    """Regression: audit.jsonl must still be gitignored after adding filter rules."""
    result = _check_ignored(f"data/intake_meta/{BATCH}/audit.jsonl")
    assert result.returncode == 0, (
        "audit.jsonl is NOT gitignored — check for accidental negation rule. "
        f"git check-ignore output: {result.stdout!r} {result.stderr!r}"
    )


# ─── Regression: requirements-intake.txt must not gain ML deps ────────────────


def test_requirements_no_ml_deps_regression() -> None:
    """Regression: requirements-intake.txt must not contain torch or ultralytics."""
    req_path = REPO_ROOT / "requirements-intake.txt"
    assert req_path.exists(), "requirements-intake.txt not found"
    content = req_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        for forbidden in ("torch", "torchvision", "ultralytics"):
            assert forbidden not in stripped.lower(), (
                f"Forbidden ML dep '{forbidden}' found in requirements-intake.txt: {line!r}"
            )
