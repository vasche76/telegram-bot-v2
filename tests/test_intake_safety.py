"""
test_intake_safety.py — Safety invariant tests for the intake pipeline.

Verifies:
- requirements-intake.txt does not pull in ML deps (torch/torchvision/ultralytics)
- data/intake/ photos are gitignored
- data/intake_meta/**/manifest.jsonl IS gitignored (contains captions/sender names)
- data/intake_meta/**/manifest_summary.json is NOT gitignored (aggregate only)
- intake_constants.py imports without error
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_requirements_no_torch() -> None:
    """requirements-intake.txt must not list torch, torchvision, or ultralytics."""
    req_path = REPO_ROOT / "requirements-intake.txt"
    assert req_path.exists(), "requirements-intake.txt not found"
    content = req_path.read_text(encoding="utf-8").lower()
    forbidden = ["torch", "torchvision", "ultralytics"]
    for dep in forbidden:
        # Ignore comment lines
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert dep not in stripped, (
                f"Forbidden ML dependency '{dep}' found in requirements-intake.txt: {line!r}"
            )


def test_intake_photos_are_gitignored() -> None:
    """Photo files under data/intake/ must be gitignored."""
    result = subprocess.run(
        [
            "git", "check-ignore", "-v",
            "data/intake/tg_2026-04-24/candidates/stage_b/pike/test.jpg",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "data/intake/ photos are NOT gitignored — check .gitignore. "
        f"git check-ignore output: {result.stdout!r} {result.stderr!r}"
    )


def test_intake_manifest_jsonl_is_gitignored() -> None:
    """Full manifest.jsonl (contains captions/sender names) MUST be gitignored."""
    result = subprocess.run(
        [
            "git", "check-ignore", "-v",
            "data/intake_meta/tg_2026-04-24/manifest.jsonl",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "data/intake_meta/ manifest.jsonl is NOT gitignored — privacy risk. "
        "Add 'data/intake_meta/**/manifest.jsonl' to .gitignore. "
        f"git check-ignore output: {result.stdout!r} {result.stderr!r}"
    )


def test_manifest_summary_not_gitignored() -> None:
    """Privacy-safe manifest_summary.json must NOT be gitignored (tracked output)."""
    result = subprocess.run(
        [
            "git", "check-ignore", "-v",
            "data/intake_meta/tg_2026-04-24/manifest_summary.json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        "data/intake_meta/ manifest_summary.json IS gitignored but should be tracked. "
        f"Matching rule: {result.stdout!r}"
    )


def test_intake_constants_imports() -> None:
    """intake_constants.py must import cleanly from any working directory."""
    result = subprocess.run(
        [sys.executable, "-c", "import intake_constants; print('ok')"],
        cwd=REPO_ROOT / "scripts",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"intake_constants.py failed to import: {result.stderr}"
    )
    assert "ok" in result.stdout


def test_stage_lists_not_empty() -> None:
    """STAGE_A_CLASSES and STAGE_B_SPECIES must be non-empty after import."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import intake_constants as c  # noqa: PLC0415
    assert len(c.STAGE_A_CLASSES) >= 5, "STAGE_A_CLASSES suspiciously short"
    assert len(c.STAGE_B_SPECIES) >= 15, "STAGE_B_SPECIES suspiciously short"


def test_dedup_clusters_not_gitignored() -> None:
    """Privacy-safe dedup_clusters.jsonl must NOT be gitignored (tracked output, ~20-50 KB)."""
    result = subprocess.run(
        [
            "git", "check-ignore", "-v",
            "data/intake_meta/tg_2026-04-24/dedup_clusters.jsonl",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        "data/intake_meta/ dedup_clusters.jsonl IS gitignored but should be tracked. "
        f"Matching rule: {result.stdout!r}"
    )


def test_dedup_summary_not_gitignored() -> None:
    """Privacy-safe dedup_summary.json must NOT be gitignored (tracked output)."""
    result = subprocess.run(
        [
            "git", "check-ignore", "-v",
            "data/intake_meta/tg_2026-04-24/dedup_summary.json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0, (
        "data/intake_meta/ dedup_summary.json IS gitignored but should be tracked. "
        f"Matching rule: {result.stdout!r}"
    )
