"""
test_intake_review_finalize.py — Tests for scripts/intake_review_finalize.py (U3).

Tests cover:
- All KEEP_DEDUP: provisional cleared, recommendation=validated
- Mixed decisions below threshold: provisional cleared, counts correct
- FP rate exceeds threshold: dedup_summary.json NOT modified, recommendation=lower_threshold
- MIXED counted as FP: same outcome as FALSE_POSITIVE for rate calculation
- Incomplete decisions without --partial: exits non-zero, missing IDs on stderr
- Incomplete decisions with --partial: proceeds, review_complete=false, provisional NOT cleared
- --dry-run: dedup_summary.json not modified even when fp_rate < threshold
- All UNSURE: review_complete=false, provisional NOT cleared
- Already finalized: warns and skips update, summary still written, exits 0
- Integration: end-to-end with synthetic clusters, review, summary counts correct and no PII
- UNSURE-only has no effect on FP rate (unsure != FP)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from intake_review_finalize import (  # noqa: E402
    DEFAULT_FP_THRESHOLD,
    _load_boundary_clusters,
    _suggest_threshold,
    run,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _write_clusters(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def _make_boundary_cluster(cid: int) -> dict:
    return {
        "cluster_id": cid,
        "cluster_type": "perceptual",
        "keep_filename": f"photos/keep_{cid}.jpg",
        "duplicate_filenames": [f"photos/dup_{cid}.jpg"],
        "hamming_distance": 8,
        "reason": "phash_hamming<=8",
    }


def _make_non_boundary(cid: int, hamming: int) -> dict:
    return {
        "cluster_id": cid,
        "cluster_type": "perceptual",
        "keep_filename": f"photos/keep_{cid}.jpg",
        "duplicate_filenames": [f"photos/dup_{cid}.jpg"],
        "hamming_distance": hamming,
        "reason": f"phash_hamming<={hamming}",
    }


def _make_dedup_summary(provisional: bool = True) -> dict:
    return {
        "input_records": 100,
        "phash_threshold": 8,
        "perceptual_clusters": 10,
        "boundary_clusters_at_threshold": 5,
        "provisional": provisional,
        "manual_review_required": provisional,
        "source": "telegram_private_2026-04-24",
        "license": "private_training_only",
    }


def _make_decisions(cluster_ids: list[int], decision: str = "KEEP_DEDUP") -> dict:
    return {
        "schema_version": 1,
        "threshold_reviewed": 8,
        "decisions": [
            {"cluster_id": cid, "decision": decision, "note": ""}
            for cid in cluster_ids
        ],
    }


def _setup(
    tmp_path: Path,
    n_boundary: int = 5,
    decisions: dict | None = None,
    provisional: bool = True,
) -> tuple[Path, Path, Path, Path]:
    clusters_path = tmp_path / "dedup_clusters.jsonl"
    _write_clusters(clusters_path, [_make_boundary_cluster(i) for i in range(1, n_boundary + 1)])

    summary_path = tmp_path / "dedup_summary.json"
    summary_path.write_text(
        json.dumps(_make_dedup_summary(provisional=provisional)), encoding="utf-8"
    )

    decisions_path = tmp_path / "decisions.json"
    if decisions is None:
        decisions = _make_decisions(list(range(1, n_boundary + 1)), "KEEP_DEDUP")
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")

    output_path = tmp_path / "dedup_review_summary.json"
    return clusters_path, summary_path, decisions_path, output_path


# ─── _suggest_threshold tests ─────────────────────────────────────────────────


def test_suggest_threshold_low_fp() -> None:
    # fp_rate=0.10 at threshold=8 → 8 - round(0.10*8)=8-1=7
    assert _suggest_threshold(0.10, 8) == 7


def test_suggest_threshold_high_fp() -> None:
    # fp_rate=0.40 at threshold=8 → 8 - round(0.40*8)=8-3=5
    assert _suggest_threshold(0.40, 8) == 5


def test_suggest_threshold_floor() -> None:
    # fp_rate=1.0 → max(1, 8-8)=1
    assert _suggest_threshold(1.0, 8) == 1


# ─── Happy path: all KEEP_DEDUP ───────────────────────────────────────────────


def test_all_keep_dedup_finalizes(tmp_path: Path) -> None:
    clusters, summary, decisions, output = _setup(tmp_path, n_boundary=5)

    rc = run(decisions, clusters, summary, output)
    assert rc == 0

    review = json.loads(output.read_text(encoding="utf-8"))
    assert review["keep_dedup_count"] == 5
    assert review["false_positive_count"] == 0
    assert review["false_positive_rate"] == 0.0
    assert review["threshold_recommendation"] == "validated"
    assert review["review_complete"] is True

    updated = json.loads(summary.read_text(encoding="utf-8"))
    assert updated["provisional"] is False
    assert updated["manual_review_required"] is False
    assert "review_completed_at" in updated


def test_all_keep_dedup_no_pii_in_summary(tmp_path: Path) -> None:
    clusters, summary, decisions, output = _setup(tmp_path, n_boundary=5)
    run(decisions, clusters, summary, output)

    text = output.read_text(encoding="utf-8")
    # Check that actual filenames (with path or extension) don't appear — not just field name substrings
    assert ".jpg" not in text, "filenames must not appear in tracked summary"
    assert "photos/" not in text, "photo paths must not appear in tracked summary"
    assert "caption" not in text.lower()


# ─── Happy path: mixed below threshold ───────────────────────────────────────


def test_mixed_below_threshold_finalizes(tmp_path: Path) -> None:
    dec = {
        "schema_version": 1,
        "threshold_reviewed": 8,
        "decisions": [
            {"cluster_id": i, "decision": "KEEP_DEDUP", "note": ""} for i in range(1, 19)
        ] + [
            {"cluster_id": 19, "decision": "FALSE_POSITIVE", "note": ""},
            {"cluster_id": 20, "decision": "FALSE_POSITIVE", "note": ""},
        ],
    }
    clusters, summary, decisions_path, output = _setup(tmp_path, n_boundary=20, decisions=dec)

    rc = run(decisions_path, clusters, summary, output, fp_threshold=0.15)
    assert rc == 0

    review = json.loads(output.read_text(encoding="utf-8"))
    assert review["keep_dedup_count"] == 18
    assert review["false_positive_count"] == 2
    assert review["false_positive_rate"] == pytest.approx(0.10)
    assert review["threshold_recommendation"] == "validated"

    updated = json.loads(summary.read_text(encoding="utf-8"))
    assert updated["provisional"] is False


# ─── FP rate exceeds threshold ────────────────────────────────────────────────


def test_fp_exceeded_does_not_modify_summary(tmp_path: Path) -> None:
    dec = {
        "schema_version": 1,
        "threshold_reviewed": 8,
        "decisions": [
            {"cluster_id": i, "decision": "KEEP_DEDUP", "note": ""} for i in range(1, 17)
        ] + [
            {"cluster_id": i, "decision": "FALSE_POSITIVE", "note": ""} for i in range(17, 21)
        ],
    }
    clusters, summary, decisions_path, output = _setup(tmp_path, n_boundary=20, decisions=dec)

    rc = run(decisions_path, clusters, summary, output, fp_threshold=0.15)
    assert rc == 1

    review = json.loads(output.read_text(encoding="utf-8"))
    assert review["threshold_recommendation"] == "lower_threshold"
    assert review["review_complete"] is False
    assert review["suggested_lower_threshold"] is not None

    unchanged = json.loads(summary.read_text(encoding="utf-8"))
    assert unchanged["provisional"] is True


# ─── MIXED counts as FP ───────────────────────────────────────────────────────


def test_mixed_counted_as_fp(tmp_path: Path) -> None:
    dec = {
        "schema_version": 1,
        "threshold_reviewed": 8,
        "decisions": [
            {"cluster_id": i, "decision": "KEEP_DEDUP", "note": ""} for i in range(1, 17)
        ] + [
            {"cluster_id": i, "decision": "MIXED", "note": ""} for i in range(17, 21)
        ],
    }
    clusters, summary, decisions_path, output = _setup(tmp_path, n_boundary=20, decisions=dec)

    rc = run(decisions_path, clusters, summary, output, fp_threshold=0.15)
    assert rc == 1

    review = json.loads(output.read_text(encoding="utf-8"))
    assert review["mixed_count"] == 4
    assert review["false_positive_count"] == 0  # tracked separately from mixed
    assert review["false_positive_rate"] == pytest.approx(0.20)  # 4/20
    assert review["threshold_recommendation"] == "lower_threshold"

    unchanged = json.loads(summary.read_text(encoding="utf-8"))
    assert unchanged["provisional"] is True


# ─── Incomplete decisions ─────────────────────────────────────────────────────


def test_incomplete_without_partial_exits_nonzero(tmp_path: Path) -> None:
    # KP-01: helpers must not call sys.exit — run() must return 1 directly
    dec = _make_decisions([1, 2, 3], "KEEP_DEDUP")  # missing clusters 4, 5
    clusters, summary, decisions_path, output = _setup(tmp_path, n_boundary=5, decisions=dec)

    rc = run(decisions_path, clusters, summary, output, partial=False)
    assert rc == 1  # hard error: incomplete without --partial


def test_incomplete_with_partial_proceeds(tmp_path: Path) -> None:
    dec = _make_decisions([1, 2, 3], "KEEP_DEDUP")  # missing 4, 5
    clusters, summary, decisions_path, output = _setup(tmp_path, n_boundary=5, decisions=dec)

    rc = run(decisions_path, clusters, summary, output, partial=True)
    # KP-02: incomplete + partial → valid but non-final preview state
    assert rc == 2

    review = json.loads(output.read_text(encoding="utf-8"))
    assert review["review_complete"] is False
    assert review["boundary_clusters_reviewed"] == 3

    unchanged = json.loads(summary.read_text(encoding="utf-8"))
    assert unchanged["provisional"] is True


# ─── --dry-run ────────────────────────────────────────────────────────────────


def test_dry_run_does_not_modify_summary(tmp_path: Path) -> None:
    clusters, summary, decisions, output = _setup(tmp_path, n_boundary=5)
    summary_before = summary.read_text(encoding="utf-8")

    rc = run(decisions, clusters, summary, output, dry_run=True)
    # KP-02: dry-run is a valid preview state
    assert rc == 2

    review = json.loads(output.read_text(encoding="utf-8"))
    assert review["threshold_recommendation"] == "validated"

    assert summary.read_text(encoding="utf-8") == summary_before


# ─── All UNSURE ───────────────────────────────────────────────────────────────


def test_all_unsure_does_not_finalize(tmp_path: Path) -> None:
    dec = _make_decisions(list(range(1, 6)), "UNSURE")
    clusters, summary, decisions_path, output = _setup(tmp_path, n_boundary=5, decisions=dec)

    rc = run(decisions_path, clusters, summary, output)
    # KP-02: UNSURE decisions block finalization → valid but non-final
    assert rc == 2

    review = json.loads(output.read_text(encoding="utf-8"))
    assert review["unsure_count"] == 5
    assert review["false_positive_rate"] == 0.0
    assert review["review_complete"] is False

    unchanged = json.loads(summary.read_text(encoding="utf-8"))
    assert unchanged["provisional"] is True


# ─── UNSURE doesn't inflate FP rate ──────────────────────────────────────────


def test_unsure_not_counted_as_fp(tmp_path: Path) -> None:
    dec = {
        "schema_version": 1,
        "threshold_reviewed": 8,
        "decisions": [
            {"cluster_id": i, "decision": "KEEP_DEDUP", "note": ""} for i in range(1, 4)
        ] + [
            {"cluster_id": 4, "decision": "UNSURE", "note": ""},
            {"cluster_id": 5, "decision": "UNSURE", "note": ""},
        ],
    }
    clusters, summary, decisions_path, output = _setup(tmp_path, n_boundary=5, decisions=dec)
    run(decisions_path, clusters, summary, output)

    review = json.loads(output.read_text(encoding="utf-8"))
    assert review["false_positive_rate"] == 0.0  # UNSURE is not FP


# ─── Already finalized guard ──────────────────────────────────────────────────


def test_already_finalized_warns_and_skips(tmp_path: Path) -> None:
    clusters, summary, decisions, output = _setup(tmp_path, n_boundary=5, provisional=False)

    rc = run(decisions, clusters, summary, output)
    # KP-02: source already final, nothing new done → valid but non-final preview
    assert rc == 2

    review = json.loads(output.read_text(encoding="utf-8"))
    assert review["threshold_recommendation"] == "validated"

    # summary must be unchanged (already provisional=false)
    final = json.loads(summary.read_text(encoding="utf-8"))
    assert final["provisional"] is False
    assert "review_completed_at" not in final  # not re-written


# ─── Integration: end-to-end ─────────────────────────────────────────────────


def test_integration_no_pii_and_correct_counts(tmp_path: Path) -> None:
    n = 10
    clusters_path = tmp_path / "clusters.jsonl"
    _write_clusters(clusters_path, [_make_boundary_cluster(i) for i in range(1, n + 1)])

    summary_path = tmp_path / "dedup_summary.json"
    summary_path.write_text(json.dumps(_make_dedup_summary()), encoding="utf-8")

    dec: dict = {
        "schema_version": 1,
        "threshold_reviewed": 8,
        "decisions": [
            {"cluster_id": i, "decision": "KEEP_DEDUP", "note": ""} for i in range(1, 8)
        ] + [
            {"cluster_id": 8, "decision": "FALSE_POSITIVE", "note": "different species"},
            {"cluster_id": 9, "decision": "MIXED", "note": "3 images but 1 looks different"},
            {"cluster_id": 10, "decision": "KEEP_DEDUP", "note": ""},
        ],
    }
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(json.dumps(dec), encoding="utf-8")

    output_path = tmp_path / "dedup_review_summary.json"
    rc = run(decisions_path, clusters_path, summary_path, output_path, fp_threshold=0.15)

    # fp_rate = (1 FP + 1 MIXED) / 10 = 0.20 ≥ 0.15 → should NOT finalize
    assert rc == 1

    review = json.loads(output_path.read_text(encoding="utf-8"))
    assert review["boundary_clusters_reviewed"] == 10
    assert review["keep_dedup_count"] == 8
    assert review["false_positive_count"] == 1
    assert review["mixed_count"] == 1
    assert review["unsure_count"] == 0
    assert review["false_positive_rate"] == pytest.approx(0.20)
    assert review["threshold_recommendation"] == "lower_threshold"
    assert review["review_complete"] is False

    # Verify no PII: no filenames or reviewer notes in the tracked summary
    text = output_path.read_text(encoding="utf-8")
    assert ".jpg" not in text, "filenames must not appear in tracked summary"
    assert "photos/" not in text, "photo paths must not appear in tracked summary"
    assert "different species" not in text, "reviewer notes must not appear in tracked summary"

    # dedup_summary.json must be unchanged
    unchanged = json.loads(summary_path.read_text(encoding="utf-8"))
    assert unchanged["provisional"] is True


# ─── ADV-001: Output overwrite protection ─────────────────────────────────────


def test_overwrite_final_output_refused(tmp_path: Path) -> None:
    """Cannot overwrite an existing finalized output without --force."""
    clusters, summary, decisions, output = _setup(tmp_path, n_boundary=5)
    # Manually seed a finalized output
    output.write_text(json.dumps({"review_complete": True}), encoding="utf-8")

    rc = run(decisions, clusters, summary, output)
    assert rc == 1  # refused to overwrite — hard error

    # Output must remain unchanged (original finalized value)
    kept = json.loads(output.read_text(encoding="utf-8"))
    assert kept == {"review_complete": True}


def test_force_allows_overwrite_of_final_output(tmp_path: Path) -> None:
    """--force allows overwriting a finalized output summary."""
    clusters, summary, decisions, output = _setup(tmp_path, n_boundary=5)
    output.write_text(json.dumps({"review_complete": True}), encoding="utf-8")

    rc = run(decisions, clusters, summary, output, force=True)
    assert rc == 0  # all KEEP_DEDUP + force → complete finalization

    review = json.loads(output.read_text(encoding="utf-8"))
    assert review["review_complete"] is True
    assert review["keep_dedup_count"] == 5


# ─── ADV-002: Contradictory state protection ──────────────────────────────────


def test_contradictory_state_refused(tmp_path: Path) -> None:
    """Source already final + decisions can't finalize → refuses to write contradictory state."""
    dec = {
        "schema_version": 1,
        "threshold_reviewed": 8,
        "decisions": [
            {"cluster_id": i, "decision": "FALSE_POSITIVE", "note": ""} for i in range(1, 6)
        ],
    }
    clusters, summary, decisions_path, output = _setup(
        tmp_path, n_boundary=5, decisions=dec, provisional=False
    )

    rc = run(decisions_path, clusters, summary, output)
    assert rc == 1  # contradictory: source=final but FP=100% → can't finalize


def test_force_allows_contradictory_state_write(tmp_path: Path) -> None:
    """--force allows writing review summary even when source is final and can't finalize."""
    dec = {
        "schema_version": 1,
        "threshold_reviewed": 8,
        "decisions": [
            {"cluster_id": i, "decision": "FALSE_POSITIVE", "note": ""} for i in range(1, 6)
        ],
    }
    clusters, summary, decisions_path, output = _setup(
        tmp_path, n_boundary=5, decisions=dec, provisional=False
    )

    rc = run(decisions_path, clusters, summary, output, force=True)
    assert rc == 1  # FP threshold exceeded → still rc=1, but output IS written

    review = json.loads(output.read_text(encoding="utf-8"))
    assert review["review_complete"] is False
    assert review["threshold_recommendation"] == "lower_threshold"


# ─── Dynamic threshold: phash_threshold=2 ────────────────────────────────────


def _make_dedup_summary_threshold2(provisional: bool = True) -> dict:
    return {
        "input_records": 100,
        "phash_threshold": 2,
        "perceptual_clusters": 30,
        "boundary_clusters_at_threshold": 5,
        "provisional": provisional,
        "manual_review_required": provisional,
        "source": "telegram_private_2026-04-24",
        "license": "private_training_only",
    }


def _make_boundary_cluster_hamming2(cid: int) -> dict:
    return {
        "cluster_id": cid,
        "cluster_type": "perceptual",
        "keep_filename": f"photos/keep_{cid}.jpg",
        "duplicate_filenames": [f"photos/dup_{cid}.jpg"],
        "hamming_distance": 2,
        "reason": "phash_hamming<=2",
    }


def test_threshold2_all_keep_dedup_finalizes(tmp_path: Path) -> None:
    """phash_threshold=2: boundary clusters at hamming=2 are selected and finalized."""
    n = 5
    clusters_path = tmp_path / "dedup_clusters.jsonl"
    # mix: some hamming=2 (boundary) and some hamming=0 (non-boundary)
    records = [_make_boundary_cluster_hamming2(i) for i in range(1, n + 1)]
    records += [_make_non_boundary(100 + i, 0) for i in range(3)]
    _write_clusters(clusters_path, records)

    summary_path = tmp_path / "dedup_summary.json"
    summary_path.write_text(json.dumps(_make_dedup_summary_threshold2()), encoding="utf-8")

    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(
        json.dumps({
            "schema_version": 1,
            "threshold_reviewed": 2,
            "decisions": [
                {"cluster_id": i, "decision": "KEEP_DEDUP", "note": ""} for i in range(1, n + 1)
            ],
        }),
        encoding="utf-8",
    )

    output_path = tmp_path / "dedup_review_summary.json"
    rc = run(decisions_path, clusters_path, summary_path, output_path)
    assert rc == 0

    review = json.loads(output_path.read_text(encoding="utf-8"))
    assert review["boundary_clusters_total"] == n
    assert review["keep_dedup_count"] == n
    assert review["false_positive_count"] == 0
    assert review["false_positive_rate"] == 0.0
    assert review["threshold_recommendation"] == "validated"
    assert review["review_complete"] is True

    updated = json.loads(summary_path.read_text(encoding="utf-8"))
    assert updated["provisional"] is False
    assert updated["manual_review_required"] is False


# ─── P1: cluster_id validation ────────────────────────────────────────────────


def test_decision_missing_cluster_id_returns_rc1(tmp_path: Path) -> None:
    """Decision entry without cluster_id must return rc=1 without raw traceback."""
    dec = {
        "schema_version": 1,
        "threshold_reviewed": 8,
        "decisions": [
            {"decision": "KEEP_DEDUP", "note": ""},  # no cluster_id at all
        ],
    }
    clusters, summary, decisions_path, output = _setup(tmp_path, n_boundary=5, decisions=dec)
    rc = run(decisions_path, clusters, summary, output)
    assert rc == 1


def test_decision_string_cluster_id_accepted(tmp_path: Path) -> None:
    """Decision entry with string cluster_id like '3' must be coerced to int and accepted."""
    dec = {
        "schema_version": 1,
        "threshold_reviewed": 8,
        "decisions": [
            {"cluster_id": str(i), "decision": "KEEP_DEDUP", "note": ""}
            for i in range(1, 6)
        ],
    }
    clusters, summary, decisions_path, output = _setup(tmp_path, n_boundary=5, decisions=dec)
    rc = run(decisions_path, clusters, summary, output)
    assert rc == 0

    review = json.loads(output.read_text(encoding="utf-8"))
    assert review["keep_dedup_count"] == 5
    assert review["review_complete"] is True


def test_decision_non_coercible_cluster_id_returns_rc1(tmp_path: Path) -> None:
    """Decision entry with non-integer string cluster_id must return rc=1."""
    dec = {
        "schema_version": 1,
        "threshold_reviewed": 8,
        "decisions": [
            {"cluster_id": "abc", "decision": "KEEP_DEDUP", "note": ""},
        ],
    }
    clusters, summary, decisions_path, output = _setup(tmp_path, n_boundary=5, decisions=dec)
    rc = run(decisions_path, clusters, summary, output)
    assert rc == 1


def test_decision_null_cluster_id_returns_rc1(tmp_path: Path) -> None:
    """Decision entry with null cluster_id must return rc=1."""
    dec = {
        "schema_version": 1,
        "threshold_reviewed": 8,
        "decisions": [
            {"cluster_id": None, "decision": "KEEP_DEDUP", "note": ""},
        ],
    }
    clusters, summary, decisions_path, output = _setup(tmp_path, n_boundary=5, decisions=dec)
    rc = run(decisions_path, clusters, summary, output)
    assert rc == 1


# ─── P2: threshold_reviewed fallback ──────────────────────────────────────────


def test_threshold_reviewed_fallback_uses_phash_threshold(tmp_path: Path) -> None:
    """Missing threshold_reviewed falls back to phash_threshold (not hardcoded 8)."""
    n = 5
    clusters_path = tmp_path / "clusters.jsonl"
    _write_clusters(clusters_path, [_make_boundary_cluster_hamming2(i) for i in range(1, n + 1)])

    summary_path = tmp_path / "dedup_summary.json"
    summary_path.write_text(json.dumps(_make_dedup_summary_threshold2()), encoding="utf-8")

    dec = {
        "schema_version": 1,
        # deliberately omit threshold_reviewed — fallback must use phash_threshold=2, not 8
        "decisions": [
            {"cluster_id": i, "decision": "KEEP_DEDUP", "note": ""} for i in range(1, n + 1)
        ],
    }
    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(json.dumps(dec), encoding="utf-8")

    output_path = tmp_path / "dedup_review_summary.json"
    rc = run(decisions_path, clusters_path, summary_path, output_path)
    assert rc == 0

    review = json.loads(output_path.read_text(encoding="utf-8"))
    assert review["threshold_reviewed"] == 2  # phash_threshold from summary, not hardcoded 8


# ─── P3: manual_review_reason updated on finalization ─────────────────────────


def test_finalization_updates_manual_review_reason(tmp_path: Path) -> None:
    """Successful finalization replaces stale manual_review_reason with completed-review wording."""
    clusters, summary, decisions, output = _setup(tmp_path, n_boundary=5)
    # Seed a stale pre-review reason
    s = json.loads(summary.read_text(encoding="utf-8"))
    s["manual_review_reason"] = "Spot-check before use."
    summary.write_text(json.dumps(s), encoding="utf-8")

    rc = run(decisions, clusters, summary, output)
    assert rc == 0

    updated = json.loads(summary.read_text(encoding="utf-8"))
    reason = updated.get("manual_review_reason", "")
    assert "Spot-check" not in reason
    assert "completed" in reason.lower()
    assert "KEEP_DEDUP" in reason
    assert "downstream staging" in reason


def test_missing_phash_threshold_returns_1(tmp_path: Path) -> None:
    """Missing phash_threshold in dedup_summary.json must produce rc=1."""
    clusters_path = tmp_path / "dedup_clusters.jsonl"
    _write_clusters(clusters_path, [_make_boundary_cluster(i) for i in range(1, 4)])

    summary_no_threshold = {
        "input_records": 100,
        "provisional": True,
        "manual_review_required": True,
        "source": "telegram_private_2026-04-24",
        "license": "private_training_only",
    }
    summary_path = tmp_path / "dedup_summary.json"
    summary_path.write_text(json.dumps(summary_no_threshold), encoding="utf-8")

    decisions_path = tmp_path / "decisions.json"
    decisions_path.write_text(json.dumps(_make_decisions([1, 2, 3])), encoding="utf-8")

    output_path = tmp_path / "dedup_review_summary.json"
    rc = run(decisions_path, clusters_path, summary_path, output_path)
    assert rc == 1
