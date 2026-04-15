"""
Fish Vision Evaluation Runner v2.

Real ML metrics: precision, recall, F1, confusion matrix, acceptance gates.

Usage:
    # Evaluate Stage A with local test images:
    python3 -m bot.fish_vision.evaluation.eval_runner --stage a --dir data/eval_cases/stage_a

    # Evaluate Stage B:
    python3 -m bot.fish_vision.evaluation.eval_runner --stage b --dir data/eval_cases/stage_b

    # Legacy JSON mode:
    python3 -m bot.fish_vision.evaluation.eval_runner --cases eval_cases.json

    # Full evaluation (both stages):
    python3 -m bot.fish_vision.evaluation.eval_runner --all

    # Check if model passes acceptance criteria (exit 0=pass, 1=fail):
    python3 -m bot.fish_vision.evaluation.eval_runner --all --strict
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from bot.fish_vision.pipeline import analyze_fish_photo

# ── Class definitions ─────────────────────────────────────────────────────────

STAGE_A_CLASSES = ["whole_fish", "lure", "fish_part", "fry", "no_fish"]

# Stage B: expanded to 15 classes (original 5 + Salmonidae + Cyprinidae + Siluriformes + fallback)
STAGE_B_CLASSES = [
    # Original
    "pike", "taimen", "grayling", "whitefish", "perch",
    # New Salmonidae
    "brown_trout", "rainbow_trout", "atlantic_salmon",
    # New Cyprinidae
    "common_carp", "crucian_carp", "bream", "roach", "ide",
    # New Siluriformes
    "wels_catfish",
    # Fallback
    "unknown_fish",
]

# Salmonid confusion pairs — extra gates for visually similar species
# These pairs are especially prone to confusion and need specific monitoring.
SALMONID_CONFUSION_PAIRS = [
    ("pike", "wels_catfish"),        # large elongated predators
    ("brown_trout", "atlantic_salmon"), # most similar Salmonidae
    ("rainbow_trout", "brown_trout"),   # similar body plan
    ("grayling", "brown_trout"),        # river salmonids
    ("whitefish", "roach"),             # silvery body plan
    ("bream", "roach"),                 # Cyprinidae look-alikes
    ("common_carp", "crucian_carp"),    # Cyprinidae look-alikes
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"}

# Pipeline version — bump when the underlying model/prompts change
PIPELINE_VERSION = 1

# ── Acceptance thresholds ─────────────────────────────────────────────────────

ACCEPTANCE_CRITERIA: dict[str, dict[str, float]] = {
    "stage_a": {
        # Critical: lure must never be called a fish (false positive that misleads user)
        "lure_recall": 0.90,
        "whole_fish_recall": 0.85,
        "fish_part_recall": 0.80,
        # Critical error rate: lure classified as whole_fish
        "lure_as_whole_fish_rate": 0.05,
        # Critical error rate: fish_part classified as whole_fish
        "fish_part_as_whole_fish_rate": 0.10,
    },
    "stage_b": {
        # Min per-species accuracy (currently relaxed for new species with small datasets)
        "per_species_accuracy_min": 0.70,
        # Rate at which unknown/OOD fish are correctly sent to unknown_fish
        "unknown_rejection_rate": 0.20,
        # Critical: taimen vs pike confusion (both rare large fish — user cares)
        "pike_taimen_confusion": 0.05,
        # Salmonid inter-species confusion gates (new — broader family)
        "brown_trout_atlantic_salmon_confusion": 0.15,
        "rainbow_trout_brown_trout_confusion": 0.20,
        # Cyprinid inter-species confusion gates
        "bream_roach_confusion": 0.25,
        "common_carp_crucian_carp_confusion": 0.20,
    },
}


# ── Legacy TestCase / TestResult (backward compat for JSON mode) ──────────────

@dataclass
class TestCase:
    image_url: str
    caption: str
    expected_object_type: str       # whole_fish | lure | fish_part | fry | no_fish
    expected_species: Optional[str] # pike|taimen|grayling|whitefish|perch|brown_trout|rainbow_trout|atlantic_salmon|common_carp|crucian_carp|bream|roach|ide|wels_catfish|unknown_fish|None
    description: str
    must_reject: bool = False
    species_must_match: bool = True


@dataclass
class TestResult:
    test_case: TestCase
    object_type_correct: bool
    species_correct: bool
    valid_catch_correct: bool
    is_valid_catch: bool
    got_object_type: str
    got_species: str
    got_detection_conf: float
    got_species_conf: float
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        if self.error:
            return False
        if not self.object_type_correct:
            return False
        if self.test_case.must_reject and self.is_valid_catch:
            return False
        if self.test_case.species_must_match and not self.species_correct:
            return False
        return True


# ── Metrics implementation (no sklearn) ──────────────────────────────────────

def build_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    classes: list[str],
) -> list[list[int]]:
    """
    Build a confusion matrix. Rows = actual, Cols = predicted.
    Returns a 2-D list of shape (n_classes, n_classes).
    """
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm: list[list[int]] = [[0] * n for _ in range(n)]
    for true_label, pred_label in zip(y_true, y_pred):
        r = idx.get(true_label, -1)
        c = idx.get(pred_label, -1)
        if r >= 0 and c >= 0:
            cm[r][c] += 1
    return cm


def class_metrics_from_cm(
    cm: list[list[int]],
    classes: list[str],
) -> dict[str, dict[str, float]]:
    """
    Compute per-class precision, recall, F1, support from confusion matrix.
    Returns dict: class_name -> {precision, recall, f1, support}.
    """
    n = len(classes)
    result: dict[str, dict[str, float]] = {}
    for i, cls in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(n)) - tp    # col sum minus TP
        fn = sum(cm[i][c] for c in range(n)) - tp    # row sum minus TP
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        result[cls] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return result


def macro_avg(per_class: dict[str, dict[str, float]]) -> dict[str, float]:
    """Unweighted average of per-class metrics."""
    if not per_class:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    keys = ["precision", "recall", "f1"]
    return {k: sum(v[k] for v in per_class.values()) / len(per_class) for k in keys}


def weighted_avg(per_class: dict[str, dict[str, float]]) -> dict[str, float]:
    """Support-weighted average of per-class metrics."""
    total_support = sum(v["support"] for v in per_class.values())
    if total_support == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    keys = ["precision", "recall", "f1"]
    return {
        k: sum(v[k] * v["support"] for v in per_class.values()) / total_support
        for k in keys
    }


def overall_accuracy(cm: list[list[int]]) -> float:
    total = sum(cm[i][j] for i in range(len(cm)) for j in range(len(cm[0])))
    correct = sum(cm[i][i] for i in range(len(cm)))
    return correct / total if total > 0 else 0.0


# ── Progress bar ──────────────────────────────────────────────────────────────

def _progress_bar(done: int, total: int, width: int = 32) -> str:
    if total == 0:
        return f"[{'=' * width}]"
    filled = int(width * done / total)
    remaining = width - filled - (1 if done < total else 0)
    arrow = ">" if done < total else "="
    bar = "=" * filled + arrow + " " * remaining
    return f"[{bar}]"


# ── Caching helpers ───────────────────────────────────────────────────────────

def _image_cache_key(image_path: Path, pipeline_version: int) -> str:
    # Include backend env vars so that switching YOLO/GPT invalidates the cache.
    detector_backend = os.environ.get("FISH_DETECTOR_BACKEND", "gpt")
    classifier_backend = os.environ.get("FISH_CLASSIFIER_BACKEND", "gpt")
    key_material = (
        f"{image_path.resolve()}"
        f"|v{pipeline_version}"
        f"|det={detector_backend}"
        f"|cls={classifier_backend}"
    )
    return hashlib.sha1(key_material.encode()).hexdigest()[:20]


def load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


# ── Image-mode evaluation ─────────────────────────────────────────────────────

@dataclass
class ImageEvalResult:
    image_path: str
    true_label: str
    pred_label: str
    confidence: float
    error: Optional[str] = None


async def _run_stage_a_image(
    image_path: Path,
    semaphore: asyncio.Semaphore,
) -> ImageEvalResult:
    """Run Stage A on a single image file, returning predicted object_type."""
    async with semaphore:
        try:
            result = await analyze_fish_photo(str(image_path), caption="")
            return ImageEvalResult(
                image_path=str(image_path),
                true_label=image_path.parent.name,
                pred_label=result.object_type,
                confidence=result.detection_confidence,
            )
        except Exception as e:
            return ImageEvalResult(
                image_path=str(image_path),
                true_label=image_path.parent.name,
                pred_label="error",
                confidence=0.0,
                error=str(e),
            )


async def _run_stage_b_image(
    image_path: Path,
    semaphore: asyncio.Semaphore,
) -> ImageEvalResult:
    """Run Stage B on a single image file, returning predicted species."""
    async with semaphore:
        try:
            result = await analyze_fish_photo(str(image_path), caption="")
            # If stage A rejected (not whole_fish), species = unknown_fish
            return ImageEvalResult(
                image_path=str(image_path),
                true_label=image_path.parent.name,
                pred_label=result.species_key,
                confidence=result.species_confidence,
            )
        except Exception as e:
            return ImageEvalResult(
                image_path=str(image_path),
                true_label=image_path.parent.name,
                pred_label="error",
                confidence=0.0,
                error=str(e),
            )


def _collect_images(base_dir: Path, classes: list[str]) -> dict[str, list[Path]]:
    """Collect image paths grouped by class (subdirectory name)."""
    grouped: dict[str, list[Path]] = {}
    for cls in classes:
        cls_dir = base_dir / cls
        if not cls_dir.is_dir():
            grouped[cls] = []
            continue
        images = sorted(
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        grouped[cls] = images
    return grouped


async def evaluate_stage(
    stage: str,
    base_dir: Path,
    concurrency: int = 3,
    use_cache: bool = False,
    cache_path: Optional[Path] = None,
) -> tuple[list[str], list[str], list[ImageEvalResult]]:
    """
    Evaluate a stage by running all images through the pipeline.
    Returns (y_true, y_pred, all_results).
    """
    classes = STAGE_A_CLASSES if stage == "a" else STAGE_B_CLASSES
    runner = _run_stage_a_image if stage == "a" else _run_stage_b_image

    grouped = _collect_images(base_dir, classes)
    total = sum(len(v) for v in grouped.values())
    stage_label = f"Stage {'A' if stage == 'a' else 'B'}"
    print(f"\nRunning {stage_label} evaluation ({total} images)...")

    cache: dict = load_cache(cache_path) if (use_cache and cache_path) else {}
    cache_dirty = False

    semaphore = asyncio.Semaphore(concurrency)
    all_results: list[ImageEvalResult] = []

    for cls in classes:
        images = grouped[cls]
        if not images:
            continue

        done = 0
        tasks = []
        cached_results: list[ImageEvalResult] = []

        for img_path in images:
            ckey = _image_cache_key(img_path, PIPELINE_VERSION) if use_cache else None
            if use_cache and ckey and ckey in cache:
                entry = cache[ckey]
                cached_results.append(ImageEvalResult(
                    image_path=entry["image_path"],
                    true_label=entry["true_label"],
                    pred_label=entry["pred_label"],
                    confidence=entry["confidence"],
                    error=entry.get("error"),
                ))
            else:
                tasks.append((img_path, ckey))

        # Print initial progress
        print(f"  {_progress_bar(len(cached_results), len(images))} "
              f"{len(cached_results)}/{len(images)} {cls} (from cache)", end="\r", flush=True)

        done = len(cached_results)
        all_results.extend(cached_results)

        # Run non-cached images
        if tasks:
            start = time.monotonic()

            async def run_one(img_p: Path, ckey: Optional[str]) -> ImageEvalResult:
                r = await runner(img_p, semaphore)
                return r, ckey

            pending = [asyncio.create_task(run_one(ip, ck)) for ip, ck in tasks]
            for coro in asyncio.as_completed(pending):
                r, ckey = await coro
                done += 1
                elapsed = time.monotonic() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(images) - done) / rate if rate > 0 else 0
                eta_str = f"ETA {eta:.0f}s" if eta > 1 else ""
                print(f"  {_progress_bar(done, len(images))} "
                      f"{done}/{len(images)} {cls}  {eta_str}   ", end="\r", flush=True)

                all_results.append(r)
                if use_cache and ckey:
                    cache[ckey] = {
                        "image_path": r.image_path,
                        "true_label": r.true_label,
                        "pred_label": r.pred_label,
                        "confidence": r.confidence,
                        "error": r.error,
                    }
                    cache_dirty = True

        # Final progress for this class
        print(f"  {_progress_bar(len(images), len(images))} "
              f"{len(images)}/{len(images)} {cls}           ")

    if use_cache and cache_dirty and cache_path:
        save_cache(cache_path, cache)

    # Build y_true / y_pred (errors counted as wrong prediction)
    y_true = [r.true_label for r in all_results]
    y_pred = [
        r.pred_label if not r.error else f"error_{r.true_label}"
        for r in all_results
    ]
    return y_true, y_pred, all_results


# ── Printing helpers ──────────────────────────────────────────────────────────

_LINE = "\u2500" * 57  # ─────


def print_confusion_matrix(
    cm: list[list[int]],
    classes: list[str],
    stage_label: str,
) -> None:
    col_w = max(len(c) for c in classes) + 2
    header_pad = 16

    print(f"\n{stage_label} Confusion Matrix (rows=actual, cols=predicted):")
    # Header row
    print(" " * header_pad + "".join(c.rjust(col_w) for c in classes))
    for i, cls in enumerate(classes):
        row_cells = "".join(str(cm[i][j]).rjust(col_w) for j in range(len(classes)))
        print(f"{cls:<{header_pad}}{row_cells}")


def print_per_class_metrics(
    per_class: dict[str, dict[str, float]],
    macro: dict[str, float],
    weighted: dict[str, float],
    stage_label: str,
) -> None:
    total_support = int(sum(v["support"] for v in per_class.values()))
    print(f"\n{stage_label} Per-Class Metrics:")
    header = f"{'Class':<16}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}"
    print(header)
    for cls, m in per_class.items():
        print(
            f"{cls:<16}"
            f"{m['precision']:>10.3f}"
            f"{m['recall']:>10.3f}"
            f"{m['f1']:>10.3f}"
            f"{int(m['support']):>10}"
        )
    print(_LINE)
    print(
        f"{'Macro avg':<16}"
        f"{macro['precision']:>10.3f}"
        f"{macro['recall']:>10.3f}"
        f"{macro['f1']:>10.3f}"
        f"{total_support:>10}"
    )
    print(
        f"{'Weighted avg':<16}"
        f"{weighted['precision']:>10.3f}"
        f"{weighted['recall']:>10.3f}"
        f"{weighted['f1']:>10.3f}"
        f"{total_support:>10}"
    )


# ── Acceptance gate ───────────────────────────────────────────────────────────

def check_stage_a_acceptance(
    per_class: dict[str, dict[str, float]],
    cm: list[list[int]],
    classes: list[str],
) -> tuple[bool, dict[str, tuple[float, bool, str]]]:
    """
    Returns (all_passed, criteria_results).
    criteria_results: name -> (actual_value, passed, detail_msg)
    """
    thresholds = ACCEPTANCE_CRITERIA["stage_a"]
    results: dict[str, tuple[float, bool, str]] = {}

    # lure_recall
    lure_recall = per_class.get("lure", {}).get("recall", 0.0)
    thresh = thresholds["lure_recall"]
    results["lure_recall"] = (lure_recall, lure_recall >= thresh, "")

    # whole_fish_recall
    wf_recall = per_class.get("whole_fish", {}).get("recall", 0.0)
    thresh = thresholds["whole_fish_recall"]
    results["whole_fish_recall"] = (wf_recall, wf_recall >= thresh, "")

    # fish_part_recall
    fp_recall = per_class.get("fish_part", {}).get("recall", 0.0)
    thresh = thresholds["fish_part_recall"]
    results["fish_part_recall"] = (fp_recall, fp_recall >= thresh, "")

    # lure_as_whole_fish_rate: how often actual lures are predicted as whole_fish
    lure_idx = classes.index("lure") if "lure" in classes else -1
    wf_idx = classes.index("whole_fish") if "whole_fish" in classes else -1
    if lure_idx >= 0 and wf_idx >= 0:
        lure_total = sum(cm[lure_idx])
        lure_as_wf = cm[lure_idx][wf_idx]
        rate = lure_as_wf / lure_total if lure_total > 0 else 0.0
    else:
        rate = 0.0
    thresh = thresholds["lure_as_whole_fish_rate"]
    results["lure_as_whole_fish_rate"] = (rate, rate < thresh, "")

    all_passed = all(v[1] for v in results.values())
    return all_passed, results


def check_stage_b_acceptance(
    per_class: dict[str, dict[str, float]],
    cm: list[list[int]],
    classes: list[str],
    y_true: list[str],
    y_pred: list[str],
) -> tuple[bool, dict[str, tuple[float, bool, str]]]:
    """
    Returns (all_passed, criteria_results).
    criteria_results: name -> (actual_value, passed, detail_msg)
    """
    thresholds = ACCEPTANCE_CRITERIA["stage_b"]
    results: dict[str, tuple[float, bool, str]] = {}

    # per_species_accuracy_min — each known species must hit ≥80% recall
    min_acc = thresholds["per_species_accuracy_min"]
    failing: list[str] = []
    worst_val = 1.0
    for cls in STAGE_B_CLASSES:
        if cls == "unknown_fish":
            continue  # unknown_fish is not a target species
        rec = per_class.get(cls, {}).get("recall", 0.0)
        if rec < worst_val:
            worst_val = rec
        if rec < min_acc and per_class.get(cls, {}).get("support", 0) > 0:
            failing.append(f"{cls}={rec:.2f}")
    if failing:
        detail = f"{', '.join(failing)} fail"
        results["per_species_accuracy_min"] = (worst_val, False, detail)
    else:
        results["per_species_accuracy_min"] = (worst_val, True, "")

    # unknown_rejection_rate — what fraction of all predictions are "unknown_fish"
    # (proxy for how often the model admits uncertainty)
    total = len(y_pred)
    unknown_preds = sum(1 for p in y_pred if p == "unknown_fish")
    rej_rate = unknown_preds / total if total > 0 else 0.0
    thresh = thresholds["unknown_rejection_rate"]
    results["unknown_rejection_rate"] = (rej_rate, rej_rate >= thresh, "")

    # pike_taimen_confusion — rate at which pike is predicted as taimen OR vice versa
    pike_idx = classes.index("pike") if "pike" in classes else -1
    taimen_idx = classes.index("taimen") if "taimen" in classes else -1
    if pike_idx >= 0 and taimen_idx >= 0:
        pike_as_taimen = cm[pike_idx][taimen_idx]
        taimen_as_pike = cm[taimen_idx][pike_idx]
        pike_total = sum(cm[pike_idx])
        taimen_total = sum(cm[taimen_idx])
        denom = pike_total + taimen_total
        confusion_rate = (pike_as_taimen + taimen_as_pike) / denom if denom > 0 else 0.0
    else:
        confusion_rate = 0.0
    thresh = thresholds["pike_taimen_confusion"]
    results["pike_taimen_confusion"] = (confusion_rate, confusion_rate < thresh, "")

    all_passed = all(v[1] for v in results.values())
    return all_passed, results


def print_acceptance_gate(
    stage_label: str,
    passed: bool,
    criteria: dict[str, tuple[float, bool, str]],
    thresholds: dict[str, float],
    strict: bool,
) -> None:
    print(f"\nACCEPTANCE CRITERIA — {stage_label}:")
    for name, (actual, ok, detail) in criteria.items():
        thresh = thresholds[name]
        icon = "OK" if ok else "FAIL"
        # Direction of comparison
        if name in ("lure_as_whole_fish_rate", "pike_taimen_confusion"):
            cmp_str = f"{actual:.3f} < {thresh:.3f}"
        elif name == "unknown_rejection_rate":
            cmp_str = f"{actual:.3f} >= {thresh:.3f}"
        else:
            cmp_str = f"{actual:.3f} >= {thresh:.3f}"
        suffix = f"  ({detail})" if detail else ""
        print(f"  [{icon}] {name}: {cmp_str}{suffix}")

    print(f"  {_LINE}")
    verdict = "PASSED" if passed else "FAILED"
    gate_note = "  (use --strict to gate deployment)" if not strict else ""
    print(f"  {stage_label}: {verdict}{gate_note}")


# ── Report export ─────────────────────────────────────────────────────────────

def build_report(
    stage_a_data: Optional[dict],
    stage_b_data: Optional[dict],
) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "stage_a": stage_a_data,
        "stage_b": stage_b_data,
    }


# ── Legacy JSON mode ──────────────────────────────────────────────────────────

async def run_test_case(tc: TestCase) -> TestResult:
    """Run one legacy test case and return result."""
    try:
        result = await analyze_fish_photo(tc.image_url, tc.caption)

        obj_correct = result.object_type == tc.expected_object_type
        species_correct = (
            tc.expected_species is None
            or result.species_key == tc.expected_species
            or (tc.expected_species == "any_fish" and result.species_key != "unknown_fish")
        )
        valid_correct = (
            (tc.must_reject and not result.is_valid_catch)
            or (not tc.must_reject and result.is_valid_catch)
        )

        return TestResult(
            test_case=tc,
            object_type_correct=obj_correct,
            species_correct=species_correct,
            valid_catch_correct=valid_correct,
            is_valid_catch=result.is_valid_catch,
            got_object_type=result.object_type,
            got_species=result.species_key,
            got_detection_conf=result.detection_confidence,
            got_species_conf=result.species_confidence,
        )
    except Exception as e:
        return TestResult(
            test_case=tc,
            object_type_correct=False,
            species_correct=False,
            valid_catch_correct=False,
            is_valid_catch=False,
            got_object_type="error",
            got_species="error",
            got_detection_conf=0.0,
            got_species_conf=0.0,
            error=str(e),
        )


def print_results(results: list[TestResult]) -> None:
    """Print formatted legacy evaluation report."""
    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print("\n" + "=" * 60)
    print("FISH VISION EVALUATION REPORT (Legacy JSON mode)")
    print(f"Passed: {passed}/{total}  ({passed / total * 100:.0f}%)")
    print("=" * 60)

    for r in results:
        status = "[PASS]" if r.passed else "[FAIL]"
        print(f"\n{status} | {r.test_case.description}")
        print(f"  Expected: type={r.test_case.expected_object_type}, "
              f"species={r.test_case.expected_species}, "
              f"reject={r.test_case.must_reject}")
        print(f"  Got:      type={r.got_object_type} ({'OK' if r.object_type_correct else 'X'}), "
              f"species={r.got_species} ({'OK' if r.species_correct else 'X'}), "
              f"valid={r.is_valid_catch} ({'OK' if r.valid_catch_correct else 'X'})")
        print(f"  Confidence: detection={r.got_detection_conf:.2f}, "
              f"species={r.got_species_conf:.2f}")
        if r.error:
            print(f"  ERROR: {r.error}")

    print("\n" + "=" * 60)
    print("\nACCEPTANCE CRITERIA:")
    criteria = [
        ("Lures rejected", [r for r in results if r.test_case.expected_object_type == "lure"]),
        ("Fish parts rejected", [r for r in results if r.test_case.expected_object_type == "fish_part"]),
        ("Fry rejected", [r for r in results if r.test_case.expected_object_type == "fry"]),
        ("Real fish accepted", [r for r in results if r.test_case.expected_object_type == "whole_fish"]),
    ]
    for name, group in criteria:
        if group:
            group_pass = sum(1 for r in group if r.passed)
            print(f"  {name}: {group_pass}/{len(group)}")
        else:
            print(f"  {name}: (no test cases)")


async def load_test_cases_from_file(path: str) -> list[TestCase]:
    """Load test cases from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [
        TestCase(
            image_url=tc["image_url"],
            caption=tc.get("caption", ""),
            expected_object_type=tc["expected_object_type"],
            expected_species=tc.get("expected_species"),
            description=tc.get("description", tc["image_url"]),
            must_reject=tc.get("must_reject", tc["expected_object_type"] != "whole_fish"),
            species_must_match=tc.get("species_must_match", True),
        )
        for tc in data
    ]


async def run_legacy_json_mode(cases_path: str) -> None:
    """Run legacy JSON smoke-test mode."""
    if not os.path.exists(cases_path):
        print(f"ERROR: test cases file not found: {cases_path}")
        sys.exit(1)

    cases = await load_test_cases_from_file(cases_path)
    print(f"Running {len(cases)} test cases (legacy JSON mode)...")
    results = []
    for tc in cases:
        print(f"  Testing: {tc.description[:50]}...", end=" ", flush=True)
        r = await run_test_case(tc)
        print("PASS" if r.passed else "FAIL")
        results.append(r)

    print_results(results)


# ── Image-mode full stage runner ──────────────────────────────────────────────

async def run_stage_eval(
    stage: str,
    base_dir: Path,
    concurrency: int,
    use_cache: bool,
    strict: bool,
    report_data: dict,
) -> bool:
    """
    Run image-mode evaluation for one stage.
    Returns True if acceptance criteria are met.
    Updates report_data in place.
    """
    cache_path = base_dir.parent / ".eval_cache.json" if use_cache else None
    classes = STAGE_A_CLASSES if stage == "a" else STAGE_B_CLASSES
    stage_label = "Stage A" if stage == "a" else "Stage B"

    y_true, y_pred, all_results = await evaluate_stage(
        stage=stage,
        base_dir=base_dir,
        concurrency=concurrency,
        use_cache=use_cache,
        cache_path=cache_path,
    )

    if not y_true:
        print(f"\nNo images found for {stage_label} in {base_dir}")
        return True  # vacuously passing

    cm = build_confusion_matrix(y_true, y_pred, classes)
    per_class = class_metrics_from_cm(cm, classes)
    macro = macro_avg(per_class)
    weighted = weighted_avg(per_class)
    acc = overall_accuracy(cm)

    print(f"\n{'=' * 60}")
    print(f"{stage_label} RESULTS  (accuracy={acc:.3f})")
    print(f"{'=' * 60}")

    print_confusion_matrix(cm, classes, stage_label)
    print_per_class_metrics(per_class, macro, weighted, stage_label)

    # Show error summary
    errors = [r for r in all_results if r.error]
    if errors:
        print(f"\n  [{len(errors)} images had errors during pipeline call]")
        for e in errors[:5]:
            print(f"    {Path(e.image_path).name}: {e.error}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")

    # Acceptance check
    if stage == "a":
        gate_passed, gate_results = check_stage_a_acceptance(per_class, cm, classes)
        thresholds = ACCEPTANCE_CRITERIA["stage_a"]
    else:
        gate_passed, gate_results = check_stage_b_acceptance(
            per_class, cm, classes, y_true, y_pred
        )
        thresholds = ACCEPTANCE_CRITERIA["stage_b"]

    print_acceptance_gate(stage_label, gate_passed, gate_results, thresholds, strict)

    # Save to report_data
    stage_key = f"stage_{stage}"
    report_data[stage_key] = {
        "accuracy": acc,
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "weighted_f1": weighted["f1"],
        "confusion_matrix": cm,
        "classes": classes,
        "per_class": {
            cls: {k: float(v) for k, v in m.items()}
            for cls, m in per_class.items()
        },
        "acceptance_criteria": {
            name: {"value": float(val), "passed": ok, "detail": detail}
            for name, (val, ok, detail) in gate_results.items()
        },
        "accepted": gate_passed,
        "total_images": len(y_true),
        "errors": len(errors),
    }

    return gate_passed


# ── Main entrypoint ───────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fish Vision Evaluation Runner v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--stage", choices=["a", "b"], help="Evaluate single stage (a or b)")
    mode.add_argument("--all", action="store_true", help="Evaluate both stages")
    mode.add_argument("--cases", metavar="FILE", help="Legacy JSON smoke-test mode")

    # Directory
    parser.add_argument(
        "--dir",
        metavar="DIR",
        help="Base directory for image-mode evaluation "
             "(default: data/eval_cases/stage_a or stage_b)",
    )

    # Options
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any acceptance criterion fails",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        metavar="N",
        help="Maximum concurrent API calls (default: 3)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        help="Cache pipeline results to .eval_cache.json (skip already-evaluated images)",
    )
    parser.add_argument(
        "--report",
        metavar="FILE",
        help="Export full results to a JSON report file",
    )

    # Legacy positional (backward compat): first positional = cases file
    parser.add_argument("cases_pos", nargs="?", metavar="CASES_FILE", help=argparse.SUPPRESS)

    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    # Set dummy env for standalone use
    os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test")

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set")
        sys.exit(1)

    # ── Legacy JSON mode ─────────────────────────────────────────────────────
    cases_file = args.cases or args.cases_pos
    if cases_file:
        await run_legacy_json_mode(cases_file)
        return

    # ── Image-mode evaluation ────────────────────────────────────────────────
    project_root = Path(__file__).parent.parent.parent.parent
    report_data: dict = {}
    all_passed = True

    stages_to_run: list[str] = []
    if args.all:
        stages_to_run = ["a", "b"]
    elif args.stage:
        stages_to_run = [args.stage]
    else:
        # Default: try both stages
        stages_to_run = ["a", "b"]

    for stage in stages_to_run:
        if args.dir:
            base_dir = Path(args.dir)
        else:
            subdir = "stage_a" if stage == "a" else "stage_b"
            base_dir = project_root / "data" / "eval_cases" / subdir

        if not base_dir.exists():
            print(f"\nDirectory not found: {base_dir}")
            print(f"  Create it with: mkdir -p {base_dir}")
            print(f"  Then add labeled images in subdirectories:")
            classes = STAGE_A_CLASSES if stage == "a" else STAGE_B_CLASSES
            for cls in classes:
                print(f"    {base_dir / cls}/")
            continue

        stage_passed = await run_stage_eval(
            stage=stage,
            base_dir=base_dir,
            concurrency=args.concurrency,
            use_cache=args.cache,
            strict=args.strict,
            report_data=report_data,
        )
        if not stage_passed:
            all_passed = False

    # ── Report export ─────────────────────────────────────────────────────────
    if args.report and report_data:
        report = build_report(
            stage_a_data=report_data.get("stage_a"),
            stage_b_data=report_data.get("stage_b"),
        )
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport written to: {report_path}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    print()
    if report_data:
        if all_passed:
            print("OVERALL: ALL CRITERIA PASSED")
        else:
            print("OVERALL: SOME CRITERIA FAILED")
            if args.strict:
                print("Exiting with code 1 (--strict mode).")
                sys.exit(1)
    else:
        print("No stages were evaluated (no data directories found or no stages specified).")


if __name__ == "__main__":
    asyncio.run(main())
