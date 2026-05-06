#!/usr/bin/env python3
"""
train_mvp_structural.py — MVP binary structural classifier (fish vs not_fish_or_other).

Trains a YOLOv8n-cls image classifier on the MVP structural dataset.
This is the correct training script for the binary MVP baseline produced by
build_mvp_training_dataset.py (as opposed to train_stage_a.py which trains
the 5-class YOLO detection model on a separate labeled dataset).

Prerequisites:
    pip install ultralytics   # includes torch, torchvision, etc.

Data layout expected (produced by build_mvp_training_dataset.py --copy-images):
    data/mvp_training/structural/
        train/fish/                ← fish images (external public only)
        train/not_fish_or_other/   ← non-fish images (iNat birds + Wikimedia lures)
        val/fish/
        val/not_fish_or_other/
        test/fish/
        test/not_fish_or_other/

NOTE: Telegram-private images cannot be copied by the builder (no local_path).
      They are counted in the manifest but absent from the physical layout.
      The training dataset is therefore external-public-only for now.

Model outputs:
    data/fish_models/mvp_structural_v1.pt   ← saved best weights
    data/fish_models/mvp_structural_report.json ← metrics report
    data/runs/mvp_structural/               ← full YOLO run directory

Experimental marker: model is saved with metadata flag experimental=True.
Do NOT deploy to production.

Usage:
    python3 scripts/train_mvp_structural.py
    python3 scripts/train_mvp_structural.py --epochs 30 --device mps

source=external_public_only, license=CC-BY-4.0_CC0-1.0_CC-BY-SA
experimental=True
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
STRUCTURAL_DIR = DATA_ROOT / "mvp_training" / "structural"
MODELS_DIR = DATA_ROOT / "fish_models"
MANIFESTS_DIR = DATA_ROOT / "mvp_training" / "manifests"

MIN_IMAGES_PER_CLASS = 10


def _check_environment() -> None:
    try:
        import ultralytics  # noqa: F401
        print(f"[OK]  ultralytics {ultralytics.__version__} available")
    except ImportError:
        print("[FAIL] ultralytics is not installed.", file=sys.stderr)
        print("       Install it with:  pip install ultralytics", file=sys.stderr)
        print("       Then re-run this script.", file=sys.stderr)
        sys.exit(1)


def _check_data() -> None:
    for split in ("train", "val"):
        for cls in ("fish", "not_fish_or_other"):
            d = STRUCTURAL_DIR / split / cls
            if not d.exists():
                print(f"[FAIL] Missing directory: {d}", file=sys.stderr)
                print(
                    "       Run:  python3 scripts/build_mvp_training_dataset.py --copy-images",
                    file=sys.stderr,
                )
                sys.exit(1)
            count = sum(1 for f in d.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
            if count < MIN_IMAGES_PER_CLASS:
                print(f"[FAIL] Too few images in {d}: {count} < {MIN_IMAGES_PER_CLASS}", file=sys.stderr)
                sys.exit(1)
            print(f"[OK]  {split}/{cls}: {count} images")


def _train(epochs: int, batch: int, device: str) -> "ultralytics.engine.results.Results":
    from ultralytics import YOLO

    print(f"\n{'=' * 60}")
    print(f"  MVP Structural Baseline — YOLOv8n-cls")
    print(f"  Epochs={epochs}  Batch={batch}  Device={device}")
    print(f"  experimental=True — DO NOT deploy to production")
    print(f"{'=' * 60}\n")

    model = YOLO("yolov8n-cls.pt")

    results = model.train(
        data=str(STRUCTURAL_DIR),
        epochs=epochs,
        imgsz=224,
        batch=batch,
        patience=10,
        project=str(DATA_ROOT / "runs"),
        name="mvp_structural",
        exist_ok=True,
        device=device,
        verbose=True,
    )
    return results


def _save_report(results, epochs: int, device: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Copy best weights
    run_dir = DATA_ROOT / "runs" / "mvp_structural"
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_dir / "weights" / "last.pt"

    if best_pt.exists():
        import shutil
        dest = MODELS_DIR / "mvp_structural_v1.pt"
        shutil.copy2(best_pt, dest)
        print(f"[OK]  Model saved: {dest}")
    else:
        print("[WARN] No weights file found in run directory", file=sys.stderr)

    # Extract metrics
    metrics: dict = {"experimental": True}
    try:
        rd = results.results_dict
        for key in ("metrics/accuracy_top1", "metrics/accuracy_top5",
                    "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
            if key in rd:
                metrics[key.replace("metrics/", "").replace("(B)", "")] = round(float(rd[key]), 4)
    except Exception:
        pass

    # Split counts from manifest
    split_report_path = MANIFESTS_DIR / "mvp_split_report.json"
    split_data: dict = {}
    if split_report_path.exists():
        split_data = json.loads(split_report_path.read_text())

    report = {
        "schema_version": "mvp_structural_report_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experimental": True,
        "model": "yolov8n-cls",
        "training_epochs": epochs,
        "device": device,
        "dataset": {
            "manifest": str(MANIFESTS_DIR / "mvp_split_report.json"),
            "note": "Telegram-private records counted in manifest but absent from physical layout",
            "physical_train_fish": sum(
                1 for f in (STRUCTURAL_DIR / "train" / "fish").iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ) if (STRUCTURAL_DIR / "train" / "fish").exists() else 0,
            "physical_train_nofish": sum(
                1 for f in (STRUCTURAL_DIR / "train" / "not_fish_or_other").iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            ) if (STRUCTURAL_DIR / "train" / "not_fish_or_other").exists() else 0,
        },
        "metrics": metrics,
        "manifest_summary": split_data.get("data_summary", {}),
        "gate_failures": split_data.get("gate_failures", []),
    }

    report_path = MODELS_DIR / "mvp_structural_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK]  Report saved: {report_path}")
    print(f"\n[NOTE] Model is marked EXPERIMENTAL. Metrics reflect external-public images only.")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train MVP binary structural classifier (fish vs not_fish_or_other)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", type=str, default="cpu",
                   help="'cpu', 'mps' (Apple Silicon), '0' (GPU)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _check_environment()
    _check_data()

    results = _train(args.epochs, args.batch, args.device)
    _save_report(results, args.epochs, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
