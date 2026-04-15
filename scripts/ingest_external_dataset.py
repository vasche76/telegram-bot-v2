#!/usr/bin/env python3
"""
ingest_external_dataset.py — Normalize and ingest externally downloaded datasets.

Handles three common external formats:
  1. Roboflow YOLO export   — images + labels/ (already YOLO format)
  2. Roboflow class-folders — images organized by class name
  3. Open Images V7 CSV     — CSV manifest + downloaded images

Usage:
    # From Roboflow YOLO export:
    python3 scripts/ingest_external_dataset.py \\
        --source ~/Downloads/FishLures.v2-yolo.zip \\
        --format roboflow_yolo \\
        --stage a \\
        --class-map lure:1,whole_fish:0

    # From Roboflow class folders:
    python3 scripts/ingest_external_dataset.py \\
        --source ~/Downloads/fish_dataset/ \\
        --format class_folders \\
        --stage b \\
        --class-map pike:pike,perch:perch

    # From Open Images downloaded images:
    python3 scripts/ingest_external_dataset.py \\
        --source ~/Downloads/openimages/ \\
        --format open_images \\
        --stage a \\
        --class-map Fish:whole_fish,Fishing_bait:lure

The script:
  - Validates each image (checks it opens correctly)
  - Deduplicates by MD5 hash
  - Normalizes filenames (no spaces, no Unicode issues)
  - Copies to the correct data/fish_dataset/stage_*/  directory
  - Records provenance in PROVENANCE_external.json

Placement:
  Stage A raw:     data/fish_dataset/stage_a/raw/{class_name}/
  Stage B species: data/fish_dataset/stage_b/{species_name}/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
STAGE_A_RAW = DATA_ROOT / "fish_dataset" / "stage_a" / "raw"
STAGE_B_DIR = DATA_ROOT / "fish_dataset" / "stage_b"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

# Valid Stage A class names
STAGE_A_CLASSES = {"whole_fish", "lure", "fish_part", "fry", "no_fish"}
# Valid Stage B species — expanded to include Salmonidae, Cyprinidae, Siluriformes
STAGE_B_SPECIES = {
    # Original 5 + unknown fallback
    "pike", "taimen", "grayling", "whitefish", "perch", "unknown_fish",
    # New Salmonidae
    "brown_trout", "rainbow_trout", "atlantic_salmon",
    # New Cyprinidae
    "common_carp", "crucian_carp", "bream", "roach", "ide",
    # New Siluriformes
    "wels_catfish",
}


def _banner(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def _ok(msg: str) -> None:
    print(f"[OK]  {msg}")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


# ─── Validation ──────────────────────────────────────────────────────────────

def _validate_image(path: Path) -> bool:
    """Return True if path is a valid openable image."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_filename(name: str) -> str:
    """Remove/replace characters unsafe for filenames."""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


# ─── Parse class-map argument ─────────────────────────────────────────────────

def parse_class_map(raw: str) -> dict[str, str]:
    """
    Parse 'src_class:dst_class,...' into a dict.
    Example: "Fish:whole_fish,Bait:lure" → {"Fish": "whole_fish", "Bait": "lure"}
    """
    result: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" not in pair:
            _warn(f"Skipping malformed class-map entry: '{pair}' (expected src:dst)")
            continue
        src, dst = pair.split(":", 1)
        result[src.strip()] = dst.strip()
    return result


# ─── Destination directory resolver ──────────────────────────────────────────

def get_dest_dir(stage: str, class_name: str) -> Path:
    """Return the destination directory for a given stage + class."""
    if stage == "a":
        if class_name not in STAGE_A_CLASSES:
            raise ValueError(
                f"Unknown Stage A class: '{class_name}'. "
                f"Valid: {sorted(STAGE_A_CLASSES)}"
            )
        return STAGE_A_RAW / class_name
    elif stage == "b":
        if class_name not in STAGE_B_SPECIES:
            raise ValueError(
                f"Unknown Stage B species: '{class_name}'. "
                f"Valid: {sorted(STAGE_B_SPECIES)}"
            )
        return STAGE_B_DIR / class_name
    else:
        raise ValueError(f"stage must be 'a' or 'b', got '{stage}'")


# ─── Provenance tracking ──────────────────────────────────────────────────────

class ProvenanceTracker:
    def __init__(self, source_name: str):
        self._by_dir: dict[Path, dict] = {}
        self._source_name = source_name

    def _get(self, dest_dir: Path) -> dict:
        if dest_dir not in self._by_dir:
            prov_path = dest_dir / "PROVENANCE_external.json"
            if prov_path.exists():
                try:
                    data = json.loads(prov_path.read_text(encoding="utf-8"))
                except Exception:
                    data = {}
            else:
                data = {}
            data.setdefault("source", self._source_name)
            data.setdefault("images", {})
            self._by_dir[dest_dir] = data
        return self._by_dir[dest_dir]

    def record(self, dest_dir: Path, filename: str, info: dict) -> None:
        self._get(dest_dir)["images"][filename] = info

    def has_md5(self, dest_dir: Path, md5: str) -> bool:
        prov = self._get(dest_dir)
        return any(v.get("md5") == md5 for v in prov["images"].values())

    def save_all(self) -> None:
        for dest_dir, data in self._by_dir.items():
            prov_path = dest_dir / "PROVENANCE_external.json"
            prov_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ─── Copy image with dedup ────────────────────────────────────────────────────

def copy_image(
    src: Path,
    dest_dir: Path,
    label: str,
    source_name: str,
    provenance: ProvenanceTracker,
    dry_run: bool = False,
) -> bool:
    """
    Copy src to dest_dir with deduplication and validation.
    Returns True if image was copied (not skipped).
    """
    # Validate image
    if not _validate_image(src):
        _warn(f"  Invalid image, skipping: {src.name}")
        return False

    # Check MD5 dedup
    md5 = _md5(src)
    dest_dir.mkdir(parents=True, exist_ok=True)

    if provenance.has_md5(dest_dir, md5):
        _info(f"  Duplicate (MD5), skipping: {src.name}")
        return False

    # Build safe destination filename
    safe_stem = _safe_filename(src.stem)[:80]
    ext = src.suffix.lower()
    if ext not in IMAGE_EXTS:
        ext = ".jpg"

    # Prefix to avoid collisions with existing files
    prefix = f"ext_{label}_"
    dest_name = prefix + safe_stem + ext

    # Resolve collision
    dest_path = dest_dir / dest_name
    counter = 1
    while dest_path.exists():
        dest_path = dest_dir / f"{prefix}{safe_stem}_{counter}{ext}"
        counter += 1

    if dry_run:
        _info(f"  [DRY-RUN] Would copy: {src.name} → {dest_path.name}")
        return True

    shutil.copy2(src, dest_path)
    provenance.record(dest_dir, dest_path.name, {
        "source": source_name,
        "original_name": src.name,
        "md5": md5,
        "label": label,
    })
    return True


# ─── Format: Roboflow YOLO export ─────────────────────────────────────────────

def ingest_roboflow_yolo(
    source: Path,
    stage: str,
    class_map: dict[str, str],
    dry_run: bool,
) -> int:
    """
    Ingest a Roboflow YOLO export.

    Expected structure (after unzip):
        train/images/*.jpg
        train/labels/*.txt
        val/images/*.jpg
        val/labels/*.txt
        data.yaml  ← contains class names

    Strategy: read data.yaml to get class name → index mapping,
    then for each image find its label, determine the dominant class,
    and copy to the appropriate raw directory.
    """
    # Auto-unzip if necessary
    work_dir = source
    if source.suffix == ".zip":
        _info(f"  Unzipping {source.name}...")
        tmp = source.parent / (source.stem + "_extracted")
        with zipfile.ZipFile(source) as zf:
            zf.extractall(tmp)
        work_dir = tmp
        _info(f"  Extracted to {work_dir}")

    # Find data.yaml
    yaml_paths = list(work_dir.rglob("data.yaml")) + list(work_dir.rglob("*.yaml"))
    if not yaml_paths:
        _fail("No data.yaml found in Roboflow export. Cannot determine class names.")
        return 0

    yaml_path = yaml_paths[0]
    _info(f"  Reading class names from {yaml_path.name}")

    # Parse YAML without PyYAML (keep it simple: read 'names:' list)
    names: list[str] = []
    in_names = False
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if stripped.startswith("-"):
                names.append(stripped.lstrip("-").strip().strip("'\""))
            elif stripped and not stripped.startswith("#"):
                in_names = False

    _info(f"  Classes from YAML: {names}")
    if not names:
        _fail("Could not parse class names from YAML file.")
        return 0

    # Build index → dest class mapping
    idx_to_dest: dict[int, Optional[str]] = {}
    for i, name in enumerate(names):
        dest_class = class_map.get(name)
        if dest_class:
            idx_to_dest[i] = dest_class
        else:
            _warn(f"  Class '{name}' not in class_map — will be skipped")
            idx_to_dest[i] = None

    provenance = ProvenanceTracker(f"Roboflow YOLO export: {source.name}")
    copied = 0

    for split in ["train", "val", "test"]:
        images_dir = work_dir / split / "images"
        labels_dir = work_dir / split / "labels"
        if not images_dir.exists():
            continue

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            label_path = labels_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                continue

            # Determine dominant class from label file (class with most area)
            dominant_cls: Optional[int] = None
            max_area = 0.0
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    try:
                        cls_id = int(parts[0])
                        w, h = float(parts[3]), float(parts[4])
                        area = w * h
                        if area > max_area:
                            max_area = area
                            dominant_cls = cls_id
                    except ValueError:
                        continue

            if dominant_cls is None:
                continue

            dest_class = idx_to_dest.get(dominant_cls)
            if dest_class is None:
                continue

            try:
                dest_dir = get_dest_dir(stage, dest_class)
            except ValueError as e:
                _warn(str(e))
                continue

            ok = copy_image(img_path, dest_dir, dest_class, source.name, provenance, dry_run)
            if ok:
                copied += 1

    provenance.save_all()
    return copied


# ─── Format: Class folders ────────────────────────────────────────────────────

def ingest_class_folders(
    source: Path,
    stage: str,
    class_map: dict[str, str],
    dry_run: bool,
) -> int:
    """
    Ingest images organized in class-named subfolders.

    source/
        pike/img1.jpg ...
        perch/img1.jpg ...
    """
    provenance = ProvenanceTracker(f"Class folder import: {source.name}")
    copied = 0

    if source.suffix == ".zip":
        _info(f"  Unzipping {source.name}...")
        tmp = source.parent / (source.stem + "_extracted")
        with zipfile.ZipFile(source) as zf:
            zf.extractall(tmp)
        source = tmp

    for folder in sorted(source.iterdir()):
        if not folder.is_dir():
            continue
        src_class = folder.name
        dest_class = class_map.get(src_class, src_class)  # identity if not in map

        try:
            dest_dir = get_dest_dir(stage, dest_class)
        except ValueError as e:
            _warn(f"  {e} — skipping folder '{src_class}'")
            continue

        n = 0
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            ok = copy_image(img_path, dest_dir, dest_class, source.name, provenance, dry_run)
            if ok:
                n += 1
                copied += 1

        _info(f"  {src_class} → {dest_class}: {n} images ingested")

    provenance.save_all()
    return copied


# ─── Format: Open Images CSV ──────────────────────────────────────────────────

def ingest_open_images(
    source: Path,
    stage: str,
    class_map: dict[str, str],
    dry_run: bool,
) -> int:
    """
    Ingest images from an Open Images V7 download directory.

    Expected structure:
        source/
            images/      ← JPG files named by image ID
            labels/      ← CSV or annotation files (optional, for bboxes)
            manifest.csv ← ImageID, OriginalURL, LabelName (from fiftyone or oi-downloader)

    If manifest.csv is absent, falls back to using folder name as class label.
    """
    provenance = ProvenanceTracker(f"Open Images: {source.name}")
    copied = 0

    images_dir = source / "images"
    if not images_dir.exists():
        # Try treating source itself as images directory
        images_dir = source

    manifest_path = source / "manifest.csv"
    id_to_label: dict[str, str] = {}

    if manifest_path.exists():
        _info(f"  Reading manifest: {manifest_path.name}")
        for line in manifest_path.read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                img_id = parts[0].strip()
                oi_label = parts[2].strip()
                dest_class = class_map.get(oi_label, class_map.get(oi_label.replace(" ", "_")))
                if dest_class:
                    id_to_label[img_id] = dest_class
    else:
        _info("  No manifest.csv found — using class_map folder-name fallback")

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        # Try to determine class from manifest or default
        img_id = img_path.stem
        dest_class = id_to_label.get(img_id)

        if dest_class is None:
            # Fallback: first value in class_map
            if len(class_map) == 1:
                dest_class = next(iter(class_map.values()))
            else:
                _warn(f"  Cannot determine class for {img_path.name} — skipping")
                continue

        try:
            dest_dir = get_dest_dir(stage, dest_class)
        except ValueError as e:
            _warn(str(e))
            continue

        ok = copy_image(img_path, dest_dir, dest_class, source.name, provenance, dry_run)
        if ok:
            copied += 1

    provenance.save_all()
    return copied


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Normalize and ingest external fish datasets (Roboflow, OpenImages, class folders).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", required=True,
                   help="Path to source dataset (zip file or directory)")
    p.add_argument("--format", required=True,
                   choices=["roboflow_yolo", "class_folders", "open_images"],
                   help="Dataset format")
    p.add_argument("--stage", required=True, choices=["a", "b"],
                   help="Target stage: 'a' (detector raw) or 'b' (species classifier)")
    p.add_argument("--class-map", required=True,
                   help="Class name mapping: 'src_class:dst_class,...'")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be done without copying files")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        _fail(f"Source not found: {source}")
        sys.exit(1)

    class_map = parse_class_map(args.class_map)
    if not class_map:
        _fail("class-map is empty or invalid. Example: 'Fish:whole_fish,Bait:lure'")
        sys.exit(1)

    _banner(f"Ingesting External Dataset")
    _info(f"  Source  : {source}")
    _info(f"  Format  : {args.format}")
    _info(f"  Stage   : {args.stage}")
    _info(f"  Class map: {class_map}")
    if args.dry_run:
        _info("  DRY-RUN mode — no files will be copied")

    if args.format == "roboflow_yolo":
        n = ingest_roboflow_yolo(source, args.stage, class_map, args.dry_run)
    elif args.format == "class_folders":
        n = ingest_class_folders(source, args.stage, class_map, args.dry_run)
    elif args.format == "open_images":
        n = ingest_open_images(source, args.stage, class_map, args.dry_run)
    else:
        _fail(f"Unknown format: {args.format}")
        sys.exit(1)

    _banner("Done")
    _ok(f"Ingested: {n} images")
    print()
    _info("Next: run scripts/build_dataset.py to rebuild labels and validate")


if __name__ == "__main__":
    main()
