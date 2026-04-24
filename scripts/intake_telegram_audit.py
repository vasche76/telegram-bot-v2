#!/usr/bin/env python3
"""
intake_telegram_audit.py — Audit every photo in manifest.jsonl.

For each photo: verifies integrity with PIL, reads dimensions, computes SHA-256
hash, records file size, and flags low-resolution images.

source=telegram_private_2026-04-24, license=private_training_only

Usage:
    python3 scripts/intake_telegram_audit.py [--manifest PATH] [--export-dir PATH] [--output-dir PATH]

Output record fields (audit.jsonl):
    filename   — relative path as in manifest (e.g. "photos/photo_1@....jpg")
    sha256     — hex SHA-256 digest
    width      — pixel width (int) or null if corrupt
    height     — pixel height (int) or null if corrupt
    max_side   — max(width, height) or null if corrupt
    file_size  — bytes from stat()
    low_res    — true when max_side < 800 (excludes corrupt images)
    corrupt    — true when PIL cannot open or verify the file

Two-open pattern (mandatory):
    (1) open → verify()  — corruption check; invalidates the handle
    (2) re-open → .size  — read dimensions from a fresh handle
    Never read .size from the same handle that called verify().
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

LOW_RES_THRESHOLD = 800  # max-side pixels; < 800 → low_res=true
CHUNK_SIZE = 65536  # 64 KB read chunks for SHA-256
PROGRESS_EVERY = 1000  # log progress every N images

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of file in 64 KB chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def audit_image(path: Path) -> dict:
    """
    Return an audit record for a single image file.

    Implements the two-open PIL pattern:
      Pass 1: Image.open(path).verify() — integrity check (invalidates handle)
      Pass 2: Image.open(path) again   — read .size (fresh handle required)
    """
    from PIL import Image, UnidentifiedImageError  # noqa: PLC0415

    file_size = path.stat().st_size

    # Pass 1: integrity verification
    try:
        with Image.open(path) as img:
            img.verify()
        corrupt = False
    except (OSError, UnidentifiedImageError, Exception):
        corrupt = True

    width: int | None = None
    height: int | None = None
    max_side: int | None = None
    low_res: bool = False

    if not corrupt:
        # Pass 2: dimension read — MUST re-open; verify() invalidates the handle
        try:
            with Image.open(path) as img2:
                width, height = img2.size
            max_side = max(width, height)
            low_res = max_side < LOW_RES_THRESHOLD
        except Exception:
            # If re-open fails after verify passed, treat as corrupt
            corrupt = True
            width = height = max_side = None

    sha = _sha256(path)

    return {
        "sha256": sha,
        "width": width,
        "height": height,
        "max_side": max_side,
        "file_size": file_size,
        "low_res": low_res,
        "corrupt": corrupt,
    }


def run(manifest_path: Path, export_dir: Path, output_dir: Path) -> int:
    """
    Main pipeline entry point. Returns number of records written.
    """
    if not manifest_path.exists():
        log.error("manifest.jsonl not found: %s", manifest_path)
        log.error("Run intake_telegram_manifest.py first.")
        sys.exit(1)

    filenames: list[str] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                filenames.append(rec["filename"])

    total = len(filenames)
    log.info("Auditing %d photos from manifest", total)

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "audit.jsonl"

    corrupt_count = 0
    low_res_count = 0
    written = 0

    with audit_path.open("w", encoding="utf-8") as out_fh:
        for idx, filename in enumerate(filenames, start=1):
            if idx % PROGRESS_EVERY == 0 or idx == total:
                log.info("Progress: %d / %d", idx, total)

            photo_path = export_dir / filename
            if not photo_path.exists():
                rec = {
                    "filename": filename,
                    "sha256": None,
                    "width": None,
                    "height": None,
                    "max_side": None,
                    "file_size": None,
                    "low_res": False,
                    "corrupt": True,
                }
                log.warning("File not found: %s", filename)
            else:
                try:
                    fields = audit_image(photo_path)
                except Exception as exc:
                    log.warning("Unexpected error auditing %s: %s", filename, exc)
                    fields = {
                        "sha256": None,
                        "width": None, "height": None, "max_side": None,
                        "file_size": None, "low_res": False, "corrupt": True,
                    }
                rec = {"filename": filename, **fields}

            if rec["corrupt"]:
                corrupt_count += 1
            if rec.get("low_res"):
                low_res_count += 1

            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1

    log.info("Audit complete: %d records written", written)
    log.info("  corrupt : %d", corrupt_count)
    log.info("  low_res : %d (max_side < %d px)", low_res_count, LOW_RES_THRESHOLD)
    log.info("Audit written → %s", audit_path)
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit photo dimensions, SHA-256, and integrity for manifest photos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--manifest",
        default=str(REPO_ROOT / "data" / "intake_meta" / "tg_2026-04-24" / "manifest.jsonl"),
        help="Path to manifest.jsonl produced by intake_telegram_manifest.py",
    )
    p.add_argument(
        "--export-dir",
        default=str(Path.home() / "Downloads" / "Telegram Desktop" / "ChatExport_2026-04-24"),
        help="Path to the Telegram export directory (read-only).",
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "intake_meta" / "tg_2026-04-24"),
        help="Directory where audit.jsonl will be written.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    export_dir = Path(args.export_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    n = run(manifest_path, export_dir, output_dir)
    print(f"[OK] audit.jsonl written: {n} records → {output_dir / 'audit.jsonl'}")


if __name__ == "__main__":
    main()
