#!/usr/bin/env python3
"""
check_duplicates.py — Detect (and optionally remove) duplicate images in the fish dataset.

Scans all images in stage_a/raw/ and stage_b/ using MD5 hash of file content.
Files with identical hashes are duplicates — same content regardless of filename.

Usage:
    python3 scripts/check_duplicates.py [--delete] [--help]

Options:
    --delete    Remove duplicate files, keeping the first file found for each hash.

Exit codes:
    0 — No duplicates found
    1 — Duplicates found (or deleted)
"""

import sys
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_A_RAW = REPO_ROOT / "data" / "fish_dataset" / "stage_a" / "raw"
STAGE_B = REPO_ROOT / "data" / "fish_dataset" / "stage_b"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SCAN_ROOTS = [STAGE_A_RAW, STAGE_B]

CHUNK_SIZE = 65536  # 64 KB read chunks for hashing large files

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def md5_file(path: Path) -> str:
    """Return hex MD5 digest of file contents."""
    h = hashlib.md5()
    try:
        with path.open("rb") as fh:
            while True:
                chunk = fh.read(CHUNK_SIZE)
                if not chunk:
                    break
                h.update(chunk)
    except OSError as exc:
        print(f"  [WARN] Cannot read {path}: {exc}", file=sys.stderr)
        return ""
    return h.hexdigest()


def collect_images(roots: list[Path]) -> list[Path]:
    """Collect all image files under the given root directories."""
    images: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(f)
    return sorted(images)


def separator(char: str = "-", width: int = 70) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def find_duplicates(image_paths: list[Path]) -> dict[str, list[Path]]:
    """
    Hash every image and group by MD5.
    Returns dict of hash -> [paths] for hashes that appear more than once.
    """
    hash_map: dict[str, list[Path]] = defaultdict(list)
    total = len(image_paths)

    for idx, path in enumerate(image_paths, start=1):
        # Progress indicator for large datasets
        if total > 100 and idx % 100 == 0:
            print(f"  Hashing: {idx}/{total} ...", flush=True)

        digest = md5_file(path)
        if digest:
            hash_map[digest].append(path)

    # Keep only groups with duplicates
    return {h: paths for h, paths in hash_map.items() if len(paths) > 1}


def delete_duplicates(duplicate_groups: dict[str, list[Path]]) -> int:
    """
    Delete all but the first file in each duplicate group.
    Returns count of deleted files.
    """
    deleted = 0
    for paths in duplicate_groups.values():
        kept = paths[0]
        for path in paths[1:]:
            try:
                path.unlink()
                print(f"  Deleted: {path}")
                deleted += 1
            except OSError as exc:
                print(f"  [ERROR] Could not delete {path}: {exc}", file=sys.stderr)
    return deleted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect (and optionally remove) duplicate images in the fish dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Remove duplicate files, keeping the first copy found for each hash.",
    )
    args = parser.parse_args()

    w = 70
    print()
    print(separator("=", w))
    print(" DUPLICATE IMAGE DETECTOR".center(w))
    print(separator("=", w))

    print(f"\n  Scanning directories:")
    for root in SCAN_ROOTS:
        status = "exists" if root.exists() else "not found"
        print(f"    {root}  [{status}]")

    print()
    images = collect_images(SCAN_ROOTS)
    print(f"  Total image files found: {len(images)}")

    if not images:
        print("\n  No images to scan. Dataset appears to be empty.")
        print(separator("=", w))
        return 0

    print(f"  Computing MD5 hashes ...\n")
    duplicate_groups = find_duplicates(images)

    if not duplicate_groups:
        print("  No duplicates found. All images are unique.")
        print()
        print(separator("=", w))
        print("  Result: PASS — 0 duplicate files.")
        print(separator("=", w))
        print()
        return 0

    # Report duplicates
    total_duplicate_files = sum(len(paths) - 1 for paths in duplicate_groups.values())
    total_groups = len(duplicate_groups)

    print(separator("-", w))
    print(f"  Found {total_groups} duplicate group(s) — {total_duplicate_files} redundant file(s):")
    print(separator("-", w))

    for group_idx, (digest, paths) in enumerate(sorted(duplicate_groups.items()), start=1):
        print(f"\n  Group {group_idx}  (MD5: {digest})")
        for file_idx, path in enumerate(paths):
            marker = "KEEP" if file_idx == 0 else "DUPE"
            try:
                size_str = f"{path.stat().st_size:,} bytes"
            except OSError:
                size_str = "unknown size"
            print(f"    [{marker}] {path}  ({size_str})")

    print()
    print(separator("=", w))

    if args.delete:
        print(f"  --delete flag set. Removing {total_duplicate_files} redundant file(s) ...")
        print()
        deleted = delete_duplicates(duplicate_groups)
        print()
        print(f"  Deleted {deleted} file(s).")
        print(f"  Result: DONE — {deleted} duplicate(s) removed.")
    else:
        print(
            f"  Result: FAIL — {total_duplicate_files} duplicate file(s) found in "
            f"{total_groups} group(s)."
        )
        print(f"  Run with --delete to remove them automatically.")

    print(separator("=", w))
    print()

    # Exit 1 whether we found duplicates (even if deleted — signals something was wrong)
    return 1


if __name__ == "__main__":
    sys.exit(main())
