#!/usr/bin/env python3
"""
fetch_wikimedia_lures.py — Download CC-licensed fishing lure photos from Wikimedia Commons.

Legal basis:
  - Uses official MediaWiki API (action=query) — no scraping.
  - Checks license per file via imageinfo. Only CC0, CC-BY, CC-BY-SA, Public Domain.
  - Provenance recorded in PROVENANCE.json.

Why Wikimedia Commons for lures?
  - iNaturalist does not have fishing tackle (only wildlife).
  - Wikimedia has high-quality encyclopedia images of fishing lures.
  - License is verifiable per file via the API.

Usage:
    python3 scripts/fetch_wikimedia_lures.py [--max 60] [--dry-run]

Output:
    data/fish_dataset/stage_a/raw/lure/    ← lure images for Stage A detector
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional

try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()
_OPENER = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_SSL_CTX))

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
LURE_DIR = DATA_ROOT / "fish_dataset" / "stage_a" / "raw" / "lure"
WHOLE_FISH_WIKI_DIR = DATA_ROOT / "fish_dataset" / "stage_a" / "raw" / "whole_fish"

WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
API_DELAY = 1.2  # seconds between API calls
IMAGE_DOWNLOAD_DELAY = 2.5  # seconds between image downloads (Wikimedia is strict)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Title fragments that indicate non-fishing results to skip
TITLE_BLOCKLIST = [
    "hybrid", "plug-in", "toyota", "kia", "car ", "automobile",
    "socket", "electric", "charger", "voltmeter", "connecto",
]

# These license strings (lowercased, partial match) are accepted.
ALLOWED_LICENSES = {
    "cc0",
    "cc-by",
    "cc by",
    "public domain",
    "pd ",
    "pd-",
    "cc-by-sa",
    "cc by-sa",
    "cc-by 4",
    "cc-by-sa 4",
    "attribution",
}

# Search queries for lure images.
# Order matters: more specific queries yield cleaner results.
LURE_SEARCH_QUERIES = [
    "fishing wobbler",
    "fishing spinner",
    "fishing spoon bait",
    "fishing jig tackle",
    "artificial fishing bait",
    "rapala lure",
    "treble hook lure",
]

# For supplementary whole_fish images
FISH_SEARCH_QUERIES = [
    "pike fish caught",
    "European perch fish",
    "grayling fish caught",
]


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


def _api_get(params: dict) -> dict:
    """Query Wikimedia Commons API."""
    params["format"] = "json"
    params["origin"] = "*"
    query = urllib.parse.urlencode(params)
    url = f"{WIKIMEDIA_API}?{query}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "telegram-fish-bot/1.0 (ML training data collection, bot owner contact)",
            "Accept": "application/json",
        },
    )
    try:
        with _OPENER.open(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        _warn(f"API error: {exc}")
        return {}


def _is_allowed_license(license_text: str) -> bool:
    """Return True if the license text indicates an open/CC license."""
    lt = license_text.lower()
    return any(allowed in lt for allowed in ALLOWED_LICENSES)


def _search_images(query: str, limit: int = 30) -> list[str]:
    """
    Search for image filenames on Wikimedia Commons matching query.
    Returns list of file titles ("File:Foo.jpg").
    """
    params = {
        "action": "query",
        "list": "search",
        "srnamespace": "6",  # File namespace
        "srsearch": query,
        "srlimit": min(limit, 50),
        "srprop": "title",
    }
    data = _api_get(params)
    results = data.get("query", {}).get("search", [])
    return [r["title"] for r in results if "title" in r]


def _get_image_info(titles: list[str]) -> list[dict]:
    """
    Retrieve imageinfo (license, URL) for a batch of file titles.
    Returns list of info dicts.
    """
    if not titles:
        return []

    # API accepts up to 50 titles at once
    titles_str = "|".join(t.replace(" ", "_") for t in titles[:50])

    params = {
        "action": "query",
        "titles": titles_str,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime",
    }
    data = _api_get(params)
    pages = data.get("query", {}).get("pages", {})

    results = []
    for page_id, page in pages.items():
        if page_id == "-1":
            continue
        imageinfo_list = page.get("imageinfo", [])
        if not imageinfo_list:
            continue
        ii = imageinfo_list[0]

        # Extract license from extmetadata
        meta = ii.get("extmetadata", {})
        license_text = (
            meta.get("License", {}).get("value", "")
            or meta.get("LicenseShortName", {}).get("value", "")
            or meta.get("UsageTerms", {}).get("value", "")
        )
        attribution = meta.get("Attribution", {}).get("value", "") or meta.get("Artist", {}).get("value", "")

        # Skip if not an allowed license
        if not license_text or not _is_allowed_license(license_text):
            continue

        # Skip non-image MIME types
        mime = ii.get("mime", "")
        if not mime.startswith("image/"):
            continue

        # Use direct url (not thumbnail) to avoid CDN bot blocking
        img_url = ii.get("url") or ii.get("thumburl")
        if not img_url:
            continue

        # Check image size — skip tiny images
        width = ii.get("thumbwidth") or ii.get("width") or 0
        height = ii.get("thumbheight") or ii.get("height") or 0
        if width < 200 or height < 200:
            continue

        title = page.get("title", "unknown")
        results.append({
            "title": title,
            "url": img_url,
            "license": license_text,
            "attribution": attribution,
            "width": width,
            "height": height,
        })

    return results


WIKIMEDIA_FILE_PATH = "https://commons.wikimedia.org/wiki/Special:FilePath/"
# Wikimedia recommends thumbnail sizes to avoid 429 on direct CDN requests.
# 800px is a good balance: enough detail for training, below the CDN rate-limit tier.
WIKIMEDIA_THUMB_WIDTH = 800


def _url_to_thumbnail_url(url: str, title: str) -> str:
    """
    Build a Wikimedia thumbnail URL via Special:FilePath?width=N.

    Per https://w.wiki/GHai, use thumbnail images at supported sizes to
    avoid HTTP 429 errors on direct CDN (upload.wikimedia.org) downloads.
    Special:FilePath with ?width= returns a redirect to the resized CDN thumb.
    """
    filename = title.replace("File:", "").replace(" ", "_")
    import urllib.parse
    encoded = urllib.parse.quote(filename)
    return f"{WIKIMEDIA_FILE_PATH}{encoded}?width={WIKIMEDIA_THUMB_WIDTH}"


def _download_image(url: str, dest_path: Path, title: str = "") -> bool:
    # Use thumbnail URL (Special:FilePath?width=800) to comply with Wikimedia
    # rate-limit guidance (avoids HTTP 429 on direct CDN requests).
    if title:
        effective_url = _url_to_thumbnail_url(url, title)
    else:
        effective_url = url

    req = urllib.request.Request(
        effective_url,
        headers={
            "User-Agent": "FishRecognitionBot/1.0 (educational research, thumbnail download)",
            "Accept": "image/jpeg,image/png,image/*",
        },
    )
    try:
        with _OPENER.open(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 5000:
            return False
        dest_path.write_bytes(data)
        return True
    except Exception as exc:
        _warn(f"Download failed: {url} — {exc}")
        return False


def _count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def _load_provenance(prov_path: Path) -> dict:
    if prov_path.exists():
        try:
            return json.loads(prov_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"source": "Wikimedia Commons", "images": {}}


def _save_provenance(prov_path: Path, data: dict) -> None:
    prov_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ─── Main download logic ────────────────────────────────────────────────────

def download_category(
    search_queries: list[str],
    dest_dir: Path,
    max_count: int,
    class_name: str,
    also_copy_to: Optional[Path] = None,
    dry_run: bool = False,
) -> int:
    """
    Download images from Wikimedia Commons using search_queries into dest_dir.
    Returns count of newly downloaded images.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    if also_copy_to:
        also_copy_to.mkdir(parents=True, exist_ok=True)

    prov_path = dest_dir / "PROVENANCE.json"
    provenance = _load_provenance(prov_path)

    already = _count_images(dest_dir)
    if already >= max_count:
        _info(f"  {class_name}: already have {already} images, skipping.")
        return 0

    remaining = max_count - already
    downloaded = 0

    for query in search_queries:
        if downloaded >= remaining:
            break

        _info(f"  Searching: '{query}'")
        titles = _search_images(query, limit=50)
        _info(f"    Found {len(titles)} file candidates")

        if not titles:
            time.sleep(API_DELAY)
            continue

        # Get imageinfo in batches of 20
        for batch_start in range(0, len(titles), 20):
            if downloaded >= remaining:
                break

            batch = titles[batch_start:batch_start + 20]
            infos = _get_image_info(batch)
            _info(f"    {len(infos)}/{len(batch)} files have open licenses")

            for info in infos:
                if downloaded >= remaining:
                    break

                title = info["title"]

                # Skip clearly non-fishing results
                title_lower = title.lower()
                if any(block in title_lower for block in TITLE_BLOCKLIST):
                    _info(f"    Skipping non-fishing result: {title[:60]}")
                    continue

                img_url = info["url"]
                license_text = info["license"]

                # Build filename from title
                title_clean = title.replace("File:", "").replace(" ", "_")
                ext = os.path.splitext(title_clean)[-1].lower()
                if ext not in IMAGE_EXTS:
                    ext = ".jpg"

                # Sanitize filename
                safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in title_clean)
                filename = f"wiki_{class_name}_{safe_name}"
                if not filename.endswith(ext):
                    filename += ext

                dest = dest_dir / filename[:200]  # cap filename length

                if dest.exists():
                    _info(f"    Skipping existing: {filename[:60]}")
                    downloaded += 1
                    break

                if dry_run:
                    _info(f"    [DRY-RUN] Would download: {filename[:60]}")
                    _info(f"      License: {license_text}")
                    downloaded += 1
                    continue

                _info(f"    Downloading [{downloaded+1}/{remaining}]: {filename[:60]}")
                success = _download_image(img_url, dest, title=title)
                if not success:
                    continue

                provenance["images"][str(dest.name)] = {
                    "title": title,
                    "source": "Wikimedia Commons",
                    "url": img_url,
                    "license": license_text,
                    "attribution": info.get("attribution", "Wikimedia Commons contributor"),
                    "label": class_name,
                }

                # Copy to also_copy_to if requested
                if also_copy_to:
                    copy_dest = also_copy_to / dest.name
                    if not copy_dest.exists():
                        copy_dest.write_bytes(dest.read_bytes())

                downloaded += 1
                time.sleep(IMAGE_DOWNLOAD_DELAY)

            time.sleep(API_DELAY)

    _save_provenance(prov_path, provenance)
    new_total = _count_images(dest_dir)
    _ok(f"  {class_name}: {downloaded} new downloads, {new_total} total")
    return downloaded


# ─── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download CC-licensed fishing lure photos from Wikimedia Commons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--max", type=int, default=60, help="Max lure images to download")
    p.add_argument("--fish-max", type=int, default=30, help="Max supplementary fish images from Wikimedia")
    p.add_argument("--skip-fish", action="store_true", help="Skip supplementary fish download")
    p.add_argument("--dry-run", action="store_true", help="Dry run — show what would be downloaded")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _banner("Wikimedia Commons Lure/Fish Downloader")
    _info(f"Lure target: {args.max} images")
    if args.dry_run:
        _info("DRY-RUN mode")

    # ── Download lures ──
    _banner("Fishing Lures (Stage A — lure class)")
    n_lures = download_category(
        search_queries=LURE_SEARCH_QUERIES,
        dest_dir=LURE_DIR,
        max_count=args.max,
        class_name="lure",
        dry_run=args.dry_run,
    )

    # ── Download supplementary fish ──
    if not args.skip_fish:
        _banner("Supplementary Fish Photos (Stage A — whole_fish)")
        n_fish = download_category(
            search_queries=FISH_SEARCH_QUERIES,
            dest_dir=WHOLE_FISH_WIKI_DIR,
            max_count=args.fish_max,
            class_name="whole_fish",
            dry_run=args.dry_run,
        )
    else:
        n_fish = 0

    # ── Summary ──
    _banner("Summary")
    _ok(f"New lure images: {n_lures}")
    _ok(f"New supplementary fish images: {n_fish}")

    lure_total = _count_images(LURE_DIR)
    fish_total = _count_images(WHOLE_FISH_WIKI_DIR)
    status_lure = "OK" if lure_total >= 20 else f"WARN (need {20 - lure_total} more)"
    status_fish = "OK" if fish_total >= 20 else f"WARN (need {20 - fish_total} more)"

    print(f"\n  lure      : {lure_total:>3} images  [{status_lure}]")
    print(f"  whole_fish: {fish_total:>3} images  [{status_fish}]")
    print()
    _info("Next: run scripts/create_stage_a_labels.py to generate YOLO labels")


if __name__ == "__main__":
    main()
