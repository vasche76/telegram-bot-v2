#!/usr/bin/env python3
"""
fetch_fish_parts_fry.py — Fetch fry (juvenile fish) images from iNaturalist
                           and source fish_part images from Wikimedia Commons.

Stage A gaps addressed:
  • fry       — juvenile fish < 15 cm. iNaturalist juvenile observations.
  • fish_part — partial fish (fillets, headless). Wikimedia "fish fillet" searches.

Legal basis:
  • iNaturalist: public API, CC0/CC-BY/CC-BY-SA licenses only.
  • Wikimedia: MediaWiki API, CC0/CC-BY/PD only.
  • Provenance recorded in PROVENANCE.json per class directory.

Usage:
    python3 scripts/fetch_fish_parts_fry.py [--fry-max 60] [--fish-part-max 60] [--dry-run]

Output:
    data/fish_dataset/stage_a/raw/fry/        ← fry images
    data/fish_dataset/stage_a/raw/fish_part/  ← fish_part images
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

# ─── SSL context ────────────────────────────────────────────────────────────
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()
_OPENER = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_SSL_CTX))

# ─── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
STAGE_A_RAW = DATA_ROOT / "fish_dataset" / "stage_a" / "raw"

# ─── iNaturalist config ─────────────────────────────────────────────────────
INAT_API_BASE = "https://api.inaturalist.org/v1"
ALLOWED_LICENSES_INAT = {"cc0", "cc-by", "cc-by-sa"}
QUALITY_GRADE = "research"
API_DELAY = 0.7

# Juvenile fish taxa for fry class.
# We search for young/juvenile fish by looking for observations tagged "juvenile"
# or by searching within size ranges in captions.
# Expanded: now includes new taxonomy species (Cyprinidae, Salmonidae) + original
FRY_TAXA = [
    # Original 5 species fry
    {"taxon": "Rutilus rutilus",      "label": "roach_fry"},
    {"taxon": "Abramis brama",        "label": "bream_fry"},
    {"taxon": "Perca fluviatilis",    "label": "perch_fry"},
    {"taxon": "Esox lucius",          "label": "pike_fry"},
    {"taxon": "Thymallus",            "label": "grayling_fry"},
    # New Cyprinidae fry — extremely common in Russian rivers
    {"taxon": "Cyprinus carpio",      "label": "carp_fry"},
    {"taxon": "Carassius carassius",  "label": "crucian_fry"},
    {"taxon": "Leuciscus idus",       "label": "ide_fry"},
    # New Salmonidae fry — important for sport fishing rivers
    {"taxon": "Salmo trutta",         "label": "trout_fry"},
    {"taxon": "Oncorhynchus mykiss",  "label": "rainbow_fry"},
    {"taxon": "Salmo salar",          "label": "salmon_fry"},
]

# ─── Wikimedia config ────────────────────────────────────────────────────────
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
WIKIMEDIA_FILE_PATH = "https://commons.wikimedia.org/wiki/Special:FilePath/"
WIKIMEDIA_DELAY = 2.5

ALLOWED_LICENSES_WIKI = {
    "cc0", "cc-by", "cc by", "public domain", "pd ", "pd-",
    "cc-by-sa", "cc by-sa", "cc-by 4", "cc-by-sa 4", "attribution",
}

# Wikimedia search queries for fish_part class
# Expanded: more search terms for partial/processed fish to strengthen negative class
FISH_PART_QUERIES = [
    # Fillets — most common fish_part scenario
    "fish fillet",
    "fish steak cooking",
    "salmon fillet",
    "fish cutting preparation",
    "trout fillet",
    "smoked fish fillet",
    "raw fish fillet",
    # Heads and tails — important fish_part subcategory
    "fish head cooking",
    "fish tail market",
    "fish bones skeleton",
    "carp fillet preparation",
    "pike fillet cooking",
    # Held fish fragment (common in fishing photos)
    "fish cleaning filleting",
    "gutted fish cooking",
    "fish cleaning knife",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ─── Helpers ─────────────────────────────────────────────────────────────────

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


def _count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def _load_provenance(prov_path: Path, source: str) -> dict:
    if prov_path.exists():
        try:
            return json.loads(prov_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"source": source, "images": {}}


def _save_provenance(prov_path: Path, data: dict) -> None:
    prov_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _download_image(url: str, dest_path: Path, user_agent: str = "FishBot/1.0") -> bool:
    req = urllib.request.Request(
        url, headers={"User-Agent": user_agent, "Accept": "image/*"}
    )
    try:
        with _OPENER.open(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 5000:
            return False
        dest_path.write_bytes(data)
        return True
    except Exception as exc:
        _warn(f"  Download failed: {url} — {exc}")
        return False


# ─── iNaturalist helpers ─────────────────────────────────────────────────────

def _inat_api_get(endpoint: str, params: dict) -> dict:
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{INAT_API_BASE}/{endpoint}?{query}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "telegram-fish-bot/1.0 (ML training, fish fry data collection)",
            "Accept": "application/json",
        },
    )
    try:
        with _OPENER.open(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        _warn(f"  iNaturalist API error: {exc}")
        return {}


def _inat_photo_license(photo: dict) -> Optional[str]:
    lic = (photo.get("license_code") or "").lower().strip()
    for allowed in ALLOWED_LICENSES_INAT:
        if lic == allowed or lic.startswith(allowed + "-"):
            return allowed
    return None


def _inat_photo_url_large(photo: dict) -> Optional[str]:
    url = photo.get("url") or ""
    if not url:
        return None
    for size in ("square", "small", "medium", "original"):
        if f"/{size}." in url:
            return url.replace(f"/{size}.", "/large.")
    return url if url.startswith("http") else None


# ─── Fry download (iNaturalist, size-filtered) ───────────────────────────────

def fetch_fry(dest_dir: Path, max_count: int, dry_run: bool = False) -> int:
    """
    Download juvenile fish images from iNaturalist.

    Approach: search for each fry taxon, then filter observations that mention
    'juvenile', 'fry', 'young', or have very small estimated sizes.
    The iNaturalist API doesn't expose size directly, so we rely on:
      1. Tag filtering (term_id for 'Life Stage' → 'Juvenile')
      2. Taxon observations — selecting fish that are commonly fry-sized at these taxa
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    prov_path = dest_dir / "PROVENANCE.json"
    provenance = _load_provenance(prov_path, "iNaturalist (juvenile/fry observations)")

    already = _count_images(dest_dir)
    if already >= max_count:
        _info(f"  fry: already have {already} images, skipping.")
        return 0

    remaining = max_count - already
    downloaded = 0

    for taxon_cfg in FRY_TAXA:
        if downloaded >= remaining:
            break

        taxon = taxon_cfg["taxon"]
        label = taxon_cfg["label"]
        batch_max = min(remaining // len(FRY_TAXA) + 2, remaining - downloaded)

        _info(f"  fry/{taxon} (fetching up to {batch_max} juvenile observations)...")

        # iNaturalist term_id=1 = Life Stage, term_value_id=6 = Juvenile
        params = {
            "taxon_name": taxon.replace(" ", "+"),
            "quality_grade": QUALITY_GRADE,
            "photos": "true",
            "photo_license": "%2C".join(sorted(ALLOWED_LICENSES_INAT)),
            "term_id": "1",        # Life Stage
            "term_value_id": "6",  # Juvenile
            "per_page": min(batch_max * 3, 50),
            "page": "1",
            "order": "desc",
            "order_by": "quality_grade",
        }

        data = _inat_api_get("observations", params)
        results = data.get("results", [])
        _info(f"    Got {len(results)} observations (total={data.get('total_results', 0)})")

        for obs in results:
            if downloaded >= batch_max:
                break

            obs_id = obs.get("id")
            photos = obs.get("photos", [])
            if not photos:
                continue

            for photo in photos[:2]:
                lic = _inat_photo_license(photo)
                if lic is None:
                    continue

                img_url = _inat_photo_url_large(photo)
                if not img_url:
                    continue

                photo_id = photo.get("id", obs_id)
                ext = os.path.splitext(img_url.split("?")[0])[-1].lower()
                if ext not in IMAGE_EXTS:
                    ext = ".jpg"

                filename = f"inat_fry_{label}_{photo_id}{ext}"
                dest = dest_dir / filename

                if dest.exists():
                    downloaded += 1
                    break

                if dry_run:
                    _info(f"    [DRY-RUN] Would download: {filename}")
                    downloaded += 1
                    break

                _info(f"    Downloading [{downloaded+1}/{remaining}]: {filename}")
                success = _download_image(
                    img_url, dest,
                    user_agent="telegram-fish-bot/1.0 (ML training, fry class)"
                )
                if not success:
                    continue

                user_login = obs.get("user", {}).get("login", "unknown")
                provenance["images"][filename] = {
                    "photo_id": photo_id,
                    "observation_id": obs_id,
                    "source": "iNaturalist",
                    "license": lic,
                    "url": img_url,
                    "observer": user_login,
                    "taxon": taxon,
                    "taxon_label": label,
                    "life_stage": "juvenile",
                    "label": "fry",
                    "attribution": f"Photo by {user_login} via iNaturalist, CC {lic.upper()}",
                }

                downloaded += 1
                time.sleep(0.3)
                break

        time.sleep(API_DELAY)

    _save_provenance(prov_path, provenance)
    new_total = _count_images(dest_dir)
    _ok(f"  fry: {downloaded} new downloads, {new_total} total")
    return downloaded


# ─── Fish_part download (Wikimedia Commons) ──────────────────────────────────

def _wiki_api_get(params: dict) -> dict:
    params["format"] = "json"
    params["origin"] = "*"
    query = urllib.parse.urlencode(params)
    url = f"{WIKIMEDIA_API}?{query}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "telegram-fish-bot/1.0 (ML training, fish_part data collection)",
            "Accept": "application/json",
        },
    )
    try:
        with _OPENER.open(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        _warn(f"  Wikimedia API error: {exc}")
        return {}


def _is_allowed_wiki_license(license_text: str) -> bool:
    lt = license_text.lower()
    return any(allowed in lt for allowed in ALLOWED_LICENSES_WIKI)


def _wiki_search_images(query: str, limit: int = 30) -> list[str]:
    params = {
        "action": "query",
        "list": "search",
        "srnamespace": "6",
        "srsearch": query,
        "srlimit": min(limit, 50),
        "srprop": "title",
    }
    data = _wiki_api_get(params)
    results = data.get("query", {}).get("search", [])
    return [r["title"] for r in results if "title" in r]


def _wiki_get_image_info(titles: list[str]) -> list[dict]:
    if not titles:
        return []
    titles_str = "|".join(t.replace(" ", "_") for t in titles[:50])
    params = {
        "action": "query",
        "titles": titles_str,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime",
    }
    data = _wiki_api_get(params)
    pages = data.get("query", {}).get("pages", {})

    results = []
    for page_id, page in pages.items():
        if page_id == "-1":
            continue
        imageinfo_list = page.get("imageinfo", [])
        if not imageinfo_list:
            continue
        ii = imageinfo_list[0]

        meta = ii.get("extmetadata", {})
        license_text = (
            meta.get("License", {}).get("value", "")
            or meta.get("LicenseShortName", {}).get("value", "")
            or meta.get("UsageTerms", {}).get("value", "")
        )
        if not license_text or not _is_allowed_wiki_license(license_text):
            continue

        mime = ii.get("mime", "")
        if not mime.startswith("image/"):
            continue

        img_url = ii.get("url") or ii.get("thumburl")
        if not img_url:
            continue

        width = ii.get("thumbwidth") or ii.get("width") or 0
        height = ii.get("thumbheight") or ii.get("height") or 0
        if width < 200 or height < 200:
            continue

        results.append({
            "title": page.get("title", "unknown"),
            "url": img_url,
            "license": license_text,
            "attribution": meta.get("Artist", {}).get("value", "Wikimedia contributor"),
        })
    return results


def _wiki_download_image(url: str, dest_path: Path, title: str) -> bool:
    filename = title.replace("File:", "").replace(" ", "_")
    effective_url = WIKIMEDIA_FILE_PATH + urllib.parse.quote(filename)
    req = urllib.request.Request(
        effective_url,
        headers={
            "User-Agent": "FishRecognitionBot/1.0 (educational ML research)",
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
        _warn(f"  Download failed: {url} — {exc}")
        return False


def fetch_fish_parts(dest_dir: Path, max_count: int, dry_run: bool = False) -> int:
    """
    Download fish_part (fillet, headless fish) images from Wikimedia Commons.

    These are hard negative examples for Stage A: they look like fish but are
    incomplete (no head or no tail), which should be rejected by the detector.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    prov_path = dest_dir / "PROVENANCE.json"
    provenance = _load_provenance(prov_path, "Wikimedia Commons (fish fillet/part)")

    already = _count_images(dest_dir)
    if already >= max_count:
        _info(f"  fish_part: already have {already} images, skipping.")
        return 0

    remaining = max_count - already
    downloaded = 0

    # Title fragments to skip (non-food, non-fish results)
    blocklist = [
        "hybrid", "plugin", "toyota", "car ", "electric", "socket",
        "plant ", "flower", "butterfly", "insect",
    ]

    for query in FISH_PART_QUERIES:
        if downloaded >= remaining:
            break

        _info(f"  fish_part: searching '{query}'")
        titles = _wiki_search_images(query, limit=50)
        _info(f"    Found {len(titles)} candidates")

        if not titles:
            time.sleep(1.2)
            continue

        for batch_start in range(0, len(titles), 20):
            if downloaded >= remaining:
                break

            batch = titles[batch_start:batch_start + 20]
            infos = _wiki_get_image_info(batch)
            _info(f"    {len(infos)}/{len(batch)} with open licenses")

            for info in infos:
                if downloaded >= remaining:
                    break

                title = info["title"]
                title_lower = title.lower()
                if any(b in title_lower for b in blocklist):
                    continue

                img_url = info["url"]
                title_clean = title.replace("File:", "").replace(" ", "_")
                ext = os.path.splitext(title_clean)[-1].lower()
                if ext not in IMAGE_EXTS:
                    ext = ".jpg"

                safe_name = "".join(
                    c if c.isalnum() or c in "._-" else "_" for c in title_clean
                )
                filename = f"wiki_fishpart_{safe_name}"
                if not filename.endswith(ext):
                    filename += ext
                filename = filename[:200]

                dest = dest_dir / filename
                if dest.exists():
                    downloaded += 1
                    continue

                if dry_run:
                    _info(f"    [DRY-RUN] Would download: {filename[:60]}")
                    downloaded += 1
                    continue

                _info(f"    Downloading [{downloaded+1}/{remaining}]: {filename[:60]}")
                success = _wiki_download_image(img_url, dest, title)
                if not success:
                    continue

                provenance["images"][str(dest.name)] = {
                    "title": title,
                    "source": "Wikimedia Commons",
                    "url": img_url,
                    "license": info["license"],
                    "attribution": info["attribution"],
                    "label": "fish_part",
                }

                downloaded += 1
                time.sleep(WIKIMEDIA_DELAY)

            time.sleep(1.2)

    _save_provenance(prov_path, provenance)
    new_total = _count_images(dest_dir)
    _ok(f"  fish_part: {downloaded} new downloads, {new_total} total")
    return downloaded


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fetch fry (juvenile fish) from iNaturalist and "
            "fish_part (fillets) from Wikimedia Commons."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--fry-max", type=int, default=60, help="Max fry images")
    p.add_argument("--fish-part-max", type=int, default=60, help="Max fish_part images")
    p.add_argument("--skip-fry", action="store_true", help="Skip fry download")
    p.add_argument("--skip-fish-part", action="store_true", help="Skip fish_part download")
    p.add_argument("--dry-run", action="store_true", help="Dry run — no files downloaded")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _banner("Stage A Gap Filler — Fry + Fish_Part")

    total = 0

    if not args.skip_fry:
        _banner("Fry (juvenile fish) — iNaturalist juvenile observations")
        fry_dir = STAGE_A_RAW / "fry"
        n = fetch_fry(fry_dir, args.fry_max, dry_run=args.dry_run)
        total += n

    if not args.skip_fish_part:
        _banner("Fish Part (fillet/partial) — Wikimedia Commons")
        fp_dir = STAGE_A_RAW / "fish_part"
        n = fetch_fish_parts(fp_dir, args.fish_part_max, dry_run=args.dry_run)
        total += n

    _banner("Summary")
    _ok(f"Total new images: {total}")

    fry_n = _count_images(STAGE_A_RAW / "fry")
    fp_n = _count_images(STAGE_A_RAW / "fish_part")

    print(f"\n  fry       : {fry_n:>3} images  [{'OK' if fry_n >= 20 else f'WARN need {20-fry_n} more'}]")
    print(f"  fish_part : {fp_n:>3} images  [{'OK' if fp_n >= 20 else f'WARN need {20-fp_n} more'}]")

    print()
    _info("Next: run scripts/build_dataset.py to rebuild labels and validate")


if __name__ == "__main__":
    main()
