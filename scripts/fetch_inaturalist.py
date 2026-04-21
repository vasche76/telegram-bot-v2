#!/usr/bin/env python3
"""
fetch_inaturalist.py — Download CC-licensed fish photos from iNaturalist public API.

Legal basis:
  - Data accessed via the official public REST API (no scraping).
  - Only photos with license IN {cc0, cc-by, cc-by-sa} are downloaded.
    These licenses explicitly allow use in training datasets.
  - Provenance is recorded per image in PROVENANCE.json in each species directory.
  - Attribution strings are stored so they can be displayed if required.

Usage:
    python3 scripts/fetch_inaturalist.py [--species pike,perch] [--max 100] [--dry-run]

Output directories:
    data/fish_dataset/stage_b/{species}/        ← species classifier images
    data/fish_dataset/stage_a/raw/whole_fish/   ← also copied here for Stage A detector

Minimum thresholds (from train_stage_b.py):
    Stage B: 15 images per species (30+ recommended)
    Stage A whole_fish: 20 images total (50+ recommended)
"""

from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

# ─── SSL context (macOS Python 3.x requires certifi) ───────────────────────
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()
_OPENER = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_SSL_CTX))

# ─── Paths ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
STAGE_B_DIR = DATA_ROOT / "fish_dataset" / "stage_b"
STAGE_A_WHOLE_FISH = DATA_ROOT / "fish_dataset" / "stage_a" / "raw" / "whole_fish"

# ─── iNaturalist API ────────────────────────────────────────────────────────

INAT_API_BASE = "https://api.inaturalist.org/v1"

# Only these licenses allow training use.
# cc-by-nc excluded — non-commercial only, bot is personal but might scale.
# Using the most permissive subset for clarity.
ALLOWED_LICENSES = {"cc0", "cc-by", "cc-by-sa"}

# Research-grade observations are community-verified by iNaturalist curators.
# This significantly improves species label accuracy.
QUALITY_GRADE = "research"

# Pause between API requests to stay within rate limits (100 req/min unauth)
API_DELAY = 0.7  # seconds

# ─── Target species configuration ──────────────────────────────────────────

# Maps our internal species key to iNaturalist search terms.
# taxon_name uses exact species name for specific species,
# or genus for genus-level search (taimen rare, grayling/whitefish polymorphic).
SPECIES_CONFIG = {
    "pike": {
        "taxon_name": "Esox lucius",
        "label_ru": "Щука",
        "notes": "Exact species match. Very well documented on iNaturalist globally.",
    },
    "taimen": {
        "taxon_name": "Hucho taimen",
        "label_ru": "Таймень",
        "notes": (
            "Rare species. Few iNaturalist observations — mostly Siberia/Mongolia. "
            "Accept lower count for this class."
        ),
    },
    "grayling": {
        "taxon_name": "Thymallus",  # genus-level to include T. thymallus + T. arcticus
        "label_ru": "Хариус",
        "notes": (
            "Genus-level search to capture both European (T. thymallus) "
            "and Arctic (T. arcticus) grayling. Both valid for Russian catches."
        ),
    },
    "whitefish": {
        "taxon_name": "Coregonus",  # genus-level: lavaretus, peled, albula, etc.
        "label_ru": "Сиг",
        "notes": (
            "Genus-level. Includes C. lavaretus (broad whitefish), C. peled (peled), "
            "C. albula (vendace). All relevant for Siberian fishing."
        ),
    },
    "perch": {
        "taxon_name": "Perca fluviatilis",
        "label_ru": "Окунь",
        "notes": "European perch. Very common, many iNaturalist observations.",
    },
    "wels_catfish": {
        "taxon_name": "Silurus glanis",
        "label_ru": "Сом",
        "notes": "Wels catfish. Large predator; moderately documented on iNaturalist.",
    },
    "zander": {
        "taxon_name": "Sander lucioperca",
        "label_ru": "Судак",
        "notes": "Common European freshwater predator. Well documented on iNaturalist.",
    },
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# ─── Helpers ───────────────────────────────────────────────────────────────

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


def _api_get(endpoint: str, params: dict) -> dict:
    """Make a GET request to the iNaturalist API. Returns parsed JSON."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{INAT_API_BASE}/{endpoint}?{query}"

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "telegram-fish-bot/1.0 (data collection for ML training; contact: bot owner)",
            "Accept": "application/json",
        },
    )
    try:
        with _OPENER.open(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        _warn(f"API HTTP error {exc.code} for {url}: {exc.reason}")
        return {}
    except Exception as exc:
        _warn(f"API error for {url}: {exc}")
        return {}


def _download_image(url: str, dest_path: Path) -> bool:
    """Download image to dest_path. Returns True on success."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "telegram-fish-bot/1.0 (ML training data collection)",
        },
    )
    try:
        with _OPENER.open(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 5000:
            # Suspiciously small — likely an error page, not an image
            _warn(f"  Image too small ({len(data)} bytes), skipping: {url}")
            return False
        dest_path.write_bytes(data)
        return True
    except Exception as exc:
        _warn(f"  Download failed: {url} — {exc}")
        return False


def _load_provenance(prov_path: Path) -> dict:
    """Load existing provenance JSON, or return empty dict."""
    if prov_path.exists():
        try:
            return json.loads(prov_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"source": "iNaturalist", "license_filter": list(ALLOWED_LICENSES), "images": {}}


def _save_provenance(prov_path: Path, data: dict) -> None:
    """Write provenance JSON."""
    prov_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTS)


# ─── Photo URL resolution ───────────────────────────────────────────────────

def _photo_url_large(photo: dict) -> Optional[str]:
    """
    Get the largest available URL for an iNaturalist photo.

    iNaturalist photo URLs follow the pattern:
      https://inaturalist-open-data.s3.amazonaws.com/photos/{id}/square.jpg
    We replace "square" with "large" to get the highest res (up to 1024px).
    """
    url = photo.get("url") or ""
    if not url:
        return None
    # Replace /square. with /large. for full resolution
    for size in ("square", "small", "medium", "original"):
        if f"/{size}." in url:
            url = url.replace(f"/{size}.", "/large.")
            break
    return url if url.startswith("http") else None


def _photo_license(photo: dict) -> Optional[str]:
    """Return normalized license string from photo dict, or None if not open."""
    lic = (photo.get("license_code") or "").lower().strip()
    # Normalize: iNaturalist uses "cc-by-4.0" style sometimes
    for allowed in ALLOWED_LICENSES:
        if lic == allowed or lic.startswith(allowed + "-"):
            return allowed
    return None


# ─── iNaturalist observation fetch ─────────────────────────────────────────

def fetch_observations(
    taxon_name: str,
    max_results: int = 100,
    page_size: int = 50,
) -> list[dict]:
    """
    Fetch research-grade observations with CC-licensed photos for a taxon.

    Returns list of observation dicts (raw iNaturalist API response).
    """
    observations = []
    page = 1
    total_seen = 0

    while len(observations) < max_results:
        params = {
            "taxon_name": taxon_name.replace(" ", "+"),
            "quality_grade": QUALITY_GRADE,
            "photos": "true",
            "photo_license": "%2C".join(sorted(ALLOWED_LICENSES)),  # URL-encoded comma
            "per_page": min(page_size, max_results - len(observations)),
            "page": page,
            "order": "desc",
            "order_by": "quality_grade",
        }

        _info(f"  Fetching page {page} for '{taxon_name}' ({len(observations)}/{max_results} so far)...")
        data = _api_get("observations", params)

        results = data.get("results", [])
        total_results = data.get("total_results", 0)

        if not results:
            _info(f"  No more results (total on iNaturalist: {total_results})")
            break

        observations.extend(results)
        total_seen += len(results)

        if total_seen >= total_results:
            _info(f"  Fetched all available results ({total_results} total)")
            break

        page += 1
        time.sleep(API_DELAY)

    return observations[:max_results]


# ─── Download species images ────────────────────────────────────────────────

def download_species(
    species_key: str,
    species_dir: Path,
    whole_fish_dir: Path,
    max_per_species: int,
    dry_run: bool = False,
) -> int:
    """
    Download images for one species.

    Returns count of newly downloaded images.
    """
    config = SPECIES_CONFIG[species_key]
    taxon_name = config["taxon_name"]
    label_ru = config["label_ru"]

    species_dir.mkdir(parents=True, exist_ok=True)
    whole_fish_dir.mkdir(parents=True, exist_ok=True)

    prov_path = species_dir / "PROVENANCE.json"
    provenance = _load_provenance(prov_path)

    already_downloaded = _count_images(species_dir)
    if already_downloaded >= max_per_species:
        _info(f"  {species_key}: already have {already_downloaded} images (>= {max_per_species}), skipping.")
        return 0

    remaining = max_per_species - already_downloaded
    _info(f"  {species_key} ({label_ru}): have {already_downloaded}, fetching up to {remaining} more")
    _info(f"    iNaturalist taxon: '{taxon_name}'")
    _info(f"    {config['notes']}")

    # We request more than we need to account for filtered-out photos
    obs_list = fetch_observations(taxon_name, max_results=remaining * 3)

    downloaded = 0
    for obs in obs_list:
        if downloaded >= remaining:
            break

        obs_id = obs.get("id")
        photos = obs.get("photos", [])
        if not photos:
            continue

        # Pick the first CC-licensed photo from this observation
        for photo in photos[:3]:  # check up to 3 photos per observation
            lic = _photo_license(photo)
            if lic is None:
                continue  # not an open license

            img_url = _photo_url_large(photo)
            if not img_url:
                continue

            photo_id = photo.get("id", obs_id)
            # Determine extension from URL
            ext = os.path.splitext(img_url.split("?")[0])[-1].lower()
            if ext not in IMAGE_EXTS:
                ext = ".jpg"

            filename = f"inat_{species_key}_{photo_id}{ext}"
            dest = species_dir / filename

            if dest.exists():
                _info(f"    Skipping existing: {filename}")
                downloaded += 1  # count as downloaded
                break

            if dry_run:
                _info(f"    [DRY-RUN] Would download: {img_url} → {filename}")
                downloaded += 1
                break

            _info(f"    Downloading [{downloaded+1}/{remaining}]: {filename}")
            success = _download_image(img_url, dest)
            if not success:
                continue

            # Record provenance
            user_login = obs.get("user", {}).get("login", "unknown")
            observed_on = obs.get("observed_on") or obs.get("time_observed_at", "unknown")
            species_guess = obs.get("taxon", {}).get("name", taxon_name)

            provenance["images"][filename] = {
                "photo_id": photo_id,
                "observation_id": obs_id,
                "source": "iNaturalist",
                "license": lic,
                "url": img_url,
                "observer": user_login,
                "observed_on": observed_on,
                "taxon_identified": species_guess,
                "quality_grade": obs.get("quality_grade", "research"),
                "attribution": f"Photo by {user_login} via iNaturalist, CC {lic.upper()}",
            }

            # Also copy to stage_a whole_fish (symlink-free: just copy)
            wa_dest = whole_fish_dir / filename
            if not wa_dest.exists():
                wa_dest.write_bytes(dest.read_bytes())

            downloaded += 1
            time.sleep(0.2)  # small pause between image downloads
            break  # one image per observation is enough

        else:
            continue  # no suitable photo found in this observation

    _save_provenance(prov_path, provenance)

    new_total = _count_images(species_dir)
    _ok(f"  {species_key}: {downloaded} new downloads, {new_total} total in {species_dir}")
    return downloaded


# ─── No-fish class (iNaturalist non-fish observations) ─────────────────────

def download_no_fish(
    no_fish_dir: Path,
    max_count: int = 60,
    dry_run: bool = False,
) -> int:
    """
    Download non-fish nature photos for the Stage A 'no_fish' class.

    Uses plant/bird/insect observations as hard negatives — real outdoor
    photos that look like fishing conditions but contain no fish.
    """
    no_fish_dir.mkdir(parents=True, exist_ok=True)
    prov_path = no_fish_dir / "PROVENANCE.json"
    provenance = _load_provenance(prov_path)
    provenance["source"] = "iNaturalist (non-fish taxa)"

    already = _count_images(no_fish_dir)
    if already >= max_count:
        _info(f"  no_fish: already have {already} images, skipping.")
        return 0

    remaining = max_count - already

    # Taxa that produce realistic outdoor nature photos (river/lake environments)
    # These are negative examples for the fish detector.
    no_fish_taxa = [
        ("Actinopterygii", False),     # fish class — use as EXCLUSION, not inclusion
        ("Aves", True),                # birds — often near water
        ("Plantae", True),             # plants — riverbank
        ("Insecta", True),             # insects — water surface, fishing context
    ]

    # Actually: for no_fish we want riverbank/nature scenes without fish.
    # Simple approach: download bird/plant/insect photos from iNaturalist.
    # These will be clearly non-fish but look like outdoor nature photography.

    no_fish_search_taxa = [
        "Aves",        # birds
        "Plantae",     # plants
        "Mammalia",    # mammals (wildlife shots)
    ]

    downloaded = 0
    for taxon in no_fish_search_taxa:
        if downloaded >= remaining:
            break

        batch_max = min(remaining // len(no_fish_search_taxa) + 1, remaining - downloaded)

        _info(f"  no_fish: fetching {batch_max} images from taxon '{taxon}'")
        obs_list = fetch_observations(taxon, max_results=batch_max * 2)

        for obs in obs_list:
            if downloaded >= remaining:
                break

            photos = obs.get("photos", [])
            if not photos:
                continue

            for photo in photos[:2]:
                lic = _photo_license(photo)
                if lic is None:
                    continue

                img_url = _photo_url_large(photo)
                if not img_url:
                    continue

                photo_id = photo.get("id", obs.get("id"))
                ext = os.path.splitext(img_url.split("?")[0])[-1].lower()
                if ext not in IMAGE_EXTS:
                    ext = ".jpg"

                filename = f"inat_nofish_{taxon.lower()}_{photo_id}{ext}"
                dest = no_fish_dir / filename

                if dest.exists():
                    downloaded += 1
                    break

                if dry_run:
                    _info(f"    [DRY-RUN] Would download: {filename}")
                    downloaded += 1
                    break

                success = _download_image(img_url, dest)
                if not success:
                    continue

                user_login = obs.get("user", {}).get("login", "unknown")
                provenance["images"][filename] = {
                    "photo_id": photo_id,
                    "observation_id": obs.get("id"),
                    "source": "iNaturalist",
                    "license": lic,
                    "url": img_url,
                    "observer": user_login,
                    "taxon": taxon,
                    "label": "no_fish",
                    "attribution": f"Photo by {user_login} via iNaturalist, CC {lic.upper()}",
                }

                downloaded += 1
                time.sleep(0.2)
                break

        time.sleep(API_DELAY)

    _save_provenance(prov_path, provenance)
    new_total = _count_images(no_fish_dir)
    _ok(f"  no_fish: {downloaded} new downloads, {new_total} total")
    return downloaded


# ─── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download CC-licensed fish photos from iNaturalist for model training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--species",
        default=",".join(SPECIES_CONFIG.keys()),
        help="Comma-separated species to download (pike,taimen,grayling,whitefish,perch)",
    )
    p.add_argument(
        "--max",
        type=int,
        default=80,
        help="Max images to download per species",
    )
    p.add_argument(
        "--no-fish-max",
        type=int,
        default=60,
        help="Max no_fish images to download for Stage A",
    )
    p.add_argument(
        "--no-fish",
        action="store_true",
        default=True,
        help="Also download no_fish examples for Stage A",
    )
    p.add_argument(
        "--skip-no-fish",
        action="store_true",
        help="Skip no_fish download",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without downloading",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _banner("iNaturalist Fish Photo Downloader")
    _info(f"Repo root: {REPO_ROOT}")
    _info(f"Licenses: {sorted(ALLOWED_LICENSES)}")
    _info(f"Quality grade: {QUALITY_GRADE} (community-verified identifications)")
    _info(f"Max per species: {args.max}")
    if args.dry_run:
        _info("DRY-RUN mode — no files will be downloaded")

    species_list = [s.strip() for s in args.species.split(",") if s.strip()]
    unknown_species = [s for s in species_list if s not in SPECIES_CONFIG]
    if unknown_species:
        _fail(f"Unknown species keys: {unknown_species}")
        _fail(f"Valid keys: {list(SPECIES_CONFIG.keys())}")
        sys.exit(1)

    total_downloaded = 0

    # ── Download each species ──
    for species_key in species_list:
        _banner(f"Species: {species_key} ({SPECIES_CONFIG[species_key]['label_ru']})")
        species_dir = STAGE_B_DIR / species_key
        n = download_species(
            species_key=species_key,
            species_dir=species_dir,
            whole_fish_dir=STAGE_A_WHOLE_FISH,
            max_per_species=args.max,
            dry_run=args.dry_run,
        )
        total_downloaded += n
        time.sleep(API_DELAY)

    # ── Download no_fish for Stage A ──
    if args.no_fish and not args.skip_no_fish:
        _banner("Stage A — no_fish class")
        no_fish_dir = DATA_ROOT / "fish_dataset" / "stage_a" / "raw" / "no_fish"
        n = download_no_fish(
            no_fish_dir=no_fish_dir,
            max_count=args.no_fish_max,
            dry_run=args.dry_run,
        )
        total_downloaded += n

    # ── Summary ──
    _banner("Download Summary")
    _ok(f"Total new images downloaded: {total_downloaded}")
    print()
    print("  Stage B (species classification):")
    for species_key in species_list:
        n = _count_images(STAGE_B_DIR / species_key)
        status = "OK" if n >= 15 else f"WARN (need {15 - n} more)"
        print(f"    {species_key:<12}: {n:>3} images  [{status}]")

    print()
    print("  Stage A (raw/whole_fish):")
    n = _count_images(STAGE_A_WHOLE_FISH)
    status = "OK" if n >= 20 else f"WARN (need {20 - n} more)"
    print(f"    whole_fish  : {n:>3} images  [{status}]")

    n = _count_images(DATA_ROOT / "fish_dataset" / "stage_a" / "raw" / "no_fish")
    status = "OK" if n >= 20 else f"WARN (need {20 - n} more)"
    print(f"    no_fish     : {n:>3} images  [{status}]")

    print()
    _info("Next steps:")
    _info("  1. Run scripts/fetch_wikimedia_lures.py to get lure images for Stage A")
    _info("  2. Run scripts/create_stage_a_labels.py to create YOLO-format labels")
    _info("  3. Run scripts/validate_dataset.py to confirm training readiness")
    _info("  4. Run scripts/train_stage_b.py to train the species classifier")
    _info("  5. Run scripts/train_stage_a.py to train the fish detector")
    print()


if __name__ == "__main__":
    main()
