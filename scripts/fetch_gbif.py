#!/usr/bin/env python3
"""
fetch_gbif.py — Download CC-licensed fish photos from GBIF public API.

Legal basis:
  - Data accessed via the official GBIF REST API (no scraping).
  - Only images with license IN {CC0_1_0, CC_BY_4_0} are downloaded.
    These licenses explicitly allow use in training datasets.
  - Provenance recorded per image in PROVENANCE_gbif.json per species directory.

Why GBIF as a complement to iNaturalist?
  - GBIF aggregates observations from hundreds of institutions worldwide.
  - Covers species where iNaturalist has few research-grade observations.
  - Especially useful for new taxonomy classes: Cyprinidae, Siluriformes, new Salmonidae.
  - No authentication required; public JSON API.

GBIF taxon keys used (verified against GBIF production):
  pike           2346633  (Esox lucius)
  taimen         2351408  (Hucho taimen)
  grayling       5203981  (Thymallus — genus)
  whitefish      2350934  (Coregonus — genus)
  perch          8140485  (Perca fluviatilis)
  brown_trout    8215487  (Salmo trutta)
  rainbow_trout  5204019  (Oncorhynchus mykiss)
  atlantic_salmon 7595433 (Salmo salar)
  common_carp    4286975  (Cyprinus carpio)
  crucian_carp   2366645  (Carassius carassius)
  bream          9809222  (Abramis brama)
  roach          2359706  (Rutilus rutilus)
  ide            4409643  (Leuciscus idus)
  wels_catfish   2337607  (Silurus glanis)

Usage:
    python3 scripts/fetch_gbif.py [--species pike,bream,roach] [--max 80] [--dry-run]
    python3 scripts/fetch_gbif.py --all-new --max 60

Output directories:
    data/fish_dataset/stage_b/{species}/         <- species classifier images
    data/fish_dataset/stage_a/raw/whole_fish/    <- also copied here for Stage A
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

# ─── SSL context (macOS certifi) ────────────────────────────────────────────
try:
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()
_OPENER = urllib.request.build_opener(urllib.request.HTTPSHandler(context=_SSL_CTX))

# ─── Paths ───────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
STAGE_B_DIR = DATA_ROOT / "fish_dataset" / "stage_b"
STAGE_A_WHOLE_FISH = DATA_ROOT / "fish_dataset" / "stage_a" / "raw" / "whole_fish"

# ─── GBIF API ────────────────────────────────────────────────────────────────
GBIF_API_BASE = "https://api.gbif.org/v1"
GBIF_OCCURRENCE_SEARCH = f"{GBIF_API_BASE}/occurrence/search"
API_DELAY = 0.5     # seconds between API calls (GBIF allows ~600 req/min)
IMAGE_DELAY = 1.5   # seconds between image downloads

# Only these GBIF license URIs allow training use (no NC restriction).
ALLOWED_LICENSES = {
    "http://creativecommons.org/publicdomain/zero/1.0/",
    "http://creativecommons.org/licenses/by/4.0/",
    "CC0_1_0",
    "CC_BY_4_0",
    "CC0 1.0",
    "CC BY 4.0",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ─── Species taxonomy registry ───────────────────────────────────────────────
# Maps canonical class name -> GBIF taxon key + metadata
SPECIES_REGISTRY: dict[str, dict] = {
    "pike": {
        "taxon_key": 2346633,
        "latin": "Esox lucius",
        "rank": "species",
    },
    "taimen": {
        "taxon_key": 2351408,
        "latin": "Hucho taimen",
        "rank": "species",
    },
    "grayling": {
        "taxon_key": 5203981,
        "latin": "Thymallus",
        "rank": "genus",
    },
    "whitefish": {
        "taxon_key": 2350934,
        "latin": "Coregonus",
        "rank": "genus",
    },
    "perch": {
        "taxon_key": 8140485,
        "latin": "Perca fluviatilis",
        "rank": "species",
    },
    "brown_trout": {
        "taxon_key": 8215487,
        "latin": "Salmo trutta",
        "rank": "species",
    },
    "rainbow_trout": {
        "taxon_key": 5204019,
        "latin": "Oncorhynchus mykiss",
        "rank": "species",
    },
    "atlantic_salmon": {
        "taxon_key": 7595433,
        "latin": "Salmo salar",
        "rank": "species",
    },
    "common_carp": {
        "taxon_key": 4286975,
        "latin": "Cyprinus carpio",
        "rank": "species",
    },
    "crucian_carp": {
        "taxon_key": 2366645,
        "latin": "Carassius carassius",
        "rank": "species",
    },
    "bream": {
        "taxon_key": 9809222,
        "latin": "Abramis brama",
        "rank": "species",
    },
    "roach": {
        "taxon_key": 2359706,
        "latin": "Rutilus rutilus",
        "rank": "species",
    },
    "ide": {
        "taxon_key": 4409643,
        "latin": "Leuciscus idus",
        "rank": "species",
    },
    "wels_catfish": {
        "taxon_key": 2337607,
        "latin": "Silurus glanis",
        "rank": "species",
    },
}

# New-taxonomy species (not in the original 5)
NEW_SPECIES = {
    "brown_trout", "rainbow_trout", "atlantic_salmon",
    "common_carp", "crucian_carp", "bream", "roach", "ide",
    "wels_catfish",
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


def _ok(msg: str) -> None:
    print(f"[OK]  {msg}")


def _info(msg: str) -> None:
    print(f"[INFO] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr)


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _existing_hashes(directory: Path) -> set[str]:
    """Return MD5 hashes of all images already in directory (dedup guard)."""
    hashes: set[str] = set()
    if not directory.exists():
        return hashes
    for p in directory.iterdir():
        if p.suffix.lower() in IMAGE_EXTS:
            try:
                hashes.add(_md5(p))
            except OSError:
                pass
    return hashes


def _api_get(url: str) -> Optional[dict]:
    """Fetch JSON from GBIF API. Returns parsed dict or None on error."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FishBotTrainingPipeline/1.0 (training data collection; cc-licensed only)"},
    )
    try:
        with _OPENER.open(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        _warn(f"API error for {url}: {exc}")
        return None


def _download_image(url: str, dest: Path) -> bool:
    """Download a single image. Returns True on success."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FishBotTrainingPipeline/1.0"},
    )
    try:
        with _OPENER.open(req, timeout=30) as resp:
            data = resp.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except Exception as exc:
        _warn(f"Download failed for {url}: {exc}")
        return False


def _is_license_allowed(license_str: Optional[str]) -> bool:
    """Return True if the GBIF license string is in our allowed set."""
    if not license_str:
        return False
    lic = license_str.strip()
    return lic in ALLOWED_LICENSES


# ─── GBIF occurrence search ───────────────────────────────────────────────────

def _search_occurrences(
    taxon_key: int,
    offset: int = 0,
    limit: int = 300,
) -> Optional[dict]:
    """
    Search GBIF occurrences with StillImage media, returning JSON page.
    """
    params = urllib.parse.urlencode({
        "taxonKey": taxon_key,
        "hasCoordinate": "true",
        "mediaType": "StillImage",
        "limit": limit,
        "offset": offset,
    })
    url = f"{GBIF_OCCURRENCE_SEARCH}?{params}"
    return _api_get(url)


def _extract_image_url(occurrence: dict) -> Optional[tuple[str, str]]:
    """
    Extract a usable image URL + license from an occurrence record.
    Returns (url, license) or None.

    GBIF media is in occurrence['media'] list.
    Each media item has: identifier (URL), license, rightsHolder, etc.
    """
    media_list = occurrence.get("media", [])
    for media in media_list:
        if media.get("type", "").lower() not in ("stillimage", "image", ""):
            continue
        url = media.get("identifier", "")
        if not url or not url.startswith("http"):
            continue
        # Check extension
        url_lower = url.lower().split("?")[0]
        if not any(url_lower.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
            # Try without extension check — some URLs lack extension
            # Accept if it's a photo service URL
            if not any(svc in url for svc in ("iNaturalist", "inaturalist", "flickr", "wikimedia",
                                               "gbif.org", "observation.org", "biodiversity")):
                # Unknown URL without image extension — skip
                continue
        license_val = media.get("license", "")
        if _is_license_allowed(license_val):
            return url, license_val
    return None


# ─── Per-species downloader ───────────────────────────────────────────────────

def fetch_species(
    species_name: str,
    taxon_key: int,
    latin_name: str,
    target_dir: Path,
    max_images: int,
    dry_run: bool = False,
    existing_hashes: Optional[set[str]] = None,
) -> tuple[int, list[dict]]:
    """
    Fetch up to max_images for one species from GBIF.
    Returns (n_downloaded, provenance_records).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    if existing_hashes is None:
        existing_hashes = _existing_hashes(target_dir)

    existing_count = sum(
        1 for p in target_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    ) if target_dir.exists() else 0

    if existing_count >= max_images:
        _info(f"  {species_name}: already has {existing_count} images (target={max_images}) — skipping")
        return 0, []

    need = max_images - existing_count
    _info(f"  {species_name}: need {need} more images (have {existing_count}, target={max_images})")

    downloaded = 0
    provenance: list[dict] = []
    offset = 0
    page_size = min(300, max(need * 3, 100))  # fetch extra to account for license filtering

    while downloaded < need:
        _info(f"    Querying GBIF offset={offset} for taxon {taxon_key} ({latin_name})...")
        page = _search_occurrences(taxon_key, offset=offset, limit=page_size)
        time.sleep(API_DELAY)

        if not page:
            _warn(f"    No response for {species_name} at offset={offset}")
            break

        results = page.get("results", [])
        if not results:
            _info(f"    No more results at offset={offset}")
            break

        for occ in results:
            if downloaded >= need:
                break

            occ_key = occ.get("key", "unknown")
            img_info = _extract_image_url(occ)
            if not img_info:
                continue

            img_url, img_license = img_info
            ext = Path(img_url.split("?")[0]).suffix.lower()
            if ext not in IMAGE_EXTS:
                ext = ".jpg"  # fallback

            if dry_run:
                _info(f"    [DRY-RUN] Would download: {img_url[:80]}...")
                downloaded += 1
                continue

            # Download to temp location first for hash check
            filename = f"gbif_{taxon_key}_{occ_key}{ext}"
            dest = target_dir / filename

            if dest.exists():
                continue  # already have it

            success = _download_image(img_url, dest)
            if not success:
                continue

            # Dedup by MD5
            try:
                h = _md5(dest)
                if h in existing_hashes:
                    dest.unlink()
                    continue
                existing_hashes.add(h)
            except OSError:
                pass

            downloaded += 1
            rights_holder = occ.get("rightsHolder", "")
            institution = occ.get("institutionCode", "")
            provenance.append({
                "file": filename,
                "source": "gbif",
                "taxon_key": taxon_key,
                "occurrence_key": occ_key,
                "species": latin_name,
                "license": img_license,
                "url": img_url,
                "rights_holder": rights_holder,
                "institution": institution,
                "attribution": f"GBIF occurrence {occ_key} ({latin_name}), {img_license}"
                               + (f", {rights_holder}" if rights_holder else ""),
            })

            if downloaded % 10 == 0:
                _info(f"    Downloaded {downloaded}/{need} for {species_name}")
            time.sleep(IMAGE_DELAY)

        end_of_records = page.get("endOfRecords", True)
        if end_of_records:
            break
        offset += page_size

    return downloaded, provenance


def _save_provenance(provenance: list[dict], directory: Path) -> None:
    """Append new records to PROVENANCE_gbif.json in the directory."""
    prov_path = directory / "PROVENANCE_gbif.json"
    existing: list[dict] = []
    if prov_path.exists():
        try:
            existing = json.loads(prov_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = []
    existing.extend(provenance)
    prov_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download CC-licensed fish photos from GBIF for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--species",
        type=str,
        default=None,
        help="Comma-separated list of species to fetch (e.g. pike,bream,roach). "
             "Use --all or --all-new for bulk fetch.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Fetch all 14 species in the registry.",
    )
    p.add_argument(
        "--all-new",
        action="store_true",
        help="Fetch only newly added taxonomy classes (Cyprinidae, Siluriformes, new Salmonidae).",
    )
    p.add_argument(
        "--max",
        type=int,
        default=80,
        help="Target images per species (script stops when directory reaches this count).",
    )
    p.add_argument(
        "--no-stage-a-copy",
        action="store_true",
        help="Do not copy fetched fish images to Stage A whole_fish directory.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fetched without downloading.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _banner("GBIF Fish Image Fetcher")
    _info(f"  Repo: {REPO_ROOT}")
    _info(f"  Max per species: {args.max}")
    _info(f"  Dry-run: {args.dry_run}")

    # Determine species list
    if args.all:
        target_species = list(SPECIES_REGISTRY.keys())
    elif args.all_new:
        target_species = sorted(NEW_SPECIES)
    elif args.species:
        target_species = [s.strip() for s in args.species.split(",") if s.strip()]
    else:
        # Default: fetch all new species
        target_species = sorted(NEW_SPECIES)
        _info("  No --species / --all specified; defaulting to --all-new")

    # Validate requested species
    unknown = [s for s in target_species if s not in SPECIES_REGISTRY]
    if unknown:
        _fail(f"Unknown species: {unknown}")
        _fail(f"Available: {sorted(SPECIES_REGISTRY.keys())}")
        sys.exit(1)

    _info(f"  Target species ({len(target_species)}): {', '.join(target_species)}")

    # Fetch
    total_downloaded = 0
    for species_name in target_species:
        info = SPECIES_REGISTRY[species_name]
        taxon_key: int = info["taxon_key"]
        latin: str = info["latin"]

        _banner(f"Fetching: {species_name} ({latin}, taxonKey={taxon_key})")

        stage_b_dir = STAGE_B_DIR / species_name
        existing_hashes = _existing_hashes(stage_b_dir)

        n, provenance = fetch_species(
            species_name=species_name,
            taxon_key=taxon_key,
            latin_name=latin,
            target_dir=stage_b_dir,
            max_images=args.max,
            dry_run=args.dry_run,
            existing_hashes=existing_hashes,
        )
        total_downloaded += n

        if not args.dry_run and provenance:
            _save_provenance(provenance, stage_b_dir)

        # Copy to Stage A whole_fish (supplement detection training)
        if not args.no_stage_a_copy and not args.dry_run and provenance:
            STAGE_A_WHOLE_FISH.mkdir(parents=True, exist_ok=True)
            stage_a_hashes = _existing_hashes(STAGE_A_WHOLE_FISH)
            copied = 0
            for rec in provenance:
                src = stage_b_dir / rec["file"]
                if not src.exists():
                    continue
                h = _md5(src)
                if h in stage_a_hashes:
                    continue
                stage_a_hashes.add(h)
                dst = STAGE_A_WHOLE_FISH / f"gbif_{species_name}_{rec['file']}"
                import shutil
                shutil.copy2(src, dst)
                copied += 1
            if copied:
                _info(f"  Copied {copied} images to stage_a/raw/whole_fish/")

        status = "OK" if not args.dry_run else "DRY-RUN"
        _ok(f"{species_name}: {n} new images [{status}]")

    _banner("GBIF Fetch Complete")
    _ok(f"Total images downloaded: {total_downloaded}")

    # Print dataset status
    print()
    print("  Stage B species counts after fetch:")
    print(f"  {'Species':<20} {'Images':>7}")
    print(f"  {'-' * 30}")
    for sp in sorted(SPECIES_REGISTRY.keys()) + ["unknown_fish"]:
        folder = STAGE_B_DIR / sp
        if folder.exists():
            n = sum(1 for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            status = "OK" if n >= 15 else "NEED MORE"
            print(f"  {sp:<20} {n:>7}  [{status}]")
        else:
            print(f"  {sp:<20} {'0':>7}  [MISSING]")


if __name__ == "__main__":
    main()
