"""
intake_external_lite.py — U5 External Dataset Intake Lite.

Produces license/provenance manifests for known public fish datasets without
requiring actual downloads. Downloads are optional and capped (--max-files).

Outputs:
  data/external_public/manifests/external_sources_manifest.json   — tracked
  data/external_public/manifests/external_license_report.json     — tracked
  data/external_public/manifests/external_dataset_inventory.json  — tracked

Rules:
  - External public data is ALWAYS kept separate from Telegram-private data.
  - Every source must have: license, citation/URL, access date.
  - No source with unclear license is marked usable_for_training.
  - No Telegram private metadata appears in external manifests.
  - Manifest-only mode is valid when actual download is blocked/skipped.
  - Supports --dry-run and --max-files.

Usage:
    python3 scripts/intake_external_lite.py            # manifest-only
    python3 scripts/intake_external_lite.py --dry-run  # print manifests, no write
    python3 scripts/intake_external_lite.py --check-availability  # test HTTP reachability

source=public_external, license=see_individual_sources
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_PUBLIC_DIR = REPO_ROOT / "data" / "external_public"
MANIFESTS_DIR = EXTERNAL_PUBLIC_DIR / "manifests"

MANIFEST_SCHEMA_VERSION = "u5_external_intake_v1"
ACCESS_DATE = "2026-05-01"


# ─── Source registry ──────────────────────────────────────────────────────────


# Each source entry describes one external dataset.
# "download_method" = "api" | "manual" | "direct_url" | "login_required" | "blocked"
EXTERNAL_SOURCES: list[dict[str, Any]] = [
    {
        "source_id": "deepfish_jcu",
        "name": "DeepFish (James Cook University)",
        "url": "https://alzayats.github.io/DeepFish/",
        "data_url": "https://data.qld.edu.au/article/c0d00059-dfcb-49f5-98ba-d0b6e27ac7ce",
        "license": "CC-BY-4.0",
        "license_allows_training": True,
        "citation": (
            "Saleh A, Laradji IH, Konovalov DA, Bradley M, Wayner D, Bekris M. "
            "A realistic fish-habitat dataset to evaluate algorithms for underwater "
            "visual analysis. Scientific Reports, 2020. DOI: 10.1038/s41598-020-71639-x"
        ),
        "access_date": ACCESS_DATE,
        "description": (
            "~40K underwater frames from Australian tropical freshwater habitats. "
            "Split into fish-present (Presence) and fish-absent (Absence) subsets."
        ),
        "domain": "underwater",
        "suitability": ["structural_fish_nonfish"],
        "usable_for_stage_a": True,
        "usable_for_stage_b": False,
        "usable_for_training": True,
        "training_note": (
            "Useful for stage A (whole_fish vs no_fish) structural training. "
            "NOT suitable for stage B — tropical species, not Russian freshwater."
        ),
        "download_method": "manual",
        "download_size_approx": "~9 GB",
        "download_status": "not_downloaded",
        "download_blocked_reason": (
            "QLD data portal requires JavaScript-rendered download page — manual step needed. "
            "See scripts/fetch_deepfish.py for download instructions."
        ),
        "local_path": None,
        "file_count": None,
        "class_counts": {
            "presence_fish": "~15000",
            "absence_no_fish": "~25000",
        },
    },
    {
        "source_id": "gbif_freshwater",
        "name": "GBIF CC0/CC-BY freshwater fish photos",
        "url": "https://www.gbif.org",
        "data_url": "https://api.gbif.org/v1/occurrence/search",
        "license": "CC0-1.0 and CC-BY-4.0 (filtered by script)",
        "license_allows_training": True,
        "citation": (
            "GBIF.org (2026), GBIF Occurrence Download. "
            "https://www.gbif.org (accessed 2026-05-01)"
        ),
        "access_date": ACCESS_DATE,
        "description": (
            "Biodiversity occurrence data with photos. Filtered to CC0 and CC-BY images only. "
            "15 Russian freshwater species registered. See scripts/fetch_gbif.py."
        ),
        "domain": "specimen",
        "suitability": ["structural_fish_nonfish", "species_classification"],
        "usable_for_stage_a": True,
        "usable_for_stage_b": True,
        "usable_for_training": True,
        "training_note": (
            "Good for stage B (species classification). "
            "Images are mostly specimen/museum/underwater — different domain from recreational photos."
        ),
        "download_method": "api",
        "download_size_approx": "variable (capped by --max-items per species)",
        "download_status": "partial",
        "download_blocked_reason": None,
        "local_path": str(REPO_ROOT / "data" / "fish_dataset" / "stage_b"),
        "file_count": None,  # counted dynamically
        "class_counts": None,  # counted dynamically
    },
    {
        "source_id": "inaturalist_api",
        "name": "iNaturalist Public API (research-grade)",
        "url": "https://www.inaturalist.org",
        "data_url": "https://api.inaturalist.org/v1/observations",
        "license": "CC0, CC-BY, CC-BY-SA (filtered by script)",
        "license_allows_training": True,
        "citation": (
            "iNaturalist contributors, iNaturalist (2026). "
            "https://www.inaturalist.org (accessed 2026-05-01). "
            "CC-licensed observations only."
        ),
        "access_date": ACCESS_DATE,
        "description": (
            "Research-grade fish observations with CC licenses. "
            "Recreational/out-of-water photos — closer domain to Telegram fishing photos. "
            "See scripts/fetch_inaturalist.py."
        ),
        "domain": "recreational_out_of_water",
        "suitability": ["structural_fish_nonfish", "species_classification"],
        "usable_for_stage_a": True,
        "usable_for_stage_b": True,
        "usable_for_training": True,
        "training_note": (
            "Recommended primary external source — domain closest to Telegram fishing photos. "
            "License is per-observation: script must filter CC0/CC-BY/CC-BY-SA."
        ),
        "download_method": "api",
        "download_size_approx": "variable",
        "download_status": "partial",
        "download_blocked_reason": None,
        "local_path": str(REPO_ROOT / "data" / "fish_dataset" / "stage_b"),
        "file_count": None,
        "class_counts": None,
    },
    {
        "source_id": "fishnet_open_images",
        "name": "FishNet Open Images Database (UC San Diego)",
        "url": "https://fishnet.ai",
        "data_url": "https://drive.google.com/drive/folders/1Vl4J5yEgfqrfWVtNHREJYtfCAMHn3Kfl",
        "license": "CC-BY-4.0 (claimed)",
        "license_allows_training": True,
        "citation": (
            "FishNet: A Large-Scale Dataset and Benchmark for Fish Recognition, "
            "Detection, and Functional Traits Prediction. "
            "Kay et al., arXiv:2304.01779 (2023)."
        ),
        "access_date": ACCESS_DATE,
        "description": (
            "~94K fish images, 17K+ species. Large coverage. "
            "NOTE: Hosted on Google Drive — requires manual download."
        ),
        "domain": "mixed_specimen_and_recreational",
        "suitability": ["structural_fish_nonfish", "species_classification"],
        "usable_for_stage_a": True,
        "usable_for_stage_b": True,
        "usable_for_training": True,
        "training_note": (
            "High potential — wide species coverage. "
            "Manual download required (Google Drive). "
            "Verify license per-image if possible."
        ),
        "download_method": "login_required",
        "download_size_approx": "~50+ GB",
        "download_status": "not_downloaded",
        "download_blocked_reason": (
            "Hosted on Google Drive, requires browser/manual download. "
            "See scripts/fetch_deepfish.py for a similar manual pattern."
        ),
        "local_path": None,
        "file_count": None,
        "class_counts": None,
    },
    {
        "source_id": "fish_vista",
        "name": "Fish-Vista (NC State / BGNN)",
        "url": "https://bgnn.tulane.edu/fish-vista",
        "data_url": "https://huggingface.co/datasets/Imageomics/fish-vista",
        "license": "CC-BY-4.0 (claimed for image data; verify per-image)",
        "license_allows_training": True,
        "citation": (
            "Mehrab et al., Fish-Vista: A Multi-Purpose Dataset for "
            "Understanding & Identification of Traits in Fish. "
            "NeurIPS 2024 Datasets and Benchmarks Track. arXiv:2407.08027."
        ),
        "access_date": ACCESS_DATE,
        "description": (
            "60K+ fish images from museum specimens. "
            "Trait-level annotations. Primarily lateral-view museum specimens. "
            "Available via HuggingFace Datasets."
        ),
        "domain": "museum_specimen",
        "suitability": ["structural_fish_nonfish", "species_classification"],
        "usable_for_stage_a": True,
        "usable_for_stage_b": True,
        "usable_for_training": True,
        "training_note": (
            "Museum specimens — very different domain from fishing photos. "
            "Good for structural 'is this a fish' classifier. "
            "Less useful for species matching with recreational photos."
        ),
        "download_method": "api",
        "download_size_approx": "~30+ GB",
        "download_status": "not_downloaded",
        "download_blocked_reason": (
            "Large dataset; HuggingFace datasets library required. "
            "MVP: manual sample download for testing."
        ),
        "local_path": None,
        "file_count": None,
        "class_counts": None,
    },
    {
        "source_id": "wikimedia_commons_fish",
        "name": "Wikimedia Commons (fish category, explicit license filter)",
        "url": "https://commons.wikimedia.org",
        "data_url": "https://commons.wikimedia.org/w/api.php",
        "license": "CC0, CC-BY, CC-BY-SA (per-file, API-filtered)",
        "license_allows_training": True,
        "citation": (
            "Wikimedia Commons contributors (2026). "
            "https://commons.wikimedia.org (accessed 2026-05-01). "
            "CC-licensed files only. Attribution required for CC-BY/CC-BY-SA."
        ),
        "access_date": ACCESS_DATE,
        "description": (
            "Encyclopedia-style fish photos. Varied quality. "
            "No login required — MediaWiki API. See scripts/fetch_wikimedia_lures.py for pattern."
        ),
        "domain": "mixed",
        "suitability": ["structural_fish_nonfish", "species_classification"],
        "usable_for_stage_a": True,
        "usable_for_stage_b": True,
        "usable_for_training": True,
        "training_note": (
            "License must be verified per-image via API. "
            "Do not use files with unclear or 'some rights reserved' license."
        ),
        "download_method": "api",
        "download_size_approx": "variable (sample possible)",
        "download_status": "not_downloaded",
        "download_blocked_reason": "Not yet downloaded — API adapter available in fetch_wikimedia_lures.py",
        "local_path": None,
        "file_count": None,
        "class_counts": None,
    },
]

# Sources that are explicitly skipped with documented reasons
SKIPPED_SOURCES: list[dict[str, Any]] = [
    {
        "source_id": "kaggle_fish_datasets",
        "name": "Kaggle fish datasets (various)",
        "skip_reason": "Requires Kaggle account login — prohibited by project rules",
    },
    {
        "source_id": "imagenet_fish",
        "name": "ImageNet (fish synsets)",
        "skip_reason": "Requires account registration — prohibited by project rules",
    },
    {
        "source_id": "fishbase_images",
        "name": "FishBase image scraping",
        "skip_reason": "No bulk API; individual scraping would violate ToS",
    },
    {
        "source_id": "google_bing_scraping",
        "name": "Google / Bing / Yandex image search",
        "skip_reason": "Random internet scraping — license unclear, prohibited by project rules",
    },
]


# ─── I/O helpers ─────────────────────────────────────────────────────────────


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ─── Dynamic counts ───────────────────────────────────────────────────────────


def _count_stage_b_images() -> dict[str, int]:
    """Count existing stage_b species images from inaturalist/gbif downloads."""
    stage_b = REPO_ROOT / "data" / "fish_dataset" / "stage_b"
    counts: dict[str, int] = {}
    if not stage_b.exists():
        return counts
    for species_dir in sorted(stage_b.iterdir()):
        if not species_dir.is_dir():
            continue
        n = sum(1 for p in species_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        if n > 0:
            counts[species_dir.name] = n
    return counts


def _check_availability(url: str) -> str:
    """Quick HTTP HEAD check. Returns 'reachable', 'unreachable', or 'skipped'."""
    try:
        import urllib.request
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "fish-bot-intake-check/1.0")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return f"reachable (HTTP {resp.status})"
    except Exception as exc:
        return f"unreachable ({exc})"


# ─── Manifest builders ────────────────────────────────────────────────────────


def build_sources_manifest(now: str, stage_b_counts: dict) -> dict:
    # Enrich GBIF/iNaturalist with current stage_b file counts
    sources = []
    for src in EXTERNAL_SOURCES:
        entry = dict(src)
        if src["source_id"] in ("gbif_freshwater", "inaturalist_api"):
            entry["file_count"] = sum(stage_b_counts.values()) if stage_b_counts else 0
            entry["class_counts"] = stage_b_counts if stage_b_counts else {}
        sources.append(entry)

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": now,
        "access_date": ACCESS_DATE,
        "source_count": len(sources),
        "skipped_count": len(SKIPPED_SOURCES),
        "sources": sources,
        "skipped_sources": SKIPPED_SOURCES,
        "data_integrity_note": (
            "External public data is ALWAYS kept separate from Telegram-private data. "
            "No Telegram metadata appears in this manifest."
        ),
    }


def build_license_report(now: str) -> dict:
    usable = [s for s in EXTERNAL_SOURCES if s["usable_for_training"]]
    unusable = [s for s in EXTERNAL_SOURCES if not s["usable_for_training"]]
    training_ready = [s for s in usable if s["download_status"] in ("partial", "downloaded")]
    training_pending = [s for s in usable if s["download_status"] == "not_downloaded"]

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": now,
        "summary": {
            "total_sources": len(EXTERNAL_SOURCES),
            "skipped_sources": len(SKIPPED_SOURCES),
            "usable_for_training": len(usable),
            "unusable_for_training": len(unusable),
            "training_data_ready": len(training_ready),
            "training_data_pending_download": len(training_pending),
        },
        "usable_sources": [
            {
                "source_id": s["source_id"],
                "name": s["name"],
                "license": s["license"],
                "download_status": s["download_status"],
                "training_note": s["training_note"],
            }
            for s in usable
        ],
        "skipped_sources": [
            {"source_id": s["source_id"], "name": s["name"], "skip_reason": s["skip_reason"]}
            for s in SKIPPED_SOURCES
        ],
    }


def build_inventory(now: str, stage_b_counts: dict) -> dict:
    total_stage_b = sum(stage_b_counts.values())
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at": now,
        "local_external_data": {
            "stage_b_species_images": {
                "location": "data/fish_dataset/stage_b/",
                "source": "GBIF + iNaturalist (CC-licensed)",
                "total_images": total_stage_b,
                "per_species": stage_b_counts,
            },
            "stage_a_raw": {
                "location": "data/fish_dataset/stage_a/raw/",
                "source": "TBD — DeepFish or manual",
                "total_images": 0,
                "note": "Not yet downloaded; manual step required for DeepFish",
            },
        },
        "pending_downloads": [
            {
                "source_id": s["source_id"],
                "name": s["name"],
                "download_method": s["download_method"],
                "blocked_reason": s.get("download_blocked_reason"),
                "size_approx": s.get("download_size_approx"),
            }
            for s in EXTERNAL_SOURCES
            if s["download_status"] == "not_downloaded"
        ],
        "training_recommendation": (
            "For MVP structural training: "
            "1) Download DeepFish (no_fish frames) for stage A negatives. "
            "2) Use existing iNaturalist/GBIF stage_b images for species diversity. "
            "3) Combine with reviewed Telegram seed (fish positives). "
            "Minimum needed: 100+ no_fish images for balanced structural training."
        ),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="U5 External Dataset Intake Lite — produce license/provenance manifests"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print manifests without writing files",
    )
    parser.add_argument(
        "--check-availability",
        action="store_true",
        help="Check HTTP reachability of each source URL",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    now = datetime.now(timezone.utc).isoformat()
    stage_b_counts = _count_stage_b_images()

    if stage_b_counts:
        log.info("Found %d stage_b images across %d species", sum(stage_b_counts.values()), len(stage_b_counts))

    if args.check_availability:
        log.info("Checking source availability...")
        for src in EXTERNAL_SOURCES:
            status = _check_availability(src["url"])
            log.info("  %s: %s", src["source_id"], status)

    manifest = build_sources_manifest(now, stage_b_counts)
    license_report = build_license_report(now)
    inventory = build_inventory(now, stage_b_counts)

    if args.dry_run:
        log.info("DRY RUN — not writing files")
        print("=== SOURCES MANIFEST ===")
        print(json.dumps(manifest, ensure_ascii=False, indent=2)[:2000])
        print("=== LICENSE REPORT ===")
        print(json.dumps(license_report, ensure_ascii=False, indent=2))
        return 0

    _write_json_atomic(MANIFESTS_DIR / "external_sources_manifest.json", manifest)
    _write_json_atomic(MANIFESTS_DIR / "external_license_report.json", license_report)
    _write_json_atomic(MANIFESTS_DIR / "external_dataset_inventory.json", inventory)

    log.info("Wrote: %s", MANIFESTS_DIR / "external_sources_manifest.json")
    log.info("Wrote: %s", MANIFESTS_DIR / "external_license_report.json")
    log.info("Wrote: %s", MANIFESTS_DIR / "external_dataset_inventory.json")
    log.info("External intake lite DONE")
    log.info(
        "Training-ready external data: %d images in stage_b (GBIF/iNaturalist)",
        sum(stage_b_counts.values()),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
