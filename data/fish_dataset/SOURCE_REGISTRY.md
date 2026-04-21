# Fish Dataset Source Registry

## Sources USED

| Source | License | Script | Notes |
|--------|---------|--------|-------|
| GBIF (Global Biodiversity Information Facility) | CC0 / CC-BY 4.0 | fetch_gbif.py | Primary source; 14 species registered |
| iNaturalist | CC0 / CC-BY / CC-BY-SA | fetch_inaturalist.py | Research-grade observations; second source |

## Sources SKIPPED

| Source | Reason |
|--------|--------|
| Kaggle datasets | Requires login / account |
| FishNet (Google Drive) | Requires login / manual download |
| ImageNet | Requires login |
| FishBase images | No bulk API; individual scraping would violate ToS |

## Potential Future Sources (no login required)

| Source | License | How to access |
|--------|---------|---------------|
| Wikimedia Commons | CC / Public domain | https://commons.wikimedia.org/w/api.php (MediaWiki API, no login) |
| Open Images V7 | CC-BY 4.0 | https://storage.googleapis.com/openimages/web/download_v7.html (CSV + direct URLs) |
| BOLD Systems (fish subset) | CC | https://v3.boldsystems.org/index.php/API_Public/combined |

## Current Per-Class Counts (stage_b)

| Class | Images |
|-------|--------|
| roach | 81 |
| rainbow_trout | 81 |
| pike | 81 |
| perch | 81 |
| grayling | 81 |
| common_carp | 81 |
| brown_trout | 81 |
| bream | 81 |
| atlantic_salmon | 81 |
| ide | 77 |
| wels_catfish | 70 |
| crucian_carp | 70 |
| whitefish | 46 |
| taimen | 1 |
| zander | 0 (NEW — directory created) |
| unknown_fish | 0 |

## Weak Classes (need more data)

- **taimen**: only 1 image — needs urgent expansion via GBIF + iNaturalist
- **whitefish**: 46 images — below ideal threshold of 80+
- **zander**: 0 images — new class, requires initial data ingestion

## Next Steps

1. Run: `python3 scripts/fetch_gbif.py --species zander --max 80`
2. Run: `python3 scripts/fetch_inaturalist.py --species zander --max 80`
3. Run: `python3 scripts/fetch_gbif.py --species taimen --max 80`
4. Run: `python3 scripts/fetch_inaturalist.py --species taimen,whitefish --max 60`
5. After data ingestion: retrain Stage B with 16 classes
