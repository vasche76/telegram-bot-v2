# Fish Dataset — Data Source Report
# Agent: Data Source Agent + Licensing/Compliance Agent
# Date: 2026-04-15

This document classifies every candidate data source by legal usability for model training.

**Legend:**
- ✅ A — Legally usable for training (clear open license)
- ⚠️ B — Usable only as reference/taxonomy guidance, NOT as training images
- ❌ C — Not safely usable (unclear/restricted license, or gray scraping)

---

## SECTION 1 — Fish Image Datasets

---

### 1.1 iNaturalist (via public API)

| Field | Value |
|-------|-------|
| URL | https://api.inaturalist.org/v1/ |
| Owner | iNaturalist / California Academy of Sciences |
| License | Per-observation: CC0, CC-BY, CC-BY-NC, CC-BY-SA, CC-BY-NC-SA |
| Commercial use | Depends on individual photo license (CC0/CC-BY: yes; CC-BY-NC: no) |
| Redistribution | Allowed with attribution for CC0/CC-BY/CC-BY-SA |
| Training use | ✅ Clearly allowed for CC0 and CC-BY licensed photos |
| Class coverage | All 5 target species + no_fish (non-fish observations) |
| Image realism | High — real wildlife photos, camera/phone, outdoor lighting |
| Stage A usefulness | ✅ whole_fish, ⚠️ no direct lure/fry/fish_part coverage |
| Stage B usefulness | ✅ Primary source for species classification |

**Download method:** Public REST API — no scraping, no authentication required.
**Filter used:** `photo_license=cc0,cc-by,cc-by-sa` — only open training-compatible licenses.
**Attribution requirement:** Must credit original observer. Tracked in PROVENANCE.json per species.

**Target species and iNaturalist taxon IDs:**

| Species | Latin name | iNaturalist taxon search |
|---------|-----------|--------------------------|
| pike / щука | Esox lucius | taxon_name=Esox+lucius |
| taimen / таймень | Hucho taimen | taxon_name=Hucho+taimen |
| grayling / хариус | Thymallus thymallus + Thymallus arcticus | taxon_name=Thymallus |
| whitefish / сиг | Coregonus lavaretus + C. peled | taxon_name=Coregonus |
| perch / окунь | Perca fluviatilis | taxon_name=Perca+fluviatilis |

**VERDICT: ✅ A — PRIMARY SOURCE for Stage B species classification.**

---

### 1.2 Wikimedia Commons (fishing lures and fish photos)

| Field | Value |
|-------|-------|
| URL | https://commons.wikimedia.org/w/api.php |
| Owner | Wikimedia Foundation (content owned by individual contributors) |
| License | CC0, CC-BY, CC-BY-SA, Public Domain (per file, retrievable via API) |
| Commercial use | Yes for CC0/PD, attribution required for CC-BY |
| Training use | ✅ Clearly allowed for CC0, CC-BY, CC-BY-SA, PD files |
| Class coverage | lure (fishing tackle), whole_fish, various species |
| Image realism | Medium — encyclopedia-quality photos, often controlled lighting |
| Stage A usefulness | ✅ PRIMARY SOURCE for lure class |
| Stage B usefulness | ⚠️ Supplementary — useful for morphology reference |

**Download method:** MediaWiki API (action=query) — structured, legal, rate-limited.
**Filter used:** Check imageinfo for license field = CC0, CC-BY-*, Public Domain.

**Useful search terms:**
- "fishing lure" / "fishing wobbler" / "fishing spinner" / "fishing spoon"
- "pike fish" / "European perch fish" / "grayling fish"

**VERDICT: ✅ A — PRIMARY SOURCE for lure class (Stage A). Supplementary for fish classes.**

---

### 1.3 Open Images V7 (Google)

| Field | Value |
|-------|-------|
| URL | https://storage.googleapis.com/openimages/web/index.html |
| Owner | Google LLC |
| License | CC BY 4.0 for annotations; images under Creative Commons (varied per image) |
| Commercial use | CC BY 4.0 annotations: yes with attribution |
| Training use | ✅ Annotations clearly CC-BY; images individually licensed but batch download via official tool |
| Class coverage | "Fish" class (bounding boxes available) |
| Image realism | High — real-world diverse images |
| Stage A usefulness | ✅ Has bounding box annotations for "Fish" → whole_fish |
| Stage B usefulness | ⚠️ No species-level labels |

**Download method:** Official `fiftyone` library or manual CSV + image downloader.
**Annotation license:** CC BY 4.0 (clearly usable).
**Image license:** Each image is individually licensed. Batch download via official tools retrieves only images the dataset creator has cleared.

**Class mapping:**
- OI "Fish" → whole_fish (Stage A)
- OI "Fishing rod" + "Fishing bait" → lure (Stage A, partial)
- OI "Sports" backgrounds → no_fish (Stage A)

**Blocker:** Requires `fiftyone` package or manual CSV pipeline. Not installed currently.
**VERDICT: ✅ A — HIGH VALUE for Stage A bounding boxes. Deferred to next phase (requires fiftyone).**

---

### 1.4 Roboflow Universe — fishing datasets

| Field | Value |
|-------|-------|
| URL | https://universe.roboflow.com |
| Owner | Individual contributors |
| License | Varies per dataset: CC-BY, CC BY 4.0, MIT, or proprietary |
| Training use | ✅ For openly licensed datasets (CC-BY, MIT) |
| Class coverage | fishing lures, fish detection, some species |
| Stage A usefulness | ✅ Multiple lure + fish detector datasets with YOLO annotations |

**Notable datasets (must verify license on each before use):**
- "Fish Detection" — multiple versions, check license per project
- "Fishing Lure Detection" — varies
- "Aquarium Fish" — mostly CC-BY

**Blocker:** Requires `roboflow` API key or manual download from website.
**VERDICT: ✅ A — VALID if license confirmed per dataset. Manual verification required per dataset.**

---

### 1.5 FishBase (fishbase.org)

| Field | Value |
|-------|-------|
| URL | https://fishbase.org |
| Owner | FishBase consortium |
| License | Species data: freely accessible; photos: © original photographers, NOT CC |
| Training use | ❌ Photo copyrights belong to individual photographers, not openly licensed |
| Reference use | ✅ Excellent for morphology, taxonomy, species descriptions |

**VERDICT: ⚠️ B — REFERENCE ONLY. Excellent taxonomy and morphology database. Do NOT use photos for training.**

---

### 1.6 ImageNet (fish synsets)

| Field | Value |
|-------|-------|
| URL | https://image-net.org |
| Owner | Stanford / Princeton |
| License | "Research only" — explicitly prohibits commercial use |
| Training use | ❌ for commercial/production systems |

**VERDICT: ❌ C — NOT SAFELY USABLE for a production bot.**

---

### 1.7 Google Images / Bing Images / Getty

| Field | Value |
|-------|-------|
| Training use | ❌ Explicit scraping violates ToS; images are typically copyrighted |

**VERDICT: ❌ C — NOT USABLE.**

---

### 1.8 iNaturalist-GBIF Export (occurrence images via GBIF)

| Field | Value |
|-------|-------|
| URL | https://www.gbif.org |
| Owner | Global Biodiversity Information Facility |
| License | CC0 for occurrence data; photos from contributing sources (often iNaturalist CC-BY) |
| Training use | ✅ Where photo license is CC0 or CC-BY |

**VERDICT: ✅ A — Redundant with direct iNaturalist API for our species. Use iNaturalist directly.**

---

### 1.9 Flickr Creative Commons

| Field | Value |
|-------|-------|
| URL | https://www.flickr.com/search/?license=1,2,3,4,5,6,9,10 |
| Owner | Individual photographers |
| License | CC0, CC-BY, CC-BY-SA, CC-BY-NC, CC-BY-NC-SA — per photo |
| Training use | ✅ For CC0 and CC-BY photos only |
| Class coverage | Fishing photos, lures, catch photos |
| Blocker | Requires Flickr API key |

**VERDICT: ✅ A — VALID with API key. Deferred to Phase 2 (requires API setup).**

---

## SECTION 2 — Academic / Reference Sources

These are usable for morphology knowledge, labeling guides, and prompt improvement only.

| Source | URL | Purpose | Classification |
|--------|-----|---------|----------------|
| FishBase | fishbase.org | Full morphology, distribution, taxonomy | ⚠️ B — Reference |
| "Рыбы России" (Lebedev, Spanovskaya) | Library/academic | Russian freshwater fish field guide | ⚠️ B — Reference |
| "Определитель рыб СССР" (Berg) | Academic archive | Classic Soviet ichthyology reference | ⚠️ B — Reference |
| "Пресноводные рыбы России" (Kotlyar) | Library | Modern Russian freshwater guide | ⚠️ B — Reference |
| "Handbook of European Freshwater Fishes" (Kottelat & Freyhof) | Academic | European species atlas | ⚠️ B — Reference |
| iNaturalist species pages | inaturalist.org/taxa/* | Morphology photos with CC license | ✅ A — Some photos usable |
| Wikipedia species pages | wikipedia.org | Overview + curated Commons photos | ⚠️ B — Reference (images on Commons may be ✅ A) |
| "Freshwater fishes of Northern Eurasia" (Froese & Pauly) | FishBase | Meristic data, fin counts | ⚠️ B — Reference |

---

## SECTION 3 — Key Species Morphology Summary
### (from academic sources — for labeling guide and classifier prompt improvement)

### Pike / Щука (Esox lucius)
- Elongated body; flat "duck-bill" snout; very large mouth
- Coloration: olive-green/brown with pale spots/stripes on flanks
- Dorsal fin: far back, near tail (key distinguishing feature)
- Tail: deeply forked, large
- Belly: pale/whitish
- **Confusion cases:** Large grayling fins can look like pike at distance; wobbler lures often mimic pike pattern

### Taimen / Таймень (Hucho taimen)
- Very large (up to 2m); salmon-family body shape
- Large flat head; powerful toothed mouth
- Coloration: silver-grey with dark X-shaped spots on flanks
- Fins: reddish/orange tail and paired fins; adipose fin present
- **Key ID feature:** Reddish tail + massive size + X-spots
- **Confusion cases:** Large Siberian salmon; other salmonids

### Grayling / Хариус (Thymallus thymallus / T. arcticus)
- **KEY FEATURE:** Extremely tall sail-like dorsal fin
- Body: silver-grey with small dark spots
- Small terminal mouth
- Adipose fin between dorsal and tail
- Tail: forked
- **Confusion cases:** Whitefish at juvenile stage; other salmonids without the dorsal "sail"

### Whitefish / Сиг (Coregonus lavaretus / C. peled)
- Silvery compressed body; "blade-like" profile
- Small subterminal (slightly inferior) mouth — no teeth visible
- Small adipose fin (salmon family)
- Forked tail
- Large silvery scales, no body spots
- **Confusion cases:** Silver bream; juvenile salmonids; sheatfish

### Perch / Окунь (Perca fluviatilis)
- **KEY FEATURES:** Dark vertical bars on greenish-yellow flanks ("striped jersey")
- Humped back; laterally compressed
- First dorsal fin: spiny, black, with black spot at rear
- Pectoral and pelvic fins: bright orange-red
- **Confusion cases:** Juvenile pike-perch (zander); ruff at small size

---

## SECTION 4 — Source Prioritization for This Phase

| Priority | Source | Stage | Why |
|----------|---------|-------|-----|
| 1 | iNaturalist API | Stage B (species) + Stage A (whole_fish) | CC-BY, public API, all target species |
| 2 | Wikimedia Commons API | Stage A (lure) | CC-BY/CC0, structured API, clear license per file |
| 3 | Open Images V7 | Stage A (fish bbox) | CC-BY annotations, production quality — Phase 2 |
| 4 | Roboflow Universe | Stage A (lure+fish) | Good datasets, manual license check — Phase 2 |
| 5 | Flickr CC | All classes | Requires API key — Phase 2 |

---

## SECTION 5 — Licensing Classification Summary

```
A — USABLE FOR TRAINING:
  ✅ iNaturalist CC0/CC-BY/CC-BY-SA photos (via API)
  ✅ Wikimedia Commons CC0/CC-BY/PD files (via API)
  ✅ Open Images V7 (via official download tools)
  ✅ Roboflow datasets with confirmed CC-BY or MIT license

B — REFERENCE ONLY (not training images):
  ⚠️ FishBase species photos
  ⚠️ Wikipedia text content (morphology descriptions)
  ⚠️ Academic books/field guides (text/illustration knowledge)
  ⚠️ Flickr CC-BY-NC (non-commercial only)

C — NOT USABLE:
  ❌ ImageNet (research only, not production)
  ❌ Google Images, Bing Images (ToS violation)
  ❌ Getty Images, Shutterstock (commercial)
  ❌ Any source without explicit license
```

---

## SECTION 6 — Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| iNaturalist photo misidentified by observer | Medium | Filter by `quality_grade=research` (community-verified) |
| CC-BY license requires attribution | Low risk | Track in PROVENANCE.json; not required for bot inference |
| Wikimedia file license incorrectly listed | Low | Check via API, skip unlicensed |
| Species label noise in wild datasets | High | Training with unknown_fish fallback; GPT fallback chain |
| Small dataset → overfitting | High | Transfer learning (ImageNet pretrained) + aggressive augmentation |

---

*Report generated by Data Source Agent + Licensing/Compliance Agent — 2026-04-15*
