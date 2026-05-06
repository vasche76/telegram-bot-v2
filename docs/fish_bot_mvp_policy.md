# Fish Bot MVP Response Policy

**Version:** 1.0  
**Date:** 2026-05-01  
**Audience:** Small group of fishing friends (~10 people), private use only  
**Target event:** September 2026 fishing trip

---

## 1. Structural Classification Policy

When a user sends a photo, the bot classifies it into one of the following categories:

| Category | Bot action |
| --- | --- |
| `fish` | Proceed to species guess + weight inquiry |
| `not_fish_or_other` | Politely inform user — no fish detected |
| `lure_gear` | "Looks like fishing gear, not a fish" |
| `fish_part` | "Visible fish fragment — full fish not detected" |
| `bad_quality` | Ask user to retake the photo (better light, less blur) |
| `poster_screenshot` | "This looks like a photo of a photo — send the original" |
| `out_of_scope` | "Cannot identify this — send a fish photo!" |
| `needs_human_review` | Fallback: treat as uncertain, do not guess |

### Confidence thresholds

- If structural confidence < 0.5: respond as `needs_human_review`
- If structural confidence ≥ 0.5 and < 0.7: add "(low confidence)" qualifier to response
- If structural confidence ≥ 0.7: respond normally

---

## 2. Species Policy

**Core rule: never overclaim. A wrong guess is worse than an honest "I don't know".**

### Wording rules

- Always use "possible species" or "похоже на" (looks like), never "this is definitely"
- Always show confidence percentage or qualitative label
- Always offer the user a way to correct the guess
- Default to `unknown_fish` if confidence < threshold or species is not in the known list

### Confidence thresholds

| Confidence | Response |
| --- | --- |
| ≥ 0.75 | "Probably a [species] (~75%+ confidence)" |
| 0.5 – 0.74 | "Could be [species], but I'm not certain" |
| < 0.5 | "I can't identify the species from this photo" |
| < 0.3 | Report as `unknown_fish` only |

### Species unknown fallback

If species cannot be identified:
- **EN:** "I can see this is a fish, but I can't identify the species from this photo. What kind of fish is it?"
- **RU:** "Вижу, что это рыба, но определить вид по фото не получается. Что это за рыба?"

### User correction flow

1. Bot shows species guess
2. User can reply with the correct species name
3. Bot confirms: "Got it — [species name]. I'll remember this for future catches."
4. Correction is saved locally for future improvement

---

## 3. Weight Estimation Policy

**Core rule: a range with uncertainty, never a single exact number.**

| Situation | Action |
| --- | --- |
| Not a fish | No weight estimate |
| Partial / bad quality photo | No reliable estimate — explain why |
| Fish visible but no measurements | Ask for length in cm |
| Length provided | Rough range ± wide margin |
| Length + girth provided | Narrower range ± 20% |
| Scale reference in photo | Note it, still provide a range |

### Always say

- This is an **approximate range**, not an exact weight
- Actual weight may differ by 20–50%
- For a better estimate, send length and girth in cm

### Never say

- "Your fish weighs X kg" (single value)
- "Exact weight: X kg"
- "Weight = X kg" (any form that implies precision)

### Example weight messages (EN)

> "⚖️ Length-based estimate for Pike at 60 cm: approximately **0.6–1.6 kg**.
> Wide range due to individual variation.
> 
> For a tighter estimate, send the girth at the thickest point in cm."

> "⚖️ No length provided — I can't give a reliable weight estimate.
> 
> Please send the fish length in cm (e.g. 45 cm)."

### Example weight messages (RU)

> "⚖️ Оценка по длине для Щуки (60 см): примерно **0.6–1.6 кг**.
> Диапазон широкий из-за индивидуальной изменчивости.
> 
> Для уточнения пришли обхват в самой толстой части в сантиметрах."

> "⚖️ Вес по фото можно оценить только грубо.
> Для более точной оценки пришли длину рыбы в см и, если можешь, обхват в самой толстой части."

---

## 4. Feedback Collection

The bot saves user corrections locally for future active learning. **No data is sent to external services.**

| User action | What bot saves |
| --- | --- |
| Confirms species | `species_confirmed = true`, `confirmed_species = <name>` |
| Corrects species | `species_corrected = true`, `corrected_species = <name>` |
| Provides length | `length_cm = <value>` |
| Provides girth | `girth_cm = <value>` |
| Provides actual weight | `actual_weight_kg = <value>` |
| Rates guess (👍/👎) | `structural_correct = true/false` |

All feedback is stored locally in `bot.db` with the catch record.

---

## 5. Full Example Conversations

### Example 1: Fish correctly identified (EN)

> **User:** [sends photo of a large pike]  
> **Bot:** 🐟 Looks like a fish! Possible species: **Pike** (confidence ~80%)  
> ⚖️ Send me the fish length in cm for a weight estimate.  
> Is this right? Reply with the species name to correct.

---

### Example 2: Fish identified with weight (RU)

> **User:** [sends photo]  
> **Bot:** 🐟 Похоже на рыбу! Возможный вид: **Щука** (уверенность ~75%)  
> ⚖️ Пришли длину в см для оценки веса.  
>  
> **User:** 58 см  
> **Bot:** ⚖️ Оценка по длине для Щуки (58 см): примерно **0.5–1.5 кг**.  
> Это приблизительный диапазон. Для уточнения пришли обхват в самой толстой части.

---

### Example 3: Not a fish (RU)

> **User:** [sends photo of lure]  
> **Bot:** 🎣 Похоже на приманку или снасть — рыбы не вижу.  
> Сфотографируй рыбу и пришли!

---

### Example 4: Bad quality photo (EN/RU)

> **User:** [sends blurry photo]  
> **Bot:** 📷 Фото слишком размытое или рыба видна частично.  
> Попробуй сфотографировать поближе при хорошем освещении.

---

### Example 5: Species unknown (RU)

> **User:** [sends photo of unusual fish]  
> **Bot:** 🐟 Вижу рыбу, но определить вид не получается.  
> Что это за рыба? Напиши название, и я запомню.

---

## 6. Tone Guidelines

- **Friendly and natural** — these are fishing friends, not app store users
- **Honest** — never bluff on species or weight
- **Practical** — short replies, actionable asks
- **Not robotic** — use casual fishing language where appropriate
- **Bilingual** — prefer Russian for the primary audience, but support English

---

## 7. What the Bot Does NOT Do

- Does not claim exact fish weight from a photo
- Does not claim certain species identification
- Does not process images on external cloud services (privacy rule)
- Does not auto-label unreviewed photos as training data
- Does not store photos or private data outside the local machine
- Does not give fishing advice, forecasts, or location recommendations (out of scope for MVP)
