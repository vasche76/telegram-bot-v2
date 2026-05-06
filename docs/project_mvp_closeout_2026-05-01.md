# Fish Bot MVP Closeout — 2026-05-01

**Project:** Telegram fish recognition bot  
**Audience:** ~10 fishing friends, September 2026 trip  
**Status:** MVP path complete — structural training blocked by class imbalance (actionable)

---

## 1. What Is Ready Today

| Component | Status | Location |
| --- | --- | --- |
| Phase D reviewed seed (500 records) | ✅ Verified | `data/intake_meta/tg_2026-04-24/filter_review_partial_summary.json` |
| Phase E Lite seed materialization | ✅ Done | `scripts/intake_phase_e_lite.py` |
| Privacy-safe reviewed seed summary | ✅ Done | `data/intake_meta/tg_2026-04-24/reviewed_seed/reviewed_seed_summary.json` |
| Fish weight estimator (range-based) | ✅ Done + tested | `bot/fish_vision/weight_estimator.py` |
| Bot MVP response policy | ✅ Done | `docs/fish_bot_mvp_policy.md` |
| External dataset intake manifests | ✅ Done | `data/external_public/manifests/` |
| MVP training dataset builder | ✅ Done | `scripts/build_mvp_training_dataset.py` |
| Training blocked report | ✅ Done (correct outcome) | `data/mvp_training/training_blocked_report.md` |
| Species v0 readiness report | ✅ Done | `docs/species_v0_readiness_report.md` |
| Bot integration todo | ✅ Done | `docs/bot_mvp_integration_todo.md` |
| 103 new tests — all pass | ✅ Done | `tests/test_intake_phase_e_lite.py`, `test_fish_weight_estimator.py`, `test_build_mvp_training_dataset.py` |
| Privacy/git verification | ✅ Clean | No private leaks, no commits, no pushes |

---

## 2. What Is Intentionally NOT Ready

| Component | Reason |
| --- | --- |
| Structural training | BLOCKED: 19.4:1 fish/non-fish imbalance (gate enforced) |
| Species classifier v0 | Optional — can train on external data; see species_v0_readiness_report.md |
| Bot photo handler integration | Low-risk; weight estimator not yet wired in |
| Telegram photos copied to training dir | Original export not in repo; requires local setup |

---

## 3. How to Run Phase E Lite

```bash
# Materialize reviewed seed from decision batches
python3 scripts/intake_phase_e_lite.py

# Dry-run (print summary without writing)
python3 scripts/intake_phase_e_lite.py --dry-run

# Output: data/intake_meta/tg_2026-04-24/reviewed_seed/
```

---

## 4. How to Run External Dataset Intake Lite

```bash
# Record manifests for all known public sources
python3 scripts/intake_external_lite.py

# Check source URL reachability
python3 scripts/intake_external_lite.py --check-availability

# Output: data/external_public/manifests/
```

---

## 5. How to Build MVP Training Dataset

```bash
# Check current data balance and training gates
python3 scripts/build_mvp_training_dataset.py

# Dry run (no files written)
python3 scripts/build_mvp_training_dataset.py --dry-run

# Output: data/mvp_training/manifests/
```

Training gates will pass once non-fish count ≥ 122 (current: 63, need +59 minimum).

---

## 6. How to Train Structural Baseline (Once Gates Pass)

```bash
# Requires Python 3.12 (PyTorch not available on Python 3.13)
# First: download DeepFish no_fish frames (9GB, manual step, CC-BY-4.0)
# See scripts/fetch_deepfish.py for instructions

# After gates pass:
~/.pyenv/versions/3.12.10/bin/python3 scripts/train_stage_a.py \
  --epochs 30 --batch 16 --device cpu

# Output: data/fish_models/detector_v2.pt (experimental)
```

**Note:** Model output is automatically marked EXPERIMENTAL.

---

## 7. How to Run Weight Estimator Tests

```bash
python3 -m pytest tests/test_fish_weight_estimator.py -v

# Or full suite (3 pre-existing failures unrelated to weight estimator):
python3 -m pytest tests/ -q
```

---

## 8. How the Bot Should Answer Friends

See: `docs/fish_bot_mvp_policy.md` for full policy.

Key rules:
- **Fish detected:** "Похоже на [вид] (~X% уверенности)" + ask for length for weight estimate
- **Weight:** Always a range, never exact. Ask for length/girth if not provided.
- **Species unknown:** "Вижу рыбу, но вид определить не могу — что за рыба?"
- **Not a fish:** "Рыбы не вижу" — be specific about what it looks like (lure, etc.)
- **Bad photo:** Ask to retake with better light

---

## 9. Species Recognition Limitations

- External model trained on specimen/museum photos — **domain mismatch with fishing photos**
- Expected accuracy on fishing photos: significantly below test-set accuracy
- Always use confidence ≥ 0.75 threshold for any species claim
- Use "possible species" / "похоже на" wording always
- Collect species corrections from friends during September trip

---

## 10. Weight Estimation Limitations

- **No length → no estimate** (ask user)
- **Length only → rough range ±40–60%** (species variation)
- **Length + girth → better range ±20%**
- **No single exact weight — ever**
- Formulas are approximate; verified biologically plausible but not scientifically calibrated
- Do not use for competition or regulatory purposes

---

## 11. How to Collect September Trip Feedback

The bot auto-saves feedback to `bot.db` when users:
1. Confirm species: reply with the fish name
2. Provide length: reply with "60 см" or similar
3. Correct a wrong guess: reply with the correct species
4. Rate a guess: react with 👍/👎

After the trip:
1. Export `bot.db` catch records
2. Review corrections as new labeled data
3. Re-run `intake_phase_e_lite.py` with any new reviewed batches
4. Use corrections to fine-tune species classifier

---

## 12. Recommended Next 3 Actions

1. **Download DeepFish no_fish frames** (~100 frames from the Absence subset)  
   `scripts/fetch_deepfish.py` — manual download step required  
   Goal: Get `not_fish_or_other` count above 122 to unblock structural training

2. **Review 10 more Telegram batches** (batches 0003–0012, ~2,500 records)  
   Focus on batches likely to contain lure/gear/out_of_scope content  
   Re-run `intake_phase_e_lite.py` + `build_mvp_training_dataset.py` after each session

3. **Wire weight estimator into bot photo handler**  
   See `docs/bot_mvp_integration_todo.md` — 30 min change, low risk  
   Test manually with a fish photo before the September trip
