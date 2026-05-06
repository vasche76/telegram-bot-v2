## Project

Telegram bot for fish recognition and weight estimation — **~10 fishing friends, September 2026 trip**.
Handle: `@Vassiliy_Chekulaev_bot`. Entry point: `main.py`.

**Owner:** Vassiliy Chekulaev (`vasche76@me.com`).

---

## Architecture

### Bot layers
| Layer | Location | Notes |
|---|---|---|
| Telegram handlers | `bot/handlers/` | `vision.py` = photo, `messages.py` = text |
| Fish vision pipeline | `bot/fish_vision/pipeline.py` | Two-stage: detector → classifier |
| AI services | `bot/services/ai.py` | GPT-4o-mini backend |
| Storage | `bot/storage/database.py` + `bot.db` | SQLite, catches/messages/expenses |
| Scheduler | `bot/scheduler.py` | Periodic reminders |
| Deployment | `deploy/com.vassiliy.telegrambot.plist` | macOS launchd |

### Fish vision (two-stage)
```
Photo → Stage A (detector/structural) → Stage B (species classifier) → weight estimator
```
- **Stage A** — currently GPT-4o-mini; local stub: `bot/fish_vision/local_detector.py`
- **Stage B** — GPT-4o-mini + `classifier_v1.pt` (EfficientNet-B0, 4 species: pike/grayling/perch/whitefish)
- **Weight estimator** — `bot/fish_vision/weight_estimator.py` (exists, **NOT wired into pipeline yet**)
- **GPT fallback must always be preserved** — never remove it

### ML model state (as of 2026-05-01)
| Model | File | Status |
|---|---|---|
| Structural binary (fish / not_fish) | `data/fish_models/mvp_structural_v1.pt` | **BLOCKED** — FP rate too high on Telegram negatives |
| Species classifier | `data/fish_models/classifier_v1.pt` | 70.45% test acc, 4 species, missing taimen |
| YOLO detector | `data/fish_models/detector_v1.pt` | Experimental |

**Structural model holdout result:** 90% overall accuracy on reviewed seed, but not_fish precision = 14% (29 FP at threshold 0.5). Root cause: training negatives were iNaturalist birds + Wikimedia lures, not real Telegram photos.
**Gate to unblock:** collect ≥200 Telegram-domain negatives, re-train, re-evaluate.

**Species threshold:** ≥ 0.75 confidence for any species claim. Always use "похоже на" / "possible species" wording.

---

## Python environments

| Env | Python | Use for |
|---|---|---|
| System / venv | **3.13** (`.python-version` = 3.12.10 via pyenv, but bot runs on system) | Bot runtime, intake scripts, tests |
| `venv_ml` | **3.12** (pyenv `~/.pyenv/versions/3.12.10`) | PyTorch training only |

```bash
# Run bot / scripts / tests
python3 ...

# ML training (PyTorch not available on 3.13)
~/.pyenv/versions/3.12.10/bin/python3 scripts/train_mvp_structural.py ...
# or activate venv_ml:
source venv_ml/bin/activate

# Required env var for MPS:
PYTORCH_ENABLE_MPS_FALLBACK=1
```

---

## Common commands

```bash
# Run tests
python3 -m pytest tests/ -q

# Run a specific test file
python3 -m pytest tests/test_fish_weight_estimator.py -v

# Start bot locally
python3 main.py

# Dataset balance check + training gates
python3 scripts/build_mvp_training_dataset.py --dry-run

# Materialize reviewed Telegram seed
python3 scripts/intake_phase_e_lite.py --dry-run
python3 scripts/intake_phase_e_lite.py

# Evaluate structural model on Telegram holdout
python3 scripts/evaluate_mvp_structural_telegram_holdout.py

# Select Telegram negative review candidates
python3 scripts/select_telegram_negative_review_candidates.py
```

---

## Nearest next actions (as of 2026-05-01)

1. **Wire weight estimator into bot** — `bot/fish_vision/pipeline.py` + `bot/handlers/vision.py`
   See `docs/bot_mvp_integration_todo.md` for exact code. ~30 min, low risk.

2. **Collect 200+ Telegram-domain negatives** (lures, landscape, gear, out-of-scope)
   Review batches `0003–0012` → re-run `intake_phase_e_lite.py` + `build_mvp_training_dataset.py`
   Goal: unblock structural model re-training.

3. **Download DeepFish no_fish frames** for not-fish balance
   See `scripts/fetch_deepfish.py`. Need +59 more not_fish to pass training gate.

See `docs/bot_mvp_integration_todo.md` and `docs/project_mvp_closeout_2026-05-01.md` for full detail.

---

## Key docs

| Doc | What it covers |
|---|---|
| `docs/project_mvp_closeout_2026-05-01.md` | Complete MVP status, all commands, limitations |
| `docs/bot_mvp_integration_todo.md` | Weight estimator wiring checklist with code snippets |
| `docs/fish_bot_mvp_policy.md` | How the bot should respond to friends (Russian) |
| `docs/ML_PIPELINE_STATUS.md` | Dataset inventory, training commands, source classification |
| `docs/species_v0_readiness_report.md` | Which species are ready for local model |
| `docs/ml/mvp_structural_telegram_holdout_summary.md` | Structural model evaluation results |
| `data/fish_models/mvp_structural_telegram_holdout_summary.json` | Raw holdout metrics |

---

## Safety rules

- **Do not remove GPT fallback** — bot must always be able to fall back to GPT-4o-mini.
- **Do not modify `deploy/com.vassiliy.telegrambot.plist`** or any launchd/service files without explicit intent.
- **Do not commit** `data/bot.db`, logs, model weights (`*.pt`), dataset images, run artifacts, or anything in `runs/`.
- **Do not auto-label** unreviewed Telegram photos with the model.
- **Never output exact weight** — always a range. Never claim species with < 0.75 confidence.
- Always provide rollback notes after risky changes.
- Always produce a technical summary: what changed, why, what was not changed, how validated, remaining risks.

---

## Compound Engineering workflow

Use CE when the task involves: architecture, ML pipeline, dataset ingestion/training, deployment/launchd, risky refactoring, pre-commit audit, or final technical summary.

**Skip CE** for: typo fixes, one-line config changes, simple explanations, trivial one-file edits with no downstream risk.

Preferred flow:
1. `/ce-code-review` — read-only audit
2. `/ce-brainstorm` — solution exploration
3. `/ce-plan` — implementation plan
4. `/ce-work` — execution
5. `/ce-code-review` — final validation
6. `/ce-compound` — preserve reusable lessons
