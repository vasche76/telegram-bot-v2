---
title: "fix: Harden Telegram bot against 7 high-risk audit findings"
type: fix
status: active
date: 2026-04-24
---

# fix: Harden Telegram bot against 7 high-risk audit findings

## Overview

Seven security and reliability issues were identified in a Compound Engineering audit of the
Telegram fishing bot. This plan fixes all seven: admin gate fails open, prompt injection via
captions, `is_valid_catch` set unconditionally after Stage B, unbounded `fish_count`, fish
photo deduplication, missing OpenAI 429 retry, and a machine-tied log path.

---

## Problem Frame

The bot runs as a launchd service on a personal Mac. Several of its internal safety assumptions
are violated in practice:

- The admin gate (`_is_admin`) returns `True` for all users when `ADMIN_USER_IDS` is unset —
  letting anyone trigger `os._exit(2)` via `/status` restart buttons.
- User-supplied photo captions flow directly into GPT prompts via bare `.replace()` and f-string
  interpolation, making prompt injection trivial.
- Stage B classifier errors produce a zero-confidence `ClassificationResult`; the pipeline then
  marks `is_valid = True` unconditionally and saves the catch to the leaderboard.
- `fish_count` returned by GPT (in both Stage A and Stage B) is accepted without an upper bound;
  a hallucinating model or injected caption can produce absurd leaderboard entries.
- Fish photos are never deduplicated by `file_id`; the same photo resent multiple times creates
  multiple leaderboard rows.
- OpenAI 429 errors are not retried; they propagate up and increment the `consecutive_logic_errors`
  counter, potentially triggering a restart after 15 such errors.
- The rotating Python log uses `os.getcwd()`-based path `data/bot.log` — implicitly tied to the
  launchd `WorkingDirectory` — rather than an explicit, configurable location.

---

## Requirements Trace

- R1. `_is_admin` MUST deny access when `ADMIN_USER_IDS` is empty (treat empty list as "no admins configured → deny all").
- R2. User-supplied caption text MUST be stripped of control characters, capped at a safe length, and never used in a way that allows it to override prompt structure.
- R3. `is_valid_catch` MUST be `False` when Stage B returns `confidence == 0.0` (error fallback). A genuine `unknown_fish` result with non-zero confidence MUST remain valid (the pipeline's documented intent).
- R4. `fish_count` MUST be capped at a configurable maximum (default 20) at the point where GPT output is parsed, before it enters the pipeline.
- R5. A fish catch MUST NOT be saved if the same `photo_file_id` has already been recorded for the same `chat_id`.
- R6. OpenAI 429 and transient network errors MUST be retried with exponential backoff (up to 3 attempts) before raising.
- R7. The Python rotating log path MUST be configurable via `LOG_FILE` env-var and MUST fall back to `/tmp/bot.log` (not `os.getcwd()`) when unset.

---

## Scope Boundaries

- No changes to launchd plist, deploy scripts, or ML training pipelines.
- No schema migration adding a `UNIQUE` constraint on `catches.photo_file_id` (existing rows may already have duplicates; dedup is enforced at application layer only, matching the receipts pattern).
- No changes to Stage A/B model logic, thresholds, or retraining.
- No rate-limiting on incoming Telegram messages (separate concern).
- No refactor of `bot/services/ai.py` beyond adding a retry wrapper (no SDK migration).

---

## Context & Research

### Relevant Code and Patterns

- `bot/handlers/status.py:45-49` — `_is_admin()`, current fail-open logic.
- `bot/config.py:42-45` — `ADMIN_USER_IDS` list comprehension (empty list when env-var unset).
- `bot/fish_vision/detector.py:168` — `_DETECTION_PROMPT.replace("{caption}", caption)` — bare substitution.
- `bot/fish_vision/classifier.py:268-273` — `_CLASSIFICATION_PROMPT.replace(...)` — both `{caption}` and `{detector_context}` unsanitized.
- `bot/handlers/vision.py:212-218` — face-register prompt f-string with raw caption.
- `bot/handlers/vision.py:307-325` — receipt OCR prompt f-string with raw caption.
- `bot/fish_vision/pipeline.py:179` — `is_valid = True` hardcoded unconditionally after Stage B.
- `bot/fish_vision/classifier.py:329` — `fish_count=int(data.get("fish_count", 1) or 1)` — no upper bound.
- `bot/fish_vision/detector.py:212` — `fish_count=int(data.get("fish_count", 0) or 0)` — no upper bound.
- `bot/fish_vision/pipeline.py:187` — `max(detection.fish_count, classification.fish_count, 1)` — no ceiling.
- `bot/handlers/vision.py:34-133` — `handle_fish_photo` — no pre-insert dedup check.
- `bot/storage/catches.py:24-75` — `save_catch()` — no uniqueness check.
- `bot/storage/expenses.py` — `is_receipt_already_added(photo_file_id)` — canonical dedup pattern to mirror.
- `bot/services/ai.py:50-68` — `chat_completion()` — 429 caught and re-raised without retry.
- `bot/utils/logging.py:55-59` — `os.getcwd()`-based log path.

### Institutional Learnings

- **Receipt dedup pattern** (from project memory): call `is_receipt_already_added(photo_file_id)` BEFORE `add_expense()`. Use Telegram `file_id` as the canonical dedup key. Apply identically for fish catches.
- **OpenAI empty-response guard** (REPORT.md, Fix 5): `choices` emptiness guard in `ai.py` is load-bearing — preserve it in any retry wrapper.
- **Separate error counters** (REPORT.md, Fix 2): 429 retry errors should not increment `consecutive_logic_errors`; transient network failures are a different budget. The retry wrapper must absorb retryable errors silently.
- **Logging from env** (REPORT.md, main.py row): `LOG_LEVEL` is already env-driven; `LOG_FILE` should follow the same pattern.

### External References

- None required — all patterns have direct codebase analogues.

---

## Key Technical Decisions

- **Admin gate default: deny, not allow.** Reversing the fail-open is the correct security posture. The `/status` command exposing `os._exit` restart buttons makes this a critical fix. The comment "No admins = everyone can use /status" represented a convenience choice, not a deliberate security decision. New behavior: empty `ADMIN_USER_IDS` → deny all (or restrict to the bot's own owner via `OWNER_USER_ID` if set).
- **Caption sanitization in a shared helper, not per-call-site.** Five injection points exist (detector, classifier, face-register, receipt, and the default photo analysis path at `vision.py:460`); a single `_sanitize_caption(text: str | None) -> str` helper avoids per-site divergence. Note: control-character stripping and length capping is the right defense for user-supplied captions, but it does NOT prevent GPT-relay injection via `detector_context` (Stage A GPT output). For that surface, structural isolation in the Stage B prompt template — placing `detector_context` inside an explicit data block (e.g., `«DETECTOR OUTPUT: ... »`) — is the correct remedy, not sanitization.
- **`is_valid_catch` gate: reject only the zero-confidence error fallback.** The Stage B error handler returns exactly `confidence=0.0, species_key="unknown_fish"`. The existing pipeline intent (documented at `pipeline.py:173-178`) is that genuine `unknown_fish` results ARE valid catches — the fish was confirmed but species is uncertain. The narrowest correct fix: `is_valid = confidence > 0.0`. This rejects error fallbacks while preserving genuine unknown-species catches. Using `SPECIES_CONFIDENCE_THRESHOLD` as the gate would incorrectly reject `unknown_fish` results that have moderate (but below-threshold) confidence — a behavior change that breaks the intended design. The constant `MIN_SPECIES_CONFIDENCE` referenced in the original audit does not exist in the codebase; the real constant is `SPECIES_CONFIDENCE_THRESHOLD` in `bot/fish_vision/classifier.py`, but it is NOT the right gate here.
- **`fish_count` cap at parse site, not pipeline.** Capping at the moment GPT JSON is parsed (detector and classifier) prevents bad values from ever entering the data structures — safer than a late ceiling in `pipeline.py`.
- **Fish dedup: application-layer pre-insert check, no schema change.** Mirrors `is_receipt_already_added()` exactly. A `UNIQUE` constraint migration would require handling existing duplicates in production data and is out of scope.
- **Retry wrapper in `ai.py`, not at call sites.** All OpenAI calls go through `chat_completion` — adding retry there covers every consumer. Three attempts with exponential backoff (1s, 2s, 4s) match common practice for 429 and transient `httpx` errors. Retry errors must NOT reach the logic-error counter.
- **`LOG_FILE` env-var with `/tmp/bot.log` fallback.** Keeps the path explicit, portable, and consistent with the launchd stdout/stderr paths already in `/tmp`.

---

## Open Questions

### Resolved During Planning

- **Should `_sanitize_caption` strip all non-ASCII or only control characters?** Resolved: strip control characters (`\x00`–`\x1f`, `\x7f`) and cap at 500 chars. Do not strip non-ASCII (breaks Russian captions).
- **What is the correct `is_valid_catch` gate for Stage B?** Resolved: `is_valid = classification.confidence > 0.0`. The constant `MIN_SPECIES_CONFIDENCE` does not exist in the codebase. `SPECIES_CONFIDENCE_THRESHOLD` (in `classifier.py`) is not the right gate here — genuine `unknown_fish` results can have moderate-but-below-threshold confidence and should remain valid. The zero-confidence test is the narrowest correct fix.
- **Should the retry wrapper use `asyncio.sleep` or a sync sleep?** Resolved: `asyncio.sleep` — the entire service layer is async.

### Deferred to Implementation

- **Exact retry-eligible HTTP status codes.** 429 is definite; whether to retry 500/502/503 from OpenAI is a judgment call for the implementer based on OpenAI's documented retry policy.
- **Whether `OWNER_USER_ID` (single-user fallback) is worth adding to `bot/config.py` for solo deployments.** Implementer can add if the guard-only change feels too restrictive for personal use — but the correct fix is to document `ADMIN_USER_IDS` in `.env.example`.

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

**Caption sanitization flow:**

```
user caption (Telegram)
  → _sanitize_caption()           [new helper in bot/utils/text.py]
       strip \x00–\x1f, \x7f
       cap at 500 chars
       strip leading/trailing whitespace
  → prompt template substitution  [detector.py, classifier.py, vision.py]
```

**`is_valid_catch` decision gate:**

```
Stage B returns ClassificationResult
  confidence > 0.0?
    YES → is_valid = True   (genuine result: identified species OR unknown_fish)
    NO  → is_valid = False  (error fallback: unknown_fish + confidence=0.0)
```

**OpenAI retry wrapper:**

```
chat_completion(...)
  attempt 1
    httpx.HTTPStatusError 429 OR httpx.TransportError?
      wait 2^(attempt-1) seconds, retry (max 3 attempts)
    other error → raise immediately
  all retries exhausted → raise last error
```

---

## Implementation Units

- U1. **Fix admin gate fail-open**

**Goal:** `_is_admin()` must return `False` (not `True`) when `ADMIN_USER_IDS` is empty.

**Requirements:** R1

**Dependencies:** None

**Files:**
- Modify: `bot/handlers/status.py`
- Modify: `bot/config.py` (add `.env.example` note if needed)
- Test: `tests/test_suite.py`

**Approach:**
- Invert the guard: when `ADMIN_USER_IDS` is empty, return `False` (deny all) rather than `True`.
- Update the docstring/comment to reflect the new intent.
- Optionally add `OWNER_USER_ID: int | None` to config as a convenience for solo deployments — if set and non-zero, treat it as the sole allowed admin. This is purely additive; the security fix does not require it.

**Patterns to follow:**
- `bot/config.py` env-var loading patterns (int cast, default None).

**Test scenarios:**
- Happy path: `ADMIN_USER_IDS = [123]`, call `_is_admin(123)` → `True`.
- Happy path: `ADMIN_USER_IDS = [123]`, call `_is_admin(456)` → `False`.
- Edge case: `ADMIN_USER_IDS = []` (unset env-var), call `_is_admin(any_id)` → `False`.
- Edge case: `ADMIN_USER_IDS = [123]`, call with non-integer-coercible string in env — must not raise at import time (already handled by the list comprehension filter).

**Verification:**
- With `ADMIN_USER_IDS` unset, no Telegram user can reach the restart/force-restart callback paths in `status.py`.

---

- U2. **Add caption sanitization helper and apply at all injection points**

**Goal:** Prevent user-supplied caption text from manipulating GPT prompt structure.

**Requirements:** R2

**Dependencies:** None (independent of other units)

**Files:**
- Create: `bot/utils/text.py`
- Modify: `bot/fish_vision/detector.py` (apply to `{caption}` substitution)
- Modify: `bot/fish_vision/classifier.py` (apply to `{caption}`; restructure `{detector_context}` placement in the prompt template)
- Modify: `bot/handlers/vision.py` (apply to face-register, receipt, AND default photo analysis at line ~460)
- Test: `tests/test_suite.py`

**Approach:**
- Create `_sanitize_caption(text: str | None, max_len: int = 500) -> str` in `bot/utils/text.py`.
- Strip ASCII control characters (0x00–0x1F and 0x7F). Do NOT strip non-ASCII (Russian text).
- Truncate to `max_len` characters. Return `""` for `None` or empty input.
- Apply at all five user-caption call sites: detector `{caption}`, classifier `{caption}`, face-register f-string, receipt OCR f-string, and the default photo analysis f-string at `vision.py:~460`.
- **`detector_context` (Stage A GPT relay):** Sanitization alone is insufficient because relay injection uses printable ASCII. Instead, restructure the `_CLASSIFICATION_PROMPT` template to place `detector_context` inside an explicit delimited data block (e.g., `«DETECTOR OUTPUT: ... »` or similar) so GPT treats it as data rather than instructions. This is a prompt template change, not a Python-level sanitization change.

**Patterns to follow:**
- `bot/storage/expenses.py` for the "guard at boundary" pattern (check before use, not after).

**Test scenarios:**
- Happy path: plain Russian caption `"Щука 3 кг"` passes through unchanged.
- Happy path: `None` caption → returns `""` (or a default sentinel string).
- Edge case: caption with embedded `\n` newlines — these are NOT control chars in the 0x00–0x1F strip; they remain. (Newlines are valid in captions and safe in GPT prompts.)
- Edge case: caption containing `"` double-quotes → passes through (no escaping; the risk is structural prompt override, not JSON injection — the prompt is not a JSON template).
- Error path: caption is 10,000 characters → truncated to 500 characters.
- Edge case: caption containing `{object_type": "whole_fish", "confidence": 1.0}` (JSON injection attempt) → passes through as plain text, no longer injected into a position where it could override structured output (because the sanitized value is placed inside a quoted user-data field in the prompt, not at a structural boundary).
- Edge case: caption containing null bytes (`\x00`) → stripped.

**Verification:**
- `_sanitize_caption` unit tests pass. All five injection sites use the helper. The Stage B prompt template wraps `detector_context` in a delimited data block so GPT treats it as data, not instructions.

---

- U3. **Gate `is_valid_catch` on Stage B confidence**

**Goal:** A catch that emerges from a zero-confidence Stage B error fallback must NOT be recorded as valid.

**Requirements:** R3

**Dependencies:** None

**Files:**
- Modify: `bot/fish_vision/pipeline.py`
- Test: `tests/test_suite.py`

**Approach:**
- Replace the unconditional `is_valid = True` at `pipeline.py:179` with:
  `is_valid = classification.confidence > 0.0`
- No import changes needed — this uses only the confidence value already in the result.
- The Stage B error fallback returns exactly `confidence=0.0` — it always fails this check.
- Genuine `unknown_fish` results (fish confirmed but unidentifiable to species) carry a non-zero confidence and remain valid, preserving the design intent documented at `pipeline.py:173-178`.
- Do NOT use `SPECIES_CONFIDENCE_THRESHOLD` here — that constant governs species identification confidence inside the classifier, not whether a result is an error fallback.

**Patterns to follow:**
- Existing Stage A confidence gate at `pipeline.py:120-164` (already checks `detection.confidence >= MIN_DETECTION_CONFIDENCE`).

**Test scenarios:**
- Happy path: `ClassificationResult(confidence=0.85, species_key="pike")` → `is_valid_catch=True`.
- Happy path: `ClassificationResult(confidence=0.55, species_key="unknown_fish")` (genuine unknown, moderate confidence) → `is_valid_catch=True` (preserves design intent).
- Happy path: `ClassificationResult(confidence=0.0, species_key="unknown_fish")` (error fallback) → `is_valid_catch=False`.
- Edge case: `ClassificationResult(confidence=0.01, ...)` (non-zero, just above error floor) → `is_valid_catch=True`.
- Integration: confirmed that a catch with `is_valid_catch=False` is NOT counted in `get_chat_leaderboard` (already filtered by `WHERE is_valid_catch = 1`).

**Verification:**
- Stage B error fallback no longer produces leaderboard entries. Existing Stage A rejection logic is unchanged.

---

- U4. **Cap `fish_count` at parse time**

**Goal:** `fish_count` from GPT is bounded at a configurable maximum before entering any data structure.

**Requirements:** R4

**Dependencies:** None

**Files:**
- Modify: `bot/fish_vision/detector.py`
- Modify: `bot/fish_vision/classifier.py`
- Modify: `bot/config.py` (add `MAX_FISH_COUNT` constant, default 20)
- Test: `tests/test_suite.py`

**Approach:**
- Add `MAX_FISH_COUNT: int = int(os.environ.get("MAX_FISH_COUNT", "20"))` to `bot/config.py`.
- In `detector.py` at the `fish_count` parse line, apply `min(..., MAX_FISH_COUNT)`.
- In `classifier.py` at the `fish_count` parse line, apply `min(..., MAX_FISH_COUNT)`.
- The `max()` in `pipeline.py:187` already selects the larger of the two — no additional cap needed there because both inputs are already capped.

**Patterns to follow:**
- `bot/handlers/photos.py:54-55` — `min(max(1, int(count)), 10)` — same clamp-at-parse pattern.

**Test scenarios:**
- Happy path: GPT returns `fish_count=3` → stored as 3.
- Edge case: GPT returns `fish_count=9999` → capped to `MAX_FISH_COUNT` (20).
- Edge case: GPT returns `fish_count=0` → kept as 0 (or 1 minimum if that is enforced; check existing `or 1` default).
- Edge case: GPT omits `fish_count` field → falls back to default (1 in classifier, 0 in detector).
- Edge case: `MAX_FISH_COUNT` env-var set to `"5"` → cap is 5.

**Verification:**
- No `fish_count` value > `MAX_FISH_COUNT` can reach `save_catch()`.

---

- U5. **Deduplicate fish photos by `file_id` before saving**

**Goal:** Resending the same photo does not create duplicate leaderboard entries.

**Requirements:** R5

**Dependencies:** None (but U3 should be applied first to avoid a dedup check on a catch that will be rejected anyway — order by dependency matters for test clarity, not correctness)

**Files:**
- Modify: `bot/storage/catches.py` (add `is_fish_photo_already_saved()`)
- Modify: `bot/handlers/vision.py` (call check before `save_catch()`)
- Test: `tests/test_suite.py`

**Approach:**
- Add `async def is_fish_photo_already_saved(chat_id: int, photo_file_id: str) -> bool` to `bot/storage/catches.py`.
  - Query: `SELECT 1 FROM catches WHERE chat_id = ? AND photo_file_id = ? LIMIT 1`.
  - Return `True` if a row exists, `False` otherwise.
- In `handle_fish_photo` (`bot/handlers/vision.py`), call `is_fish_photo_already_saved()` before calling `analyze_fish_photo()` (skip expensive ML pipeline on duplicate).
- On duplicate detected: log at INFO level and send a brief user-facing message ("Эта фотография уже была добавлена" or similar).

**Patterns to follow:**
- `bot/storage/expenses.py` — `is_receipt_already_added(photo_file_id)` — identical pattern: pre-insert boolean check, `file_id` as key.
- `bot/handlers/expenses.py` — the call site pattern: check → early return if duplicate.

**Test scenarios:**
- Happy path: first submission with new `file_id` → `is_fish_photo_already_saved` returns `False`, proceeds to pipeline.
- Error path (dedup): second submission with same `file_id` in same `chat_id` → returns `True`, handler returns early without calling `analyze_fish_photo`.
- Edge case: same `file_id` in different `chat_id` → NOT considered duplicate (per-chat scope). `is_fish_photo_already_saved` must include `chat_id` in the query.
- Edge case: `photo_file_id = None` (photo submitted without tracked ID) → `is_fish_photo_already_saved` should return `False` (do not treat `None` as a dedup key).

**Verification:**
- Resending the same photo produces one leaderboard entry regardless of how many times it is sent.

---

- U6. **Add OpenAI 429 / transient error retry in `chat_completion`**

**Goal:** Transient OpenAI errors (429, network blips) are retried with backoff rather than propagating to the logic-error counter.

**Requirements:** R6

**Dependencies:** None

**Files:**
- Modify: `bot/services/ai.py`
- Test: `tests/test_suite.py`

**Approach:**
- Wrap the `client.post(...)` call in a retry loop: up to 3 attempts, backoff of `2^(attempt-1)` seconds (0s, 1s, 2s before attempts 1, 2, 3).
- Retry on: `httpx.HTTPStatusError` with `status_code == 429`, `httpx.TimeoutException`, `httpx.ConnectError`, `httpx.RemoteProtocolError`.
- Do NOT retry on: `httpx.HTTPStatusError` with other status codes (e.g., 400 Bad Request, 401 Unauthorized) — these are non-transient.
- On final failure (all retries exhausted), re-raise the last exception as before.
- Preserve the existing `choices` emptiness guard (load-bearing per institutional learning).
- Use `asyncio.sleep` (not `time.sleep`) since `chat_completion` is async.
- Log at WARNING level on each retry attempt with the status code and attempt number.

**Patterns to follow:**
- Existing `except httpx.HTTPStatusError` block in `ai.py:63-65`.
- `main.py` `RetryAfter` handler for Telegram polling (structural analogue, not code to copy).

**Test scenarios:**
- Happy path: first attempt succeeds → no retry, response returned normally.
- Error path: first attempt returns 429, second attempt succeeds → warning logged, response returned.
- Error path: all 3 attempts return 429 → exception raised after third attempt (caller's existing error handling takes over).
- Error path: `httpx.TimeoutException` on first attempt, second succeeds → retried.
- Error path: `httpx.HTTPStatusError` 401 (auth error) → raised immediately, no retry.
- Edge case: verify `asyncio.sleep` delay increases: attempt 1 = 0s, attempt 2 = 1s, attempt 3 = 2s.

**Verification:**
- For the fish pipeline path (Stage A / Stage B): a 429 that exhausts retries is caught by the local `except Exception` handlers in detector/classifier and returned as a safe fallback — it does NOT reach `main.py`'s `consecutive_logic_errors` counter.
- For the text chat path (`generate_response`): a 429 that exhausts retries still propagates to the PTB error handler and WILL increment `consecutive_logic_errors`. This is an accepted limitation at 3 retries — the counter has a budget of 15, so isolated bursts will not cause a restart.

---

- U7. **Make Python log path configurable, default to `/tmp/bot.log`**

**Goal:** The rotating log file path is explicit, portable, and does not depend on `os.getcwd()`.

**Requirements:** R7

**Dependencies:** None

**Files:**
- Modify: `bot/utils/logging.py`
- Modify: `bot/config.py` (add `LOG_FILE` constant)

**Approach:**
- Add `LOG_FILE: str = os.environ.get("LOG_FILE", "/tmp/bot.log")` to `bot/config.py`.
- In `bot/utils/logging.py`, replace the `os.getcwd()`-based path construction with `LOG_FILE` from config.
- Remove the `os.makedirs(log_dir, ...)` call for a non-`/tmp` directory (the `/tmp` parent always exists; if `LOG_FILE` is set to a custom path, the parent directory must pre-exist — document this in `.env.example`).
- Keep the `except Exception` guard that silently continues if the file handler cannot be created (allows the bot to run without logging on read-only filesystems).

**Patterns to follow:**
- `bot/config.py` env-var loading (string with default).
- Existing `LOG_LEVEL` env-var pattern.

**Test scenarios:**
- Test expectation: none — this is a configuration-only change with no behavioral logic to unit-test. Manual verification: after the change, `bot.log` appears at `/tmp/bot.log` (or `LOG_FILE` value) rather than `data/bot.log`.

**Verification:**
- On a fresh launchd restart, `data/bot.log` is no longer created. `/tmp/bot.log` receives rotating log output. `LOG_FILE=/some/path` env-var redirects accordingly.

---

## System-Wide Impact

- **Interaction graph:** U6 (retry wrapper) is called by every consumer of `chat_completion` — Stage A detector, Stage B classifier, `generate_response`, `structured_extraction`, `vision_structured`. All benefit; none are broken by the wrapper since the interface is unchanged.
- **Error propagation:** After U6, a 429 that exhausts all retries still propagates as an exception. Stage A's `except Exception` → returns `no_fish` result (safe). Stage B's `except Exception` → returns `unknown_fish` at `confidence=0.0` (safe after U3). `generate_response` has no local catch → reaches PTB error handler in `main.py`.
- **State lifecycle risks:** U5 adds a pre-insert read before write. This introduces a TOCTOU window (two concurrent sends of the same photo could both pass the check). At the bot's single-process, single-event-loop scale, this is not a practical concern — PTB dispatches updates serially per chat.
- **API surface parity:** `is_fish_photo_already_saved` (U5) should be exported from `bot/storage/catches.py` consistently with `is_receipt_already_added` in `expenses.py`.
- **Integration coverage:** U3's `is_valid_catch` change must be validated end-to-end: a zero-confidence Stage B result must produce a row with `is_valid_catch=0` in the `catches` table, confirmed absent from `get_chat_leaderboard`.
- **Unchanged invariants:** GPT fallback path in Stage B is NOT removed (safety rule from `CLAUDE.md`). The `choices` emptiness guard in `ai.py` is preserved. The `consecutive_logic_errors` / `consecutive_network_errors` counter split from Fix 2 is preserved.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| U1: Reversing the admin gate could lock out the bot owner if `ADMIN_USER_IDS` is not set in `.env` | Document `ADMIN_USER_IDS` clearly in `.env.example`. Consider logging a startup WARNING if the env-var is empty so the owner is alerted. |
| U3: Setting `is_valid_catch=False` for low-confidence catches may reject catches that were previously accepted | This is intentional and correct behavior. The fix eliminates a bug. Monitor `catches` table after deploy for unexpected `is_valid_catch=0` entries. |
| U6: Retry wrapper adds latency on 429 errors (up to 0+1+2=3 seconds overhead for 3 failures) | Acceptable — without the retry the error propagates immediately anyway. The Stage A/B pipeline already has a 60s httpx timeout. |
| U7: Moving log from `data/bot.log` to `/tmp/bot.log` means historical logs at the old path stop updating silently | Acceptable. The new path is consistent with launchd stdout/stderr in `/tmp`. |
| Schema: no UNIQUE constraint on `catches.photo_file_id` — application-layer dedup is the only guard | Acceptable at this scale. If the SQLite file is ever manipulated directly, duplicates could re-enter. Document as a known limitation. |

---

## Documentation / Operational Notes

- **`.env.example`** — add `ADMIN_USER_IDS=` (empty, with comment explaining fill-in), `MAX_FISH_COUNT=20`, `LOG_FILE=/tmp/bot.log`.
- **Rollback notes:**
  - U1: revert `_is_admin` change; set `ADMIN_USER_IDS=<owner_id>` in `.env` before re-enabling.
  - U3: `is_valid = True` was the prior behavior; revert one line in `pipeline.py`.
  - U6: remove the retry loop; restore `await client.post(...)` call directly.
  - U7: revert `bot/utils/logging.py` to `os.getcwd()`-based path.
- **Recommended commit grouping:**
  1. `fix(security): admin gate denies when ADMIN_USER_IDS is empty (U1)`
  2. `fix(security): sanitize captions before GPT prompt substitution (U2)`
  3. `fix(ml): gate is_valid_catch on Stage B confidence threshold (U3)`
  4. `fix(ml): cap fish_count at MAX_FISH_COUNT (U4), deduplicate fish photos by file_id (U5)`
  5. `fix(reliability): retry OpenAI 429 and transient httpx errors with backoff (U6)`
  6. `fix(ops): make log file path configurable via LOG_FILE env-var (U7)`

---

## Sources & References

- Related code: `bot/handlers/status.py`, `bot/fish_vision/pipeline.py`, `bot/services/ai.py`, `bot/utils/logging.py`, `bot/storage/catches.py`
- Institutional: `REPORT.md` (sections 5, 9), project memory `project_fixes_2026_04_11.md`
- Receipt dedup pattern: `bot/storage/expenses.py` — `is_receipt_already_added()`
