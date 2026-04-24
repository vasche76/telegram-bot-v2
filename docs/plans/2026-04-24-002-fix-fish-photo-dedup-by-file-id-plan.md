---
title: "fix: Deduplicate fish photos by Telegram file_id before pipeline execution"
type: fix
status: active
date: 2026-04-24
origin: docs/plans/2026-04-24-001-fix-bot-security-hardening-plan.md
---

# fix: Deduplicate fish photos by Telegram file_id before pipeline execution

## Overview

This plan covers **U5 only** from the security hardening plan. Resending the same fish
photo currently creates multiple leaderboard entries. The fix adds an application-layer
pre-insert check in `bot/storage/catches.py` and a guard call in `bot/handlers/vision.py`,
mirroring the existing receipt deduplication pattern exactly.

---

## Problem Frame

`handle_fish_photo()` runs the full two-stage ML pipeline and calls `save_catch()` on
every photo message without checking whether that Telegram `file_id` was already processed.
A user who forwards or resends the same photo gets multiple leaderboard rows. The `catches`
table already stores `photo_file_id TEXT` for every row, so the data needed for dedup is
present — only the check is missing.

---

## Requirements Trace

- R5. A fish catch MUST NOT be saved if the same `photo_file_id` has already been recorded
  for the same `chat_id`.

---

## Scope Boundaries

- No `UNIQUE` constraint or migration on `catches.photo_file_id` — existing rows may already
  contain duplicates; schema changes would require a destructive dedup migration.
- No dedup check for rejected catches (lures, low-confidence, etc.) — only valid catches
  matter for leaderboard integrity; rejected rows can accumulate without harm.
- No rate-limiting or per-user quotas — separate concern.
- No changes to the ML pipeline stages (A or B).
- No changes to `save_catch()` signature — the guard is applied before calling it.

---

## Context & Research

### Relevant Code and Patterns

- `bot/storage/expenses.py:227-235` — `is_receipt_already_added(chat_id, session_id, photo_file_id)`:
  canonical dedup pattern. Single `SELECT id ... LIMIT 1`, returns `bool`.
- `bot/handlers/vision.py:363-366` — receipt dedup call site: check → skip `add_expense()`,
  append user-facing message. The fish handler must follow the same guard shape.
- `bot/handlers/vision.py:31-135` — `handle_fish_photo()`: current flow downloads the photo,
  calls `analyze_fish_photo()`, then `save_catch()`. Guard must be inserted before
  `analyze_fish_photo()` to avoid the expensive ML round-trip on duplicates.
- `bot/storage/catches.py:24-75` — `save_catch()`: stores `photo_file_id` in every row;
  no uniqueness check. Function signature unchanged by this plan.
- `bot/storage/database.py:226-239` — `catches` table DDL: `photo_file_id TEXT` column
  present, no `UNIQUE` constraint.
- `tests/test_suite.py:340-346` — `test_is_receipt_already_added_helper`: the exact test
  shape to mirror for fish catches.

### Institutional Learnings

- **Receipt dedup pattern** (project memory): use `file_id` as the canonical dedup key,
  call the boolean check BEFORE the insert operation, return `False` for `None` input so
  `None` is never treated as a valid dedup key.
- **PTB serial dispatch**: PTB dispatches updates serially per chat on a single asyncio
  event loop — TOCTOU risk is negligible at this scale.

### External References

- None required — the codebase contains the complete reference implementation.

---

## Key Technical Decisions

- **Pre-insert check only; no schema UNIQUE constraint.** Existing rows may already have
  duplicate `photo_file_id` values from before this fix. Adding a `UNIQUE` constraint
  requires a dedup migration (identify the row to keep per duplicate, delete the rest)
  before the constraint can be applied without `UNIQUE constraint failed` at startup.
  That is a separate, riskier migration not warranted by this fix. Application-layer
  guard is sufficient at single-process scale.

- **Scope the check to `(chat_id, photo_file_id)`, not globally.** Telegram `file_id`
  values are unique per file but not guaranteed to be globally stable across bots or
  API versions. Scoping to `chat_id` matches the leaderboard scope and is consistent
  with how `is_receipt_already_added` scopes to `(chat_id, session_id, photo_file_id)`.

- **Guard before `analyze_fish_photo()`, not before `save_catch()`.** Placing the check
  before the ML call avoids the full Stage A + Stage B GPT round-trip on a known
  duplicate. This is a latency and cost optimization, not just a correctness fix.

- **Guard returns `False` for `None` photo_file_id.** Telegram always provides a
  `file_id` for photo messages, but defensive code should not treat `None` as a valid
  dedup key (two `None` values would incorrectly match).

- **Do not save a row for duplicates.** Unlike the receipt handler (which appends a
  warning to an existing session response), the fish handler should return early with
  a user-facing message and no database write. A rejected duplicate row adds noise
  to the audit trail without value.

---

## Open Questions

### Resolved During Planning

- **Should `is_fish_photo_already_saved` include `user_id` in the key?** Resolved: No.
  Scope is `(chat_id, photo_file_id)`. Different users uploading the same photo to the
  same chat is the same dedup case — the photo was already recorded.
- **Should the check apply to rejected catches too?** Resolved: No — only valid catches
  matter for leaderboard integrity. A lure photo resent twice wastes a GPT call but
  produces no leaderboard harm. Keeping the check simple (any row with matching keys)
  is still correct: if the first submission was a valid catch, the duplicate is blocked;
  if the first was rejected, the duplicate is allowed to retry (useful if the user
  sends a cleaner photo).

### Deferred to Implementation

- **Whether to include rejected rows in the dedup scope.** The current plan blocks only
  on any existing row with matching `(chat_id, photo_file_id)`. Implementer may narrow
  to `AND is_valid_catch = 1` if "retry after rejection" behavior is desired. Either
  is defensible; the plan leaves this as a judgment call.

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not
> implementation specification. The implementing agent should treat it as context, not
> code to reproduce.*

**Dedup guard flow in `handle_fish_photo()`:**

```
photo message received
  → photo = msg.photo[-1]          (largest size)
  → is_fish_photo_already_saved(chat_id, photo.file_id)
        SELECT 1 FROM catches WHERE chat_id=? AND photo_file_id=? LIMIT 1
        returns True?
          → log.info("duplicate photo_file_id")
          → reply "Эта фотография уже была добавлена"
          → return   (no ML call, no DB write)
        returns False?
          → proceed to analyze_fish_photo() → save_catch()
```

---

## Implementation Units

- U5. **Add `is_fish_photo_already_saved` and guard call in fish photo handler**

**Goal:** Prevent duplicate leaderboard entries when the same Telegram photo is submitted
more than once to the same chat.

**Requirements:** R5

**Dependencies:** None

**Files:**
- Modify: `bot/storage/catches.py`
- Modify: `bot/handlers/vision.py`
- Test: `tests/test_suite.py`

**Approach:**
- Add `async def is_fish_photo_already_saved(chat_id: int, photo_file_id: str) -> bool`
  to `bot/storage/catches.py`, immediately below `save_catch()`. Query:
  `SELECT 1 FROM catches WHERE chat_id = ? AND photo_file_id = ? LIMIT 1`.
  Return `True` if a row exists, `False` if not. Guard: if `photo_file_id` is falsy
  (None or empty string), return `False` immediately without querying.
- In `bot/handlers/vision.py`, add `is_fish_photo_already_saved` to the import from
  `bot.storage.catches` (line 18).
- In `handle_fish_photo()`, insert the guard block after `photo = msg.photo[-1]` and
  before `file = await context.bot.get_file(photo.file_id)`. On duplicate: log at INFO
  with the truncated `file_id`, reply with a Russian user-facing message
  ("Эта фотография уже была добавлена" or similar), and `return`.
- Do NOT call `analyze_fish_photo()` or `save_catch()` for duplicates.

**Patterns to follow:**
- `bot/storage/expenses.py:227-235` — `is_receipt_already_added()`: the SELECT query,
  `fetch_one`, and `return row is not None` pattern.
- `bot/handlers/vision.py:363-366` — receipt guard call site: the check-then-early-return
  shape.

**Test scenarios:**
- Happy path: `is_fish_photo_already_saved(chat_id=555, photo_file_id="abc123")` before
  any save → returns `False`.
- Happy path: after `save_catch(chat_id=555, ..., photo_file_id="abc123")`, calling
  `is_fish_photo_already_saved(555, "abc123")` → returns `True`.
- Edge case: same `photo_file_id` in a different `chat_id` → returns `False` (per-chat
  scope; each chat has its own leaderboard).
- Edge case: `photo_file_id=None` → returns `False` without hitting the database.
- Edge case: `photo_file_id=""` (empty string) → returns `False` without hitting the
  database.
- Integration: `is_fish_photo_already_saved` returns `True` only after `save_catch` has
  committed — verifies the check reads from the same table that `save_catch` writes to.

**Verification:**
- `is_fish_photo_already_saved` unit tests pass (all 5 scenarios above).
- `handle_fish_photo()` called twice with the same `photo_file_id`: the `catches` table
  contains exactly one row for that `photo_file_id`/`chat_id` pair after both calls.
- The leaderboard for the chat shows the catch counted once, not twice.

---

## System-Wide Impact

- **Interaction graph:** Only `handle_fish_photo()` calls this guard. No other handler
  submits fish catches. The new function is additive to `catches.py` — existing callers
  (`get_chat_leaderboard`, `save_catch`, etc.) are untouched.
- **Error propagation:** If `is_fish_photo_already_saved()` raises (e.g., DB connection
  failure), the exception propagates to PTB's error handler in `main.py` — same behavior
  as any other storage failure in the handler. No special handling needed.
- **State lifecycle risks:** TOCTOU window between check and insert is negligible —
  PTB dispatches updates serially per chat on a single event loop. No concurrent writes
  to the same `(chat_id, photo_file_id)` pair within a single bot process.
- **Unchanged invariants:** `save_catch()` signature is unchanged. The fish vision
  pipeline (Stage A / Stage B) is unchanged. The leaderboard query is unchanged.
  Rejected-catch audit trail is unchanged (duplicate guard fires before the ML call,
  so no rejected-catch row is written for a duplicate submission).

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Existing duplicate rows in `catches` could cause the first legitimate resend (before this fix deploys) to be silently blocked | Acceptable — the duplicate was created by the bug being fixed. The second send of the same photo after deploy will be blocked correctly. |
| `photo_file_id` column could be `NULL` for old rows (pre-pipeline saves) | The guard returns `False` for `None` input, so old NULL rows are never matched and do not block new submissions with a real `file_id`. |
| User confusion when a valid photo is blocked as "already added" after a bot restart | Telegram `file_id` values are stable for the same file — this is by design. Document in user-facing message that each photo can only be submitted once per chat. |

---

## Documentation / Operational Notes

- **Rollback procedure:**
  1. In `bot/storage/catches.py`: remove the `is_fish_photo_already_saved` function
     (approximately 6 lines after `save_catch`).
  2. In `bot/handlers/vision.py`: remove the guard block (approximately 4 lines) and
     remove `is_fish_photo_already_saved` from the import on line 18.
  3. No schema change to revert.
  4. Restart launchd service: `launchctl kickstart -k gui/$(id -u)/com.chekulaev.telegrambot`

- **Commit boundary:** This unit ships as one atomic commit, per the parent plan's
  recommended grouping (U4 + U5 in one commit, or U5 alone if implemented separately):
  `fix(ml): deduplicate fish photos by file_id before pipeline (U5)`

---

## Sources & References

- **Origin document:** [docs/plans/2026-04-24-001-fix-bot-security-hardening-plan.md](docs/plans/2026-04-24-001-fix-bot-security-hardening-plan.md) (U5, R5)
- Related code: `bot/storage/catches.py`, `bot/handlers/vision.py:31-135`
- Canonical pattern: `bot/storage/expenses.py:227-235` — `is_receipt_already_added()`
- Test reference: `tests/test_suite.py:340-346` — `test_is_receipt_already_added_helper`
