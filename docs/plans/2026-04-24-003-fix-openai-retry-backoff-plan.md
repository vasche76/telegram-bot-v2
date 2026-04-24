---
title: "fix: Add OpenAI 429 / transient error retry with exponential backoff"
type: fix
status: active
date: 2026-04-24
origin: docs/plans/2026-04-24-001-fix-bot-security-hardening-plan.md
---

# fix: Add OpenAI 429 / transient error retry with exponential backoff

## Overview

This plan implements U6 from the security hardening audit. `chat_completion()` in
`bot/services/ai.py` currently re-raises OpenAI 429 errors immediately. These errors
propagate to callers and — for the text chat path — eventually increment
`consecutive_logic_errors`, risking a watchdog restart after 15 such errors.

The fix wraps the HTTP call in a retry loop (≤ 3 attempts, exponential backoff) that
absorbs transient failures before they escape `chat_completion`. All existing callers
are covered with no interface changes.

---

## Problem Frame

`chat_completion` at `bot/services/ai.py:50-68` makes a single `await client.post()`
with no retry. On `httpx.HTTPStatusError` 429 or network-level failures it logs and
re-raises immediately. Callers handle this differently:

- **Stage A detector** (`detector.py:192`) — `except Exception` → returns `no_fish`
  fallback (safe, but wastes the user's photo submission).
- **Stage B classifier** (`classifier.py:297`) — `except Exception` → returns
  `unknown_fish, confidence=0.0` (safe after U3, but the catch is silently rejected).
- **`generate_response`** — no local handler → reaches PTB error handler in `main.py`
  → increments `consecutive_logic_errors`. Fifteen such errors trigger a restart.

A retry in `chat_completion` absorbs all of these before they escape, covering every
consumer uniformly without touching any caller code.

---

## Requirements Trace

- R6. OpenAI 429 and transient network errors MUST be retried with exponential backoff
  (up to 3 attempts) before raising. *(see origin: docs/plans/2026-04-24-001-fix-bot-security-hardening-plan.md)*

---

## Scope Boundaries

- No changes to `detector.py`, `classifier.py`, `handlers/`, or `main.py`.
- No retry for `transcribe_audio` — it is not routed through `chat_completion` and has
  its own 120s timeout; Whisper 429s are rare and out of scope.
- No retry for HTTP 500/502/503 from OpenAI — those are server errors that may indicate
  a content policy decision or model overload, not transient blips. The implementer may
  choose to add them; the plan does not require it.
- No changes to the `consecutive_logic_errors` / `consecutive_network_errors` counter
  logic in `main.py` — counters are unaffected because the retry absorbs errors inside
  `chat_completion` before they propagate.

---

## Context & Research

### Relevant Code and Patterns

- `bot/services/ai.py:32-68` — `chat_completion()`: the sole function to change.
  - `ai.py:51` — `resp = await client.post("/chat/completions", json=body)` — retry target.
  - `ai.py:52` — `resp.raise_for_status()` — raises `httpx.HTTPStatusError` on 4xx/5xx.
  - `ai.py:54-61` — `choices` emptiness guard — load-bearing, must be preserved inside
    the retry loop (checked after each successful response, not just the first).
  - `ai.py:63-65` — existing `httpx.HTTPStatusError` catch — will be replaced by the
    retry-aware handler.
  - `ai.py:66-68` — existing `Exception` catch — will be replaced by the retry-aware
    handler.
- `main.py:92-93` — `consecutive_network_errors`, `consecutive_logic_errors` counters.
- `main.py:116-120` — PTB error handler: `NetworkError`/`TimedOut` → network counter;
  everything else → logic counter. A 429 that escapes `chat_completion` and reaches a
  PTB update handler would increment `consecutive_logic_errors`.
- `bot/handlers/messages.py:245` — `except (NetworkError, TimedOut, RetryAfter)` —
  Telegram-level retry analogue (structural reference, not code to copy).

### Institutional Learnings

- **Preserve `choices` emptiness guard** (REPORT.md Fix 5): the guard at `ai.py:54-57`
  is load-bearing. It must remain inside the retry loop, applied to every successful
  response, not moved outside.
- **Separate error counters** (REPORT.md Fix 2): the retry wrapper must absorb retryable
  errors silently — they must not reach `main.py`'s counters.
- **Service is fully async**: use `asyncio.sleep`, not `time.sleep`.

### External References

- None required — local patterns are sufficient.

---

## Key Technical Decisions

- **Retry in `chat_completion`, not at call sites.** Every OpenAI consumer goes through
  this one function. Adding retry here covers Stage A, Stage B, `generate_response`,
  `structured_extraction`, `vision_analyze`, and `vision_structured` with a single change.
- **Three attempts, backoff `2^(attempt-1)` seconds (0 s, 1 s, 2 s before attempts 2 and 3).**
  Total worst-case latency overhead: 3 s. The `httpx.AsyncClient` already has a 60 s
  timeout, so three retries fit comfortably.
- **Retry-eligible exceptions: `httpx.HTTPStatusError` 429 only (not 5xx), plus
  `httpx.TimeoutException`, `httpx.ConnectError`, `httpx.RemoteProtocolError`.**
  These are the transient error classes observable on this client. 400/401/403/5xx are
  non-transient or ambiguous and are raised immediately.
- **Log at WARNING on each retry attempt** with status code (or exception type) and
  attempt number. Log at ERROR only on final failure (existing behavior).
- **Do not add a `503` retry path.** OpenAI 503 can indicate content policy decisions or
  sustained outages. Retrying 503 blindly could mask quota exhaustion signals. This
  decision is revisable at the call-site level without touching this plan.

---

## Open Questions

### Resolved During Planning

- **`asyncio.sleep` or `time.sleep`?** Resolved: `asyncio.sleep` — `chat_completion`
  is `async def` and runs inside the PTB async event loop.
- **Should the `choices` guard move outside the loop?** Resolved: No. It must stay inside
  the retry loop because each attempt returns its own response; the guard validates the
  result of the specific attempt that succeeded.
- **What happens when all retries are exhausted?** Resolved: re-raise the last exception.
  Callers' existing error handling (Stage A/B `except Exception` fallbacks, PTB error
  handler) is unaffected — they see the same exception type they would have seen before,
  just delayed by up to 3 s.

### Deferred to Implementation

- **Whether to also retry HTTP 503 from OpenAI.** The plan excludes 503 by default.
  The implementer can add `status_code in (429, 503)` if production observation warrants
  it — the change is one integer comparison.

---

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not
> implementation specification. The implementing agent should treat it as context, not
> code to reproduce.*

```
chat_completion(messages, model, ...)
  for attempt in 1..MAX_ATTEMPTS (3):
    try:
      resp = await client.post("/chat/completions", json=body)
      resp.raise_for_status()
      validate choices guard
      return content
    except (httpx.HTTPStatusError where status == 429
            OR httpx.TimeoutException
            OR httpx.ConnectError
            OR httpx.RemoteProtocolError) as retryable:
      if attempt < MAX_ATTEMPTS:
        log WARNING "Retrying OpenAI (attempt {attempt}/{MAX_ATTEMPTS}): ..."
        await asyncio.sleep(2 ** (attempt - 1))   # 1s, 2s before attempts 2, 3
        continue
      else:
        log ERROR (existing)
        raise retryable
    except httpx.HTTPStatusError (other codes):
      log ERROR (existing)
      raise immediately          # 400, 401, 403, 5xx — no retry
    except Exception:
      log ERROR (existing)
      raise immediately
```

Callers after the change:

```
Stage A detector.py:192  except Exception → no_fish fallback (unchanged)
Stage B classifier.py:297  except Exception → unknown_fish / confidence=0.0 (unchanged)
generate_response → PTB error handler → consecutive_logic_errors++ (unchanged path,
    but 429 bursts now reach it far less often)
```

---

## Implementation Units

- U1. **Wrap `client.post` in retry loop inside `chat_completion`**

**Goal:** `chat_completion` retries transient OpenAI errors up to 3 times with
exponential backoff before raising.

**Requirements:** R6

**Dependencies:** None

**Files:**
- Modify: `bot/services/ai.py`
- Test: `tests/test_suite.py`

**Approach:**
- Add a module-level constant `_MAX_RETRIES = 3` near the top of `ai.py` (after imports,
  before `_client`).
- Replace the existing `try/except` block in `chat_completion` (lines 50-68) with a
  `for attempt in range(1, _MAX_RETRIES + 1)` loop.
- Inside the loop: `await client.post(...)`, `raise_for_status()`, `choices` guard,
  return `content` — exactly as today when the call succeeds.
- On `httpx.HTTPStatusError` with `status_code == 429`: if `attempt < _MAX_RETRIES`,
  log WARNING and `await asyncio.sleep(2 ** (attempt - 1))`; else log ERROR and re-raise.
- On `httpx.TimeoutException`, `httpx.ConnectError`, `httpx.RemoteProtocolError`:
  same pattern as 429 above.
- On `httpx.HTTPStatusError` with any other status code: log ERROR and re-raise
  immediately (no retry). This preserves the existing error log format.
- On any other `Exception`: log ERROR and re-raise immediately (existing behavior).
- `asyncio` is already importable; add `import asyncio` at the top of `ai.py` if not
  already present (check first — it may be implicitly available via other imports but
  is not currently an explicit import).

**Patterns to follow:**
- Existing `except httpx.HTTPStatusError` block at `ai.py:63-65` for error log format.
- `main.py` watchdog loop structure (retry-with-sleep pattern; not code to copy, just
  structural analogy).

**Test scenarios:**
- Happy path: mock `client.post` succeeds on first call → response returned, no sleep,
  `asyncio.sleep` not called.
- Retry on 429 (second attempt succeeds): mock returns 429 on first call, 200 on second
  → WARNING logged once, `asyncio.sleep(1)` called once, response returned.
- All 3 attempts fail 429: mock always returns 429 → WARNING logged twice (attempts 1, 2),
  exception raised after third attempt; total `asyncio.sleep` calls: `sleep(1)` +
  `sleep(2)`.
- Retry on `httpx.TimeoutException`: first call raises `TimeoutException`, second
  succeeds → retried, response returned.
- No retry on 401: mock returns 401 → `httpx.HTTPStatusError` raised immediately,
  `asyncio.sleep` not called.
- No retry on 500: mock returns 500 → raised immediately.
- `choices` guard preserved: mock returns 200 with `{"choices": []}` (empty list) →
  `ValueError("OpenAI returned empty choices list")` raised, not retried (it is an
  `Exception`, not a retryable httpx type — confirmed by the except chain).
- Edge case: all 3 attempts fail with `httpx.ConnectError` → raised after third attempt;
  `asyncio.sleep` called with 1 and 2.

**Verification:**
- `pytest tests/test_suite.py -v -k "openai_retry"` passes (or equivalent test class name).
- In the fish pipeline path (Stage A / Stage B), a mocked 429-then-success scenario
  completes without raising — the caller's `except Exception` fallback is NOT triggered.
- `asyncio` is explicitly imported in `ai.py`.

---

## System-Wide Impact

- **Interaction graph:** Only `chat_completion` changes. `vision_analyze`,
  `vision_structured`, `structured_extraction` all delegate to it and benefit
  automatically. `transcribe_audio` does NOT go through `chat_completion` and is
  unaffected.
- **Error propagation:** After retries are exhausted, the exception propagates exactly as
  before. Stage A/B `except Exception` handlers absorb it. `generate_response` lets it
  reach the PTB error handler. No change in propagation topology.
- **Latency:** Worst case adds 3 s of sleep (1 + 2) per call when all 3 attempts fail.
  Stage A/B have 60 s httpx timeout budget; 3 s overhead is acceptable.
- **Unchanged invariants:** `choices` emptiness guard preserved. GPT Stage B fallback
  path NOT removed. `consecutive_logic_errors` / `consecutive_network_errors` split
  preserved — the retry wrapper absorbs errors before they reach `main.py`'s counters.

---

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Retry adds up to 3 s latency per call on sustained 429 bursts | Acceptable — the Stage A/B pipeline has a 60 s timeout and the user waits regardless |
| Test mocking of `asyncio.sleep` must not break the event loop | Use `unittest.mock.patch("asyncio.sleep", new=AsyncMock(return_value=None))` scoped to the test; verify the real event loop is unaffected |
| A 429 that exhausts all retries still reaches `consecutive_logic_errors` for the text chat path | Accepted limitation — 3 retries drastically reduce frequency; the budget of 15 errors provides sufficient buffer |

---

## Documentation / Operational Notes

- **Rollback:** Remove the `for attempt in range(...)` loop from `chat_completion`;
  restore the original flat `try/except` block (lines 50-68 before this change).
  No config changes needed — `_MAX_RETRIES` is a code constant, not an env-var.
- **Commit boundary:** Single commit covering `bot/services/ai.py` + new tests:
  `fix(reliability): retry OpenAI 429 and transient httpx errors with backoff (U6)`
- **No env-var added.** `_MAX_RETRIES = 3` is intentionally a code constant. The
  retry policy is a reliability decision, not a deployment-time parameter.

---

## Sources & References

- **Origin document:** [docs/plans/2026-04-24-001-fix-bot-security-hardening-plan.md](docs/plans/2026-04-24-001-fix-bot-security-hardening-plan.md) — U6 section
- Related code: `bot/services/ai.py:32-68` — `chat_completion`
- Related code: `bot/fish_vision/detector.py:183-202` — Stage A exception handler
- Related code: `bot/fish_vision/classifier.py:288-309` — Stage B exception handler
- Related code: `main.py:92-93, 116-120` — health counter logic
- Institutional: `REPORT.md` Fix 2 (separate error counters), Fix 5 (`choices` guard)
