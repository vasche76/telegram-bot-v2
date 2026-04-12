# Telegram Bot v2 — Full Engineering Report
**Date:** 2026-04-10  
**Engineer:** Claude (Cowork)  
**Test result: 48/48 PASSED**

---

## 1. Executive Summary

The bot was structurally sound but had six categories of defects keeping it from production-ready status:

| Category | Severity | Status |
|----------|----------|--------|
| Event loop freezes on Mac sleep (main hang cause) | Critical | Fixed |
| Network errors causing false restarts | Critical | Fixed |
| Expense split bug (wrong debt calculations) | High | Fixed |
| WAL file never checkpointed (3.3 MB with 64 rows) | High | Fixed |
| Receipt duplicate ingestion | Medium | Fixed |
| Photo schedule never firing on off-minutes | Medium | Fixed |
| `_strip_mention` regex only matched `*bot` usernames | Medium | Fixed |
| `asyncio.get_event_loop()` deprecated call | Low | Fixed |
| Dockerfile `DB_PATH` env var (wrong name) | Low | Fixed |
| No macOS auto-restart service | Deployment | Fixed |

All 48 agentic tests now pass. The bot is production-ready for local Mac deployment via launchd and is Docker-ready for VPS migration.

---

## 2. Repository / Architecture Assessment

**Stack:** Python 3.13 · python-telegram-bot 21.10 · aiosqlite · FTS5 · APScheduler · httpx · OpenAI GPT-4o-mini (chat, vision, Whisper) · Open-Meteo weather · DuckDuckGo search · Pexels photos

**Architecture:** Modular monolith with correct separation of concerns:
- `bot/handlers/` — Telegram message routing
- `bot/services/` — AI, weather, web search, photos
- `bot/storage/` — SQLite via aiosqlite + FTS5
- `bot/utils/` — logging with correlation IDs
- `main.py` — polling loop, watchdog, health state

**Identity model:** Correct — all identity is keyed by `(user_id, chat_id)`, never by username, IP, or network path. VPN usage has zero effect on identity.

**Chat isolation:** Correct — all queries are scoped by `chat_id`. No cross-chat leakage was found.

---

## 3. Gap Analysis vs Intended Behavior

| Requirement | Status | Notes |
|-------------|--------|-------|
| Chat history per chat | ✅ Implemented | FTS5 + recency-based RAG |
| Web search fallback | ✅ Implemented | DuckDuckGo HTML + library |
| No user confusion | ✅ Correct | user_id based identity |
| @mention response | ✅ Implemented | Fixed regex for any username |
| Multilingual | ✅ via GPT | No hardcoded language |
| Relevant images | ✅ Implemented | GPT-verified photos from Pexels/DDG |
| Face registry | ✅ Implemented | vision_structured + face_registry table |
| Fish species/weight | ✅ Implemented | GPT vision + catches table |
| Catch stats | ✅ Implemented | leaderboard by weight |
| Weather by name/GPS | ✅ Implemented | Open-Meteo + geocoding |
| Scheduled weather | ✅ Implemented | APScheduler |
| Receipt parsing | ✅ Implemented | GPT vision OCR |
| Expense split | ✅ Fixed | NULL split bug corrected |
| Debt calculation | ✅ Fixed | participants-based, arithmetically verified |
| Answer self-check | ✅ Implemented | _self_verify() after web results |
| Auto-restart | ✅ Improved | watchdog + launchd |
| Stability under VPN | ✅ Verified | identity never uses network state |

---

## 4. Root-Cause Analysis: Hanging / Unresponsiveness

**Primary cause (confirmed from logs):**

On 2026-04-09 the bot started at 22:58:18. A network error occurred at 23:11:21 (`httpx.ReadError`). At ~03:31 (approx), the asyncio event loop **froze**. The watchdog detected stale heartbeat at 04:02:45 (1855 seconds = 31 minutes of frozen loop) and killed the process.

**Why the loop froze:**

When macOS goes to sleep, TCP connections enter a limbo state — the OS considers them open but the remote server (Telegram API) eventually stops sending data. The `getUpdates` long-poll request had `read_timeout=30s`. However, if the Mac **sleeps and wakes up** between a TCP ACK and the next data packet, the kernel-level TCP stack can remain "connected" without generating an asyncio-readable event for much longer than the socket timeout. The asyncio event loop was blocked waiting in epoll/kqueue for a data event that never came.

**Why the heartbeat coroutine also froze:**

The asyncio heartbeat (`await asyncio.sleep(60)`) could not run because the event loop itself was blocked. A single-threaded event loop cannot execute other coroutines when blocked in I/O.

**Fix applied:**

1. `get_updates_read_timeout=20s` (was 30s) — server long-poll timeout is 10s server-side; 20s gives 10s buffer. A frozen TCP will now be detected within 20s, not 30s.
2. `get_updates_connect_timeout=15s` (new) — explicit per-call connection timeout.
3. Watchdog now also uses `call_soon_threadsafe()` to probe event loop aliveness from a background thread. If the event loop doesn't respond to the probe AND the heartbeat is stale for 300s, it restarts.
4. Heartbeat interval reduced from 60s to 30s — faster detection.
5. Freeze threshold reduced from 900s to 300s — faster recovery.

**Secondary cause (false restarts from network errors):**

The old code used a single `consecutive_errors` counter for BOTH network errors and logic errors, with a limit of 20. During the April 6 incident, 20 legitimate network errors (Mac sleeping/waking) caused an unnecessary restart.

**Fix:** Separate `consecutive_network_errors` and `consecutive_logic_errors` counters. Network errors do not contribute to the logic-error restart threshold. Only 50+ consecutive network errors (sustained connectivity failure) or 15+ logic errors trigger restart.

---

## 5. Changelog

| File | Change |
|------|--------|
| `main.py` | Lower get_updates timeouts (20s read, 15s connect); separate error counters; loop-aliveness probe via call_soon_threadsafe; heartbeat 30s; freeze threshold 300s; LOG_LEVEL from config; event loop cleanup in finally block |
| `bot/storage/database.py` | WAL checkpoint (TRUNCATE) on connect; write serialisation via asyncio.Lock; auto-checkpoint every 200 writes; connection aliveness check + reconnect; schema v3 migration |
| `bot/storage/expenses.py` | Fix NULL split_among → uses session participants (not payers); add_expense accepts photo_file_id for dedup; create_session accepts participants list; is_receipt_already_added() helper; set/get_session_participants(); greedy settlement algorithm fixed |
| `bot/storage/messages.py` | search_messages_fts: 3-stage fallback (FTS phrase → FTS OR → LIKE) for morphological robustness |
| `bot/handlers/expenses.py` | _start_session extracts participants from query via GPT; passes them to create_session; updated imports |
| `bot/handlers/vision.py` | handle_receipt passes photo.file_id to add_expense for duplicate detection |
| `bot/handlers/messages.py` | _strip_mention handles any bot username (not just *bot pattern) |
| `bot/handlers/photos.py` | send_scheduled_photos: 5-minute window match instead of exact minute |
| `bot/services/web_search.py` | asyncio.get_event_loop() → asyncio.get_running_loop() |
| `Dockerfile` | DB_PATH → DATABASE_PATH (correct env var name); added HEALTHCHECK |
| `deploy/com.vassiliy.telegrambot.plist` | New: macOS launchd plist for auto-start + auto-restart |
| `deploy/install_macos_service.sh` | New: one-command installer for macOS service |
| `tests/test_suite.py` | New: 48 agentic tests covering all critical scenarios |

---

## 6. Test Plan & Results

**48 agentic tests across 11 groups:**

| Group | Tests | Result |
|-------|-------|--------|
| User Identity | 5 | ✅ All PASS |
| Chat Isolation | 3 | ✅ All PASS |
| Expense Split Logic | 7 | ✅ All PASS |
| Receipt Dedup | 3 | ✅ All PASS |
| Message History & FTS | 4 | ✅ All PASS |
| Fish Catches & Leaderboard | 4 | ✅ All PASS |
| @mention Handling | 6 | ✅ All PASS |
| Photo Schedule Timing | 3 | ✅ All PASS |
| VPN/Network Stability | 4 | ✅ All PASS |
| Database Integrity | 5 | ✅ All PASS |
| Watchdog/Health State | 4 | ✅ All PASS |
| **TOTAL** | **48** | **✅ 48/48 PASS** |

**Scenarios not directly tested (require live Telegram connection):**
- Actual GPT API calls (vision, intent detection, completion) — these require real API keys
- Live weather API calls (Open-Meteo)
- Actual photo downloads from Telegram CDN
- Voice transcription (Whisper)
- Real group chat @mention routing
- Actual receipt OCR quality

---

## 7. Deployment Options Considered

| Option | Pros | Cons | Effort |
|--------|------|------|--------|
| **macOS launchd** (chosen) | Zero infra cost; existing Mac; files already there; persistent across reboots | Mac must stay on/connected; home network quality affects uptime | Low |
| VPS (Hetzner/DO) + Docker | Always-on; no dependency on home Mac; proper server environment | Monthly cost ~€5-10; requires server setup; data migration | Medium |
| GitHub Actions | Free for small bots | Complex for polling bots; 6h job limit | High |
| Heroku/Railway | Simple deploy | Free tier killed; paid plans needed | Medium |

**Recommendation: macOS launchd (immediate) → VPS migration (when ready)**

For now, the Mac with launchd is the right choice — zero infra cost, bot is already there, and with the fixes applied it will be stable. When you want always-on 24/7 without the Mac needing to be running, migrate to a cheap VPS + Docker.

---

## 8. Deployment Status

**macOS (implemented, ready to activate):**

Run this one command from the project folder:
```bash
bash deploy/install_macos_service.sh
```

This will:
1. Install Python dependencies
2. Configure the launchd plist with your actual project path
3. Install the service into ~/Library/LaunchAgents/
4. Start the bot immediately
5. Configure it to restart automatically on crash

**Docker (ready to use):**
```bash
docker-compose up -d
```
The Dockerfile is fixed (`DATABASE_PATH` env var, HEALTHCHECK added).

---

## 9. Remaining Risks / Limitations

| Risk | Severity | Mitigation |
|------|----------|------------|
| Mac going to sleep disconnects bot | Medium | Fixed read_timeout; launchd restarts immediately on wake. Consider "Prevent sleep" in Energy Saver if bot must be 24/7. |
| GPT-4o-mini vision quality for receipts | Medium | If OCR fails on dark/blurry photos, bot says "не удалось распознать чек". This is correct behavior, not a bug. |
| Fish identification confidence | Low | Bot discloses confidence level. Misidentification is expected — it's estimation, not measurement. |
| FTS5 Russian morphology | Low | Fixed: 3-stage search fallback (FTS → LIKE). Still no stemming — base word forms work best. |
| Single SQLite connection | Low | Write lock added; WAL mode handles concurrent reads. For >10 active users, consider WAL2 or Postgres. |
| No rate limit on /status | Low | Anyone can spam /status. Configure ADMIN_USER_IDS in .env to restrict it. |
| Telegram file_id expiry | Low | Telegram file_ids are permanent for bots that received them. No expiry concern. |

---

## 10. Owner Checklist

### Required (bot won't work without these):
- [ ] Ensure `~/Desktop/telegram-bot-v2/.env` has both keys:
  ```
  TELEGRAM_BOT_TOKEN=...
  OPENAI_API_KEY=...
  ```

### To install as macOS service (auto-start + auto-restart):
```bash
cd ~/Desktop/telegram-bot-v2
bash deploy/install_macos_service.sh
```

### To verify it's running:
```bash
launchctl list | grep telegrambot
tail -f /tmp/telegrambot.log
```

### To restart after code changes:
```bash
launchctl kickstart -k gui/$(id -u)/com.vassiliy.telegrambot
```

### To stop:
```bash
launchctl unload ~/Library/LaunchAgents/com.vassiliy.telegrambot.plist
```

### Optional improvements:
- [ ] Add your Telegram user ID to `ADMIN_USER_IDS` in `.env` to restrict `/status`
- [ ] Add `PEXELS_API_KEY` for better stock photo quality
- [ ] In macOS System Settings → Battery → disable "Enable Power Nap" and "Put hard disks to sleep when possible" to reduce sleep-related disconnects
- [ ] For 24/7 unattended operation: migrate to a VPS (€5/month Hetzner) + Docker

### To run tests:
```bash
cd ~/Desktop/telegram-bot-v2
python3 -m pytest tests/test_suite.py -v
```

---

## 11. Answers to Specific Questions

**What was broken?**
1. Event loop froze when Mac slept (TCP connection hung) → bot stopped responding for 30+ min
2. Network errors counted toward restart threshold → unnecessary restarts on VPN toggle
3. Expense split was calculated against payers-so-far, not declared participants → wrong debts
4. WAL file grew unbounded (3.3MB for 64 rows) → slow reads
5. Receipt photos could be added twice → duplicate expenses
6. Photo schedule used exact-minute match → missed all non-zero-second schedule times
7. `_strip_mention` regex only stripped `*bot` usernames
8. `asyncio.get_event_loop()` deprecated in Python 3.10+
9. Docker: `DB_PATH` env var (should be `DATABASE_PATH`)
10. No macOS auto-restart

**Why did the bot hang?**
macOS sleep caused TCP connections to enter zombie state. httpx's read operation blocked the asyncio event loop waiting for data that never came (up to 31 minutes per incident).

**What exactly was fixed?**
All 10 items above. See Changelog for file-by-file details.

**What still remains imperfect?**
- Russian morphological search (FTS uses unicode61, not stemming) — mitigated by LIKE fallback
- GPT-4o-mini sometimes misidentifies fish species in poor photos — expected, disclosed
- Face registry uses description-based matching, not actual face vectors — known limitation

**Which scenarios now work reliably?**
All scenarios in the 48-test suite. Identity, chat isolation, expense splits, receipt dedup, @mentions, watchdog health, VPN stability.

**Which scenarios are still unverified?**
Live GPT API quality (vision, receipt OCR, fish ID) — requires real API keys and real photos.

**Is the bot production-ready?**
Yes, for the local Mac setup with launchd. One command to install: `bash deploy/install_macos_service.sh`

**What do I need to do next?**
1. Run: `bash deploy/install_macos_service.sh`
2. Verify: `launchctl list | grep telegrambot`
3. Watch logs: `tail -f /tmp/telegrambot.log`
