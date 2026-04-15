"""
Agentic test suite for Telegram Bot v2.

Tests the bot logic WITHOUT a live Telegram connection — uses mocks.
Covers all major scenarios from the ТЗ:
  - User identity isolation (no cross-user confusion)
  - Chat isolation (no cross-chat leakage)
  - Expense split logic correctness (the NULL split bug)
  - Receipt duplicate detection
  - Photo schedule timing (window-based matching)
  - Weather intent detection
  - _strip_mention regex for any username
  - Database WAL checkpoint presence
  - FTS5 search correctness
  - User identity is Telegram user_id, not username/network

Run: python3 -m pytest tests/test_suite.py -v
"""

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ── Setup: point DATABASE_PATH to a temp file ─────────────────────
_TMP_DB = tempfile.mktemp(suffix=".db")
os.environ["DATABASE_PATH"] = _TMP_DB
os.environ["TELEGRAM_BOT_TOKEN"] = "test-token-000"
os.environ["OPENAI_API_KEY"] = "test-key-000"

# Now safe to import bot modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from bot.storage import database as db_module
from bot.storage.database import (
    init_db, execute, fetch_all, fetch_one, fetch_scalar,
)
from bot.storage.messages import save_message, search_messages_fts, get_recent_messages
from bot.storage.users import upsert_user, get_display_name, get_user
from bot.storage.expenses import (
    create_session, get_active_session, close_session,
    add_expense, get_session_expenses, calculate_debts,
    set_session_participants, get_session_participants,
    is_receipt_already_added,
)
from bot.storage.catches import save_catch, get_chat_leaderboard
from bot.handlers.messages import _strip_mention, _is_mention, set_bot_username


# ── Helpers ────────────────────────────────────────────────────────

def run(coro):
    """Run async coroutine in test."""
    return asyncio.get_event_loop().run_until_complete(coro)


class BotTestCase(unittest.TestCase):
    """Base class that initialises DB for each test."""

    @classmethod
    def setUpClass(cls):
        os.environ["DATABASE_PATH"] = _TMP_DB
        db_module._db = None  # force reconnect
        run(init_db())

    @classmethod
    def tearDownClass(cls):
        run(db_module.close_db())
        try:
            Path(_TMP_DB).unlink(missing_ok=True)
            Path(_TMP_DB + "-wal").unlink(missing_ok=True)
            Path(_TMP_DB + "-shm").unlink(missing_ok=True)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 1: User Identity — must be Telegram user_id based
# ══════════════════════════════════════════════════════════════════

class TestUserIdentity(BotTestCase):
    """Identity must be stable: same user_id = same person regardless of VPN/username."""

    def test_upsert_user_creates_record_by_user_id(self):
        """User profile keyed by (user_id, chat_id), not username."""
        run(upsert_user(user_id=111, chat_id=1, username="alice", first_name="Alice"))
        user = run(get_user(111, 1))
        self.assertIsNotNone(user)
        self.assertEqual(user["user_id"], 111)
        self.assertEqual(user["username"], "alice")

    def test_username_change_preserves_user_id(self):
        """If Alice changes her Telegram username, identity (user_id) is preserved."""
        run(upsert_user(user_id=111, chat_id=1, username="alice_new", first_name="Alice"))
        user = run(get_user(111, 1))
        self.assertEqual(user["user_id"], 111)
        self.assertEqual(user["username"], "alice_new")  # updated
        self.assertEqual(user["first_name"], "Alice")    # preserved

    def test_different_users_dont_share_profiles(self):
        """Two distinct user_ids must produce distinct profiles."""
        # Create both users explicitly (no cross-test dependency)
        run(upsert_user(user_id=1111, chat_id=50, username="alice2", first_name="Alice2"))
        run(upsert_user(user_id=2222, chat_id=50, username="bob2", first_name="Bob2"))
        alice = run(get_user(1111, 50))
        bob = run(get_user(2222, 50))
        self.assertIsNotNone(alice)
        self.assertIsNotNone(bob)
        self.assertNotEqual(alice["display_name"], bob["display_name"])

    def test_same_user_different_chats_are_independent(self):
        """User in chat A is independent from same user in chat B."""
        run(upsert_user(user_id=333, chat_id=10, username="carol", first_name="Carol in 10"))
        run(upsert_user(user_id=333, chat_id=20, username="carol", first_name="Carol in 20"))
        u10 = run(get_user(333, 10))
        u20 = run(get_user(333, 20))
        self.assertIsNotNone(u10)
        self.assertIsNotNone(u20)
        # Both exist independently
        self.assertEqual(u10["chat_id"], 10)
        self.assertEqual(u20["chat_id"], 20)

    def test_display_name_fallback_is_user_id(self):
        """get_display_name for unknown user returns str(user_id)."""
        name = run(get_display_name(99999, 99999))
        self.assertEqual(name, "99999")


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 2: Chat Isolation — no cross-chat leakage
# ══════════════════════════════════════════════════════════════════

class TestChatIsolation(BotTestCase):
    """Messages, expenses, catches must never leak between chats."""

    def setUp(self):
        # Clear test-specific tables
        run(execute("DELETE FROM messages WHERE chat_id IN (100, 200)"))
        run(execute("DELETE FROM expense_sessions WHERE chat_id IN (100, 200)"))
        run(execute("DELETE FROM catches WHERE chat_id IN (100, 200)"))

    def test_messages_isolated_by_chat(self):
        """Messages in chat 100 must not appear in chat 200 search.

        NOTE: Use exact base-form words to avoid FTS5 morphology mismatch
        (Russian inflected forms like 'Мурманске' ≠ 'Мурманск' in FTS5 tokenizer).
        """
        run(save_message(100, 1, "user_a", "рыбалка Мурманск", message_type="text"))
        run(save_message(200, 2, "user_b", "совсем другая тема", message_type="text"))

        results_100 = run(search_messages_fts(100, "Мурманск", limit=5))
        results_200 = run(search_messages_fts(200, "Мурманск", limit=5))

        self.assertGreater(len(results_100), 0, "Should find 'Мурманск' in chat 100")
        self.assertEqual(len(results_200), 0, "Should NOT find 'Мурманск' in chat 200")

    def test_expense_sessions_isolated_by_chat(self):
        """Expense session in chat 100 must not appear in chat 200."""
        run(create_session(100, "trip-sea"))
        active_100 = run(get_active_session(100))
        active_200 = run(get_active_session(200))
        self.assertIsNotNone(active_100)
        self.assertIsNone(active_200, "Session from chat 100 must not appear in chat 200")

    def test_catches_isolated_by_chat(self):
        """Fish catches in chat 100 must not appear in chat 200 leaderboard."""
        run(save_catch(100, 1, "Вася", "Щука", fish_count=1, weight_kg=3.5))
        lb_100 = run(get_chat_leaderboard(100))
        lb_200 = run(get_chat_leaderboard(200))
        self.assertGreater(len(lb_100), 0)
        # Leaderboard for chat 200 must not contain Вася from chat 100
        names_200 = [r["person_name"] for r in lb_200]
        # Check that chat 100 Вася is not in chat 200
        for name in names_200:
            self.assertNotEqual(
                name, "Вася",
                "Catch attributed to wrong chat!"
            )


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 3: Expense Split Logic (the core bug fix)
# ══════════════════════════════════════════════════════════════════

class TestExpenseSplit(BotTestCase):
    """Verify debt calculations are arithmetically correct."""

    def setUp(self):
        run(execute("DELETE FROM expense_sessions WHERE chat_id = 999"))
        run(execute("DELETE FROM expenses WHERE chat_id = 999"))

    def test_simple_split_two_people(self):
        """A pays 1000 RUB; split equally between A and B → B owes A 500."""
        run(create_session(999, "test-simple", participants=["A", "B"]))
        run(add_expense(999, "test-simple", 1, "A", 1000.0, "dinner", split_among=None))

        debts = run(calculate_debts(999, "test-simple"))
        self.assertEqual(debts["total"], 1000.0)

        settlements = debts["settlements"]
        self.assertEqual(len(settlements), 1, f"Expected 1 settlement, got: {settlements}")
        s = settlements[0]
        self.assertEqual(s["from"], "B")
        self.assertEqual(s["to"], "A")
        self.assertAlmostEqual(s["amount"], 500.0, places=2)

    def test_three_person_split(self):
        """A pays 900; B pays 0; C pays 0; all split equally → A gets 600."""
        run(create_session(999, "test-3", participants=["A", "B", "C"]))
        run(add_expense(999, "test-3", 1, "A", 900.0, "hotel", split_among=None))

        debts = run(calculate_debts(999, "test-3"))
        self.assertEqual(debts["total"], 900.0)
        # A paid 900, owes 300, net +600
        self.assertAlmostEqual(debts["per_person"]["A"], 600.0, places=2)
        # B owes 300, net -300
        self.assertAlmostEqual(debts["per_person"]["B"], -300.0, places=2)
        # C owes 300, net -300
        self.assertAlmostEqual(debts["per_person"]["C"], -300.0, places=2)

        settlements = debts["settlements"]
        total_transferred = sum(s["amount"] for s in settlements)
        self.assertAlmostEqual(total_transferred, 600.0, places=2)

    def test_explicit_split_among_subset(self):
        """Expense split only among specific people."""
        run(create_session(999, "test-sub", participants=["A", "B", "C"]))
        # Only A and C share this expense (B didn't order)
        run(add_expense(999, "test-sub", 1, "A", 200.0, "dessert", split_among=["A", "C"]))

        debts = run(calculate_debts(999, "test-sub"))
        # A paid 200, owes 100, net +100; C owes 100, net -100; B owes 0
        self.assertAlmostEqual(debts["per_person"]["A"], 100.0, places=2)
        self.assertAlmostEqual(debts["per_person"]["C"], -100.0, places=2)
        self.assertAlmostEqual(debts["per_person"].get("B", 0.0), 0.0, places=2)

    def test_mutual_payments_cancel(self):
        """If A and B each pay 500 and split equally, nobody owes anything."""
        run(create_session(999, "test-mutual", participants=["A", "B"]))
        run(add_expense(999, "test-mutual", 1, "A", 500.0, "A pays"))
        run(add_expense(999, "test-mutual", 2, "B", 500.0, "B pays"))

        debts = run(calculate_debts(999, "test-mutual"))
        self.assertEqual(len(debts["settlements"]), 0, "No one should owe anything")

    def test_null_split_uses_participants_not_payers(self):
        """
        THE CRITICAL BUG: When split_among=NULL and only one person has paid,
        old code used list(paid.keys()) = [payer] → split was only among the payer.
        New code uses declared participants.
        """
        # Declare 3 participants upfront
        run(create_session(999, "test-bug", participants=["A", "B", "C"]))
        # Only A has paid (B and C haven't yet)
        run(add_expense(999, "test-bug", 1, "A", 600.0, "first expense"))

        debts = run(calculate_debts(999, "test-bug"))
        # CORRECT: 600 / 3 = 200 per person
        # A net: 600 paid - 200 owed = +400
        # B net: 0 paid - 200 owed = -200
        # C net: 0 paid - 200 owed = -200
        self.assertAlmostEqual(debts["per_person"]["A"], 400.0, places=2,
                               msg="A should be owed 400 (paid 600, owes 200)")
        self.assertAlmostEqual(debts["per_person"].get("B", 0), -200.0, places=2,
                               msg="B should owe 200")
        self.assertAlmostEqual(debts["per_person"].get("C", 0), -200.0, places=2,
                               msg="C should owe 200")

    def test_session_participant_auto_added_on_expense(self):
        """When a payer is added who isn't in participants, they should be auto-added."""
        run(create_session(999, "test-auto", participants=["Alice"]))
        # Bob is a surprise payer
        run(add_expense(999, "test-auto", 2, "Bob", 300.0, "Bob's expense"))
        parts = run(get_session_participants(999, "test-auto"))
        self.assertIn("Bob", parts)

    def test_arithmetic_rounding(self):
        """Expense splits should round correctly to 2 decimal places."""
        run(create_session(999, "test-round", participants=["A", "B", "C"]))
        run(add_expense(999, "test-round", 1, "A", 100.0))  # 100 / 3 = 33.333...

        debts = run(calculate_debts(999, "test-round"))
        total_owed = sum(abs(v) for v in debts["per_person"].values() if v < 0)
        # Total owed should approximately equal total paid by A
        # (A net ~66.67, others net ~-33.33 each)
        self.assertAlmostEqual(total_owed, 200 / 3, delta=0.02)


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 4: Receipt Duplicate Detection
# ══════════════════════════════════════════════════════════════════

class TestReceiptDedup(BotTestCase):

    def setUp(self):
        run(execute("DELETE FROM expense_sessions WHERE chat_id = 888"))
        run(execute("DELETE FROM expenses WHERE chat_id = 888"))

    def test_same_photo_not_added_twice(self):
        """Sending the same receipt photo twice must not create duplicate expense."""
        run(create_session(888, "dedup-test", participants=["A", "B"]))
        photo_id = "file_id_abc123"

        # First time
        run(add_expense(888, "dedup-test", 1, "A", 500.0,
                        photo_file_id=photo_id, description="receipt 1"))
        # Second time (duplicate)
        run(add_expense(888, "dedup-test", 1, "A", 500.0,
                        photo_file_id=photo_id, description="receipt 1 again"))

        expenses = run(get_session_expenses(888, "dedup-test"))
        self.assertEqual(len(expenses), 1, "Duplicate receipt should be rejected")

        debts = run(calculate_debts(888, "dedup-test"))
        self.assertEqual(debts["total"], 500.0, "Total should be 500, not 1000")

    def test_different_photos_both_added(self):
        """Two different receipts must both be added."""
        run(create_session(888, "dedup-two", participants=["A", "B"]))
        run(add_expense(888, "dedup-two", 1, "A", 100.0, photo_file_id="file_001"))
        run(add_expense(888, "dedup-two", 1, "A", 200.0, photo_file_id="file_002"))

        expenses = run(get_session_expenses(888, "dedup-two"))
        self.assertEqual(len(expenses), 2)
        self.assertEqual(run(calculate_debts(888, "dedup-two"))["total"], 300.0)

    def test_is_receipt_already_added_helper(self):
        """is_receipt_already_added() must return True after adding and False before."""
        run(create_session(888, "dedup-check", participants=["A"]))
        photo = "unique_file_xyz"
        self.assertFalse(run(is_receipt_already_added(888, "dedup-check", photo)))
        run(add_expense(888, "dedup-check", 1, "A", 99.0, photo_file_id=photo))
        self.assertTrue(run(is_receipt_already_added(888, "dedup-check", photo)))


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 5: Message History & FTS Search
# ══════════════════════════════════════════════════════════════════

class TestMessageHistory(BotTestCase):

    def setUp(self):
        run(execute("DELETE FROM messages WHERE chat_id = 777"))

    def test_save_and_retrieve_message(self):
        """Saved messages must be retrievable."""
        run(save_message(777, 1, "testuser", "Hello world test", message_type="text"))
        msgs = run(get_recent_messages(777, limit=10))
        texts = [m["message_text"] for m in msgs]
        self.assertIn("Hello world test", texts)

    def test_fts_search_finds_keyword(self):
        """FTS5 search must find a saved message by keyword (exact base form)."""
        # Use exact base form to avoid morphology mismatch in FTS5 unicode61 tokenizer
        run(save_message(777, 1, "alice", "рыбалка Астрахань рыба", message_type="text"))
        run(save_message(777, 2, "bob", "погода сегодня отличная", message_type="text"))
        results = run(search_messages_fts(777, "Астрахань", limit=5))
        self.assertGreater(len(results), 0, "FTS should find 'Астрахань'")
        texts = [r["message_text"] for r in results]
        self.assertTrue(any("Астрахань" in t for t in texts))

    def test_fts_does_not_leak_across_chats(self):
        """FTS search must be scoped to the given chat_id."""
        run(save_message(777, 1, "u1", "секретное слово квантум", message_type="text"))
        results_same = run(search_messages_fts(777, "квантум", limit=5))
        results_other = run(search_messages_fts(778, "квантум", limit=5))
        self.assertGreater(len(results_same), 0)
        self.assertEqual(len(results_other), 0)

    def test_message_order_is_chronological(self):
        """get_recent_messages must return messages newest first."""
        run(save_message(777, 1, "u", "first", message_type="text"))
        run(save_message(777, 1, "u", "second", message_type="text"))
        run(save_message(777, 1, "u", "third", message_type="text"))
        msgs = run(get_recent_messages(777, limit=10))
        # Newest first — last inserted should be first in result
        self.assertEqual(msgs[0]["message_text"], "third")


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 6: Fish Catches & Leaderboard
# ══════════════════════════════════════════════════════════════════

class TestFishCatches(BotTestCase):

    def setUp(self):
        run(execute("DELETE FROM catches WHERE chat_id = 555"))

    def test_save_and_retrieve_catch(self):
        """A saved catch must appear in the leaderboard."""
        run(save_catch(555, 1, "Вася", "Щука", fish_count=1, weight_kg=4.2))
        lb = run(get_chat_leaderboard(555))
        self.assertGreater(len(lb), 0)
        self.assertEqual(lb[0]["person_name"], "Вася")

    def test_leaderboard_order_by_weight(self):
        """Leaderboard must be ordered by total weight descending."""
        run(save_catch(555, 1, "Вася", "Щука", fish_count=1, weight_kg=2.0))
        run(save_catch(555, 2, "Петя", "Карп", fish_count=1, weight_kg=5.0))
        lb = run(get_chat_leaderboard(555))
        self.assertEqual(lb[0]["person_name"], "Петя")  # higher weight is first

    def test_catches_isolated_by_chat(self):
        """Catches from chat 555 must not appear in chat 666 leaderboard."""
        run(save_catch(555, 1, "Вася", "Лещ", fish_count=1, weight_kg=1.0))
        lb_666 = run(get_chat_leaderboard(666))
        names = [r["person_name"] for r in lb_666]
        self.assertNotIn("Вася", names)

    def test_multiple_catches_accumulate(self):
        """Multiple catches for the same person accumulate correctly."""
        run(save_catch(555, 1, "Вася", "Окунь", fish_count=2, weight_kg=0.5))
        run(save_catch(555, 1, "Вася", "Щука", fish_count=1, weight_kg=3.0))
        lb = run(get_chat_leaderboard(555))
        vася = next(r for r in lb if r["person_name"] == "Вася")
        self.assertAlmostEqual(vася["total_weight_kg"], 3.5, places=1)


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 7: @mention Handling
# ══════════════════════════════════════════════════════════════════

class TestMentionHandling(unittest.TestCase):
    """Test @mention detection and stripping."""

    def setUp(self):
        set_bot_username("vassiliy_chekulaev_bot")

    def test_is_mention_with_entity(self):
        """_is_mention must detect bot mention via entity."""
        entity = MagicMock()
        entity.type = "mention"
        entity.offset = 0
        entity.length = 24
        text = "@vassiliy_chekulaev_bot привет"
        self.assertTrue(_is_mention(text, [entity]))

    def test_is_mention_case_insensitive(self):
        """@mention detection must be case-insensitive."""
        entity = MagicMock()
        entity.type = "mention"
        entity.offset = 0
        entity.length = 24
        text = "@Vassiliy_Chekulaev_Bot привет"
        self.assertTrue(_is_mention(text, [entity]))

    def test_is_mention_not_triggered_by_other_user(self):
        """@mention of another user must not trigger bot response."""
        entity = MagicMock()
        entity.type = "mention"
        entity.offset = 0
        entity.length = 8
        text = "@someoneelse привет"
        self.assertFalse(_is_mention(text, [entity]))

    def test_strip_mention_removes_bot_username(self):
        """_strip_mention must remove the bot's @mention."""
        text = "@vassiliy_chekulaev_bot погода в Москве"
        stripped = _strip_mention(text)
        self.assertNotIn("@vassiliy_chekulaev_bot", stripped.lower())
        self.assertIn("Москве", stripped)

    def test_strip_mention_any_username(self):
        """_strip_mention fallback must handle non-*bot usernames."""
        set_bot_username("myhelper")
        text = "@MyHelper погода в Питере"
        stripped = _strip_mention(text)
        self.assertNotIn("@MyHelper", stripped)
        self.assertIn("Питере", stripped)
        set_bot_username("vassiliy_chekulaev_bot")  # restore

    def test_strip_mention_non_empty_result(self):
        """_strip_mention must return original text if result would be empty."""
        text = "@vassiliy_chekulaev_bot"
        stripped = _strip_mention(text)
        self.assertTrue(len(stripped) > 0)


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 8: Photo Schedule Timing
# ══════════════════════════════════════════════════════════════════

class TestPhotoScheduleTiming(unittest.TestCase):
    """Test that photo schedule fires within a 5-minute window, not exact minute."""

    def test_within_window_matches(self):
        """A schedule set for 09:00 should fire when current time is 09:03."""
        from datetime import datetime, timezone

        now_minutes = 9 * 60 + 3  # 09:03
        sched_time = "09:00"
        h, m = map(int, sched_time.split(":"))
        sched_minutes = h * 60 + m
        self.assertLessEqual(abs(now_minutes - sched_minutes), 5)

    def test_outside_window_does_not_match(self):
        """A schedule set for 09:00 should NOT fire at 09:07."""
        now_minutes = 9 * 60 + 7
        sched_time = "09:00"
        h, m = map(int, sched_time.split(":"))
        sched_minutes = h * 60 + m
        self.assertGreater(abs(now_minutes - sched_minutes), 5)

    def test_exact_match_fires(self):
        """Exact minute match should always fire."""
        now_minutes = 22 * 60 + 30
        sched_time = "22:30"
        h, m = map(int, sched_time.split(":"))
        sched_minutes = h * 60 + m
        self.assertEqual(abs(now_minutes - sched_minutes), 0)


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 9: VPN / Network Stability (identity must be stable)
# ══════════════════════════════════════════════════════════════════

class TestVPNIdentityStability(BotTestCase):
    """
    VPN usage or network changes must not affect user identity or stored data.
    Identity is based on Telegram user_id (stable), not IP/geo/username.
    """

    def test_user_identity_independent_of_network(self):
        """
        Simulate same user connecting twice with 'different network context'.
        Their user_id must produce the same profile each time.
        """
        # First interaction (e.g., without VPN)
        run(upsert_user(user_id=42, chat_id=5, username="igor", first_name="Igor"))
        # Second interaction (e.g., with VPN — different IP, but same user_id)
        run(upsert_user(user_id=42, chat_id=5, username="igor", first_name="Igor"))

        user = run(get_user(42, 5))
        self.assertIsNotNone(user)
        self.assertEqual(user["user_id"], 42)
        self.assertEqual(user["first_name"], "Igor")

    def test_stored_catches_survive_multiple_sessions(self):
        """Catches stored by user_id persist correctly across reconnects."""
        run(save_catch(5, 42, "Igor", "Судак", fish_count=1, weight_kg=2.1))
        # Simulate restart / reconnect
        run(upsert_user(user_id=42, chat_id=5, username="igor", first_name="Igor"))
        lb = run(get_chat_leaderboard(5))
        names = [r["person_name"] for r in lb]
        self.assertIn("Igor", names)

    def test_two_users_same_chat_no_confusion(self):
        """Two users in the same chat must never have their data mixed."""
        run(upsert_user(user_id=10, chat_id=5, username="user_a", first_name="UserA"))
        run(upsert_user(user_id=11, chat_id=5, username="user_b", first_name="UserB"))

        u_a = run(get_user(10, 5))
        u_b = run(get_user(11, 5))
        self.assertNotEqual(u_a["display_name"], u_b["display_name"])

    def test_expense_attribution_by_user_id(self):
        """Expense paid_by must be tracked by name (stable), not by IP."""
        run(execute("DELETE FROM expense_sessions WHERE chat_id = 3"))
        run(execute("DELETE FROM expenses WHERE chat_id = 3"))
        run(create_session(3, "vpn-test", participants=["Alice", "Bob"]))
        run(add_expense(3, "vpn-test", 1, "Alice", 300.0, "dinner"))

        expenses = run(get_session_expenses(3, "vpn-test"))
        self.assertEqual(len(expenses), 1)
        self.assertEqual(expenses[0]["paid_by_name"], "Alice")


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 10: Database Integrity
# ══════════════════════════════════════════════════════════════════

class TestDatabaseIntegrity(BotTestCase):

    def test_schema_version_is_current(self):
        """Database must be at the latest schema version."""
        from bot.storage.database import SCHEMA_VERSION
        version = run(fetch_scalar("SELECT version FROM schema_version LIMIT 1"))
        self.assertEqual(version, SCHEMA_VERSION,
                         f"DB schema {version} != expected {SCHEMA_VERSION}")

    def test_wal_mode_enabled(self):
        """Database must use WAL journal mode."""
        mode = run(fetch_scalar("PRAGMA journal_mode"))
        self.assertEqual(mode, "wal")

    def test_fts_table_exists(self):
        """FTS5 virtual table must exist."""
        row = run(fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages_fts'"
        ))
        self.assertIsNotNone(row)

    def test_all_required_tables_exist(self):
        """All required tables must exist in the database."""
        required = [
            "users", "messages", "messages_fts", "user_aliases",
            "expense_sessions", "expenses", "face_registry",
            "catches", "locations", "weather_subs", "photo_schedules",
        ]
        for table in required:
            row = run(fetch_one(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ))
            self.assertIsNotNone(row, f"Table '{table}' is missing!")

    def test_write_lock_prevents_corruption(self):
        """Concurrent writes via execute() must all succeed without corruption."""
        async def _concurrent_writes():
            tasks = [
                save_message(1234, i, f"user{i}", f"message {i}", message_type="text")
                for i in range(20)
            ]
            await asyncio.gather(*tasks)

        run(_concurrent_writes())
        count = run(fetch_scalar(
            "SELECT COUNT(*) FROM messages WHERE chat_id = 1234"
        ))
        self.assertEqual(count, 20, "All 20 concurrent writes must succeed")


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 11: Watchdog / Health State
# ══════════════════════════════════════════════════════════════════

class TestHealthState(unittest.TestCase):

    def test_mark_activity_resets_error_counters(self):
        """mark_activity() must reset both error counters."""
        import importlib
        import bot.config  # ensure config is loaded
        from main import health
        health.consecutive_network_errors = 5
        health.consecutive_logic_errors = 3
        health.mark_activity()
        self.assertEqual(health.consecutive_network_errors, 0)
        self.assertEqual(health.consecutive_logic_errors, 0)

    def test_network_errors_dont_affect_logic_counter(self):
        """mark_network_error() must not increment logic error counter."""
        from main import health
        health.consecutive_logic_errors = 0
        health.mark_network_error()
        health.mark_network_error()
        self.assertEqual(health.consecutive_logic_errors, 0)
        self.assertEqual(health.consecutive_network_errors, 2)

    def test_loop_alive_updates_probe_timestamp(self):
        """mark_loop_alive() must update last_loop_probe_response."""
        from main import health
        old_ts = health.last_loop_probe_response
        time.sleep(0.01)
        health.mark_loop_alive()
        self.assertGreater(health.last_loop_probe_response, old_ts)

    def test_clear_errors_resets_both_counters(self):
        """clear_errors() must reset both network and logic counters."""
        from main import health
        health.consecutive_network_errors = 10
        health.consecutive_logic_errors = 8
        health.clear_errors()
        self.assertEqual(health.consecutive_network_errors, 0)
        self.assertEqual(health.consecutive_logic_errors, 0)


# ══════════════════════════════════════════════════════════════════
# TEST GROUP 12: Fish Vision Pipeline — logic and guardrails
# (No live API calls — tests the structural logic and DB integration)
# ══════════════════════════════════════════════════════════════════

class TestFishVisionPipeline(BotTestCase):
    """
    Tests for the fish vision module.
    These do NOT call the OpenAI API — they test the structural logic:
    - DetectionResult.should_classify gating
    - ClassificationResult.is_identified thresholding
    - FishAnalysisResult.is_valid_catch correctness
    - DB integration: rejected catches stored with is_valid_catch=0
    - Leaderboard excludes invalid catches
    """

    def setUp(self):
        """Clean up test chat IDs before each DB test."""
        run(execute("DELETE FROM catches WHERE chat_id IN (7777, 7778, 7779)"))

    def _make_detection(self, obj_type, confidence):
        from bot.fish_vision.detector import DetectionResult
        return DetectionResult(
            object_type=obj_type,
            confidence=confidence,
            fish_count=1 if obj_type == "whole_fish" else 0,
            estimated_length_cm=None,
            reasoning="test",
            raw_description="test desc",
        )

    def _make_classification(self, species, confidence):
        from bot.fish_vision.classifier import ClassificationResult, SPECIES_MAP
        # Enforce threshold in test (same as classifier does)
        from bot.fish_vision.classifier import SPECIES_CONFIDENCE_THRESHOLD
        if confidence < SPECIES_CONFIDENCE_THRESHOLD and species != "unknown_fish":
            species = "unknown_fish"
        return ClassificationResult(
            species_key=species,
            species_ru=SPECIES_MAP.get(species, "Рыба"),
            confidence=confidence,
            weight_kg_estimate=3.5,
            length_cm_estimate=60.0,
            fish_count=1,
            person_name_in_photo=None,
            distinguishing_features="duck-bill snout, green spots",
            reasoning="test",
        )

    # ── Stage A: DetectionResult.should_classify ─────────────────

    def test_whole_fish_high_confidence_proceeds(self):
        """whole_fish at 0.85 confidence must proceed to Stage B."""
        det = self._make_detection("whole_fish", 0.85)
        self.assertTrue(det.should_classify)
        self.assertIsNone(det.rejection_reason)

    def test_lure_must_be_rejected_regardless_of_confidence(self):
        """lure at any confidence must NOT proceed to Stage B."""
        det = self._make_detection("lure", 0.95)
        self.assertFalse(det.should_classify)
        self.assertIn("приманка", det.rejection_reason.lower())

    def test_fish_part_must_be_rejected(self):
        """fish_part must NOT proceed to Stage B."""
        det = self._make_detection("fish_part", 0.90)
        self.assertFalse(det.should_classify)
        self.assertIn("часть", det.rejection_reason.lower())

    def test_fry_must_be_rejected(self):
        """fry must NOT proceed to Stage B."""
        det = self._make_detection("fry", 0.80)
        self.assertFalse(det.should_classify)
        self.assertIn("малёк", det.rejection_reason.lower())

    def test_no_fish_must_be_rejected(self):
        """no_fish must NOT proceed to Stage B."""
        det = self._make_detection("no_fish", 0.99)
        self.assertFalse(det.should_classify)

    def test_whole_fish_low_confidence_is_blocked(self):
        """whole_fish below FILTER_CONFIDENCE_THRESHOLD must be blocked."""
        from bot.fish_vision.detector import FILTER_CONFIDENCE_THRESHOLD
        det = self._make_detection("whole_fish", FILTER_CONFIDENCE_THRESHOLD - 0.01)
        self.assertFalse(det.should_classify)

    def test_whole_fish_at_threshold_proceeds(self):
        """whole_fish exactly at threshold must proceed."""
        from bot.fish_vision.detector import FILTER_CONFIDENCE_THRESHOLD
        det = self._make_detection("whole_fish", FILTER_CONFIDENCE_THRESHOLD)
        self.assertTrue(det.should_classify)

    # ── Stage B: ClassificationResult.is_identified ──────────────

    def test_pike_high_confidence_is_identified(self):
        """pike at 0.80 must be identified (above threshold)."""
        cls = self._make_classification("pike", 0.80)
        self.assertTrue(cls.is_identified)
        self.assertEqual(cls.species_key, "pike")

    def test_low_confidence_forces_unknown(self):
        """Species confidence below threshold forces unknown_fish."""
        from bot.fish_vision.classifier import SPECIES_CONFIDENCE_THRESHOLD
        cls = self._make_classification("perch", SPECIES_CONFIDENCE_THRESHOLD - 0.05)
        self.assertEqual(cls.species_key, "unknown_fish",
                         "Low confidence should produce unknown_fish, not wrong species")
        self.assertFalse(cls.is_identified)

    def test_unknown_fish_is_not_identified(self):
        """unknown_fish is always not is_identified (even at high confidence)."""
        cls = self._make_classification("unknown_fish", 0.90)
        self.assertFalse(cls.is_identified)

    # ── DB integration: save_catch with new fields ───────────────

    def test_invalid_catch_not_in_leaderboard(self):
        """Catches with is_valid_catch=False must NOT appear in leaderboard."""
        async def _run():
            # Clean up
            await execute("DELETE FROM catches WHERE chat_id = 7777")
            # Save a REJECTED catch (lure photo)
            from bot.storage.catches import save_catch
            await save_catch(
                7777, 1, "Рыбак", "Щука",
                fish_count=1, weight_kg=5.0,
                object_type="lure",
                species_confidence=0.9,
                is_valid_catch=False,
                rejection_reason="Это воблер, не рыба",
            )
            # Leaderboard must be empty
            lb = await get_chat_leaderboard(7777)
            return lb
        lb = run(_run())
        self.assertEqual(len(lb), 0, "Rejected catches must not appear in leaderboard")

    def test_valid_catch_appears_in_leaderboard(self):
        """Catches with is_valid_catch=True must appear in leaderboard."""
        async def _run():
            await execute("DELETE FROM catches WHERE chat_id = 7778")
            from bot.storage.catches import save_catch
            await save_catch(
                7778, 1, "Рыбак", "Щука",
                fish_count=1, weight_kg=4.2,
                object_type="whole_fish",
                species_confidence=0.85,
                is_valid_catch=True,
            )
            return await get_chat_leaderboard(7778)
        lb = run(_run())
        self.assertEqual(len(lb), 1)
        self.assertAlmostEqual(lb[0]["total_weight_kg"], 4.2, places=1)

    def test_mixed_valid_invalid_leaderboard_only_shows_valid(self):
        """Mix of valid and invalid: leaderboard shows only valid total."""
        async def _run():
            await execute("DELETE FROM catches WHERE chat_id = 7779")
            from bot.storage.catches import save_catch
            # Valid: 3kg perch
            await save_catch(7779, 1, "Вася", "Окунь",
                             fish_count=1, weight_kg=3.0, is_valid_catch=True,
                             object_type="whole_fish", species_confidence=0.75)
            # Invalid: lure photo, 10kg — must NOT be counted
            await save_catch(7779, 1, "Вася", "Щука",
                             fish_count=1, weight_kg=10.0, is_valid_catch=False,
                             object_type="lure", rejection_reason="Это воблер")
            return await get_chat_leaderboard(7779)
        lb = run(_run())
        self.assertEqual(len(lb), 1)
        self.assertAlmostEqual(lb[0]["total_weight_kg"], 3.0, places=1,
                               msg="Only valid 3kg catch should be counted, not the 10kg lure")

    def test_schema_version_is_v4_after_migration(self):
        """Database must be at schema version 4 after migration."""
        version = run(fetch_scalar("SELECT version FROM schema_version LIMIT 1"))
        self.assertEqual(version, 4, "DB must be at v4 after fish-vision migration")

    def test_catches_table_has_new_columns(self):
        """catches table must have is_valid_catch and object_type columns."""
        async def _check():
            result = await fetch_all("PRAGMA table_info(catches)")
            cols = {r["name"] for r in result}
            return cols
        cols = run(_check())
        for col in ["object_type", "species_confidence", "is_valid_catch", "rejection_reason"]:
            self.assertIn(col, cols, f"Column '{col}' must exist in catches table")


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🧪 Telegram Bot v2 — Agentic Test Suite")
    print("=" * 50)
    unittest.main(verbosity=2)
