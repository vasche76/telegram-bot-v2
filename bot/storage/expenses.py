"""
Expense tracking and debt settlement.

FIXES vs original:
- calculate_debts: when split_among is NULL, use the session's declared participants
  (not list(paid.keys()) which only includes people who have paid so far — wrong).
- add_expense: auto-update session participants when a new payer is seen.
- create_session / set_session_participants: explicit participant management.
- add_expense: photo_file_id for receipt deduplication.
- get_session_by_receipt: check if a receipt photo was already added.
"""

import json
from typing import Optional
from bot.storage.database import execute, fetch_all, fetch_one, fetch_scalar
from bot.utils.logging import get_logger

log = get_logger("storage.expenses")


async def create_session(
    chat_id: int,
    session_id: str,
    participants: Optional[list[str]] = None,
) -> None:
    """Create a new expense session with optional participant list."""
    parts_json = json.dumps(participants, ensure_ascii=False) if participants else None
    await execute(
        "INSERT OR IGNORE INTO expense_sessions (chat_id, session_id, participants) VALUES (?, ?, ?)",
        (chat_id, session_id, parts_json),
    )


async def set_session_participants(
    chat_id: int, session_id: str, participants: list[str]
) -> None:
    """Set (or update) the participant list for a session."""
    parts_json = json.dumps(participants, ensure_ascii=False)
    await execute(
        "UPDATE expense_sessions SET participants = ? WHERE chat_id = ? AND session_id = ?",
        (parts_json, chat_id, session_id),
    )


async def get_session_participants(chat_id: int, session_id: str) -> list[str]:
    """Get the declared participant list for a session.
    Falls back to unique payers if no explicit list is set."""
    row = await fetch_one(
        "SELECT participants FROM expense_sessions WHERE chat_id = ? AND session_id = ?",
        (chat_id, session_id),
    )
    if row and row.get("participants"):
        try:
            parts = json.loads(row["participants"])
            if parts:
                return parts
        except Exception:
            pass
    # Fall back to distinct payers in this session
    rows = await fetch_all(
        "SELECT DISTINCT paid_by_name FROM expenses WHERE chat_id = ? AND session_id = ?",
        (chat_id, session_id),
    )
    return [r["paid_by_name"] for r in rows] if rows else []


async def get_active_session(chat_id: int) -> Optional[dict]:
    """Get the active expense session for a chat."""
    return await fetch_one(
        "SELECT * FROM expense_sessions "
        "WHERE chat_id = ? AND status = 'active' "
        "ORDER BY created_at DESC LIMIT 1",
        (chat_id,),
    )


async def close_session(chat_id: int, session_id: str) -> None:
    """Close an expense session."""
    await execute(
        "UPDATE expense_sessions SET status = 'closed', closed_at = CURRENT_TIMESTAMP "
        "WHERE chat_id = ? AND session_id = ?",
        (chat_id, session_id),
    )


async def add_expense(
    chat_id: int,
    session_id: str,
    paid_by_user_id: Optional[int],
    paid_by_name: str,
    amount: float,
    description: str = "",
    merchant: str = "",
    receipt_date: str = "",
    currency: str = "RUB",
    split_among: Optional[list[str]] = None,
    photo_file_id: Optional[str] = None,
) -> int:
    """Add an expense to a session.

    Auto-adds the payer to the session's participant list so future
    NULL split_among expenses are divided correctly.
    """
    split_json = json.dumps(split_among, ensure_ascii=False) if split_among else None

    # If photo_file_id is provided, check for duplicate receipt
    if photo_file_id:
        existing = await fetch_one(
            "SELECT id FROM expenses WHERE chat_id = ? AND session_id = ? AND photo_file_id = ?",
            (chat_id, session_id, photo_file_id),
        )
        if existing:
            log.warning(
                f"Duplicate receipt detected (file_id={photo_file_id[:20]}...). Skipping."
            )
            return existing["id"]

    cursor = await execute(
        """INSERT INTO expenses (chat_id, session_id, paid_by_user_id, paid_by_name,
           amount, description, merchant, receipt_date, currency, split_among, photo_file_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chat_id, session_id, paid_by_user_id, paid_by_name, amount,
            description, merchant, receipt_date, currency, split_json, photo_file_id,
        ),
    )

    # Auto-update session participants to include the payer
    current_parts = await get_session_participants(chat_id, session_id)
    if paid_by_name and paid_by_name not in current_parts:
        current_parts.append(paid_by_name)
        await set_session_participants(chat_id, session_id, current_parts)

    log.info(
        f"Expense added: {paid_by_name} paid {amount} {currency} for '{description}'"
    )
    return cursor.lastrowid


async def get_session_expenses(chat_id: int, session_id: str) -> list[dict]:
    """Get all expenses in a session."""
    return await fetch_all(
        "SELECT * FROM expenses WHERE chat_id = ? AND session_id = ? ORDER BY created_at",
        (chat_id, session_id),
    )


async def calculate_debts(chat_id: int, session_id: str) -> dict:
    """Calculate who owes whom in a session.

    FIX: When split_among is NULL (split equally among all), uses the
    session's declared participants — NOT just the payers seen so far.
    Previously the code used list(paid.keys()) which was wrong when
    some participants hadn't paid yet.
    """
    expenses = await get_session_expenses(chat_id, session_id)
    if not expenses:
        return {"total": 0, "per_person": {}, "settlements": []}

    # Get the full participant list for this session
    all_participants = await get_session_participants(chat_id, session_id)

    paid: dict[str, float] = {}
    owed: dict[str, float] = {}
    total = 0.0

    for exp in expenses:
        payer = exp["paid_by_name"]
        amount = float(exp["amount"])
        total += amount
        paid[payer] = paid.get(payer, 0.0) + amount

        if exp["split_among"]:
            try:
                split = json.loads(exp["split_among"])
            except Exception:
                split = all_participants or [payer]
        else:
            # NULL split_among = split equally among ALL session participants
            split = all_participants if all_participants else [payer]

        if not split:
            split = [payer]

        share = amount / len(split)
        for person in split:
            owed[person] = owed.get(person, 0.0) + share

    # Include people who only owe (didn't pay) in the balances
    for person in all_participants:
        if person not in paid and person not in owed:
            owed[person] = 0.0

    all_people = set(list(paid.keys()) + list(owed.keys()))
    balances: dict[str, float] = {}
    for person in all_people:
        balances[person] = paid.get(person, 0.0) - owed.get(person, 0.0)

    # Greedy settlement minimisation
    settlements = []
    debtors = {p: -b for p, b in balances.items() if b < -0.01}
    creditors = {p: b for p, b in balances.items() if b > 0.01}

    for debtor in sorted(debtors, key=lambda p: -debtors[p]):
        debt = debtors[debtor]
        for creditor in sorted(creditors, key=lambda p: -creditors[p]):
            credit = creditors[creditor]
            if debt <= 0.01 or credit <= 0.01:
                continue
            transfer = min(debt, credit)
            settlements.append({
                "from": debtor,
                "to": creditor,
                "amount": round(transfer, 2),
            })
            debt -= transfer
            creditors[creditor] = credit - transfer
        debtors[debtor] = debt

    return {
        "total": round(total, 2),
        "per_person": {p: round(v, 2) for p, v in balances.items()},
        "settlements": settlements,
    }


async def is_receipt_already_added(
    chat_id: int, session_id: str, photo_file_id: str
) -> bool:
    """Check if a receipt photo was already added to prevent duplicates."""
    row = await fetch_one(
        "SELECT id FROM expenses WHERE chat_id = ? AND session_id = ? AND photo_file_id = ?",
        (chat_id, session_id, photo_file_id),
    )
    return row is not None
