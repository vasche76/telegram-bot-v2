#!/usr/bin/env python3
"""
intake_telegram_manifest.py — Parse Telegram export HTML → manifest.jsonl.

Reads all messages*.html files from the export directory, extracts one record
per unique photo attachment, and writes data/intake_meta/tg_2026-04-24/manifest.jsonl.

source=telegram_private_2026-04-24, license=private_training_only

Usage:
    python3 scripts/intake_telegram_manifest.py [--export-dir PATH] [--output-dir PATH]

Output record fields:
    filename      — relative path within export (e.g. "photos/photo_1@....jpg")
    msg_id        — integer message ID
    timestamp     — Unix epoch (int) or null if unparseable
    timestamp_iso — ISO-8601 UTC string or null
    sender_name   — sender display name or null for joined messages
    caption       — text caption (may be empty string)
    source        — "telegram_private_2026-04-24"
    license       — "private_training_only"
    parse_error   — true if photo file path does not exist in export
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from calendar import timegm
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parent.parent

SOURCE_TAG = "telegram_private_2026-04-24"
LICENSE_TAG = "private_training_only"

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# Date format used by Telegram: "25.12.2017 19:47:37 UTC+03:00"
_DATE_RE = re.compile(
    r"^(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2}):(\d{2})\s+UTC([+-])(\d{2}):(\d{2})$"
)


def _parse_telegram_date(raw: str) -> tuple[int | None, str | None]:
    """Return (unix_epoch, iso_utc) from a Telegram date title string, or (None, None)."""
    m = _DATE_RE.match(raw.strip())
    if not m:
        return None, None
    day, mon, year, hh, mm, ss, sign, tz_hh, tz_mm = m.groups()
    tz_offset_secs = (int(tz_hh) * 60 + int(tz_mm)) * 60
    if sign == "-":
        tz_offset_secs = -tz_offset_secs
    naive_utc = datetime(
        int(year), int(mon), int(day),
        int(hh), int(mm), int(ss),
        tzinfo=timezone.utc,
    )
    # Adjust for the local timezone offset recorded in the export
    from datetime import timedelta  # noqa: PLC0415
    dt_utc = naive_utc - timedelta(seconds=tz_offset_secs)
    epoch = int(timegm(dt_utc.timetuple()))
    iso = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    return epoch, iso


# Matches _thumb.jpg  AND  _thumb (N).jpg  (space-paren-digit-paren variants)
_THUMB_RE = re.compile(r"_thumb(?:\s*\(\d+\))?\.jpg$", re.IGNORECASE)


def _is_thumbnail(href: str) -> bool:
    """Return True for any thumbnail variant: *_thumb.jpg or *_thumb (N).jpg."""
    return bool(_THUMB_RE.search(href))


def _is_photo_href(href: str | None) -> bool:
    if not href:
        return False
    return (
        href.startswith("photos/")
        and href.lower().endswith(".jpg")
        and not _is_thumbnail(href)
    )


class ParseResult(NamedTuple):
    messages_scanned: int
    jpg_refs: int        # total photos/*.jpg hrefs found (main + thumbnails)
    thumbs_skipped: int  # thumbnail hrefs excluded
    dupe_skipped: int    # main refs excluded as cross-file duplicates
    added: int           # records appended to manifest


LONG_CAPTION_THRESHOLD = 100  # characters; captions exceeding this counted as "long"


def _write_manifest_summary(
    output_dir: Path,
    total_jpg_refs: int,
    main_photo_refs: int,
    thumbnails_skipped: int,
    duplicate_main_refs_skipped: int,
    final_manifest_records: int,
    records: list[dict],
) -> Path:
    """
    Write a privacy-safe aggregate summary (no captions, no sender names).

    Returns the path of the written file.
    """
    captions = [rec.get("caption", "") or "" for rec in records]
    caption_count = sum(1 for c in captions if c)
    long_caption_count = sum(1 for c in captions if len(c) > LONG_CAPTION_THRESHOLD)
    max_caption_length = max((len(c) for c in captions), default=0)

    summary = {
        "total_jpg_refs": total_jpg_refs,
        "main_photo_refs": main_photo_refs,
        "thumbnails_skipped": thumbnails_skipped,
        "duplicate_main_refs_skipped": duplicate_main_refs_skipped,
        "final_manifest_records": final_manifest_records,
        "caption_count": caption_count,
        "long_caption_count": long_caption_count,
        "max_caption_length": max_caption_length,
        "source": SOURCE_TAG,
        "license": LICENSE_TAG,
        "usage_scope": "private_training_only",
    }

    summary_path = output_dir / "manifest_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    return summary_path


def _sort_key(p: Path) -> tuple[int, str]:
    """Sort messages*.html numerically: messages.html=1, messages2.html=2, ..."""
    m = re.match(r"messages(\d*)\.html$", p.name, re.IGNORECASE)
    if not m:
        return (999999, p.name)
    n = int(m.group(1)) if m.group(1) else 1
    return (n, p.name)


def parse_html_file(
    html_path: Path,
    export_dir: Path,
    seen_filenames: set[str],
    records: list[dict],
) -> ParseResult:
    """
    Parse one messages*.html file and append new photo records to `records`.

    Returns a ParseResult namedtuple:
        (messages_scanned, jpg_refs, thumbs_skipped, dupe_skipped, added)
    """
    log.info("Parsing %s", html_path.name)
    text = html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(text, "html.parser")

    # All message divs — both leading ("message default clearfix") and
    # continuation ("message default clearfix joined").
    all_msgs = [
        d for d in soup.find_all("div")
        if d.get("class") and "message" in d["class"] and "default" in d["class"]
    ]

    jpg_refs = 0
    thumbs_skipped = 0
    dupe_skipped = 0
    added = 0

    current_sender: str | None = None

    for msg_div in all_msgs:
        # Track sender across joined messages (continuation msgs have no from_name)
        fn_div = msg_div.find("div", class_="from_name")
        if fn_div:
            current_sender = fn_div.get_text(strip=True) or None

        # Photo attachment?
        photo_a = msg_div.find("a", class_="photo_wrap")
        href: str | None = photo_a.get("href") if photo_a else None

        if not href or not href.startswith("photos/") or not href.lower().endswith(".jpg"):
            continue

        jpg_refs += 1

        if _is_thumbnail(href):
            thumbs_skipped += 1
            continue

        filename: str = href  # relative: "photos/photo_NNN@....jpg"

        # Deduplicate: same filename appearing in multiple HTML files
        if filename in seen_filenames:
            dupe_skipped += 1
            continue
        seen_filenames.add(filename)

        # Message ID
        raw_id: str = msg_div.get("id", "")
        try:
            msg_id = int(raw_id.replace("message", ""))
        except ValueError:
            msg_id = None

        # Timestamp
        date_div = msg_div.find("div", class_="date")
        raw_date: str | None = date_div.get("title") if date_div else None
        if raw_date:
            epoch, iso = _parse_telegram_date(raw_date)
            if epoch is None:
                log.warning("Unparseable date in %s msg %s: %r", html_path.name, raw_id, raw_date)
        else:
            epoch, iso = None, None

        # Caption
        text_div = msg_div.find("div", class_="text")
        caption = text_div.get_text(strip=True) if text_div else ""

        # Existence check
        photo_path = export_dir / filename
        parse_error = not photo_path.exists()

        records.append({
            "filename": filename,
            "msg_id": msg_id,
            "timestamp": epoch,
            "timestamp_iso": iso,
            "sender_name": current_sender,
            "caption": caption,
            "source": SOURCE_TAG,
            "license": LICENSE_TAG,
            "parse_error": parse_error,
        })
        added += 1

    return ParseResult(len(all_msgs), jpg_refs, thumbs_skipped, dupe_skipped, added)


def run(export_dir: Path, output_dir: Path) -> int:
    """
    Main pipeline entry point. Returns number of records written.
    """
    if not export_dir.exists():
        log.error("Export directory not found: %s", export_dir)
        sys.exit(1)

    html_files = sorted(
        [f for f in export_dir.glob("messages*.html")],
        key=_sort_key,
    )
    if not html_files:
        log.error("No messages*.html files found in %s", export_dir)
        sys.exit(1)
    log.info("Found %d HTML file(s): %s … %s", len(html_files), html_files[0].name, html_files[-1].name)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    records: list[dict] = []
    seen_filenames: set[str] = set()
    total_scanned = 0
    total_jpg_refs = 0
    total_thumbs_skipped = 0
    total_dupe_skipped = 0

    for html_path in html_files:
        result = parse_html_file(html_path, export_dir, seen_filenames, records)
        total_scanned += result.messages_scanned
        total_jpg_refs += result.jpg_refs
        total_thumbs_skipped += result.thumbs_skipped
        total_dupe_skipped += result.dupe_skipped

    main_refs = total_jpg_refs - total_thumbs_skipped

    log.info("Messages scanned:          %d", total_scanned)
    log.info("Total JPG refs:            %d", total_jpg_refs)
    log.info("  Main photo refs:         %d", main_refs)
    log.info("  Thumbnails skipped:      %d", total_thumbs_skipped)
    log.info("  Duplicate main refs:     %d", total_dupe_skipped)
    log.info("Manifest records written:  %d", len(records))

    with manifest_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info("Manifest written → %s", manifest_path)

    summary_path = _write_manifest_summary(
        output_dir=output_dir,
        total_jpg_refs=total_jpg_refs,
        main_photo_refs=main_refs,
        thumbnails_skipped=total_thumbs_skipped,
        duplicate_main_refs_skipped=total_dupe_skipped,
        final_manifest_records=len(records),
        records=records,
    )
    log.info("Summary written  → %s", summary_path)

    return len(records)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parse Telegram export HTML files into a photo manifest (JSONL).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--export-dir",
        default=str(Path.home() / "Downloads" / "Telegram Desktop" / "ChatExport_2026-04-24"),
        help="Path to the Telegram export directory (read-only).",
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "data" / "intake_meta" / "tg_2026-04-24"),
        help="Directory where manifest.jsonl will be written.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    export_dir = Path(args.export_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    n = run(export_dir, output_dir)
    print(f"[OK] manifest.jsonl written: {n} records → {output_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
