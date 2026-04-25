"""
test_intake_manifest.py — Unit tests for intake_telegram_manifest.py (U1).
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from intake_telegram_manifest import (  # noqa: E402
    LONG_CAPTION_THRESHOLD,
    _ALL_PHOTO_JPG_RE,
    _is_thumbnail,
    _parse_telegram_date,
    _sort_key,
    _write_manifest_summary,
    parse_html_file,
    run,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_html(messages: list[dict], title: str = "Test Export") -> str:
    """
    Build a minimal Telegram-export-style HTML page.

    Each message dict supports keys:
        id, date_title, from_name, photo_href, caption, joined
    """
    msg_blocks = []
    for msg in messages:
        joined = " joined" if msg.get("joined") else ""
        userpic = ""
        from_name_block = ""
        if not msg.get("joined") and msg.get("from_name"):
            userpic = f"""
      <div class="pull_left userpic_wrap">
       <div class="userpic userpic5" style="width: 42px">
        <div class="initials">X</div>
       </div>
      </div>"""
            from_name_block = f"""
       <div class="from_name">{msg["from_name"]}</div>"""

        photo_block = ""
        if msg.get("photo_href"):
            photo_block = f"""
       <div class="media_wrap clearfix">
        <a class="photo_wrap clearfix pull_left" href="{msg['photo_href']}">
         <img class="photo" src="{msg['photo_href'].replace('.jpg', '_thumb.jpg')}"/>
        </a>
       </div>"""

        caption_block = ""
        if msg.get("caption") is not None:
            caption_block = f"""
       <div class="text">{msg["caption"]}</div>"""

        date_title = msg.get("date_title", "01.01.2020 12:00:00 UTC+00:00")
        block = f"""
    <div class="message default clearfix{joined}" id="message{msg['id']}">
     {userpic}
     <div class="body">
      <div class="pull_right date details" title="{date_title}">12:00</div>
      {from_name_block}
      {photo_block}
      {caption_block}
      <div class="signature details">sender</div>
     </div>
    </div>"""
        msg_blocks.append(block)

    body = "\n".join(msg_blocks)
    return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html>
        <head><meta charset="utf-8"/><title>{title}</title></head>
        <body>
        <div class="page_wrap">
         <div class="page_body chat_page">
          <div class="history">
          {body}
          </div>
         </div>
        </div>
        </body>
        </html>
    """)


# ─── _parse_telegram_date ─────────────────────────────────────────────────────


def test_parse_date_utc_plus() -> None:
    epoch, iso = _parse_telegram_date("25.12.2017 19:47:37 UTC+03:00")
    assert iso == "2017-12-25T16:47:37Z"
    assert epoch == 1514220457


def test_parse_date_utc_zero() -> None:
    epoch, iso = _parse_telegram_date("01.01.2020 12:00:00 UTC+00:00")
    assert iso == "2020-01-01T12:00:00Z"
    assert epoch == 1577880000


def test_parse_date_utc_minus() -> None:
    epoch, iso = _parse_telegram_date("01.01.2020 12:00:00 UTC-05:00")
    assert epoch is not None
    assert iso is not None
    assert iso == "2020-01-01T17:00:00Z"


def test_parse_date_malformed() -> None:
    epoch, iso = _parse_telegram_date("not a date")
    assert epoch is None
    assert iso is None


# ─── _sort_key ────────────────────────────────────────────────────────────────


def test_sort_key_ordering() -> None:
    paths = [Path(n) for n in ["messages10.html", "messages2.html", "messages.html", "messages20.html"]]
    sorted_names = [p.name for p in sorted(paths, key=_sort_key)]
    assert sorted_names == ["messages.html", "messages2.html", "messages10.html", "messages20.html"]


# ─── parse_html_file ──────────────────────────────────────────────────────────


def test_happy_path_10_photos(tmp_path: Path) -> None:
    """10 photo messages → 10 JSONL records with all required fields."""
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()

    msgs = []
    for i in range(1, 11):
        fname = f"photos/photo_{i}@01-01-2020_12-00-00.jpg"
        (tmp_path / fname).touch()
        msgs.append({"id": 1000 + i, "from_name": "Alice", "photo_href": fname, "caption": f"cap{i}"})

    html_path = tmp_path / "messages.html"
    html_path.write_text(_make_html(msgs), encoding="utf-8")

    records: list[dict] = []
    seen: set[str] = set()
    result = parse_html_file(html_path, tmp_path, seen, records, set())

    assert result.added == 10
    assert result.dupe_skipped == 0
    assert result.thumbs_skipped == 0
    assert len(records) == 10
    required_fields = {"filename", "msg_id", "timestamp", "source", "license"}
    for rec in records:
        assert required_fields.issubset(rec.keys()), f"Missing fields in {rec}"
    assert records[0]["source"] == "telegram_private_2026-04-24"
    assert records[0]["license"] == "private_training_only"


def test_multi_file_no_overlap(tmp_path: Path) -> None:
    """messages.html + messages2.html with distinct photos → combined, no duplicates."""
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()

    msgs1 = []
    for i in range(1, 6):
        fname = f"photos/photo_{i}@01-01-2020_12-00-00.jpg"
        (tmp_path / fname).touch()
        msgs1.append({"id": 100 + i, "from_name": "Alice", "photo_href": fname})

    msgs2 = []
    for i in range(6, 11):
        fname = f"photos/photo_{i}@01-01-2020_12-00-00.jpg"
        (tmp_path / fname).touch()
        msgs2.append({"id": 200 + i, "from_name": "Bob", "photo_href": fname})

    (tmp_path / "messages.html").write_text(_make_html(msgs1), encoding="utf-8")
    (tmp_path / "messages2.html").write_text(_make_html(msgs2), encoding="utf-8")

    records: list[dict] = []
    seen: set[str] = set()
    all_refs: set[str] = set()
    for html_path in sorted((tmp_path / f for f in ["messages.html", "messages2.html"])):
        parse_html_file(html_path, tmp_path, seen, records, all_refs)

    assert len(records) == 10
    assert len({r["filename"] for r in records}) == 10


def test_no_caption_message(tmp_path: Path) -> None:
    """Photo with no text → record created, caption field is empty string."""
    (tmp_path / "photos").mkdir()
    fname = "photos/photo_1@01-01-2020_12-00-00.jpg"
    (tmp_path / fname).touch()
    msg = [{"id": 500, "from_name": "X", "photo_href": fname}]  # no caption key

    html = _make_html(msg)
    html_path = tmp_path / "messages.html"
    html_path.write_text(html, encoding="utf-8")

    records: list[dict] = []
    parse_html_file(html_path, tmp_path, set(), records, set())
    assert len(records) == 1
    assert records[0]["caption"] == ""


def test_text_only_message_skipped(tmp_path: Path) -> None:
    """Text-only message (no photo href) → not in output."""
    msg = [{"id": 600, "from_name": "X", "photo_href": None, "caption": "just text"}]
    html_path = tmp_path / "messages.html"
    html_path.write_text(_make_html(msg), encoding="utf-8")

    records: list[dict] = []
    parse_html_file(html_path, tmp_path, set(), records, set())
    assert len(records) == 0


def test_duplicate_filename_deduplicated(tmp_path: Path) -> None:
    """Same photo href in two HTML files → only one record written."""
    (tmp_path / "photos").mkdir()
    fname = "photos/photo_1@01-01-2020_12-00-00.jpg"
    (tmp_path / fname).touch()

    msg = [{"id": 700, "from_name": "X", "photo_href": fname}]
    for name in ["messages.html", "messages2.html"]:
        (tmp_path / name).write_text(_make_html(msg), encoding="utf-8")

    records: list[dict] = []
    seen: set[str] = set()
    for name in ["messages.html", "messages2.html"]:
        parse_html_file(tmp_path / name, tmp_path, seen, records, set())

    assert len(records) == 1


def test_malformed_date_sets_null(tmp_path: Path) -> None:
    """Malformed date string → timestamp=None, record still created."""
    (tmp_path / "photos").mkdir()
    fname = "photos/photo_1@01-01-2020_12-00-00.jpg"
    (tmp_path / fname).touch()

    msg = [{"id": 800, "from_name": "X", "photo_href": fname,
             "date_title": "not-a-date"}]
    html_path = tmp_path / "messages.html"
    html_path.write_text(_make_html(msg), encoding="utf-8")

    records: list[dict] = []
    parse_html_file(html_path, tmp_path, set(), records, set())
    assert len(records) == 1
    assert records[0]["timestamp"] is None


def test_missing_photo_file_sets_parse_error(tmp_path: Path) -> None:
    """href pointing to non-existent file → parse_error=True, record still written."""
    # Do NOT create the file
    msg = [{"id": 900, "from_name": "X", "photo_href": "photos/missing.jpg"}]
    html_path = tmp_path / "messages.html"
    html_path.write_text(_make_html(msg), encoding="utf-8")

    records: list[dict] = []
    parse_html_file(html_path, tmp_path, set(), records, set())
    assert len(records) == 1
    assert records[0]["parse_error"] is True


def test_joined_messages_inherit_sender(tmp_path: Path) -> None:
    """Joined messages (no from_name) inherit the last known sender."""
    (tmp_path / "photos").mkdir()
    for i in range(1, 3):
        (tmp_path / f"photos/photo_{i}@01-01-2020_12-00-00.jpg").touch()

    msgs = [
        {"id": 1, "from_name": "Alice", "photo_href": "photos/photo_1@01-01-2020_12-00-00.jpg"},
        {"id": 2, "joined": True, "photo_href": "photos/photo_2@01-01-2020_12-00-00.jpg"},
    ]
    html_path = tmp_path / "messages.html"
    html_path.write_text(_make_html(msgs), encoding="utf-8")

    records: list[dict] = []
    parse_html_file(html_path, tmp_path, set(), records, set())
    assert len(records) == 2
    assert records[1]["sender_name"] == "Alice"


# ─── run() integration ────────────────────────────────────────────────────────


def test_run_writes_jsonl(tmp_path: Path) -> None:
    """run() writes a valid JSONL manifest and returns correct count."""
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    for i in range(1, 4):
        (tmp_path / f"photos/photo_{i}@01-01-2020_12-00-00.jpg").touch()

    msgs = [
        {"id": 1000 + i, "from_name": "Test",
         "photo_href": f"photos/photo_{i}@01-01-2020_12-00-00.jpg"}
        for i in range(1, 4)
    ]
    (tmp_path / "messages.html").write_text(_make_html(msgs), encoding="utf-8")

    output_dir = tmp_path / "out"
    n = run(tmp_path, output_dir)

    assert n == 3
    manifest = output_dir / "manifest.jsonl"
    assert manifest.exists()
    lines = manifest.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        rec = json.loads(line)
        assert rec["source"] == "telegram_private_2026-04-24"
        assert rec["license"] == "private_training_only"


# ─── _is_thumbnail ────────────────────────────────────────────────────────────


def test_is_thumbnail_simple() -> None:
    assert _is_thumbnail("photos/photo_1_thumb.jpg") is True


def test_is_thumbnail_numbered() -> None:
    assert _is_thumbnail("photos/photo_1_thumb (1).jpg") is True


def test_is_thumbnail_numbered_higher() -> None:
    assert _is_thumbnail("photos/photo_1_thumb (12).jpg") is True


def test_is_thumbnail_main_photo_not_thumb() -> None:
    assert _is_thumbnail("photos/photo_1@01-01-2020_12-00-00.jpg") is False


# ─── thumbnail skipping in parse_html_file ────────────────────────────────────


def _make_html_with_href(href: str, msg_id: int = 1) -> str:
    """Minimal HTML with a photo_wrap anchor pointing to `href`."""
    return _make_html([{"id": msg_id, "from_name": "X", "photo_href": href}])


def test_simple_thumb_skipped(tmp_path: Path) -> None:
    """*_thumb.jpg href → not added to manifest, counted as thumbs_skipped."""
    thumb_href = "photos/photo_1_thumb.jpg"
    (tmp_path / "photos").mkdir()
    (tmp_path / thumb_href).touch()

    html_path = tmp_path / "messages.html"
    html_path.write_text(_make_html_with_href(thumb_href), encoding="utf-8")

    records: list[dict] = []
    result = parse_html_file(html_path, tmp_path, set(), records, set())

    assert result.added == 0
    assert result.thumbs_skipped == 1
    assert len(records) == 0


def test_numbered_thumb_skipped(tmp_path: Path) -> None:
    """*_thumb (1).jpg href → not added to manifest, counted as thumbs_skipped."""
    thumb_href = "photos/photo_1_thumb (1).jpg"
    (tmp_path / "photos").mkdir()
    (tmp_path / thumb_href).touch()

    html_path = tmp_path / "messages.html"
    html_path.write_text(_make_html_with_href(thumb_href), encoding="utf-8")

    records: list[dict] = []
    result = parse_html_file(html_path, tmp_path, set(), records, set())

    assert result.added == 0
    assert result.thumbs_skipped == 1
    assert len(records) == 0


def test_mixed_main_and_thumbs(tmp_path: Path) -> None:
    """2 main photos + 1 plain thumb + 1 numbered thumb → 2 records, 2 thumbs_skipped."""
    (tmp_path / "photos").mkdir()
    main1 = "photos/photo_1@01-01-2020.jpg"
    main2 = "photos/photo_2@01-01-2020.jpg"
    thumb1 = "photos/photo_1_thumb.jpg"
    thumb2 = "photos/photo_1_thumb (1).jpg"
    for f in [main1, main2, thumb1, thumb2]:
        (tmp_path / f).touch()

    msgs = [
        {"id": 1, "from_name": "X", "photo_href": main1},
        {"id": 2, "from_name": "X", "photo_href": thumb1},
        {"id": 3, "from_name": "X", "photo_href": thumb2},
        {"id": 4, "from_name": "X", "photo_href": main2},
    ]
    html_path = tmp_path / "messages.html"
    html_path.write_text(_make_html(msgs), encoding="utf-8")

    records: list[dict] = []
    result = parse_html_file(html_path, tmp_path, set(), records, set())

    assert result.jpg_refs == 4
    assert result.thumbs_skipped == 2
    assert result.dupe_skipped == 0
    assert result.added == 2
    assert len(records) == 2


def test_parse_result_dupe_counter(tmp_path: Path) -> None:
    """Duplicate main photo across two files → dupe_skipped=1 in second call."""
    (tmp_path / "photos").mkdir()
    fname = "photos/photo_1@01-01-2020.jpg"
    (tmp_path / fname).touch()

    msg = [{"id": 10, "from_name": "X", "photo_href": fname}]
    (tmp_path / "messages.html").write_text(_make_html(msg), encoding="utf-8")
    (tmp_path / "messages2.html").write_text(_make_html(msg), encoding="utf-8")

    seen: set[str] = set()
    records: list[dict] = []
    r1 = parse_html_file(tmp_path / "messages.html", tmp_path, seen, records, set())
    r2 = parse_html_file(tmp_path / "messages2.html", tmp_path, seen, records, set())

    assert r1.added == 1 and r1.dupe_skipped == 0
    assert r2.added == 0 and r2.dupe_skipped == 1


# ─── manifest_summary.json ────────────────────────────────────────────────────


def test_run_writes_manifest_summary(tmp_path: Path) -> None:
    """run() writes manifest_summary.json alongside manifest.jsonl."""
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    for i in range(1, 4):
        (tmp_path / f"photos/photo_{i}@01-01-2020_12-00-00.jpg").touch()

    msgs = [
        {"id": 1000 + i, "from_name": "Test",
         "photo_href": f"photos/photo_{i}@01-01-2020_12-00-00.jpg",
         "caption": f"cap{i}"}
        for i in range(1, 4)
    ]
    (tmp_path / "messages.html").write_text(_make_html(msgs), encoding="utf-8")

    output_dir = tmp_path / "out"
    run(tmp_path, output_dir)

    summary_path = output_dir / "manifest_summary.json"
    assert summary_path.exists(), "manifest_summary.json was not written"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    required_keys = {
        "all_html_jpg_refs", "raw_thumbnail_jpg_refs", "raw_non_thumbnail_jpg_refs",
        "parser_candidate_photo_refs", "duplicate_parser_candidate_refs_skipped",
        "final_manifest_records",
        "caption_count", "long_caption_count", "max_caption_length",
        "source", "license", "usage_scope",
    }
    assert required_keys.issubset(summary.keys()), (
        f"Missing keys: {required_keys - summary.keys()}"
    )
    assert summary["final_manifest_records"] == 3
    assert summary["caption_count"] == 3
    assert summary["source"] == "telegram_private_2026-04-24"
    assert summary["license"] == "private_training_only"


def test_manifest_summary_no_pii_fields(tmp_path: Path) -> None:
    """manifest_summary.json must not contain caption text or sender_name."""
    records = [
        {"caption": "This is a personal caption", "sender_name": "John Doe",
         "filename": "photos/x.jpg"},
    ]
    _write_manifest_summary(
        output_dir=tmp_path,
        all_html_jpg_refs=1,
        raw_thumbnail_jpg_refs=0,
        raw_non_thumbnail_jpg_refs=1,
        parser_candidate_photo_refs=1,
        duplicate_parser_candidate_refs_skipped=0,
        final_manifest_records=1,
        records=records,
    )
    summary = json.loads((tmp_path / "manifest_summary.json").read_text(encoding="utf-8"))
    # Aggregate count keys are fine; raw text and name fields must be absent
    assert "sender_name" not in summary
    assert "John Doe" not in summary.values()
    assert "This is a personal caption" not in summary.values()
    # Ensure no key stores the raw caption string or sender name
    for v in summary.values():
        assert not isinstance(v, str) or v not in {"This is a personal caption", "John Doe"}


def test_manifest_summary_caption_counts(tmp_path: Path) -> None:
    """Caption aggregate counts are computed correctly without exposing text."""
    long_cap = "x" * (LONG_CAPTION_THRESHOLD + 1)
    records = [
        {"caption": "short", "sender_name": "A", "filename": "photos/1.jpg"},
        {"caption": long_cap, "sender_name": "B", "filename": "photos/2.jpg"},
        {"caption": "", "sender_name": "C", "filename": "photos/3.jpg"},
    ]
    _write_manifest_summary(
        output_dir=tmp_path,
        all_html_jpg_refs=3,
        raw_thumbnail_jpg_refs=0,
        raw_non_thumbnail_jpg_refs=3,
        parser_candidate_photo_refs=3,
        duplicate_parser_candidate_refs_skipped=0,
        final_manifest_records=3,
        records=records,
    )
    summary = json.loads((tmp_path / "manifest_summary.json").read_text(encoding="utf-8"))
    assert summary["caption_count"] == 2          # "short" + long_cap; "" is not counted
    assert summary["long_caption_count"] == 1     # only long_cap exceeds threshold
    assert summary["max_caption_length"] == len(long_cap)


# ─── raw all_jpg_refs scan ────────────────────────────────────────────────────


def test_raw_scan_counts_img_src_thumbnails(tmp_path: Path) -> None:
    """
    Raw scan (_ALL_PHOTO_JPG_RE) finds photos/*.jpg in both <a href> and <img src>.

    The helper _make_html embeds the main photo in <a href> and its _thumb.jpg
    variant in <img src>.  Both must appear in all_jpg_refs; only the img src one
    is a thumbnail.
    """
    (tmp_path / "photos").mkdir()
    main_href = "photos/photo_1@01-01-2020.jpg"
    (tmp_path / main_href).touch()

    html_path = tmp_path / "messages.html"
    html_path.write_text(_make_html_with_href(main_href), encoding="utf-8")

    all_refs: set[str] = set()
    records: list[dict] = []
    parse_html_file(html_path, tmp_path, set(), records, all_refs)

    # main photo ref captured from <a href>
    assert main_href in all_refs
    # thumbnail ref captured from <img src> (auto-generated by _make_html)
    thumb_href = main_href.replace(".jpg", "_thumb.jpg")
    assert thumb_href in all_refs

    # thumbnail identified correctly by _is_thumbnail
    assert _is_thumbnail(thumb_href) is True
    assert _is_thumbnail(main_href) is False


def test_raw_scan_thumbnail_variants(tmp_path: Path) -> None:
    """
    Raw scan correctly categorises both *_thumb.jpg and *_thumb (N).jpg as thumbnails.
    """
    html = (
        '<html><body>'
        '<a href="photos/photo_1@2020.jpg">'
        '<img src="photos/photo_1@2020_thumb.jpg"/>'
        '</a>'
        '<a href="photos/photo_2@2020_thumb.jpg">'
        '<img src="photos/photo_2@2020_thumb (1).jpg"/>'
        '</a>'
        '</body></html>'
    )
    html_path = tmp_path / "messages.html"
    html_path.write_text(html, encoding="utf-8")

    all_refs: set[str] = set()
    parse_html_file(html_path, tmp_path, set(), [], all_refs)

    thumbs = {r for r in all_refs if _is_thumbnail(r)}
    mains = {r for r in all_refs if not _is_thumbnail(r)}

    # plain _thumb.jpg and space-paren variant both detected
    assert any("_thumb.jpg" in r and " (" not in r for r in thumbs), \
        f"Expected plain _thumb.jpg in thumbs, got {thumbs}"
    assert any("_thumb (1).jpg" in r for r in thumbs), \
        f"Expected _thumb (1).jpg in thumbs, got {thumbs}"
    assert len(mains) == 1  # only photo_1@2020.jpg is a main photo


# ─── summary counter arithmetic ──────────────────────────────────────────────


def test_summary_arithmetic_holds(tmp_path: Path) -> None:
    """parser_candidate_photo_refs - duplicate_parser_candidate_refs_skipped == final_manifest_records."""
    (tmp_path / "photos").mkdir()
    for i in range(1, 4):
        (tmp_path / f"photos/photo_{i}@01-01-2020_12-00-00.jpg").touch()

    # file 1: photos 1, 2  |  file 2: photo 2 (dup), photo 3
    msgs1 = [
        {"id": 1, "from_name": "A", "photo_href": "photos/photo_1@01-01-2020_12-00-00.jpg"},
        {"id": 2, "from_name": "A", "photo_href": "photos/photo_2@01-01-2020_12-00-00.jpg"},
    ]
    msgs2 = [
        {"id": 3, "from_name": "B", "photo_href": "photos/photo_2@01-01-2020_12-00-00.jpg"},
        {"id": 4, "from_name": "B", "photo_href": "photos/photo_3@01-01-2020_12-00-00.jpg"},
    ]
    (tmp_path / "messages.html").write_text(_make_html(msgs1), encoding="utf-8")
    (tmp_path / "messages2.html").write_text(_make_html(msgs2), encoding="utf-8")

    output_dir = tmp_path / "out"
    run(tmp_path, output_dir)

    summary = json.loads((output_dir / "manifest_summary.json").read_text(encoding="utf-8"))
    assert (
        summary["parser_candidate_photo_refs"]
        - summary["duplicate_parser_candidate_refs_skipped"]
        == summary["final_manifest_records"]
    ), "Arithmetic invariant violated"
    assert summary["final_manifest_records"] == 3      # 3 unique photos
    assert summary["duplicate_parser_candidate_refs_skipped"] == 1  # photo_2 seen twice
    assert summary["parser_candidate_photo_refs"] == 4  # 4 parser-selected non-thumb hrefs


def test_raw_thumbnail_counting_in_summary(tmp_path: Path) -> None:
    """raw_thumbnail_jpg_refs counts both *_thumb.jpg and *_thumb (N).jpg from the raw scan."""
    (tmp_path / "photos").mkdir()
    (tmp_path / "photos/photo_1@2020.jpg").touch()
    (tmp_path / "photos/photo_2@2020.jpg").touch()

    # HTML with explicit thumb variants in img src (no proper message divs, so parser finds 0 records)
    html = (
        "<html><body>"
        '<a class="photo_wrap" href="photos/photo_1@2020.jpg">'
        '<img src="photos/photo_1@2020_thumb.jpg"/>'
        "</a>"
        '<a class="photo_wrap" href="photos/photo_2@2020.jpg">'
        '<img src="photos/photo_2@2020_thumb (3).jpg"/>'
        "</a>"
        "</body></html>"
    )
    (tmp_path / "messages.html").write_text(html, encoding="utf-8")

    output_dir = tmp_path / "out"
    run(tmp_path, output_dir)

    summary = json.loads((output_dir / "manifest_summary.json").read_text(encoding="utf-8"))
    # Raw scan: 2 main + 1 plain thumb + 1 numbered thumb = 4 unique refs
    assert summary["all_html_jpg_refs"] == 4
    assert summary["raw_thumbnail_jpg_refs"] == 2   # both _thumb.jpg and _thumb (N).jpg counted
    assert summary["raw_non_thumbnail_jpg_refs"] == 2
