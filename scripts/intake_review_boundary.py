#!/usr/bin/env python3
"""
intake_review_boundary.py — Generate a local-only HTML contact sheet for manual
inspection of perceptual dedup boundary clusters.

Reads dedup_clusters.jsonl and generates an HTML file displaying each boundary
cluster (hamming_distance == 8 by default) as side-by-side images with radio-button
decision controls. The reviewer marks each cluster KEEP_DEDUP / FALSE_POSITIVE /
UNSURE (or MIXED for multi-member clusters) and clicks "Export Decisions" to
download review_decisions.json.

PRIVACY: The HTML uses file:// references to photos in the original Telegram export
directory. Photos are never copied. The HTML output is gitignored — it must not be
committed or shared.

source=telegram_private_2026-04-24, license=private_training_only

Usage:
    python3 scripts/intake_review_boundary.py \\
        --clusters data/intake_meta/tg_2026-04-24/dedup_clusters.jsonl \\
        --export-dir "/path/to/ChatExport_2026-04-24" \\
        [--output data/intake_meta/tg_2026-04-24/review/boundary_review.html] \\
        [--sample 30] \\
        [--hamming-min 8] [--hamming-max 8] \\
        [--unsure-from review_decisions.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parent.parent

SOURCE = "telegram_private_2026-04-24"
LICENSE = "private_training_only"

SAMPLE_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


# ─── File URI helpers ─────────────────────────────────────────────────────────


def _file_uri(path: Path) -> str:
    """Return a browser-safe file:// URI for an absolute path, percent-encoding spaces."""
    parts = path.parts  # ('/', 'Users', 'imac', 'Downloads', 'Telegram Desktop', ...)
    encoded = "/".join(quote(p, safe="") for p in parts[1:])  # skip leading '/'
    return f"file:///{encoded}"


# ─── Cluster loading ──────────────────────────────────────────────────────────


def _load_clusters(path: Path) -> list[dict]:
    clusters: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                clusters.append(json.loads(line))
    return clusters


def _filter_boundary(
    clusters: list[dict],
    hamming_min: int,
    hamming_max: int,
) -> list[dict]:
    return [
        c for c in clusters
        if c.get("cluster_type") == "perceptual"
        and c.get("hamming_distance") is not None
        and hamming_min <= c["hamming_distance"] <= hamming_max
    ]


def _print_hamming_distribution(clusters: list[dict]) -> None:
    perceptual = [c for c in clusters if c.get("cluster_type") == "perceptual"]
    dist: Counter[int] = Counter(
        c["hamming_distance"] for c in perceptual if c.get("hamming_distance") is not None
    )
    log.info("Hamming distance distribution across all %d perceptual clusters:", len(perceptual))
    for d in sorted(dist):
        bar = "█" * dist[d]
        log.info("  hamming=%d  %4d  %s", d, dist[d], bar[:60])


# ─── HTML generation ──────────────────────────────────────────────────────────


_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dedup Boundary Cluster Review</title>
<style>
  body {{ font-family: monospace; font-size: 13px; background: #1a1a1a; color: #e0e0e0; padding: 16px; }}
  h1 {{ color: #f0a030; }}
  .notice {{ background: #3a1a00; border: 1px solid #f0a030; padding: 10px; margin-bottom: 16px; border-radius: 4px; }}
  .cluster {{ border: 1px solid #444; margin-bottom: 12px; padding: 10px; border-radius: 4px; background: #242424; }}
  .cluster.decided-keep {{ border-color: #2a7a2a; }}
  .cluster.decided-fp {{ border-color: #7a2a2a; }}
  .cluster.decided-unsure {{ border-color: #7a7a2a; }}
  .cluster.decided-mixed {{ border-color: #2a4a7a; }}
  .cluster-header {{ font-weight: bold; margin-bottom: 8px; color: #b0b0ff; }}
  .images {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }}
  .img-wrap {{ text-align: center; }}
  .img-wrap img {{ max-width: 280px; max-height: 280px; border: 1px solid #555; display: block; }}
  .img-missing {{ display: inline-block; width: 140px; height: 140px; background: #333; border: 1px dashed #666;
                  color: #888; font-size: 11px; text-align: center; padding-top: 60px; word-break: break-all; }}
  .img-label {{ font-size: 11px; color: #888; max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .controls label {{ cursor: pointer; }}
  .controls input[type=radio] {{ margin-right: 4px; }}
  .note-input {{ background: #333; color: #ccc; border: 1px solid #555; padding: 4px 6px;
                 border-radius: 3px; font-family: monospace; font-size: 12px; width: 300px; }}
  .badge {{ display: inline-block; font-size: 11px; padding: 1px 6px; border-radius: 3px;
            background: #444; color: #ccc; margin-left: 6px; }}
  #export-btn {{ margin-top: 16px; padding: 10px 24px; background: #2a6a2a; color: white;
                 border: none; border-radius: 4px; cursor: pointer; font-size: 14px; font-family: monospace; }}
  #export-btn:disabled {{ background: #444; color: #888; cursor: default; }}
  #progress {{ margin-top: 8px; color: #888; font-size: 12px; }}
</style>
</head>
<body>
<h1>Dedup Boundary Cluster Review</h1>
<div class="notice">
  ⚠️ LOCAL-ONLY — This file contains <code>file://</code> references to private Telegram export photos.<br>
  Do NOT commit, email, upload, or share this file. Discard after review is complete.<br>
  <strong>Open with Safari</strong> on macOS for best <code>file://</code> compatibility.
</div>
{meta_block}
<div id="progress">Loading...</div>
{clusters_html}
<button id="export-btn" disabled onclick="exportDecisions()">Export Decisions</button>
<script>
const TOTAL = {total};
const THRESHOLD_REVIEWED = {threshold_reviewed};
const EXPORT_DIR = {export_dir_json};
const SAMPLE_SIZE = {sample_size_json};
const UNSURE_REPASS = {unsure_repass_json};

function updateProgress() {{
  const decided = document.querySelectorAll(
    'input[type=radio][name^="dec-"]:checked'
  ).length;
  const clusters = document.querySelectorAll('.cluster').length;
  document.getElementById('progress').textContent =
    `Decided: ${{decided}} / ${{clusters}} clusters`;
  document.getElementById('export-btn').disabled = (decided < clusters);
}}

document.addEventListener('DOMContentLoaded', function() {{
  document.querySelectorAll('input[type=radio]').forEach(r => {{
    r.addEventListener('change', function() {{
      const cid = this.name.replace('dec-', '');
      const cls = this.closest('.cluster');
      cls.className = 'cluster decided-' + this.value.toLowerCase().replace('_','-');
      updateProgress();
    }});
  }});
  updateProgress();
}});

function exportDecisions() {{
  const rows = document.querySelectorAll('.cluster');
  const decisions = [];
  rows.forEach(row => {{
    const cid = parseInt(row.dataset.clusterId);
    const ham = parseInt(row.dataset.hamming);
    const isMulti = row.dataset.multi === 'true';
    const keepFn = row.dataset.keepFilename;
    const checked = row.querySelector('input[type=radio]:checked');
    if (!checked) return;
    const note = row.querySelector('.note-input')?.value || '';
    decisions.push({{
      cluster_id: cid,
      hamming_distance: ham,
      is_multi_member: isMulti,
      keep_filename: keepFn,
      decision: checked.value,
      note: note
    }});
  }});
  const now = new Date().toISOString().slice(0, 10);
  const payload = {{
    schema_version: 1,
    reviewed_at: now,
    export_dir: EXPORT_DIR,
    threshold_reviewed: THRESHOLD_REVIEWED,
    sample_size: SAMPLE_SIZE,
    unsure_repass: UNSURE_REPASS,
    decisions: decisions
  }};
  const blob = new Blob([JSON.stringify(payload, null, 2)], {{type: 'application/json'}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'review_decisions.json';
  a.click();
  URL.revokeObjectURL(url);
}}
</script>
</body>
</html>
"""

_CLUSTER_ROW = """\
<div class="cluster" data-cluster-id="{cid}" data-hamming="{hamming}"
     data-multi="{is_multi}" data-keep-filename="{keep_fn_escaped}">
  <div class="cluster-header">
    Cluster #{cid} &nbsp;·&nbsp; hamming={hamming}
    {multi_badge}
  </div>
  <div class="images">{images_html}</div>
  <div class="controls">
    <label><input type="radio" name="dec-{cid}" value="KEEP_DEDUP"> KEEP_DEDUP</label>
    <label><input type="radio" name="dec-{cid}" value="FALSE_POSITIVE"> FALSE_POSITIVE</label>
    <label><input type="radio" name="dec-{cid}" value="UNSURE"> UNSURE</label>
    {mixed_option}
    <input class="note-input" type="text" placeholder="note (optional)" />
  </div>
</div>
"""

_MIXED_OPTION = '<label><input type="radio" name="dec-{cid}" value="MIXED"> MIXED</label>'

_IMG_TAG = (
    '<div class="img-wrap">'
    '<img src="{uri}" loading="lazy" alt="{label}">'
    '<div class="img-label" title="{label}">{label_short}</div>'
    '</div>'
)

_IMG_MISSING = (
    '<div class="img-wrap">'
    '<span class="img-missing">[missing]<br>{label_short}</span>'
    '<div class="img-label" title="{label}">{label_short}</div>'
    '</div>'
)


def _img_html(photo_path: Path, label: str) -> str:
    label_short = label.split("/")[-1] if "/" in label else label
    if photo_path.exists():
        return _IMG_TAG.format(
            uri=_file_uri(photo_path),
            label=label,
            label_short=label_short,
        )
    return _IMG_MISSING.format(label=label, label_short=label_short)


def _cluster_html(cluster: dict, export_dir: Path) -> str:
    cid = cluster["cluster_id"]
    hamming = cluster["hamming_distance"]
    keep_fn = cluster["keep_filename"]
    dup_fns: list[str] = cluster.get("duplicate_filenames", [])
    is_multi = len(dup_fns) > 1

    images_parts = [_img_html(export_dir / keep_fn, f"KEEP: {keep_fn}")]
    for dup in dup_fns:
        images_parts.append(_img_html(export_dir / dup, f"DUP: {dup}"))

    multi_badge = (
        f'<span class="badge">multi-member: {len(dup_fns) + 1} images</span>'
        if is_multi else ""
    )
    mixed_option = _MIXED_OPTION.format(cid=cid) if is_multi else ""

    return _CLUSTER_ROW.format(
        cid=cid,
        hamming=hamming,
        is_multi=str(is_multi).lower(),
        keep_fn_escaped=keep_fn.replace('"', "&quot;"),
        multi_badge=multi_badge,
        images_html="".join(images_parts),
        mixed_option=mixed_option,
    )


def _no_clusters_html() -> str:
    return '<p style="color:#888">No clusters to review with the current filter settings.</p>'


# ─── Main ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate local HTML contact sheet for boundary cluster review.",
    )
    p.add_argument(
        "--clusters",
        required=True,
        type=Path,
        help="Path to dedup_clusters.jsonl",
    )
    p.add_argument(
        "--export-dir",
        required=True,
        type=Path,
        help="Root directory of the Telegram ChatExport (contains photos/ subdirectory)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: <clusters-dir>/review/boundary_review.html)",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Randomly sample N clusters for calibration preview. "
            "NOTE: sample results are discarded before full review — "
            "finalization always requires all boundary clusters to be reviewed."
        ),
    )
    p.add_argument("--hamming-min", type=int, default=8, metavar="N")
    p.add_argument("--hamming-max", type=int, default=8, metavar="N")
    p.add_argument(
        "--unsure-from",
        type=Path,
        default=None,
        metavar="DECISIONS_JSON",
        help=(
            "Filter to only clusters marked UNSURE in a prior decisions file. "
            "Use to resolve uncertain entries without re-reviewing all clusters."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    clusters_path: Path = args.clusters
    if not clusters_path.exists():
        log.error("Clusters file not found: %s", clusters_path)
        return 1

    all_clusters = _load_clusters(clusters_path)
    if not all_clusters:
        log.error("Clusters file is empty: %s", clusters_path)
        return 1

    _print_hamming_distribution(all_clusters)

    boundary = _filter_boundary(all_clusters, args.hamming_min, args.hamming_max)
    log.info(
        "Filtered to hamming [%d, %d]: %d clusters (from %d total)",
        args.hamming_min, args.hamming_max, len(boundary), len(all_clusters),
    )

    unsure_repass = False
    if args.unsure_from is not None:
        if not args.unsure_from.exists():
            log.error("--unsure-from file not found: %s", args.unsure_from)
            return 1
        with args.unsure_from.open(encoding="utf-8") as fh:
            prior = json.load(fh)
        unsure_ids = {
            d["cluster_id"] for d in prior.get("decisions", [])
            if d.get("decision") == "UNSURE"
        }
        boundary = [c for c in boundary if c["cluster_id"] in unsure_ids]
        unsure_repass = True
        log.info("--unsure-from: filtered to %d UNSURE clusters", len(boundary))

    sample_size: int | None = None
    if args.sample is not None and not unsure_repass:
        sample_size = args.sample
        if sample_size < len(boundary):
            rng = random.Random(SAMPLE_SEED)
            boundary = rng.sample(boundary, sample_size)
            log.info(
                "SAMPLE MODE (seed=%d): %d of %d clusters selected. "
                "This is a calibration preview — full review required for finalization.",
                SAMPLE_SEED, sample_size, len(boundary),
            )

    output_path = args.output
    if output_path is None:
        output_path = clusters_path.parent / "review" / "boundary_review.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_dir: Path = args.export_dir.resolve()

    if boundary:
        clusters_html_parts = [_cluster_html(c, export_dir) for c in boundary]
        clusters_html = "\n".join(clusters_html_parts)
    else:
        clusters_html = _no_clusters_html()

    sample_notice = ""
    if sample_size is not None:
        sample_notice = (
            f'<div class="notice" style="background:#1a3a1a; border-color:#2a7a2a;">'
            f"SAMPLE MODE: showing {sample_size} randomly selected clusters (seed={SAMPLE_SEED}). "
            f"This is a calibration preview — full 285-cluster review required before finalization. "
            f"Discard these sample decisions before running the full pass."
            f"</div>"
        )
    if unsure_repass:
        sample_notice = (
            f'<div class="notice" style="background:#1a2a3a; border-color:#2a4a7a;">'
            f"UNSURE RE-PASS: showing {len(boundary)} clusters previously marked UNSURE. "
            f"Merge this file's decisions with your prior review_decisions.json before finalizing."
            f"</div>"
        )

    hamming_info = (
        f"hamming [{args.hamming_min}, {args.hamming_max}]"
        if args.hamming_min != args.hamming_max
        else f"hamming={args.hamming_min}"
    )
    meta_block = (
        f'<p style="color:#888; font-size:12px;">'
        f"Clusters: {len(boundary)} | Filter: {hamming_info} | "
        f"Export dir: {export_dir}"
        f"</p>"
        + sample_notice
    )

    html = _HTML_HEAD.format(
        meta_block=meta_block,
        clusters_html=clusters_html,
        total=len(boundary),
        threshold_reviewed=args.hamming_max,
        export_dir_json=json.dumps(str(export_dir)),
        sample_size_json=json.dumps(sample_size),
        unsure_repass_json=json.dumps(unsure_repass),
    )

    output_path.write_text(html, encoding="utf-8")
    log.info("Written: %s", output_path)
    log.info("Clusters included: %d", len(boundary))
    print(f"Written: {output_path}")
    print(f"Clusters included: {len(boundary)}")
    print(f"Open in browser: open -a Safari '{output_path}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
