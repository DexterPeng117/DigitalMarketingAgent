"""
ad_tracker.py — Publishing & analytics tracker for the ad-video pipeline.

Fills the gap after finalize_ad.py: right now a finished ad (outputs/<title>_full.mp4)
has no record of whether/where it was published, or how it performed. Since the
project requires posting each ad across multiple platforms for comparison (Insta,
FB, Twitter, Pinterest, websites, etc.), this uses TWO linked CSV tables:

  ads.csv          one row per finalized ad (fixed info: brand, video, duration)
  publications.csv one row per (ad x platform) posting, so the same ad can have
                    many publication rows — one per platform it's posted to.

Commands:
  scan       Batch-import unregistered videos from outputs/ (best-effort spec match)
  register   Register a newly finalized ad (reads brand from its workflow spec)
  publish    Add a new publication record for an ad on a given platform
  metrics    Update performance numbers for one ad+platform publication
  report     Print a joined summary table (optionally filtered)
  export     Write both tables out as sheets in one .xlsx for easy sharing

Typical flow:
  python scripts/ad_tracker.py scan --dry-run
  python scripts/ad_tracker.py scan

  python scripts/ad_tracker.py register workflows/watch_wan_flf.json \
      --video outputs/watch_wan_flf_full.mp4

  python scripts/ad_tracker.py publish watch_wan_flf_full \
      --platform instagram --post-id 17912345 \
      --url https://instagram.com/p/xxxxx

  python scripts/ad_tracker.py publish watch_wan_flf_full \
      --platform tiktok --post-id 89213 --url https://tiktok.com/@x/video/89213

  python scripts/ad_tracker.py metrics watch_wan_flf_full --platform instagram \
      --views 12000 --likes 340 --comments 12 --shares 5 --clicks 88

  python scripts/ad_tracker.py report
  python scripts/ad_tracker.py report --platform instagram
  python scripts/ad_tracker.py export --out ad_tracking.xlsx
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ADS_PATH = REPO_ROOT / "outputs" / "ads.csv"
PUBS_PATH = REPO_ROOT / "outputs" / "publications.csv"

AD_FIELDS = [
    "ad_id",          # unique key, derived from the video filename stem
    "spec_file",      # workflows/xxx.json this ad was generated from
    "video_path",     # path to the finalized mp4
    "brand",          # product/brand name, extracted from the spec
    "duration_s",     # video length in seconds (best-effort, may be blank)
    "status",         # generated -> published (has >=1 publication) -> archived
    "created_at",     # when this ad was registered (ISO 8601, UTC)
    "last_updated",
    "notes",
]

PUB_FIELDS = [
    "pub_id",         # "<ad_id>__<platform>", unique per ad+platform
    "ad_id",          # foreign key into ads.csv
    "platform",       # instagram / tiktok / youtube_shorts / etc
    "post_id",
    "post_url",
    "published_at",
    "views",
    "likes",
    "comments",
    "shares",
    "clicks",
    "last_updated",
]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _ffprobe_duration(video_path: Path) -> str:
    """Best-effort duration lookup; blank if ffprobe isn't available or fails."""
    try:
        import subprocess
        out = subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(video_path),
        ])
        return f"{float(out.decode().strip()):.1f}"
    except Exception:
        return ""


def _extract_brand(spec: dict) -> str:
    """Best-effort brand/product name lookup.

    Real spec files (see workflows/*.json) don't have an explicit "brand"
    field. The most reliable signal is the folder name under
    product.views[].image, e.g. "assets/rolex/front.png" -> "rolex".
    Falls back to parsing the trailing "— BRAND" in audio.tagline, then to
    an empty string if neither is present.
    """
    views = spec.get("product", {}).get("views", [])
    for v in views:
        image = v.get("image", "")
        if image:
            parts = Path(image).parts
            if len(parts) >= 2:
                return parts[-2]  # folder just above the filename

    tagline = spec.get("audio", {}).get("tagline", "")
    if "—" in tagline:
        return tagline.split("—")[-1].strip()

    return ""


def _find(rows: list[dict], **kwargs) -> dict | None:
    for r in rows:
        if all(r.get(k) == v for k, v in kwargs.items()):
            return r
    return None


def _guess_spec_path(video_stem: str, workflows_dir: Path) -> Path | None:
    """Guess which workflows/*.json produced a video, based on naming
    convention <title>_full.mp4 / <title>_final.mp4 (see finalize_ad.py)."""
    base = video_stem
    for suffix in ("_full", "_final"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    candidate = workflows_dir / f"{base}.json"
    return candidate if candidate.exists() else None


def _register_one(ad_id: str, spec_path: Path, video_path: Path) -> dict:
    spec = {}
    if spec_path and spec_path.exists():
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    brand = _extract_brand(spec)
    notes = "" if spec_path and spec_path.exists() else "no matching spec found - brand unknown, please fill in manually"
    return {
        "ad_id": ad_id,
        "spec_file": str(spec_path) if spec_path else "",
        "video_path": str(video_path),
        "brand": brand,
        "duration_s": _ffprobe_duration(video_path) if video_path.exists() else "",
        "status": "generated",
        "created_at": _now(),
        "last_updated": _now(),
        "notes": notes,
    }


def cmd_scan(args: argparse.Namespace) -> None:
    outputs_dir = Path(args.dir) if args.dir else (REPO_ROOT / "outputs")
    workflows_dir = Path(args.workflows_dir) if args.workflows_dir else (REPO_ROOT / "workflows")

    ads = _load_csv(ADS_PATH)
    existing_ids = {a["ad_id"] for a in ads}

    videos = sorted(outputs_dir.glob("*.mp4"))
    new_rows = []
    for video in videos:
        ad_id = video.stem
        if ad_id in existing_ids:
            continue
        spec_path = _guess_spec_path(ad_id, workflows_dir)
        new_rows.append(_register_one(ad_id, spec_path, video))

    if not new_rows:
        print(f"No new videos found in {outputs_dir} (checked {len(videos)}, all already registered).")
        return

    print(f"Found {len(new_rows)} unregistered video(s) in {outputs_dir}:")
    for r in new_rows:
        flag = "  [!] no spec matched — brand unknown" if not r["spec_file"] else ""
        print(f"  - {r['ad_id']} (brand={r['brand'] or 'unknown'}){flag}")

    if args.dry_run:
        print("\n[dry-run] Nothing written. Re-run without --dry-run to actually register these.")
        return

    ads.extend(new_rows)
    _save_csv(ADS_PATH, ads, AD_FIELDS)
    print(f"\n[scanned] Registered {len(new_rows)} new ad(s).")


def cmd_register(args: argparse.Namespace) -> None:
    spec_path = Path(args.spec)
    video_path = Path(args.video)

    if not video_path.exists():
        print(f"[error] video file not found: {video_path}. Check the path and try again.")
        return
    if not spec_path.exists():
        print(f"[warn] spec file not found: {spec_path}. Registering anyway, but brand "
              f"will be blank - you can edit it manually in outputs/ads.csv afterward.")

    ad_id = video_path.stem
    ads = _load_csv(ADS_PATH)
    if _find(ads, ad_id=ad_id):
        print(f"[skip] '{ad_id}' is already registered. Use 'publish'/'metrics' to update it.")
        return

    row = _register_one(ad_id, spec_path, video_path)
    ads.append(row)
    _save_csv(ADS_PATH, ads, AD_FIELDS)
    print(f"[registered] {ad_id} (brand={row['brand'] or 'unknown'})")


def cmd_publish(args: argparse.Namespace) -> None:
    platform = args.platform.strip().lower()
    if not platform:
        print("[error] --platform cannot be empty.")
        return
    if args.url and not (args.url.startswith("http://") or args.url.startswith("https://")):
        print(f"[warn] --url '{args.url}' doesn't look like a full URL (expected it to start "
              f"with http:// or https://). Saving it as-is.")

    ads = _load_csv(ADS_PATH)
    ad = _find(ads, ad_id=args.ad_id)
    if ad is None:
        print(f"[error] no registered ad with id '{args.ad_id}'. Run 'register' first.")
        return

    pubs = _load_csv(PUBS_PATH)
    pub_id = f"{args.ad_id}__{platform}"
    existing = _find(pubs, pub_id=pub_id)
    if existing:
        print(f"[skip] '{args.ad_id}' is already marked published on {platform}. "
              f"Use 'metrics' to update its numbers, or pick a different platform.")
        return

    pubs.append({
        "pub_id": pub_id,
        "ad_id": args.ad_id,
        "platform": platform,
        "post_id": args.post_id or "",
        "post_url": args.url or "",
        "published_at": _now(),
        "views": "", "likes": "", "comments": "", "shares": "", "clicks": "",
        "last_updated": _now(),
    })
    _save_csv(PUBS_PATH, pubs, PUB_FIELDS)

    ad["status"] = "published"
    ad["last_updated"] = _now()
    _save_csv(ADS_PATH, ads, AD_FIELDS)
    print(f"[published] {args.ad_id} -> {platform}")


def cmd_metrics(args: argparse.Namespace) -> None:
    platform = args.platform.strip().lower()
    metric_fields = ("views", "likes", "comments", "shares", "clicks")
    for field in metric_fields:
        val = getattr(args, field)
        if val is not None and val < 0:
            print(f"[error] --{field} cannot be negative (got {val}). Nothing was updated.")
            return

    pubs = _load_csv(PUBS_PATH)
    pub_id = f"{args.ad_id}__{platform}"
    pub = _find(pubs, pub_id=pub_id)
    if pub is None:
        print(f"[error] no publication of '{args.ad_id}' on '{platform}'. "
              f"Run 'publish' for that platform first.")
        return
    for field in metric_fields:
        val = getattr(args, field)
        if val is not None:
            pub[field] = str(val)
    pub["last_updated"] = _now()
    _save_csv(PUBS_PATH, pubs, PUB_FIELDS)
    print(f"[updated metrics] {args.ad_id} on {platform}")


def _joined_rows(ads: list[dict], pubs: list[dict]) -> list[dict]:
    """Left-join ads -> publications so ads with no publications still show up."""
    ads_by_id = {a["ad_id"]: a for a in ads}
    rows = []
    seen_ad_ids = set()
    for p in pubs:
        ad = ads_by_id.get(p["ad_id"], {})
        rows.append({**ad, **p})
        seen_ad_ids.add(p["ad_id"])
    for a in ads:
        if a["ad_id"] not in seen_ad_ids:
            rows.append({**a, "platform": "", "views": "", "likes": "", "clicks": "", "published_at": ""})
    return rows


def cmd_report(args: argparse.Namespace) -> None:
    ads = _load_csv(ADS_PATH)
    pubs = _load_csv(PUBS_PATH)
    rows = _joined_rows(ads, pubs)
    if args.platform:
        platform = args.platform.strip().lower()
        rows = [r for r in rows if r.get("platform") == platform]
    if args.status:
        rows = [r for r in rows if r.get("status") == args.status]
    if not rows:
        print("No matching ads/publications found.")
        return
    cols = ["ad_id", "brand", "status", "platform", "views", "likes", "clicks", "published_at"]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0)) for c in cols}
    header = " | ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def cmd_export(args: argparse.Namespace) -> None:
    ads = _load_csv(ADS_PATH)
    pubs = _load_csv(PUBS_PATH)
    if not ads and not pubs:
        print("Nothing to export yet.")
        return
    try:
        import pandas as pd
    except ImportError:
        print("[error] pandas is required for export (pip install pandas openpyxl).")
        return
    out_path = Path(args.out)
    with pd.ExcelWriter(out_path) as writer:
        pd.DataFrame(ads, columns=AD_FIELDS).to_excel(writer, sheet_name="ads", index=False)
        pd.DataFrame(pubs, columns=PUB_FIELDS).to_excel(writer, sheet_name="publications", index=False)
        pd.DataFrame(_joined_rows(ads, pubs)).to_excel(writer, sheet_name="summary", index=False)
    print(f"[exported] {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Batch-import unregistered videos from outputs/")
    p_scan.add_argument("--dir", help="Directory to scan for .mp4 files (default: outputs/)")
    p_scan.add_argument("--workflows-dir", dest="workflows_dir",
                         help="Directory to look for matching spec JSONs (default: workflows/)")
    p_scan.add_argument("--dry-run", action="store_true", help="Preview without writing to ads.csv")
    p_scan.set_defaults(func=cmd_scan)

    p_reg = sub.add_parser("register", help="Register a newly finalized ad")
    p_reg.add_argument("spec", help="Path to the workflow spec JSON used to generate this ad")
    p_reg.add_argument("--video", required=True, help="Path to the finalized mp4")
    p_reg.set_defaults(func=cmd_register)

    p_pub = sub.add_parser("publish", help="Add a publication record for an ad on a platform")
    p_pub.add_argument("ad_id", help="ad_id (video filename stem, e.g. watch_wan_flf_full)")
    p_pub.add_argument("--platform", required=True, help="e.g. instagram, tiktok, youtube_shorts")
    p_pub.add_argument("--post-id", dest="post_id", help="Platform-native post ID")
    p_pub.add_argument("--url", help="Public URL to the post")
    p_pub.set_defaults(func=cmd_publish)

    p_met = sub.add_parser("metrics", help="Update performance metrics for one ad+platform publication")
    p_met.add_argument("ad_id")
    p_met.add_argument("--platform", required=True, help="Which platform's publication to update")
    p_met.add_argument("--views", type=int)
    p_met.add_argument("--likes", type=int)
    p_met.add_argument("--comments", type=int)
    p_met.add_argument("--shares", type=int)
    p_met.add_argument("--clicks", type=int)
    p_met.set_defaults(func=cmd_metrics)

    p_rep = sub.add_parser("report", help="Print a joined summary table")
    p_rep.add_argument("--platform")
    p_rep.add_argument("--status")
    p_rep.set_defaults(func=cmd_report)

    p_exp = sub.add_parser("export", help="Export ads + publications + summary to .xlsx")
    p_exp.add_argument("--out", default="ad_tracking.xlsx")
    p_exp.set_defaults(func=cmd_export)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
