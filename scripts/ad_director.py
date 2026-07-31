"""
ad_director.py — LLM "ad director": product views -> storyboard spec JSON.

Input:
    A directory of classified product view images (as produced by
    extract_product_views.py, e.g. assets/<brand>/front.png) and an
    optional free-text creative brief.

Output:
    A single JSON spec file at workflows/<title>.json describing the ad:
    copy/tagline, shot-by-shot camera/animation instructions per view,
    and render configuration.

    The spec MUST always include an "animate_backend" field (e.g.
    "wan_flf" for the paid cloud render backend, or "interp" for the
    free local placeholder/interpolation backend). This field was
    historically dropped by mistake in the prior implementation — do not
    repeat that bug; render_pipeline.py depends on it to pick a backend.

CLI:
    python scripts/ad_director.py --views assets/<brand> \
        [--brief "free-text creative brief"] \
        [--backend wan_flf|interp]

    Prints a line "   spec   -> workflows/<title>.json" on success (this
    exact "key -> value" shape is relied on by run_full_pipeline.sh to
    scrape the generated spec path).

TODO(design confirmed, implementation pending):
    - Define the full spec JSON schema (storyboard shots, per-shot
      camera/animation instructions, audio/tagline fields, assemble mode).
    - Implement the actual LLM call(s) in `generate_storyboard`.
    - Decide default value / selection logic for animate_backend when
      --backend is not passed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANIMATE_BACKEND = "interp"


def load_views(views_dir: Path) -> dict[str, Path]:
    """Load the view_name -> image_path mapping from a views directory
    (as written by extract_product_views.py).

    TODO: implement (glob *.png / *.jpg in views_dir, key by file stem).
    """
    raise NotImplementedError


def generate_storyboard(views: dict[str, Path], brief: str | None, animate_backend: str) -> dict:
    """Call the LLM to turn product views (+ optional brief) into a
    storyboard spec dict. Must set spec["animate_backend"] = animate_backend.

    TODO: implement the actual LLM call and spec schema.
    """
    raise NotImplementedError


def write_spec(spec: dict, out_dir: Path) -> Path:
    """Write `spec` to workflows/<title>.json and return the path.

    TODO: implement (derive filename from spec["title"], json.dump).
    """
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--views", required=True, help="Directory of classified view images, e.g. assets/<brand>")
    ap.add_argument("--brief", default=None, help="Optional free-text creative brief")
    ap.add_argument("--backend", dest="animate_backend", default=DEFAULT_ANIMATE_BACKEND,
                     choices=["wan_flf", "interp"], help="Render backend to record in the spec")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "workflows"), help="Directory to write the spec JSON into")
    args = ap.parse_args()

    views = load_views(Path(args.views))
    spec = generate_storyboard(views, args.brief, args.animate_backend)
    assert "animate_backend" in spec, "spec must always include animate_backend"

    spec_path = write_spec(spec, Path(args.out_dir))
    print(f"   spec   -> {spec_path}")


if __name__ == "__main__":
    main()
