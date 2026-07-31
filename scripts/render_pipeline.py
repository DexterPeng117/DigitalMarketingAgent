"""
render_pipeline.py — Spec + view images -> rendered silent video.

Input:
    A spec JSON produced by ad_director.py (workflows/<title>.json),
    containing the storyboard/camera instructions and an
    "animate_backend" field.

Output:
    An intermediate, silent (no narration/music yet) rendered video at
    outputs/<title>_silent.mp4, ready for finalize_ad.py.

Backend selection:
    spec["animate_backend"] picks which renderer implementation is used
    for each shot, e.g.:
      - "wan_flf" — paid cloud AI render (first/last-frame video model).
      - "interp"  — free local placeholder/interpolation render, usable
                    with no paid API and no dedicated local render
                    environment (e.g. no ComfyUI install required).

    Both backends are implemented against the abstraction layer in
    lib/story_reel/ (sr_lib / sr_keyframe / sr_segment / sr_concat) so
    this script itself has no hard dependency on any specific local
    render environment — only the backend actually selected by the spec
    needs its underlying service (e.g. ComfyUI) to be reachable.

CLI:
    python scripts/render_pipeline.py <spec_path> \
        --story-reel-dir lib/story_reel \
        --out outputs/<title>_silent.mp4

TODO(design confirmed, implementation pending):
    - Implement `render_wan_flf` (cloud backend) and `render_interp`
      (free local backend) against lib/story_reel/.
    - Implement segment concatenation (spec["assemble"]: "cut" | "xfade")
      using lib/story_reel/sr_concat.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_spec(spec_path: Path) -> dict:
    return json.loads(spec_path.read_text(encoding="utf-8"))


def render_wan_flf(spec: dict, story_reel_dir: Path, out_path: Path) -> Path:
    """Render all shots using the paid cloud AI backend (wan_flf).

    TODO: implement using lib/story_reel/ as the rendering dependency
    abstraction (keyframe generation, segment animation, concatenation).
    """
    raise NotImplementedError


def render_interp(spec: dict, story_reel_dir: Path, out_path: Path) -> Path:
    """Render all shots using the free local placeholder/interpolation
    backend — must work with no paid API and no dedicated local render
    environment.

    TODO: implement using lib/story_reel/ as the rendering dependency
    abstraction.
    """
    raise NotImplementedError


BACKENDS = {
    "wan_flf": render_wan_flf,
    "interp": render_interp,
}


def render(spec: dict, story_reel_dir: Path, out_path: Path) -> Path:
    backend_name = spec.get("animate_backend")
    if backend_name not in BACKENDS:
        raise ValueError(
            f"spec has no usable animate_backend (got {backend_name!r}); "
            f"expected one of {sorted(BACKENDS)}"
        )
    return BACKENDS[backend_name](spec, story_reel_dir, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="Path to the workflow spec JSON (from ad_director.py)")
    ap.add_argument("--story-reel-dir", default=str(REPO_ROOT / "lib" / "story_reel"),
                     help="Path to the story_reel rendering abstraction layer")
    ap.add_argument("--out", required=True, help="Output path for the rendered silent video")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    spec = load_spec(spec_path)

    out_path = render(spec, Path(args.story_reel_dir), Path(args.out))
    print(f"   render -> {out_path}")


if __name__ == "__main__":
    main()
