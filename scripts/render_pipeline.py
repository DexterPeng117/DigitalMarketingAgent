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
    for each shot:
      - "interp"  — free, fully local placeholder/interpolation render.
                    For each shot, crossfades between its start_view and
                    end_view product photos (spec["product"]["views"]):
                    PIL blends the two images into a frame sequence,
                    ffmpeg encodes that sequence into a duration_s-long
                    clip. No paid API, no ComfyUI. lib/story_reel/'s
                    ComfyUI-only stubs (sr_keyframe.gen_t2i,
                    sr_segment.stage_input/build_prompt) are never
                    called by this backend — they are permanently
                    NotImplementedError here (see their own docstrings:
                    they require a live ComfyUI instance that this repo
                    has no access to) and calling them would just crash.
      - "wan_flf" — Wan 2.2 (alibaba/wan-2.7) first-last-frame-to-video
                    via OpenRouter's video generation API
                    (openrouter.ai/api/v1/videos) — see render_wan_flf's
                    docstring. Same OpenRouter account/key already used
                    by extract_product_views.py and ad_director.py; no
                    GPU, no ComfyUI, no separate account.

    Per-shot clips are then concatenated according to spec["assemble"]:
      - "cut"   — straight concatenation (ffmpeg concat demuxer).
      - "xfade" — crossfade between consecutive clips, implemented in
                  lib/story_reel/sr_concat.py's concat_xfade() with
                  ffmpeg's xfade filter.

CLI:
    python scripts/render_pipeline.py <spec_path> \
        --story-reel-dir lib/story_reel \
        --out outputs/<title>_silent.mp4
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FPS = 24
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920
DEFAULT_SHOT_DURATION_S = 3.0
DEFAULT_XFADE_DURATION_S = 0.5

# OpenRouter's video generation API (openrouter.ai/api/v1/videos) — see
# render_wan_flf's docstring for the documentation this is based on.
OPENROUTER_VIDEO_URL = "https://openrouter.ai/api/v1/videos"
DEFAULT_WAN_VIDEO_MODEL = "alibaba/wan-2.7"
WAN_VIDEO_POLL_INTERVAL_S = 20
WAN_VIDEO_POLL_TIMEOUT_S = 900


def load_spec(spec_path: Path) -> dict:
    return json.loads(spec_path.read_text(encoding="utf-8"))


def _load_story_reel_module(story_reel_dir: Path, module_name: str):
    """Load a lib/story_reel/<module_name>.py module by file path, so
    render_pipeline.py doesn't hard-depend on it being importable as a
    package (--story-reel-dir is caller-configurable)."""
    module_path = story_reel_dir / f"{module_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"{module_name}.py not found in story-reel dir: {story_reel_dir}")
    module_spec = importlib.util.spec_from_file_location(f"story_reel_{module_name}", module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg command failed: {' '.join(cmd)}\n{result.stderr}")


def _view_images(spec: dict) -> dict[str, str]:
    views = spec.get("product", {}).get("views", [])
    return {v["view"]: v["image"] for v in views}


def _shot_view_paths(shot: dict, view_images: dict[str, str], index: int) -> tuple[Path, Path]:
    start_view = shot.get("start_view")
    end_view = shot.get("end_view")
    start_path = view_images.get(start_view)
    end_path = view_images.get(end_view)
    if start_path is None or end_path is None:
        raise ValueError(
            f"Shot #{index} references an unknown view (start_view={start_view!r}, "
            f"end_view={end_view!r}); known views: {sorted(view_images)}"
        )
    return Path(start_path), Path(end_path)


def _concat_cut(clips: list[Path], out_path: Path) -> Path:
    """Straight ffmpeg concat-demuxer join, no transitions."""
    list_path = clips[0].parent / "concat_list.txt"
    list_path.write_text("".join(f"file '{c.resolve()}'\n" for c in clips), encoding="utf-8")
    _run_ffmpeg([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path),
    ])
    return out_path


def _concat_shots(clips: list[Path], spec: dict, story_reel_dir: Path, out_path: Path) -> Path:
    assemble = spec.get("assemble", "cut")
    if assemble == "xfade":
        sr_concat = _load_story_reel_module(story_reel_dir, "sr_concat")
        width = int(spec.get("width", DEFAULT_WIDTH))
        height = int(spec.get("height", DEFAULT_HEIGHT))
        fps = int(spec.get("fps", DEFAULT_FPS))
        xfade_duration = float(spec.get("xfade_duration_s", DEFAULT_XFADE_DURATION_S))
        return sr_concat.concat_xfade(clips, out_path, width, height, fps, xfade_duration)
    return _concat_cut(clips, out_path)


def _load_and_cover(path: Path, width: int, height: int) -> Image.Image:
    """Resize+center-crop `path` to exactly width x height, preserving
    aspect ratio (like CSS `object-fit: cover`), so shots never distort
    the product photo."""
    image = Image.open(path).convert("RGB")
    src_w, src_h = image.size
    scale = max(width / src_w, height / src_h)
    new_w, new_h = max(round(src_w * scale), width), max(round(src_h * scale), height)
    image = image.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    return image.crop((left, top, left + width, top + height))


def _render_shot_crossfade(shot: dict, view_images: dict[str, str], fps: int, width: int,
                            height: int, tmp_dir: Path, index: int) -> Path:
    """Crossfade shot['start_view'] -> shot['end_view'] over shot['duration_s']:
    PIL blends the two cover-cropped images into a frame sequence, ffmpeg
    encodes the sequence into a clip."""
    start_path, end_path = _shot_view_paths(shot, view_images, index)
    start_img = _load_and_cover(start_path, width, height)
    end_img = _load_and_cover(end_path, width, height)

    duration_s = float(shot.get("duration_s") or DEFAULT_SHOT_DURATION_S)
    num_frames = max(round(duration_s * fps), 1)

    frames_dir = tmp_dir / f"shot_{index:03d}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_frames):
        alpha = i / (num_frames - 1) if num_frames > 1 else 1.0
        Image.blend(start_img, end_img, alpha).save(frames_dir / f"frame_{i:05d}.png")

    clip_path = tmp_dir / f"shot_{index:03d}.mp4"
    _run_ffmpeg([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%05d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
        str(clip_path),
    ])
    return clip_path


def render_interp(spec: dict, story_reel_dir: Path, out_path: Path) -> Path:
    """Render all shots using the free local placeholder/interpolation
    backend — crossfades between each shot's start_view/end_view product
    photos. Works with no paid API and no ComfyUI; never touches the
    ComfyUI-only stubs in lib/story_reel/sr_keyframe.py / sr_segment.py.
    """
    fps = int(spec.get("fps", DEFAULT_FPS))
    width = int(spec.get("width", DEFAULT_WIDTH))
    height = int(spec.get("height", DEFAULT_HEIGHT))
    shots = spec.get("shots")
    if not shots:
        raise ValueError("spec has no shots to render")
    view_images = _view_images(spec)
    if not view_images:
        raise ValueError("spec has no product.views to render from")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="render_interp_") as tmp:
        tmp_dir = Path(tmp)
        clips = [
            _render_shot_crossfade(shot, view_images, fps, width, height, tmp_dir, i)
            for i, shot in enumerate(shots)
        ]
        _concat_shots(clips, spec, story_reel_dir, out_path)

    return out_path


def _load_openrouter_api_key() -> str:
    """Same key/config used by extract_product_views.py and ad_director.py
    (config/settings.json's "llm.api_key") — this backend is just another
    OpenRouter call, not a separate account."""
    settings_path = REPO_ROOT / "config" / "settings.json"
    if not settings_path.exists():
        raise RuntimeError(
            "config/settings.json not found. Copy config/settings.example.json "
            "to config/settings.json and fill in 'llm.api_key'."
        )
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    api_key = settings.get("llm", {}).get("api_key")
    if not api_key:
        raise RuntimeError("config/settings.json is missing a non-empty 'llm.api_key'.")
    return api_key


def _wan_video_image_url(path: Path) -> str:
    import base64
    import mimetypes
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{data}"


def _wan_video_submit(api_key: str, shot: dict, start_path: Path, end_path: Path,
                       width: int, height: int) -> dict:
    duration_s = float(shot.get("duration_s") or DEFAULT_SHOT_DURATION_S)
    payload = {
        "model": DEFAULT_WAN_VIDEO_MODEL,
        "prompt": shot.get("prompt", ""),
        "duration": round(duration_s),
        "size": f"{width}x{height}",
        "generate_audio": False,
        "frame_images": [
            {"type": "image_url", "image_url": {"url": _wan_video_image_url(start_path)}, "frame_type": "first_frame"},
            {"type": "image_url", "image_url": {"url": _wan_video_image_url(end_path)}, "frame_type": "last_frame"},
        ],
    }
    seed = shot.get("seed")
    if seed is not None:
        payload["seed"] = int(seed)

    response = requests.post(
        OPENROUTER_VIDEO_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=60,
    )
    if not response.ok:
        raise RuntimeError(f"OpenRouter video submission failed ({response.status_code}): {response.text}")
    return response.json()


def _wan_video_poll(api_key: str, polling_url: str) -> dict:
    """Poll until the job reaches a terminal state, bounded by
    WAN_VIDEO_POLL_TIMEOUT_S so this can never hang indefinitely."""
    deadline = time.monotonic() + WAN_VIDEO_POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        response = requests.get(polling_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        response.raise_for_status()
        data = response.json()
        status = data.get("status")
        if status == "completed":
            return data
        if status in ("failed", "cancelled", "expired"):
            raise RuntimeError(f"OpenRouter video job {data.get('id')} ended with status={status!r}: {data.get('error')}")
        time.sleep(WAN_VIDEO_POLL_INTERVAL_S)
    raise TimeoutError(f"Timed out after {WAN_VIDEO_POLL_TIMEOUT_S}s waiting for OpenRouter video job at {polling_url}")


def _wan_video_download(api_key: str, result: dict, out_path: Path) -> None:
    """Download the finished clip and strip its audio track.

    Verified against a real generation: even with generate_audio=false
    in the submit payload, the returned mp4 still carried an AAC audio
    stream. This backend's whole job is a *silent* clip (finalize_ad.py
    adds narration/music later), so the track is stripped unconditionally
    here rather than trusted to already be silent/absent.
    """
    urls = result.get("unsigned_urls") or []
    if not urls:
        raise RuntimeError(f"OpenRouter video job completed but returned no unsigned_urls: {result}")
    response = requests.get(urls[0], headers={"Authorization": f"Bearer {api_key}"}, timeout=120)
    response.raise_for_status()

    raw_path = out_path.with_name(out_path.stem + ".raw" + out_path.suffix)
    raw_path.write_bytes(response.content)
    try:
        _run_ffmpeg(["ffmpeg", "-y", "-i", str(raw_path), "-c:v", "copy", "-an", str(out_path)])
    finally:
        raw_path.unlink(missing_ok=True)


def render_wan_flf(spec: dict, story_reel_dir: Path, out_path: Path) -> Path:
    """Render all shots using Wan 2.2 (alibaba/wan-2.7) first-last-frame-
    to-video through OpenRouter's video generation API — no GPU, no
    ComfyUI, no separate account: same config/settings.json "llm.api_key"
    already used by extract_product_views.py/ad_director.py. Pricing is
    $0.10/second (openrouter.ai/alibaba/wan-2.7).

    Endpoints/fields below are from openrouter.ai/docs/guides/overview/
    multimodal/video-generation, github.com/OpenRouterTeam/skills'
    openrouter-video SKILL.md, and openrouter.ai/docs/cookbook/
    video-generation/image-to-video — not guessed:
      POST https://openrouter.ai/api/v1/videos
        {"model", "prompt", "duration" (int, must match the model's
         supported_durations — not published anywhere we could find, so
         this just passes round(shot["duration_s"]) and lets OpenRouter's
         own validation error surface if it's unsupported), "size"
         ("WxH"), "generate_audio" (false here — this backend renders a
         silent clip; finalize_ad.py adds narration/music later), "seed",
         "frame_images": [{"type": "image_url", "image_url": {"url": ...},
         "frame_type": "first_frame"|"last_frame"}]}
        -> 202 {"id", "polling_url", "status": "pending"}
      GET {polling_url}  (Bearer auth)
        -> {"status": "pending"|"in_progress"|"completed"|"failed"|
            "cancelled"|"expired", "unsigned_urls": [...] (on completion),
            "error": "..." (on failure), "usage": {...}}
      GET {unsigned_urls[0]}  (Bearer auth) -> the video file's bytes.

    shot["negative_prompt"] is intentionally NOT sent: none of the above
    sources document a negative_prompt (or equivalent) field on this API
    — unlike OpenRouter's chat/vision endpoints, which do — so rather
    than guess an undocumented field name, it's dropped for this backend.

    Polling is bounded by WAN_VIDEO_POLL_TIMEOUT_S (15 min) so a stuck
    job can't hang the pipeline forever; a missing/empty llm.api_key
    raises immediately, before any shot is touched.
    """
    api_key = _load_openrouter_api_key()

    width = int(spec.get("width", DEFAULT_WIDTH))
    height = int(spec.get("height", DEFAULT_HEIGHT))
    shots = spec.get("shots")
    if not shots:
        raise ValueError("spec has no shots to render")
    view_images = _view_images(spec)
    if not view_images:
        raise ValueError("spec has no product.views to render from")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="render_wan_flf_") as tmp:
        tmp_dir = Path(tmp)
        clips = []
        for i, shot in enumerate(shots):
            start_path, end_path = _shot_view_paths(shot, view_images, i)
            try:
                submitted = _wan_video_submit(api_key, shot, start_path, end_path, width, height)
                polling_url = submitted.get("polling_url") or f"{OPENROUTER_VIDEO_URL}/{submitted['id']}"
                result = _wan_video_poll(api_key, polling_url)
            except Exception as exc:
                print(f"[error] wan_flf render call failed for shot #{i}: {exc}")
                raise

            clip_path = tmp_dir / f"shot_{i:03d}.mp4"
            _wan_video_download(api_key, result, clip_path)
            clips.append(clip_path)

        _concat_shots(clips, spec, story_reel_dir, out_path)

    return out_path


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
