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
      - "wan_flf" — Wan 2.2 first-last-frame-to-video, run locally
                    through a ComfyUI instance (Apache-2.0, Alibaba,
                    commercially usable) — see render_wan_flf's
                    docstring. No paid API; cost is just local/rented
                    GPU time. Untested end-to-end in this environment
                    (no GPU/ComfyUI/model weights here) — see that
                    docstring for exactly what's been verified.

    Per-shot clips are then concatenated according to spec["assemble"]:
      - "cut"   — straight concatenation (ffmpeg concat demuxer).
      - "xfade" — crossfade between consecutive clips, implemented in
                  lib/story_reel/sr_concat.py's concat_xfade() with
                  ffmpeg's xfade filter. That function is pure ffmpeg
                  (no ComfyUI dependency), unlike the three stubs above.

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
import uuid
from pathlib import Path

import requests
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FPS = 24
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920
DEFAULT_SHOT_DURATION_S = 3.0
DEFAULT_XFADE_DURATION_S = 0.5

# lib/story_reel/comfy_workflows/wan_flf2v.json is the API-format node graph
# for Wan 2.2 FLF2V (the 4-step LoRA-accelerated variant of the official
# Comfy-Org "video_wan2_2_14B_flf2v" template — see render_wan_flf's
# docstring for provenance/model list). These are the node ids inside that
# JSON that render_wan_flf patches per shot.
COMFY_WORKFLOW_PATH = REPO_ROOT / "lib" / "story_reel" / "comfy_workflows" / "wan_flf2v.json"
WAN_FLF2V_POSITIVE_NODE = "6"
WAN_FLF2V_NEGATIVE_NODE = "7"
WAN_FLF2V_START_IMAGE_NODE = "68"
WAN_FLF2V_END_IMAGE_NODE = "62"
WAN_FLF2V_FLF_NODE = "67"
WAN_FLF2V_SEED_NODE = "57"
WAN_FLF2V_VIDEO_NODE = "60"

COMFY_HEALTHCHECK_TIMEOUT_S = 5
COMFY_SUBMIT_TIMEOUT_S = 30
COMFY_POLL_INTERVAL_S = 5
COMFY_POLL_TIMEOUT_S = 900


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


def _comfy_url() -> str:
    settings_path = REPO_ROOT / "config" / "settings.json"
    if not settings_path.exists():
        raise RuntimeError(
            "config/settings.json not found. Copy config/settings.example.json "
            "to config/settings.json (render.comfy_url defaults to http://127.0.0.1:8188)."
        )
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    comfy_url = settings.get("render", {}).get("comfy_url")
    if not comfy_url:
        raise RuntimeError("config/settings.json is missing a non-empty 'render.comfy_url'.")
    return comfy_url.rstrip("/")


def _check_comfy_reachable(comfy_url: str) -> None:
    """Fail fast (bounded by COMFY_HEALTHCHECK_TIMEOUT_S) instead of
    hanging on ComfyUI's default request timeout if it's offline."""
    try:
        response = requests.get(f"{comfy_url}/system_stats", timeout=COMFY_HEALTHCHECK_TIMEOUT_S)
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"ComfyUI not reachable at {comfy_url}; make sure it's running locally "
            f"with the Wan 2.2 models installed."
        ) from exc


def _comfy_upload_image(comfy_url: str, path: Path) -> str:
    with path.open("rb") as f:
        response = requests.post(
            f"{comfy_url}/upload/image",
            files={"image": (path.name, f, "image/png")},
            data={"type": "input", "overwrite": "true"},
            timeout=60,
        )
    response.raise_for_status()
    return response.json()["name"]


def _comfy_submit(comfy_url: str, workflow: dict) -> str:
    response = requests.post(
        f"{comfy_url}/prompt",
        json={"prompt": workflow, "client_id": str(uuid.uuid4())},
        timeout=COMFY_SUBMIT_TIMEOUT_S,
    )
    response.raise_for_status()
    return response.json()["prompt_id"]


def _comfy_wait_for_result(comfy_url: str, prompt_id: str) -> dict:
    """Poll /history/{prompt_id} until ComfyUI reports this prompt done,
    then return its "outputs" dict.

    Known caveat (unverified here, flagging for whoever runs this for
    real): ComfyUI's /history reporting for SaveVideo specifically has
    had reliability issues in some versions — a render can finish and
    the file can exist in ComfyUI's output/ folder while /history never
    shows it. If this raises TimeoutError despite ComfyUI's own
    console/log clearly showing the prompt finished, check its output
    folder directly before assuming the call itself is broken.
    """
    deadline = time.monotonic() + COMFY_POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        response = requests.get(f"{comfy_url}/history/{prompt_id}", timeout=30)
        response.raise_for_status()
        history = response.json()
        entry = history.get(prompt_id)
        if entry is not None:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI reported an error for prompt {prompt_id}: {status}")
            if status.get("completed") or status.get("status_str") == "success":
                return entry.get("outputs", {})
        time.sleep(COMFY_POLL_INTERVAL_S)
    raise TimeoutError(f"Timed out after {COMFY_POLL_TIMEOUT_S}s waiting for ComfyUI prompt {prompt_id}")


def _comfy_find_video_file(outputs: dict) -> dict:
    """outputs is node_id -> {field_name: value, ...}; scan every
    list-valued field for a dict with a "filename" key, since the exact
    field name SaveVideo's output lands under has moved between ComfyUI
    versions (images/gifs/videos) — see _comfy_wait_for_result's caveat.
    """
    for node_output in outputs.values():
        for value in node_output.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "filename" in item:
                        return item
    raise RuntimeError(f"No output file found in ComfyUI history outputs: {outputs}")


def _comfy_download(comfy_url: str, file_ref: dict, out_path: Path) -> None:
    params = {
        "filename": file_ref["filename"],
        "subfolder": file_ref.get("subfolder", ""),
        "type": file_ref.get("type", "output"),
    }
    response = requests.get(f"{comfy_url}/view", params=params, timeout=120)
    response.raise_for_status()
    out_path.write_bytes(response.content)


def render_wan_flf(spec: dict, story_reel_dir: Path, out_path: Path) -> Path:
    """Render all shots using Wan 2.2 FLF2V (first-last-frame-to-video)
    through a locally-running ComfyUI instance — Apache-2.0, Alibaba,
    commercially usable, no paid API; cost is just local/rented GPU time.

    Drives lib/story_reel/comfy_workflows/wan_flf2v.json, the API-format
    node graph for the 4-step LoRA-accelerated variant of the official
    Comfy-Org "video_wan2_2_14B_flf2v" template (~70-110s/shot on an
    RTX 4090-class GPU, vs 500s+ without the LoRA — chosen here since the
    whole point of this backend is near-zero cost per shot, and less GPU
    time is less cost). Every node in it is comfy-core (no custom node
    packs); requires these model files under ComfyUI/models/:
      diffusion_models/  wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors
                          wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors
      loras/              wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors
                          wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors
      text_encoders/      umt5_xxl_fp8_e4m3fn_scaled.safetensors
      vae/                wan_2.1_vae.safetensors
    (all from huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged, per
    docs.comfy.org/tutorials/video/wan/wan2_2).

    Per shot: uploads the start/end view images to ComfyUI (/upload/image),
    patches a copy of the workflow template with this shot's prompt/
    negative_prompt/width/height/length/seed, submits it (/prompt), polls
    for completion (/history), and downloads the resulting clip (/view).

    Fails fast with a clear error if ComfyUI isn't reachable at
    config/settings.json's render.comfy_url — checked once up front via
    /system_stats, bounded by COMFY_HEALTHCHECK_TIMEOUT_S — rather than
    hanging on a connection timeout or failing partway through a shot.

    What's actually been verified without a GPU/ComfyUI/model weights
    available in this environment: the unreachable-ComfyUI fail-fast path
    (tested against a real connection-refused address), and that the
    workflow JSON's node graph faithfully reproduces the official
    template's wiring (traced node-by-node from its links, not guessed).
    The live HTTP round-trip (submit/poll/view) against a real ComfyUI
    instance is NOT verified — see this repo's notes on what Ricard's
    GPU box needs before that's possible.
    """
    comfy_url = _comfy_url()
    _check_comfy_reachable(comfy_url)

    fps = int(spec.get("fps", DEFAULT_FPS))
    width = int(spec.get("width", DEFAULT_WIDTH))
    height = int(spec.get("height", DEFAULT_HEIGHT))
    shots = spec.get("shots")
    if not shots:
        raise ValueError("spec has no shots to render")
    view_images = _view_images(spec)
    if not view_images:
        raise ValueError("spec has no product.views to render from")

    template = json.loads(COMFY_WORKFLOW_PATH.read_text(encoding="utf-8"))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="render_wan_flf_") as tmp:
        tmp_dir = Path(tmp)
        clips = []
        for i, shot in enumerate(shots):
            start_path, end_path = _shot_view_paths(shot, view_images, i)
            try:
                start_name = _comfy_upload_image(comfy_url, start_path)
                end_name = _comfy_upload_image(comfy_url, end_path)

                workflow = json.loads(json.dumps(template))  # deep copy
                workflow[WAN_FLF2V_START_IMAGE_NODE]["inputs"]["image"] = start_name
                workflow[WAN_FLF2V_END_IMAGE_NODE]["inputs"]["image"] = end_name
                workflow[WAN_FLF2V_POSITIVE_NODE]["inputs"]["text"] = shot.get("prompt", "")
                workflow[WAN_FLF2V_NEGATIVE_NODE]["inputs"]["text"] = shot.get("negative_prompt", "")
                workflow[WAN_FLF2V_FLF_NODE]["inputs"]["width"] = width
                workflow[WAN_FLF2V_FLF_NODE]["inputs"]["height"] = height
                duration_s = float(shot.get("duration_s") or DEFAULT_SHOT_DURATION_S)
                # WanFirstLastFrameToVideo's "length" is output frame count + 1
                # (see the official template: length=81 @ fps=16 -> 80 output
                # frames -> 5s; the +1 covers the boundary first-frame latent).
                workflow[WAN_FLF2V_FLF_NODE]["inputs"]["length"] = max(round(duration_s * fps), 1) + 1
                seed = shot.get("seed")
                workflow[WAN_FLF2V_SEED_NODE]["inputs"]["noise_seed"] = int(seed) if seed is not None else 0
                workflow[WAN_FLF2V_VIDEO_NODE]["inputs"]["fps"] = fps

                prompt_id = _comfy_submit(comfy_url, workflow)
                outputs = _comfy_wait_for_result(comfy_url, prompt_id)
                file_ref = _comfy_find_video_file(outputs)
            except Exception as exc:
                print(f"[error] wan_flf render call failed for shot #{i}: {exc}")
                raise

            clip_path = tmp_dir / f"shot_{i:03d}.mp4"
            _comfy_download(comfy_url, file_ref, clip_path)
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
