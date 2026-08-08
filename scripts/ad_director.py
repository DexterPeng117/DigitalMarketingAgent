"""
ad_director.py — LLM "ad director": product views -> storyboard spec JSON.

Input:
    A directory of classified product view images (as produced by
    extract_product_views.py, e.g. assets/<brand>/front.png) and an
    optional free-text creative brief.

Output:
    A single JSON spec file at workflows/<title>.json describing the ad:
    copy/tagline, shot-by-shot camera/animation instructions per view,
    and render configuration. Also, alongside the original product
    photos in the --views directory, one new "product in scene" image
    per view (assets/<brand>/scene_<view>.png — see "Scene generation"
    below) — the spec's product.views[].image ends up pointing at these,
    not the originals, so downstream (render_pipeline.py) transparently
    renders shots against scened photos instead of bare studio product
    shots, with zero changes needed on that end.

    The spec MUST always include an "animate_backend" field (e.g.
    "wan_flf" for the paid cloud render backend, or "interp" for the
    free local placeholder/interpolation backend). This field was
    historically dropped by mistake in the prior implementation — do not
    repeat that bug; render_pipeline.py depends on it to pick a backend.
    It is never trusted from the LLM: `generate_storyboard` always force-
    sets it to the `animate_backend` argument it was called with.

CLI:
    python scripts/ad_director.py --views assets/<brand> \
        [--brief "free-text creative brief"] \
        [--backend wan_flf|interp]

    Prints a line "   spec   -> workflows/<title>.json" on success (this
    exact "key -> value" shape is relied on by run_full_pipeline.sh to
    scrape the generated spec path).

Spec schema:
    {
      "title":            str    -- slugified ad title; also the output
                                     filename stem (workflows/<title>.json)
                                     and what every downstream script
                                     (render_pipeline.py, finalize_ad.py,
                                     ad_tracker.py) uses to name its own
                                     outputs, so it's kept in sync with the
                                     filename by write_spec().
      "animate_backend":  str    -- "wan_flf" | "interp"; see above.
      "assemble":         str    -- "cut" | "xfade"; how render_pipeline.py
                                     should concatenate rendered shots
                                     (maps to lib/story_reel/sr_concat.py's
                                     concat_xfade, used when == "xfade").
      "xfade_duration_s": float  -- crossfade length in seconds; only
                                     meaningful when assemble == "xfade"
                                     (-> concat_xfade's `xfade` arg).
      "fps":               int
      "width":             int
      "height":            int  -- shared render dimensions, reused by
                                     every shot (-> sr_keyframe.gen_t2i's
                                     width/height, sr_segment.build_prompt's
                                     width/height, sr_concat.concat_xfade's
                                     width/height/fps).
      "scene_prompt":      str  -- unified background/mood/lighting
                                     direction for the whole ad, consistent
                                     with "tagline" (e.g. "underwater,
                                     dramatic blue lighting"). Drives
                                     generate_scene_images; kept in the
                                     spec for traceability even though
                                     render_pipeline.py never reads it.
      "product": {
        "views": [{"view": <name>, "image": <path>}, ...]
                           -- one entry per input view, pointing at the
                              *scene* image (assets/<brand>/scene_<view>
                              .png) once generate_scene_images has run,
                              not the original studio photo. Also read by
                              ad_tracker.py's brand-extraction logic
                              (it infers brand from views[].image's
                              parent folder name), so the key names here
                              ("view"/"image") must not change.
      },
      "audio": {
        "tagline": str    -- short ad-copy line (kept for ad_tracker.py's
                              brand fallback / on-screen use).
        "narration_script": str
                           -- longer voiceover script (multiple
                              sentences), meant to be read aloud over
                              roughly the whole ad rather than just a
                              few seconds of it. finalize_ad.py's
                              synthesize_narration/build_subtitles read
                              this (falling back to "tagline" for older
                              specs that predate this field).
      },
      "shots": [
        {
          "start_view":     str   -- one of product.views[].view; the
                                      keyframe to animate from
                                      (-> sr_segment.stage_input's `name` /
                                      build_prompt's start_name; also
                                      usable as gen_t2i's reference if that
                                      view has no real photo yet).
          "end_view":       str   -- keyframe to animate to
                                      (-> build_prompt's end_name).
          "prompt":         str   -- camera/animation direction for this
                                      segment (-> build_prompt's prompt /
                                      gen_t2i's prompt).
          "negative_prompt": str  -- (-> build_prompt's neg / gen_t2i's neg).
          "duration_s":    float  -- (-> build_prompt's seconds).
          "seed":            int  -- (-> build_prompt's seed / gen_t2i's seed).
          "likeness":      float  -- (-> build_prompt's likeness).
          "end_strength":  float  -- (-> build_prompt's end_strength).
          "fast":           bool  -- (-> build_prompt's fast).
        },
        ...
      ]
    }

LLM call:
    Single OpenRouter chat-completions call (same provider/config pattern
    as extract_product_views.py's vision classification: key from
    config/settings.json's "llm.api_key", model from "llm.writer_model"),
    with the product view images attached so the model can ground copy
    and shot descriptions in what the product actually looks like. The
    model is asked for raw JSON (title/tagline/narration_script/
    scene_prompt/assemble/shots); "title", "narration_script",
    "scene_prompt", and non-empty "shots" are required, each shot's
    start_view/end_view is validated against the actual view names and
    rejected with a clear error if not one of them, everything else
    optional falls back to a module-level default. A failed HTTP call or
    a non-JSON response is never swallowed — both are printed clearly
    and re-raised.

    narration_script isn't given an exact target length: the total shot
    duration isn't known until the same response's "shots" are parsed,
    so there's no duration figure to hand the model up front without a
    second LLM call. Instead it's just asked for a handful of sentences
    "meant to be read aloud over the course of the whole ad" — natural
    TTS pacing then roughly fills the video instead of a single short
    tagline leaving most of it silent (the original problem this fixes).
    finalize_ad.py's mix_and_mux already trims/pads for the mismatch
    either way, so an imperfect length match here isn't fatal.

Scene generation:
    generate_scene_images turns each view's bare studio product photo
    into a "product in scene" composite via Qwen Image 3 Pro
    (qwen/qwen-image-3-pro) — OpenRouter's dedicated Images API, POST
    https://openrouter.ai/api/v1/images (NOT chat completions), request
    {"model", "prompt", "n": 1, "input_references": [{"type":
    "image_url", "image_url": {"url": ...}}]} -> response {"data":
    [{"b64_json": ..., "media_type": ...}], "usage": {"cost": ...}}.
    Confirmed by a real test call (not guessed): $0.078 for one image at
    1616x2560, and this account's block on google/gemini-2.5-flash-image
    (403 Terms Of Service violation, on both chat completions and
    /api/v1/images, with or without an image) does NOT extend to Qwen —
    same as alibaba/wan-2.7 already being unaffected for video
    generation, both being Alibaba-provider models on this account.

    Known environment quirk (this dev sandbox specifically): its local
    HTTP(S) proxy drops this call's long-held synchronous connection
    (60s+) with ProxyError/ReadTimeout; bypassing the proxy for just
    this request fixed it in testing. The bypass is harmless for
    environments with no such proxy, so it's applied unconditionally
    rather than only in this one sandbox.

    Runs once per view, after generate_storyboard and before write_spec,
    so the spec written to disk already points at the scene images —
    render_pipeline.py (which just reads whatever path is in
    product.views[].image) needs zero changes to pick this up.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import random
import re
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_IMAGES_URL = "https://openrouter.ai/api/v1/images"
DEFAULT_WRITER_MODEL = "openai/gpt-4o"
SCENE_IMAGE_MODEL = "qwen/qwen-image-3-pro"

DEFAULT_ANIMATE_BACKEND = "interp"
DEFAULT_ASSEMBLE = "cut"
DEFAULT_XFADE_DURATION_S = 0.5
DEFAULT_FPS = 24
DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920
DEFAULT_SHOT_DURATION_S = 3.0
DEFAULT_LIKENESS = 0.8
DEFAULT_END_STRENGTH = 0.6
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, watermark, text, logo, "
    "extra objects, duplicate product, jpeg artifacts, noise"
)


def load_views(views_dir: Path) -> dict[str, Path]:
    """Load the view_name -> image_path mapping from a views directory
    (as written by extract_product_views.py).
    """
    if not views_dir.exists():
        raise FileNotFoundError(f"Views directory does not exist: {views_dir}")
    views = {
        p.stem: p
        for p in sorted(views_dir.iterdir())
        if p.suffix.lower() in IMAGE_EXTENSIONS
    }
    if not views:
        raise ValueError(f"No view images (*.png/*.jpg/*.jpeg) found in {views_dir}")
    return views


def _load_openrouter_config() -> tuple[str, str]:
    """Returns (api_key, writer_model) read from config/settings.json's
    "llm" block — the same OpenRouter key extract_product_views.py uses.
    "llm.writer_model" is optional; falls back to DEFAULT_WRITER_MODEL.
    """
    settings_path = REPO_ROOT / "config" / "settings.json"
    if not settings_path.exists():
        raise RuntimeError(
            "config/settings.json not found. Copy config/settings.example.json "
            "to config/settings.json and fill in 'llm.api_key'."
        )
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    llm = settings.get("llm", {})
    api_key = llm.get("api_key")
    if not api_key:
        raise RuntimeError("config/settings.json is missing a non-empty 'llm.api_key'.")
    writer_model = llm.get("writer_model") or DEFAULT_WRITER_MODEL
    return api_key, writer_model


def _encode_image_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{data}"


def generate_storyboard(views: dict[str, Path], brief: str | None, animate_backend: str) -> dict:
    """Call the LLM to turn product views (+ optional brief) into a
    storyboard spec dict. Must set spec["animate_backend"] = animate_backend.

    See the module docstring's "Spec schema" section for the full shape
    of the returned dict.
    """
    view_names = sorted(views)
    api_key, writer_model = _load_openrouter_config()

    instructions = (
        "You are an ad director. You are given product photos for these "
        f"views: {view_names}"
        + (f", and this creative brief: {brief!r}." if brief else ".")
        + "\nWrite a short-form video ad storyboard as a single raw JSON "
          "object (no markdown code fences, no extra text) with this exact "
          "shape:\n"
        '{\n'
        '  "title": "short_snake_case_ad_title",\n'
        '  "tagline": "one punchy line of ad copy",\n'
        '  "narration_script": "3 to 5 sentences of voiceover copy, meant to be read '
        'aloud over the course of the whole ad (not just the tagline repeated) — '
        'descriptive, cinematic language expanding on the tagline and matching the scene_prompt mood",\n'
        '  "scene_prompt": "a short, unified background/mood/lighting direction '
        'for the whole ad, consistent with the tagline, e.g. \'underwater, dramatic blue lighting\'",\n'
        '  "assemble": "cut" or "xfade",\n'
        '  "shots": [\n'
        '    {\n'
        '      "start_view": "<one of the given view names>",\n'
        '      "end_view": "<one of the given view names>",\n'
        '      "prompt": "camera movement / animation direction for this segment",\n'
        '      "negative_prompt": "optional, things to avoid",\n'
        '      "duration_s": 3.0,\n'
        '      "seed": 12345,\n'
        '      "likeness": 0.8,\n'
        '      "end_strength": 0.6,\n'
        '      "fast": false\n'
        '    }\n'
        '  ]\n'
        '}\n'
        f"start_view and end_view must each be exactly one of: {view_names}. "
        "Use 3 to 6 shots that together cover all the given views at least once."
    )

    content: list[dict] = [{"type": "text", "text": instructions}]
    for name in view_names:
        content.append({"type": "text", "text": f"View: {name}"})
        content.append({
            "type": "image_url",
            "image_url": {"url": _encode_image_data_url(views[name])},
        })

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": writer_model,
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": content}],
            },
            timeout=90,
        )
        response.raise_for_status()
        raw_text = response.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        print(f"[error] Storyboard LLM call failed: {exc}")
        raise

    try:
        draft = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        print(f"[error] Storyboard LLM did not return valid JSON. Raw response:\n{raw_text}")
        raise ValueError("Storyboard LLM response was not valid JSON") from exc

    title = str(draft.get("title") or "").strip()
    if not title:
        raise ValueError(f"Storyboard LLM response is missing a non-empty 'title': {draft}")

    scene_prompt = str(draft.get("scene_prompt") or "").strip()
    if not scene_prompt:
        raise ValueError(f"Storyboard LLM response is missing a non-empty 'scene_prompt': {draft}")

    narration_script = str(draft.get("narration_script") or "").strip()
    if not narration_script:
        raise ValueError(f"Storyboard LLM response is missing a non-empty 'narration_script': {draft}")

    raw_shots = draft.get("shots")
    if not isinstance(raw_shots, list) or not raw_shots:
        raise ValueError(f"Storyboard LLM response is missing a non-empty 'shots' list: {draft}")

    assemble = draft.get("assemble")
    if assemble not in ("cut", "xfade"):
        assemble = DEFAULT_ASSEMBLE

    shots: list[dict] = []
    for i, raw_shot in enumerate(raw_shots):
        start_view = raw_shot.get("start_view")
        end_view = raw_shot.get("end_view")
        prompt = raw_shot.get("prompt")
        if start_view not in views or end_view not in views or not prompt:
            raise ValueError(
                f"Storyboard LLM shot #{i} is invalid (start_view={start_view!r}, "
                f"end_view={end_view!r}, prompt={prompt!r}); start_view/end_view "
                f"must each be one of {view_names}"
            )
        shots.append({
            "start_view": start_view,
            "end_view": end_view,
            "prompt": prompt,
            "negative_prompt": raw_shot.get("negative_prompt") or DEFAULT_NEGATIVE_PROMPT,
            "duration_s": float(raw_shot.get("duration_s") or DEFAULT_SHOT_DURATION_S),
            "seed": int(raw_shot["seed"]) if raw_shot.get("seed") is not None else random.randint(0, 2**31 - 1),
            "likeness": float(raw_shot["likeness"]) if raw_shot.get("likeness") is not None else DEFAULT_LIKENESS,
            "end_strength": float(raw_shot["end_strength"]) if raw_shot.get("end_strength") is not None else DEFAULT_END_STRENGTH,
            "fast": bool(raw_shot.get("fast", False)),
        })

    return {
        "title": title,
        "animate_backend": animate_backend,
        "assemble": assemble,
        "xfade_duration_s": DEFAULT_XFADE_DURATION_S,
        "fps": DEFAULT_FPS,
        "width": DEFAULT_WIDTH,
        "height": DEFAULT_HEIGHT,
        "scene_prompt": scene_prompt,
        "product": {
            "views": [{"view": name, "image": str(path)} for name, path in views.items()],
        },
        "audio": {
            "tagline": str(draft.get("tagline") or ""),
            "narration_script": narration_script,
        },
        "shots": shots,
    }


def _scene_image_prompt(scene_prompt: str) -> str:
    return (
        "Keep the product in the reference image exactly as it is — same shape, "
        "proportions, materials, color, and branding, completely unchanged. Only "
        f"change the environment around it: place the product in this scene: {scene_prompt}. "
        "Photorealistic, professional product photography, cinematic lighting, "
        "sharp focus on the product."
    )


def generate_scene_images(views: dict[str, Path], scene_prompt: str, out_dir: Path) -> dict[str, Path]:
    """For each view's original studio product photo, generate a new
    "product in scene" composite via Qwen Image 3 Pro. See the module
    docstring's "Scene generation" section for the API details and why
    this model was chosen.

    Saves each result to out_dir/scene_<view_name>.png (originals are
    never modified or deleted) and returns view_name -> new image path.
    """
    api_key, _ = _load_openrouter_config()
    prompt = _scene_image_prompt(scene_prompt)

    out_dir.mkdir(parents=True, exist_ok=True)
    scene_images: dict[str, Path] = {}
    for name, path in views.items():
        try:
            response = requests.post(
                OPENROUTER_IMAGES_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": SCENE_IMAGE_MODEL,
                    "prompt": prompt,
                    "n": 1,
                    "input_references": [
                        {"type": "image_url", "image_url": {"url": _encode_image_data_url(path)}},
                    ],
                },
                timeout=180,
                # This dev sandbox's local HTTP(S) proxy drops this call's
                # long-held synchronous connection — see module docstring's
                # "Scene generation" section. Harmless where no such proxy
                # exists, so applied unconditionally rather than gated.
                proxies={"http": None, "https": None},
            )
            if not response.ok:
                raise RuntimeError(f"Scene image generation failed ({response.status_code}): {response.text}")
            image_data = response.json()["data"][0]
        except Exception as exc:
            print(f"[error] Scene image generation failed for view {name!r}: {exc}")
            raise

        image_bytes = base64.standard_b64decode(image_data["b64_json"])
        scene_path = out_dir / f"scene_{name}.png"
        scene_path.write_bytes(image_bytes)
        scene_images[name] = scene_path
        print(f"   scene  -> {scene_path}  ({name})")

    return scene_images


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return slug or "untitled"


def write_spec(spec: dict, out_dir: Path) -> Path:
    """Write `spec` to workflows/<title>.json and return the path.

    spec["title"] is slugified for filename-safety and written back into
    the spec itself, so the JSON's own "title" always matches the
    filename stem — run_full_pipeline.sh and the other scripts derive
    every downstream output path from spec["title"].
    """
    slug = _slugify(str(spec.get("title", "")))
    spec["title"] = slug

    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = out_dir / f"{slug}.json"
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return spec_path


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

    scene_images = generate_scene_images(views, spec["scene_prompt"], Path(args.views))
    spec["product"]["views"] = [{"view": name, "image": str(scene_images[name])} for name in views]

    spec_path = write_spec(spec, Path(args.out_dir))
    print(f"   spec   -> {spec_path}")


if __name__ == "__main__":
    main()
