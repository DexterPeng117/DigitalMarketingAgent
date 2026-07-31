"""
extract_product_views.py — Sort a product's raw photos into named views.

Input:
    One or more source images of a single product, covering different
    angles/perspectives (e.g. a raw multi-view reference sheet, or a
    directory containing one image per angle already).

Output:
    assets/<brand>/<view_name>.png for each recognized view (e.g.
    assets/rolex/front.png, assets/rolex/side.png, assets/rolex/back.png),
    ready to be read by ad_director.py.

CLI:
    python scripts/extract_product_views.py <input_path> \
        --out assets/<brand> --names front side back

    <input_path> may be a single image file or a directory of images.
    --names lists the expected view names, in the order/labels the
    classifier should try to assign.

Classification:
    - If <input_path> is a directory (one image per angle already, the
      common case), a single vision-LLM call (Anthropic API) classifies
      all images against `names` at once. Images the model can't match
      to any name are skipped with a CLI warning, not an error.
    - If <input_path> is a single file (a multi-view reference sheet),
      it's split into len(names) equal-width vertical strips, left to
      right, matched in order to `names`. This is a naive fallback — see
      the comment on `_classify_views_grid` for its limitations.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import tempfile
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
VISION_MODEL = "claude-sonnet-4-6"


def _load_anthropic_api_key() -> str:
    settings_path = REPO_ROOT / "config" / "settings.json"
    if not settings_path.exists():
        raise RuntimeError(
            "config/settings.json not found. Copy config/settings.example.json "
            "to config/settings.json and fill in 'anthropic_api_key'."
        )
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    api_key = settings.get("anthropic_api_key")
    if not api_key:
        raise RuntimeError("config/settings.json is missing a non-empty 'anthropic_api_key'.")
    return api_key


def _encode_image(path: Path) -> tuple[str, str]:
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return media_type, data


def _classify_views_dir(input_dir: Path, names: list[str]) -> dict[str, Path]:
    """Case A: `input_dir` holds one already-separate image per angle.

    Sends all images to the vision LLM in a single call along with
    `names`, and asks it to return a filename -> view_name (or null)
    mapping as raw JSON.
    """
    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"No image files found in {input_dir}")

    from anthropic import Anthropic

    client = Anthropic(api_key=_load_anthropic_api_key())

    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "You are given several photos of a single product from different "
                f"angles, and this list of expected view names: {names}.\n"
                "For each image, decide which view name (if any) it best matches.\n"
                "Respond with ONLY a raw JSON object (no markdown code fences, no "
                "extra text), mapping each image's filename to the matching view "
                "name from the list, or null if none of the names fit. Example:\n"
                '{"front.jpg": "front", "IMG_002.jpg": null}'
            ),
        }
    ]
    for path in image_paths:
        media_type, data = _encode_image(path)
        content.append({"type": "text", "text": f"Filename: {path.name}"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        })

    try:
        response = client.messages.create(
            model=VISION_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}],
        )
        raw_text = response.content[0].text
    except Exception as exc:
        print(f"[error] Vision LLM call failed: {exc}")
        raise

    try:
        assignments: dict[str, str | None] = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        print(f"[error] Vision LLM did not return valid JSON. Raw response:\n{raw_text}")
        raise ValueError("Vision LLM response was not valid JSON") from exc

    by_name = {p.name: p for p in image_paths}
    views: dict[str, Path] = {}
    for filename, view_name in assignments.items():
        path = by_name.get(filename)
        if path is None:
            continue
        if view_name is None or view_name not in names:
            print(f"[!] {filename} did not match any view name, skipped")
            continue
        if view_name in views:
            # Model assigned this name to more than one image; keep the first.
            continue
        views[view_name] = path

    unassigned = [p.name for p in image_paths if p.name not in assignments]
    for filename in unassigned:
        print(f"[!] {filename} was not classified by the model, skipped")

    return views


def _classify_views_grid(input_path: Path, names: list[str]) -> dict[str, Path]:
    """Case B: `input_path` is a single multi-view reference sheet.

    Naive fallback: splits the image into len(names) equal-width vertical
    strips, left-to-right, matched in order to `names`. This is NOT smart
    layout detection — if the reference image isn't a simple horizontal
    grid of views, the crops will be wrong. A future improvement would be
    to use a vision LLM to detect the actual view regions instead.
    """
    image = Image.open(input_path)
    width, height = image.size
    n = len(names)
    strip_width = width // n

    tmp_dir = Path(tempfile.mkdtemp(prefix="extract_product_views_"))
    views: dict[str, Path] = {}
    for i, name in enumerate(names):
        left = i * strip_width
        right = width if i == n - 1 else (i + 1) * strip_width
        crop = image.crop((left, 0, right, height))
        crop_path = tmp_dir / f"{name}.png"
        crop.save(crop_path)
        views[name] = crop_path

    return views


def classify_views(input_path: Path, names: list[str]) -> dict[str, Path]:
    """Classify source image(s) at `input_path` into named product views.

    Returns a mapping of view_name -> path of the (possibly temporary)
    image to use for that view. `names` is the ordered list of expected
    view labels (e.g. ["front", "side", "back"]).
    """
    if input_path.is_dir():
        return _classify_views_dir(input_path, names)
    return _classify_views_grid(input_path, names)


def save_views(views: dict[str, Path], out_dir: Path) -> dict[str, Path]:
    """Write each classified view image to `out_dir/<view_name>.png`.

    Returns a mapping of view_name -> final saved path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for name, path in views.items():
        image = Image.open(path)
        out_path = out_dir / f"{name}.png"
        image.save(out_path, format="PNG")
        saved[name] = out_path
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Source image file or directory of source images")
    ap.add_argument("--out", required=True, help="Output directory, e.g. assets/<brand>")
    ap.add_argument("--names", nargs="+", required=True, help="Expected view names, e.g. front side back")
    args = ap.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)

    views = classify_views(input_path, args.names)
    saved = save_views(views, out_dir)

    for name, path in saved.items():
        print(f"   view   -> {path}  ({name})")


if __name__ == "__main__":
    main()
