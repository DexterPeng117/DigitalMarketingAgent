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

TODO(design confirmed, implementation pending):
    - Decide classification strategy (vision-LLM angle detection vs.
      simple grid-split vs. filename heuristics) and implement it in
      `classify_views`.
    - Decide how to handle inputs with fewer/more images than `--names`.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def classify_views(input_path: Path, names: list[str]) -> dict[str, Path]:
    """Classify source image(s) at `input_path` into named product views.

    Returns a mapping of view_name -> path of the (possibly temporary)
    image to use for that view. `names` is the ordered list of expected
    view labels (e.g. ["front", "side", "back"]).

    TODO: implement. For a single multi-view sheet this likely means
    detecting and cropping sub-regions; for a directory of separate
    images it likely means matching each file to the closest label.
    """
    raise NotImplementedError


def save_views(views: dict[str, Path], out_dir: Path) -> dict[str, Path]:
    """Write each classified view image to `out_dir/<view_name>.png`.

    Returns a mapping of view_name -> final saved path.

    TODO: implement (load via Pillow, convert/save as PNG).
    """
    raise NotImplementedError


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
