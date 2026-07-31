#!/usr/bin/env bash
# run_full_pipeline.sh — one-shot, end-to-end run of the ad pipeline.
#
# Covers every stage:
#   1) extract_product_views.py   raw product photo(s) -> named view PNGs
#   2) ad_director.py              views -> spec JSON + storyboard (LLM)
#   3) render_pipeline.py          spec -> silent rendered video
#                                   (backend picked from spec["animate_backend"],
#                                   implemented via lib/story_reel/)
#   4) finalize_ad.py              silent video -> narration + subtitles + music
#   5) ad_tracker.py register      finished ad -> tracked in outputs/ads.csv
#
# Run this from the ad-pipeline-independent repo root (same level as
# scripts/, workflows/, assets/, lib/).
#
# Prereqs:
#   - lib/story_reel/ already in place (sr_lib.py etc.)
#   - config/settings.json filled in (copy from config/settings.example.json)
#   - pip install -r requirements.txt done, ffmpeg on PATH
#
# Usage:
#   ./scripts/run_full_pipeline.sh <raw_product_photo.png> <product_name> ["optional brief"]
#
# Example:
#   ./scripts/run_full_pipeline.sh raw/rolex_threeview_src.png rolex \
#       "targets upwardly-mobile professionals, emphasize dive-grade precision and understated luxury"

set -euo pipefail

RAW_PHOTO="${1:?Usage: $0 <raw_photo.png> <product_name> [brief]}"
PRODUCT_NAME="${2:?Usage: $0 <raw_photo.png> <product_name> [brief]}"
BRIEF="${3:-}"

VIEWS_DIR="assets/${PRODUCT_NAME}"
STORY_REEL_DIR="lib/story_reel"

echo "=================================================================="
echo "STEP 1/5 — extract_product_views.py"
echo "=================================================================="
python3 scripts/extract_product_views.py "$RAW_PHOTO" \
    --out "$VIEWS_DIR" --names front side back

echo ""
echo "=================================================================="
echo "STEP 2/5 — ad_director.py"
echo "=================================================================="
DIRECTOR_ARGS=(--views "$VIEWS_DIR")
if [[ -n "$BRIEF" ]]; then
    DIRECTOR_ARGS+=(--brief "$BRIEF")
fi
DIRECTOR_OUTPUT="$(python3 scripts/ad_director.py "${DIRECTOR_ARGS[@]}" | tee /dev/stderr)"

# Pull the generated spec path out of ad_director.py's own printed output
# (it prints a line like "   spec   -> workflows/<title>.json")
SPEC_PATH="$(echo "$DIRECTOR_OUTPUT" | python3 -c "
import sys
for line in sys.stdin:
    if 'spec' in line and '->' in line and line.strip().endswith('.json'):
        print(line.split('->', 1)[1].strip())
" | tail -1)"
if [[ -z "$SPEC_PATH" ]]; then
    echo "[fatal] Could not find the generated spec path in ad_director.py's output." >&2
    exit 1
fi
echo ""
echo ">>> Generated spec: $SPEC_PATH"

TITLE="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['title'])" "$SPEC_PATH")"
ANIMATE_BACKEND="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['animate_backend'])" "$SPEC_PATH")"

echo ""
echo "=================================================================="
echo "STEP 3/5 — render_pipeline.py (backend: ${ANIMATE_BACKEND})"
echo "=================================================================="
if [[ "$ANIMATE_BACKEND" == "wan_flf" ]]; then
    echo "NOTE: this step calls a paid cloud render API."
    read -r -p "Proceed with real render cost? [y/N] " CONFIRM
    if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
        echo "Aborted before incurring render cost. Spec is saved at: $SPEC_PATH"
        exit 0
    fi
fi

RENDERED_VIDEO="outputs/${TITLE}_silent.mp4"
python3 scripts/render_pipeline.py "$SPEC_PATH" \
    --story-reel-dir "$STORY_REEL_DIR" \
    --out "$RENDERED_VIDEO"

if [[ ! -f "$RENDERED_VIDEO" ]]; then
    echo "[fatal] Expected rendered video not found at $RENDERED_VIDEO" >&2
    exit 1
fi
echo ""
echo ">>> Rendered (silent) video: $RENDERED_VIDEO"

echo ""
echo "=================================================================="
echo "STEP 4/5 — finalize_ad.py (narration + subtitles + music)"
echo "=================================================================="
FINAL_VIDEO="outputs/${TITLE}_full.mp4"
python3 scripts/finalize_ad.py "$SPEC_PATH" \
    --video "$RENDERED_VIDEO" \
    --out "$FINAL_VIDEO"

echo ""
echo ">>> Final video: $FINAL_VIDEO"

echo ""
echo "=================================================================="
echo "STEP 5/5 — ad_tracker.py register"
echo "=================================================================="
python3 scripts/ad_tracker.py register "$SPEC_PATH" --video "$FINAL_VIDEO"

echo ""
echo "=================================================================="
echo "DONE"
echo "=================================================================="
echo "Spec:          $SPEC_PATH"
echo "Silent render: $RENDERED_VIDEO"
echo "Final video:   $FINAL_VIDEO"
echo "Tracked in:    outputs/ads.csv"
