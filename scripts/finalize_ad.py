"""
finalize_ad.py — Silent rendered video -> finished ad.

Input:
    The spec JSON (workflows/<title>.json, for tagline/copy) and the
    silent rendered video produced by render_pipeline.py
    (outputs/<title>_silent.mp4).

Output:
    The final ad video at outputs/<title>_full.mp4, with narration (TTS),
    burned-in or soft subtitles, and background music mixed in.

CLI:
    python scripts/finalize_ad.py <spec_path> \
        --video outputs/<title>_silent.mp4 \
        --out outputs/<title>_full.mp4

TODO(design confirmed, implementation pending):
    - Implement `synthesize_narration` (TTS call, config in
      config/settings.json under "tts").
    - Implement `build_subtitles` (from spec copy + narration timing).
    - Implement `mix_and_mux` (ffmpeg: overlay subtitles, mix narration +
      background music with the silent video track).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_spec(spec_path: Path) -> dict:
    return json.loads(spec_path.read_text(encoding="utf-8"))


def synthesize_narration(spec: dict, out_dir: Path) -> Path:
    """Generate a narration audio track from the spec's tagline/copy.

    Returns the path to the generated audio file.

    TODO: implement (call TTS provider configured in config/settings.json).
    """
    raise NotImplementedError


def build_subtitles(spec: dict, narration_path: Path, out_dir: Path) -> Path:
    """Generate a subtitle file (e.g. .srt) aligned to the narration audio.

    TODO: implement.
    """
    raise NotImplementedError


def mix_and_mux(video_path: Path, narration_path: Path, subtitles_path: Path,
                 music_path: Path | None, out_path: Path) -> Path:
    """Combine the silent video with narration, subtitles, and optional
    background music into the final ad video.

    TODO: implement (ffmpeg audio mix + subtitle burn-in/mux).
    """
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="Path to the workflow spec JSON")
    ap.add_argument("--video", required=True, help="Path to the silent rendered video")
    ap.add_argument("--music", default=None, help="Optional path to a background music track")
    ap.add_argument("--out", required=True, help="Output path for the finished ad video")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    spec = load_spec(spec_path)
    video_path = Path(args.video)
    out_path = Path(args.out)

    narration_path = synthesize_narration(spec, out_path.parent)
    subtitles_path = build_subtitles(spec, narration_path, out_path.parent)
    music_path = Path(args.music) if args.music else None

    final_path = mix_and_mux(video_path, narration_path, subtitles_path, music_path, out_path)
    print(f"   final  -> {final_path}")


if __name__ == "__main__":
    main()
