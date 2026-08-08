"""
finalize_ad.py — Silent rendered video -> finished ad.

Input:
    The spec JSON (workflows/<title>.json, for tagline/copy) and the
    silent rendered video produced by render_pipeline.py
    (outputs/<title>_silent.mp4).

Output:
    The final ad video at outputs/<title>_full.mp4: the silent video with
    narration (TTS) and background music mixed in, and subtitles
    soft-mounted as an MP4 subtitle track.

CLI:
    python scripts/finalize_ad.py <spec_path> \
        --video outputs/<title>_silent.mp4 \
        [--music path/to/bed.mp3] \
        --out outputs/<title>_full.mp4

TTS:
    OpenRouter's text-to-speech endpoint (POST
    https://openrouter.ai/api/v1/audio/speech, OpenAI Audio Speech
    API-compatible — openrouter.ai/docs/guides/overview/multimodal/tts).
    Same account as the rest of the pipeline: config/settings.json's
    "llm.api_key", not a separate "tts.api_key". Model defaults to
    "fish-audio/s2.1-pro-free:free" (openrouter.ai/collections/
    text-to-speech-models — the only $0/M-character option there;
    OpenRouter's own docs note free-tier TTS is meant for
    testing/prototyping, not production volume — swap "tts.model" in
    config/settings.json for a paid model when that matters).
    "tts.voice" is provider/model-dependent and left out of the request
    entirely when unset, rather than guessing a voice name.

Subtitles:
    One .srt cue per sentence of spec["audio"]["narration_script"]
    (falling back to "tagline" for older specs from before that field
    existed), evenly distributed across the narration's full duration —
    not driven by real per-word TTS timestamps (the TTS response is raw
    audio bytes, no timing metadata), so this is an even split by
    sentence count rather than true audio-aligned timing. Cheap and
    honest given the data actually available; revisit if word-level
    timestamps ever become available from the TTS provider.

    Soft-mounted (MP4 mov_text track) rather than burned into the
    picture: burn-in needs ffmpeg built with libass (the `subtitles`
    filter) or at least libfreetype (`drawtext`) — verified neither is
    present in this environment's ffmpeg build (`ffmpeg -filters` shows
    no subtitle-related filter at all), so relying on either would make
    this backend silently unportable to any ffmpeg build without those
    optional deps. mov_text muxing is core ffmpeg with no such
    dependency. Tradeoff: some short-form platforms re-encode uploads
    and drop soft subtitle tracks, so burn-in may still be worth
    revisiting once it's clear what ffmpeg build is actually available
    in production.

Mixing:
    Narration (+ background music, ducked under it if --music is given)
    mixed into one audio track; video is stream-copied unmodified
    (no re-encode needed since nothing touches the picture). Output
    length is capped to the silent video's own duration (the
    storyboard's intended pacing) — narration/music are trimmed if
    longer, or just leave trailing silence if shorter.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENROUTER_TTS_URL = "https://openrouter.ai/api/v1/audio/speech"
DEFAULT_TTS_MODEL = "fish-audio/s2.1-pro-free:free"
DEFAULT_TTS_RESPONSE_FORMAT = "mp3"
DEFAULT_MUSIC_VOLUME = 0.25


def load_spec(spec_path: Path) -> dict:
    return json.loads(spec_path.read_text(encoding="utf-8"))


def _run_ffmpeg(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg command failed: {' '.join(cmd)}\n{result.stderr}")


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path),
    ])
    return float(out.decode().strip())


def _load_tts_config() -> tuple[str, str, str, str]:
    """Returns (api_key, model, voice, response_format)."""
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
    tts = settings.get("tts", {})
    model = tts.get("model") or DEFAULT_TTS_MODEL
    voice = tts.get("voice") or ""
    response_format = tts.get("response_format") or DEFAULT_TTS_RESPONSE_FORMAT
    return api_key, model, voice, response_format


def _narration_text(spec: dict) -> str:
    audio = spec.get("audio", {})
    text = str(audio.get("narration_script") or audio.get("tagline") or "").strip()
    if not text:
        raise ValueError("spec['audio'] has neither a non-empty 'narration_script' nor 'tagline'; nothing to narrate")
    return text


def synthesize_narration(spec: dict, out_dir: Path) -> Path:
    """Generate a narration audio track from spec["audio"]["narration_script"]
    (falling back to "tagline" for older specs) via OpenRouter's TTS
    endpoint. Returns the path to the generated audio file.
    """
    text = _narration_text(spec)

    api_key, model, voice, response_format = _load_tts_config()

    payload = {"model": model, "input": text, "response_format": response_format}
    if voice:
        payload["voice"] = voice

    try:
        response = requests.post(
            OPENROUTER_TTS_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(f"OpenRouter TTS request failed ({response.status_code}): {response.text}")
    except Exception as exc:
        print(f"[error] TTS synthesis failed: {exc}")
        raise

    out_dir.mkdir(parents=True, exist_ok=True)
    narration_path = out_dir / f"narration.{response_format}"
    narration_path.write_bytes(response.content)
    return narration_path


def _srt_timestamp(seconds: float) -> str:
    total_ms = round(max(seconds, 0) * 1000)
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter: break after '.', '!', or '?' followed by
    whitespace. No NLP — good enough for the short narration scripts this
    generates, and a wrong split just means a slightly off cue boundary,
    not broken output."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def build_subtitles(spec: dict, narration_path: Path, out_dir: Path) -> Path:
    """Write one .srt cue per sentence of the narration text, evenly
    distributed across the narration's full duration (see module
    docstring's "Subtitles" section for why this is an even split
    rather than true audio-aligned timing).
    """
    text = _narration_text(spec)
    duration_s = _ffprobe_duration(narration_path)
    sentences = _split_sentences(text) or [text]
    per_cue = duration_s / len(sentences)

    cues = []
    for i, sentence in enumerate(sentences):
        start = i * per_cue
        end = duration_s if i == len(sentences) - 1 else (i + 1) * per_cue
        cues.append(f"{i + 1}\n{_srt_timestamp(start)} --> {_srt_timestamp(end)}\n{sentence}")

    out_dir.mkdir(parents=True, exist_ok=True)
    srt_path = out_dir / "subtitles.srt"
    srt_path.write_text("\n\n".join(cues) + "\n", encoding="utf-8")
    return srt_path


def mix_and_mux(video_path: Path, narration_path: Path, subtitles_path: Path,
                 music_path: Path | None, out_path: Path) -> Path:
    """Mix narration (+ optional background music, ducked under it) into
    one audio track, soft-mount the subtitles as an MP4 subtitle track,
    and mux all three with the (stream-copied, unmodified) video. See
    module docstring's "Subtitles"/"Mixing" sections for why.
    """
    video_duration = _ffprobe_duration(video_path)

    inputs = ["-i", str(video_path), "-i", str(narration_path)]
    if music_path is not None:
        inputs += ["-i", str(music_path)]
        audio_args = [
            "-filter_complex",
            "[1:a]volume=1.0[narr];"
            f"[2:a]volume={DEFAULT_MUSIC_VOLUME}[music];"
            "[narr][music]amix=inputs=2:duration=first:dropout_transition=2[aout]",
            "-map", "[aout]",
        ]
    else:
        audio_args = ["-map", "1:a"]

    subtitle_input_index = len(inputs) // 2
    inputs += ["-i", str(subtitles_path)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _run_ffmpeg([
        "ffmpeg", "-y", *inputs,
        "-map", "0:v", *audio_args,
        "-map", f"{subtitle_input_index}:s",
        "-t", f"{video_duration:.3f}",
        "-c:v", "copy", "-c:a", "aac", "-c:s", "mov_text",
        str(out_path),
    ])
    return out_path


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
