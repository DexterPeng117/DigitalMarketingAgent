"""story_reel `sr_concat` module — pure ffmpeg clip concatenation.

concat_xfade() is only used when spec["assemble"] == "xfade"; the "cut"
assemble mode (this pipeline's default) never calls it (render_pipeline.py
handles "cut" itself via a plain ffmpeg concat demuxer).

Unlike sr_keyframe.py/sr_segment.py, this module has no ComfyUI
dependency — concat_xfade is just ffmpeg's xfade video filter chained
across consecutive clips, so it works with any backend's output
(render_interp's crossfade clips today, render_wan_flf's cloud-rendered
clips once that backend is live).
"""
import subprocess
from pathlib import Path


def _ffprobe_duration(path: Path) -> float:
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path),
    ])
    return float(out.decode().strip())


def concat_xfade(clips, final, width, height, fps, xfade):
    """Concatenate `clips` (in playback order) into `final`, crossfading
    each pair of consecutive clips over `xfade` seconds using ffmpeg's
    xfade filter. `width`/`height`/`fps` normalize every clip before
    chaining, so this doesn't assume all clips were encoded identically.
    """
    clips = [Path(c) for c in clips]
    final = Path(final)
    if not clips:
        raise ValueError("concat_xfade requires at least one clip")

    if len(clips) == 1:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(clips[0]), "-c", "copy", str(final)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed copying single clip to {final}:\n{result.stderr}")
        return final

    durations = [_ffprobe_duration(c) for c in clips]

    inputs: list[str] = []
    for clip in clips:
        inputs += ["-i", str(clip)]

    filter_parts = [
        f"[{i}:v]scale={width}:{height},setsar=1,fps={fps}[v{i}]" for i in range(len(clips))
    ]

    prev_label = "v0"
    cumulative = durations[0]
    for i in range(1, len(clips)):
        offset = max(cumulative - xfade, 0.0)
        out_label = f"x{i}"
        filter_parts.append(
            f"[{prev_label}][v{i}]xfade=transition=fade:duration={xfade}:offset={offset:.3f}[{out_label}]"
        )
        cumulative = cumulative - xfade + durations[i]
        prev_label = out_label

    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", ";".join(filter_parts),
        "-map", f"[{prev_label}]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(final),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg xfade concat failed:\n{result.stderr}")

    return final
