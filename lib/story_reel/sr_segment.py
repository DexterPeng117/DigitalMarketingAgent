"""Minimal local stub for the external story_reel `sr_segment` module.

NEG_DEFAULT is read unconditionally as a dict.get() default inside
stage_animate(), so it must be a real string even when unused.
stage_input() and build_prompt() are only used by the "ltx" animate
backend (LTX FLF via ComfyUI), which requires ComfyUI online and is not
exercised by the "interp" (default) or "composite" stages.
"""

NEG_DEFAULT = (
    "blurry, low quality, distorted, deformed, watermark, text, logo, "
    "extra objects, duplicate product, jpeg artifacts, noise"
)


def stage_input(path, name):
    raise NotImplementedError(
        "sr_segment.stage_input is a stub in lib/story_reel — it is only "
        "needed by the ltx animate backend, which requires ComfyUI online."
    )


def build_prompt(start_name, end_name, prompt, neg, prefix, seconds, fps, width,
                  height, likeness, end_strength, seed, fast):
    raise NotImplementedError(
        "sr_segment.build_prompt is a stub in lib/story_reel — it is only "
        "needed by the ltx animate backend, which requires ComfyUI online."
    )
