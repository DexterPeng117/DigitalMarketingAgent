"""Minimal local stub for the external story_reel `sr_keyframe` module.

NEG_DEFAULT is read unconditionally as a dict.get() default inside
stage_backgrounds(), so it must be a real string even when unused.
gen_t2i() is only called when ComfyUI is online and no background image
was provided/cached; with ComfyUI offline this stub should never be hit.
"""

NEG_DEFAULT = (
    "blurry, low quality, distorted, deformed, watermark, text, logo, "
    "extra objects, duplicate product, jpeg artifacts, noise"
)


def gen_t2i(prompt, filename, width, height, neg, seed):
    raise NotImplementedError(
        "sr_keyframe.gen_t2i is a stub in lib/story_reel — it requires "
        "a live ComfyUI instance and should not be called while ComfyUI is offline."
    )
