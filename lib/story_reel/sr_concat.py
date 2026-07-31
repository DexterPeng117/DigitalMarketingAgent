"""Minimal local stub for the external story_reel `sr_concat` module.

concat_xfade() is only used when spec["assemble"] == "xfade"; the "cut"
assemble mode (this pipeline's default) never calls it.
"""


def concat_xfade(clips, final, width, height, fps, xfade):
    raise NotImplementedError(
        "sr_concat.concat_xfade is a stub in lib/story_reel — it is only "
        "needed when spec['assemble'] == 'xfade'; use assemble='cut' instead."
    )
