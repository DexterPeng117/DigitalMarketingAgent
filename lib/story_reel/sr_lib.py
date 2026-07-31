"""Minimal local stub for the external story_reel `sr_lib` module.

Real story_reel lives at a Windows-only path
(C:/Users/31660/.openclaw/workspace/ComfyUI/story_reel) that does not exist
on this machine. render_pipeline.py only relies on ROOT / OUTPUT_DIR / COMFY
unconditionally (via load_story_reel()), so this stub provides real working
values for those and nothing else.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMFY = "http://127.0.0.1:8188"
