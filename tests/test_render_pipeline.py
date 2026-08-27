"""Tests for scripts/render_pipeline.py.

Pure-logic coverage: _shot_view_paths, _load_and_cover (real PIL, no
network/ffmpeg), and render()'s backend dispatch (BACKENDS entries
monkeypatched to fakes, so no real rendering/network/ffmpeg happens).
"""
from pathlib import Path

import pytest
from PIL import Image

import render_pipeline as m


class TestShotViewPaths:
    def test_returns_paths_for_known_views(self):
        shot = {"start_view": "front", "end_view": "back"}
        view_images = {"front": "a.png", "back": "b.png"}
        start, end = m._shot_view_paths(shot, view_images, 0)
        assert start == Path("a.png")
        assert end == Path("b.png")

    def test_unknown_start_view_raises_with_shot_index(self):
        shot = {"start_view": "nope", "end_view": "back"}
        view_images = {"front": "a.png", "back": "b.png"}
        with pytest.raises(ValueError, match=r"Shot #2.*unknown view"):
            m._shot_view_paths(shot, view_images, 2)

    def test_unknown_end_view_raises(self):
        shot = {"start_view": "front", "end_view": "nope"}
        view_images = {"front": "a.png", "back": "b.png"}
        with pytest.raises(ValueError):
            m._shot_view_paths(shot, view_images, 0)


class TestLoadAndCover:
    def test_crops_wide_source_to_exact_target_size(self, tmp_path):
        src = tmp_path / "wide.png"
        Image.new("RGB", (400, 200), "red").save(src)
        result = m._load_and_cover(src, 100, 100)
        assert result.size == (100, 100)

    def test_crops_tall_source_to_exact_target_size(self, tmp_path):
        src = tmp_path / "tall.png"
        Image.new("RGB", (200, 400), "blue").save(src)
        result = m._load_and_cover(src, 300, 150)
        assert result.size == (300, 150)

    def test_never_upscales_below_target(self, tmp_path):
        src = tmp_path / "small.png"
        Image.new("RGB", (50, 50), "green").save(src)
        result = m._load_and_cover(src, 200, 400)
        assert result.size == (200, 400)


class TestRenderDispatch:
    def test_routes_interp_to_the_right_function(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setitem(m.BACKENDS, "interp", lambda spec, d, o: calls.append("interp") or o)
        m.render({"animate_backend": "interp"}, tmp_path, tmp_path / "out.mp4")
        assert calls == ["interp"]

    def test_routes_wan_flf_to_the_right_function(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setitem(m.BACKENDS, "wan_flf", lambda spec, d, o: calls.append("wan_flf") or o)
        m.render({"animate_backend": "wan_flf"}, tmp_path, tmp_path / "out.mp4")
        assert calls == ["wan_flf"]

    def test_routes_wan_flf_local_to_the_right_function(self, monkeypatch, tmp_path):
        calls = []
        monkeypatch.setitem(m.BACKENDS, "wan_flf_local", lambda spec, d, o: calls.append("wan_flf_local") or o)
        m.render({"animate_backend": "wan_flf_local"}, tmp_path, tmp_path / "out.mp4")
        assert calls == ["wan_flf_local"]

    def test_unknown_backend_raises_and_lists_valid_options(self, tmp_path):
        with pytest.raises(ValueError, match="animate_backend"):
            m.render({"animate_backend": "not-a-real-backend"}, tmp_path, tmp_path / "out.mp4")

    def test_missing_animate_backend_key_raises(self, tmp_path):
        with pytest.raises(ValueError):
            m.render({}, tmp_path, tmp_path / "out.mp4")

    def test_does_not_call_other_backends(self, monkeypatch, tmp_path):
        wrong_calls = []
        monkeypatch.setitem(m.BACKENDS, "interp", lambda spec, d, o: wrong_calls.append("interp") or o)
        monkeypatch.setitem(m.BACKENDS, "wan_flf_local", lambda spec, d, o: calls_right.append(1) or o)
        calls_right = []
        m.render({"animate_backend": "wan_flf_local"}, tmp_path, tmp_path / "out.mp4")
        assert wrong_calls == []
        assert calls_right == [1]
