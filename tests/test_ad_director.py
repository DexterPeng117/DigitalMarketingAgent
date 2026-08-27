"""Tests for scripts/ad_director.py.

generate_storyboard's validation logic is exercised with a mocked LLM
response (requests.post replaced) -- no real OpenRouter calls, no cost.
"""
import json
from unittest.mock import MagicMock

import pytest

import ad_director as m


@pytest.fixture
def views(tmp_path):
    paths = {}
    for name in ("front", "back"):
        p = tmp_path / f"{name}.png"
        p.write_bytes(b"fake png bytes")
        paths[name] = p
    return paths


VALID_DRAFT = {
    "title": "test_ad",
    "tagline": "Punchy tagline.",
    "narration_script": "Sentence one. Sentence two.",
    "scene_prompt": "underwater, dramatic blue lighting",
    "assemble": "cut",
    "shots": [{"start_view": "front", "end_view": "back", "prompt": "slow pan"}],
}


def _fake_response(content):
    """content: dict (will be json-dumped) or a raw string (e.g. non-JSON)."""
    text = json.dumps(content) if isinstance(content, dict) else content
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"choices": [{"message": {"content": text}}]}
    return resp


@pytest.fixture(autouse=True)
def fake_openrouter_config(monkeypatch):
    monkeypatch.setattr(m, "_load_openrouter_config", lambda: ("fake-key", "fake-model"))


def _mock_llm(monkeypatch, content):
    monkeypatch.setattr(m.requests, "post", lambda *a, **k: _fake_response(content))


class TestGenerateStoryboardValidation:
    def test_animate_backend_is_forced_from_argument_not_llm(self, views, monkeypatch):
        _mock_llm(monkeypatch, VALID_DRAFT)
        assert m.generate_storyboard(views, None, "wan_flf")["animate_backend"] == "wan_flf"
        assert m.generate_storyboard(views, None, "interp")["animate_backend"] == "interp"

    def test_missing_title_raises(self, views, monkeypatch):
        _mock_llm(monkeypatch, {**VALID_DRAFT, "title": ""})
        with pytest.raises(ValueError, match="title"):
            m.generate_storyboard(views, None, "interp")

    def test_missing_scene_prompt_raises(self, views, monkeypatch):
        _mock_llm(monkeypatch, {**VALID_DRAFT, "scene_prompt": ""})
        with pytest.raises(ValueError, match="scene_prompt"):
            m.generate_storyboard(views, None, "interp")

    def test_missing_narration_script_raises(self, views, monkeypatch):
        _mock_llm(monkeypatch, {**VALID_DRAFT, "narration_script": ""})
        with pytest.raises(ValueError, match="narration_script"):
            m.generate_storyboard(views, None, "interp")

    def test_empty_shots_list_raises(self, views, monkeypatch):
        _mock_llm(monkeypatch, {**VALID_DRAFT, "shots": []})
        with pytest.raises(ValueError, match="shots"):
            m.generate_storyboard(views, None, "interp")

    def test_missing_shots_key_raises(self, views, monkeypatch):
        draft = {k: v for k, v in VALID_DRAFT.items() if k != "shots"}
        _mock_llm(monkeypatch, draft)
        with pytest.raises(ValueError, match="shots"):
            m.generate_storyboard(views, None, "interp")

    def test_invalid_start_view_raises(self, views, monkeypatch):
        draft = {**VALID_DRAFT, "shots": [{"start_view": "nope", "end_view": "back", "prompt": "p"}]}
        _mock_llm(monkeypatch, draft)
        with pytest.raises(ValueError, match="start_view"):
            m.generate_storyboard(views, None, "interp")

    def test_invalid_end_view_raises(self, views, monkeypatch):
        draft = {**VALID_DRAFT, "shots": [{"start_view": "front", "end_view": "nope", "prompt": "p"}]}
        _mock_llm(monkeypatch, draft)
        with pytest.raises(ValueError, match="end_view"):
            m.generate_storyboard(views, None, "interp")

    def test_shot_missing_prompt_raises(self, views, monkeypatch):
        draft = {**VALID_DRAFT, "shots": [{"start_view": "front", "end_view": "back", "prompt": ""}]}
        _mock_llm(monkeypatch, draft)
        with pytest.raises(ValueError):
            m.generate_storyboard(views, None, "interp")

    def test_non_json_response_raises_clear_error(self, views, monkeypatch, capsys):
        _mock_llm(monkeypatch, "this is not json")
        with pytest.raises(ValueError, match="not valid JSON"):
            m.generate_storyboard(views, None, "interp")
        assert "did not return valid JSON" in capsys.readouterr().out

    def test_product_views_reflect_input_views(self, views, monkeypatch):
        _mock_llm(monkeypatch, VALID_DRAFT)
        spec = m.generate_storyboard(views, None, "interp")
        assert {v["view"] for v in spec["product"]["views"]} == {"front", "back"}

    def test_audio_block_has_tagline_and_narration_script(self, views, monkeypatch):
        _mock_llm(monkeypatch, VALID_DRAFT)
        spec = m.generate_storyboard(views, None, "interp")
        assert spec["audio"]["tagline"] == "Punchy tagline."
        assert spec["audio"]["narration_script"] == "Sentence one. Sentence two."

    def test_invalid_assemble_falls_back_to_default(self, views, monkeypatch):
        _mock_llm(monkeypatch, {**VALID_DRAFT, "assemble": "not-a-real-mode"})
        spec = m.generate_storyboard(views, None, "interp")
        assert spec["assemble"] == m.DEFAULT_ASSEMBLE


class TestWriteSpec:
    def test_slugifies_title_and_keeps_json_in_sync_with_filename(self, tmp_path):
        spec = {"title": "My Cool Ad! (v2)"}
        path = m.write_spec(spec, tmp_path)
        assert path.name == "my_cool_ad_v2.json"
        assert spec["title"] == "my_cool_ad_v2"
        assert json.loads(path.read_text())["title"] == "my_cool_ad_v2"

    def test_empty_title_falls_back_to_untitled(self, tmp_path):
        path = m.write_spec({"title": ""}, tmp_path)
        assert path.name == "untitled.json"

    def test_creates_out_dir_if_missing(self, tmp_path):
        out_dir = tmp_path / "nested" / "workflows"
        path = m.write_spec({"title": "x"}, out_dir)
        assert path.exists()


class TestLoadViews:
    def test_loads_image_files_keyed_by_stem(self, tmp_path):
        (tmp_path / "front.png").write_bytes(b"x")
        (tmp_path / "side.jpg").write_bytes(b"x")
        (tmp_path / "back.jpeg").write_bytes(b"x")
        (tmp_path / "notes.txt").write_text("ignore me")
        views = m.load_views(tmp_path)
        assert set(views.keys()) == {"front", "side", "back"}
        assert views["front"] == tmp_path / "front.png"

    def test_missing_directory_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            m.load_views(tmp_path / "does-not-exist")

    def test_directory_with_no_images_raises_value_error(self, tmp_path):
        (tmp_path / "readme.txt").write_text("no images here")
        with pytest.raises(ValueError):
            m.load_views(tmp_path)
