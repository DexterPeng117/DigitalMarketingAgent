"""Tests for scripts/finalize_ad.py.

_ffprobe_duration is monkeypatched to a fixed value throughout (real
ffprobe isn't needed to test the sentence-splitting/cue-timing logic,
and this keeps the suite from depending on ffmpeg being on PATH).
synthesize_narration's TTS call is mocked -- no real OpenRouter calls.
"""
from unittest.mock import MagicMock

import pytest

import finalize_ad as m


class TestSplitSentences:
    def test_splits_on_period_exclaim_question(self):
        assert m._split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_single_sentence_stays_one_cue(self):
        assert m._split_sentences("Just one sentence.") == ["Just one sentence."]

    def test_empty_string_returns_empty_list(self):
        assert m._split_sentences("") == []

    def test_ignores_extra_whitespace_between_sentences(self):
        assert m._split_sentences("One.   Two.") == ["One.", "Two."]


class TestNarrationTextFallback:
    def test_prefers_narration_script_over_tagline(self):
        spec = {"audio": {"tagline": "short", "narration_script": "long script"}}
        assert m._narration_text(spec) == "long script"

    def test_falls_back_to_tagline_for_specs_without_narration_script(self):
        spec = {"audio": {"tagline": "short tagline only"}}
        assert m._narration_text(spec) == "short tagline only"

    def test_raises_when_both_are_empty(self):
        with pytest.raises(ValueError):
            m._narration_text({"audio": {}})

    def test_raises_when_audio_key_missing_entirely(self):
        with pytest.raises(ValueError):
            m._narration_text({})


class TestBuildSubtitles:
    def test_multi_sentence_narration_script_creates_proportional_cues(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "_ffprobe_duration", lambda path: 10.0)
        spec = {"audio": {"narration_script": "First sentence. Second sentence."}}
        narration_path = tmp_path / "narration.mp3"
        narration_path.write_bytes(b"fake audio bytes")

        srt_path = m.build_subtitles(spec, narration_path, tmp_path)
        content = srt_path.read_text()

        assert "1\n00:00:00,000 --> 00:00:05,000\nFirst sentence." in content
        assert "2\n00:00:05,000 --> 00:00:10,000\nSecond sentence." in content

    def test_single_sentence_spans_full_duration(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "_ffprobe_duration", lambda path: 3.6)
        spec = {"audio": {"narration_script": "Just one line."}}
        narration_path = tmp_path / "narration.mp3"
        narration_path.write_bytes(b"x")

        content = m.build_subtitles(spec, narration_path, tmp_path).read_text()
        assert "00:00:00,000 --> 00:00:03,600" in content

    def test_falls_back_to_tagline_for_older_specs(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m, "_ffprobe_duration", lambda path: 3.0)
        spec = {"audio": {"tagline": "Old style tagline."}}
        narration_path = tmp_path / "narration.mp3"
        narration_path.write_bytes(b"x")

        content = m.build_subtitles(spec, narration_path, tmp_path).read_text()
        assert "Old style tagline." in content


class TestSynthesizeNarration:
    @pytest.fixture(autouse=True)
    def fake_tts_config(self, monkeypatch):
        monkeypatch.setattr(m, "_load_tts_config", lambda: ("fake-key", "fake-model", "", "mp3"))

    def _fake_ok_response(self, content=b"audio bytes"):
        resp = MagicMock()
        resp.ok = True
        resp.content = content
        return resp

    def test_sends_narration_script_not_tagline_when_both_present(self, tmp_path, monkeypatch):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured["json"] = json
            return self._fake_ok_response()

        monkeypatch.setattr(m.requests, "post", fake_post)
        spec = {"audio": {"tagline": "short", "narration_script": "the long script"}}
        m.synthesize_narration(spec, tmp_path)

        assert captured["json"]["input"] == "the long script"

    def test_voice_omitted_from_payload_when_unset(self, tmp_path, monkeypatch):
        captured = {}

        def fake_post(url, headers, json, timeout):
            captured["json"] = json
            return self._fake_ok_response()

        monkeypatch.setattr(m.requests, "post", fake_post)
        m.synthesize_narration({"audio": {"tagline": "hi"}}, tmp_path)

        assert "voice" not in captured["json"]

    def test_writes_response_bytes_to_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(m.requests, "post", lambda *a, **k: self._fake_ok_response(b"the mp3 bytes"))
        path = m.synthesize_narration({"audio": {"tagline": "hi"}}, tmp_path)
        assert path.read_bytes() == b"the mp3 bytes"

    def test_raises_clear_error_on_non_ok_response(self, tmp_path, monkeypatch):
        resp = MagicMock()
        resp.ok = False
        resp.status_code = 402
        resp.text = "insufficient credits"
        monkeypatch.setattr(m.requests, "post", lambda *a, **k: resp)

        with pytest.raises(RuntimeError, match="402"):
            m.synthesize_narration({"audio": {"tagline": "hi"}}, tmp_path)

    def test_raises_when_no_text_available_at_all(self, tmp_path):
        with pytest.raises(ValueError):
            m.synthesize_narration({"audio": {}}, tmp_path)
