"""Tests for scripts/ad_tracker.py.

Pure-logic coverage only (no real ffmpeg/network calls): ADS_PATH/PUBS_PATH
are monkeypatched to a tmp_path per test via the `isolated_csvs` fixture,
so nothing here ever touches the real outputs/ directory. Video files
created for these tests are fake bytes, not real videos — _ffprobe_duration
already degrades gracefully (returns "") when ffprobe can't parse them, so
duration_s just comes back blank, which doesn't affect anything tested here.
"""
import json
from types import SimpleNamespace

import pytest

import ad_tracker as m


@pytest.fixture
def isolated_csvs(tmp_path, monkeypatch):
    """Point ad_tracker's module-level ADS_PATH/PUBS_PATH at tmp_path for
    the duration of one test, instead of the real repo's outputs/."""
    ads_path = tmp_path / "ads.csv"
    pubs_path = tmp_path / "publications.csv"
    monkeypatch.setattr(m, "ADS_PATH", ads_path)
    monkeypatch.setattr(m, "PUBS_PATH", pubs_path)
    return ads_path, pubs_path


def _make_video_and_spec(tmp_path, title="my_ad", brand="rolex"):
    spec = {
        "title": title,
        "product": {"views": [{"view": "front", "image": f"assets/{brand}/front.png"}]},
        "audio": {"tagline": "test tagline"},
    }
    spec_path = tmp_path / f"{title}.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    video_path = tmp_path / f"{title}_full.mp4"
    video_path.write_bytes(b"not a real video, just needs to exist")
    return spec_path, video_path


def _register(tmp_path, isolated_csvs, title="my_ad", brand="rolex"):
    spec_path, video_path = _make_video_and_spec(tmp_path, title, brand)
    m.cmd_register(SimpleNamespace(spec=str(spec_path), video=str(video_path)))
    return video_path.stem


class TestExtractBrand:
    def test_from_view_image_parent_folder(self):
        spec = {"product": {"views": [{"view": "front", "image": "assets/rolex/front.png"}]}}
        assert m._extract_brand(spec) == "rolex"

    def test_falls_back_to_tagline_em_dash(self):
        spec = {"audio": {"tagline": "Some ad copy — Acme"}}
        assert m._extract_brand(spec) == "Acme"

    def test_no_signal_returns_empty_string(self):
        assert m._extract_brand({}) == ""

    def test_view_image_takes_priority_over_tagline(self):
        spec = {
            "product": {"views": [{"view": "front", "image": "assets/rolex/front.png"}]},
            "audio": {"tagline": "Some copy — OtherBrand"},
        }
        assert m._extract_brand(spec) == "rolex"


class TestGuessSpecPath:
    def test_matches_full_suffix(self, tmp_path):
        (tmp_path / "my_ad.json").write_text("{}")
        assert m._guess_spec_path("my_ad_full", tmp_path) == tmp_path / "my_ad.json"

    def test_matches_final_suffix(self, tmp_path):
        (tmp_path / "my_ad.json").write_text("{}")
        assert m._guess_spec_path("my_ad_final", tmp_path) == tmp_path / "my_ad.json"

    def test_silent_suffix_is_not_recognized(self, tmp_path):
        # _silent is render_pipeline.py's intermediate output, not a
        # finished ad — deliberately not one of the guessed suffixes.
        (tmp_path / "my_ad.json").write_text("{}")
        assert m._guess_spec_path("my_ad_silent", tmp_path) is None

    def test_no_matching_spec_file_returns_none(self, tmp_path):
        assert m._guess_spec_path("my_ad_full", tmp_path) is None


class TestRegister:
    def test_registers_and_extracts_brand(self, tmp_path, isolated_csvs):
        ads_path, _ = isolated_csvs
        ad_id = _register(tmp_path, isolated_csvs)

        rows = m._load_csv(ads_path)
        assert len(rows) == 1
        assert rows[0]["ad_id"] == ad_id
        assert rows[0]["brand"] == "rolex"
        assert rows[0]["status"] == "generated"

    def test_duplicate_register_is_skipped(self, tmp_path, isolated_csvs, capsys):
        ads_path, _ = isolated_csvs
        spec_path, video_path = _make_video_and_spec(tmp_path)
        args = SimpleNamespace(spec=str(spec_path), video=str(video_path))
        m.cmd_register(args)
        m.cmd_register(args)

        rows = m._load_csv(ads_path)
        assert len(rows) == 1
        assert "already registered" in capsys.readouterr().out


class TestPublish:
    def test_publish_marks_ad_published_and_writes_pub_row(self, tmp_path, isolated_csvs):
        ads_path, pubs_path = isolated_csvs
        ad_id = _register(tmp_path, isolated_csvs)

        m.cmd_publish(SimpleNamespace(ad_id=ad_id, platform="instagram", post_id="1", url="https://instagram.com/p/1"))

        ads = m._load_csv(ads_path)
        pubs = m._load_csv(pubs_path)
        assert ads[0]["status"] == "published"
        assert len(pubs) == 1
        assert pubs[0]["pub_id"] == f"{ad_id}__instagram"

    def test_same_ad_multiple_platforms(self, tmp_path, isolated_csvs):
        _, pubs_path = isolated_csvs
        ad_id = _register(tmp_path, isolated_csvs)

        m.cmd_publish(SimpleNamespace(ad_id=ad_id, platform="instagram", post_id="1", url="https://instagram.com/p/1"))
        m.cmd_publish(SimpleNamespace(ad_id=ad_id, platform="tiktok", post_id="2", url="https://tiktok.com/@x/2"))

        pubs = m._load_csv(pubs_path)
        assert len(pubs) == 2
        assert {p["platform"] for p in pubs} == {"instagram", "tiktok"}
        assert {p["ad_id"] for p in pubs} == {ad_id}

    def test_duplicate_platform_publish_skipped_case_insensitively(self, tmp_path, isolated_csvs, capsys):
        _, pubs_path = isolated_csvs
        ad_id = _register(tmp_path, isolated_csvs)

        m.cmd_publish(SimpleNamespace(ad_id=ad_id, platform="instagram", post_id="1", url="https://instagram.com/p/1"))
        m.cmd_publish(SimpleNamespace(
            ad_id=ad_id, platform="Instagram",  # different case, same platform
            post_id="should-not-be-written", url="https://instagram.com/p/should-not-be-written",
        ))

        pubs = m._load_csv(pubs_path)
        assert len(pubs) == 1
        assert pubs[0]["post_id"] == "1"
        assert "already marked published" in capsys.readouterr().out

    def test_publish_unregistered_ad_errors_without_writing(self, tmp_path, isolated_csvs, capsys):
        _, pubs_path = isolated_csvs
        m.cmd_publish(SimpleNamespace(ad_id="does-not-exist", platform="instagram", post_id=None, url=None))
        assert m._load_csv(pubs_path) == []
        assert "no registered ad" in capsys.readouterr().out


class TestMetrics:
    def _publish(self, tmp_path, isolated_csvs):
        ad_id = _register(tmp_path, isolated_csvs)
        m.cmd_publish(SimpleNamespace(ad_id=ad_id, platform="instagram", post_id="1", url="https://instagram.com/p/1"))
        return ad_id

    def test_updates_all_metric_fields(self, tmp_path, isolated_csvs):
        _, pubs_path = isolated_csvs
        ad_id = self._publish(tmp_path, isolated_csvs)

        m.cmd_metrics(SimpleNamespace(
            ad_id=ad_id, platform="instagram",
            views=12000, likes=340, comments=12, shares=5, clicks=88,
        ))

        pub = m._load_csv(pubs_path)[0]
        assert pub["views"] == "12000"
        assert pub["likes"] == "340"
        assert pub["comments"] == "12"
        assert pub["shares"] == "5"
        assert pub["clicks"] == "88"

    def test_negative_metric_rejected_and_nothing_written(self, tmp_path, isolated_csvs, capsys):
        _, pubs_path = isolated_csvs
        ad_id = self._publish(tmp_path, isolated_csvs)

        m.cmd_metrics(SimpleNamespace(
            ad_id=ad_id, platform="instagram",
            views=-1, likes=None, comments=None, shares=None, clicks=None,
        ))

        pub = m._load_csv(pubs_path)[0]
        assert pub["views"] == ""  # unchanged, not "-1"
        assert "cannot be negative" in capsys.readouterr().out

    def test_metrics_on_unpublished_platform_errors(self, tmp_path, isolated_csvs, capsys):
        ad_id = self._publish(tmp_path, isolated_csvs)
        m.cmd_metrics(SimpleNamespace(
            ad_id=ad_id, platform="tiktok",  # never published there
            views=100, likes=None, comments=None, shares=None, clicks=None,
        ))
        assert "no publication" in capsys.readouterr().out


class TestReport:
    def test_platform_filter(self, tmp_path, isolated_csvs, capsys):
        ad_id = _register(tmp_path, isolated_csvs)
        m.cmd_publish(SimpleNamespace(ad_id=ad_id, platform="instagram", post_id="1", url="https://instagram.com/p/1"))
        m.cmd_publish(SimpleNamespace(ad_id=ad_id, platform="tiktok", post_id="2", url="https://tiktok.com/@x/2"))

        capsys.readouterr()  # discard register/publish's own prints
        m.cmd_report(SimpleNamespace(platform="tiktok", status=None))
        out = capsys.readouterr().out
        assert "tiktok" in out
        assert "instagram" not in out

    def test_status_filter(self, tmp_path, isolated_csvs, capsys):
        _register(tmp_path, isolated_csvs, title="unpublished_ad")
        published_id = _register(tmp_path, isolated_csvs, title="published_ad")
        m.cmd_publish(SimpleNamespace(ad_id=published_id, platform="instagram", post_id="1", url="https://instagram.com/p/1"))

        capsys.readouterr()  # discard register/publish's own prints
        m.cmd_report(SimpleNamespace(platform=None, status="published"))
        out = capsys.readouterr().out
        assert "published_ad" in out
        assert "unpublished_ad" not in out

    def test_no_matches_prints_message(self, tmp_path, isolated_csvs, capsys):
        m.cmd_report(SimpleNamespace(platform=None, status=None))
        assert "No matching ads/publications found." in capsys.readouterr().out


class TestExport:
    def test_export_writes_three_sheets_matching_csv(self, tmp_path, isolated_csvs):
        ads_path, pubs_path = isolated_csvs
        ad_id = _register(tmp_path, isolated_csvs)
        m.cmd_publish(SimpleNamespace(ad_id=ad_id, platform="instagram", post_id="1", url="https://instagram.com/p/1"))

        out_path = tmp_path / "export.xlsx"
        m.cmd_export(SimpleNamespace(out=str(out_path)))
        assert out_path.exists()

        import pandas as pd
        xl = pd.ExcelFile(out_path)
        assert set(xl.sheet_names) == {"ads", "publications", "summary"}

        ads_csv_rows = m._load_csv(ads_path)
        pubs_csv_rows = m._load_csv(pubs_path)

        ads_df = pd.read_excel(xl, sheet_name="ads")
        assert len(ads_df) == len(ads_csv_rows) == 1
        assert ads_df.iloc[0]["ad_id"] == ad_id

        pubs_df = pd.read_excel(xl, sheet_name="publications")
        assert len(pubs_df) == len(pubs_csv_rows) == 1
        assert pubs_df.iloc[0]["pub_id"] == f"{ad_id}__instagram"

        summary_df = pd.read_excel(xl, sheet_name="summary")
        assert len(summary_df) == 1
