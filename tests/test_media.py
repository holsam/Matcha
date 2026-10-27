"""Tests for matcha.review.media (uses a tiny ffmpeg-generated clip)."""

import shutil, subprocess
import pytest

from matcha.review import media

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("clip") / "c.mp4"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=duration=2:size=64x48:rate=5", str(path)],
        check=True,
    )
    return str(path)


def test_human_helpers():
    assert media.human_size(512) == "512 B"
    assert media.human_size(1536) == "1.5 KB"
    assert media.human_duration(65) == "1:05"
    assert media.human_duration(3725) == "1:02:05"


def test_file_info(tmp_path):
    f = tmp_path / "a"
    f.write_text("abc")
    assert media.file_info(str(f)).size == 3
    assert media.file_info(str(tmp_path / "nope")) is None


@needs_ffmpeg
def test_probe_and_thumbnail(clip, tmp_path):
    p = media.probe(clip)
    assert (p.width, p.height) == (64, 48) and p.codec
    shot = media.thumbnail(str(tmp_path), 7, clip, 2.0)
    assert shot and shot.endswith("thumbs/7.jpg")
    assert media.thumbnail(str(tmp_path), 7, "/nonexistent.mp4", 2.0) == shot  # cache hit, no ffmpeg


@needs_ffmpeg
def test_probe_reports_fps_and_duration(clip):
    p = media.probe(clip)
    assert p.fps == pytest.approx(5.0)
    assert p.duration == pytest.approx(2.0, abs=0.2)


@needs_ffmpeg
def test_unreadable_file_degrades(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_text("not a video")
    assert media.probe(str(bad)) is None
    assert media.thumbnail(str(tmp_path), 1, str(bad), 1.0) is None
    assert not list((tmp_path / "thumbs").glob("*.tmp.jpg"))
