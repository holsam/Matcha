
"""Tests for matcha.reencode (hand-built Info values, plus tiny generated clips when ffmpeg is installed)."""

import os, pytest
from dataclasses import replace

from matcha import reencode
from matcha.reencode import Info, _info, _videos
from matcha.review.media import Probe

# H264: 1080p30 at 8 Mbps for 100 s, about 0.13 bpp
H264 = Info('a.mp4', 105_000_000, 100.0, 1920, 1080, 30.0, 'h264', 8_000_000)


def test_bpp():
    assert H264.bpp == pytest.approx(8_000_000 / (1920 * 1080 * 30))


def test_saving_is_video_bytes_minus_target_bytes():
    expected = 8_000_000 * 100 / 8 - 1920 * 1080 * 30 * 0.06 * 100 / 8
    assert abs(H264.saving(0.06) - expected) <= 1


def test_saving_is_zero_when_already_efficient():
    assert replace(H264, vbitrate=1_000_000).saving(0.06) == 0


def test_saving_is_zero_without_a_frame_rate_or_bitrate():
    assert replace(H264, fps=0.0).saving(0.06) == 0
    assert replace(H264, vbitrate=0).saving(0.06) == 0


def test_bloated_hevc_still_saves():
    assert replace(H264, codec='hevc').saving(0.06) > 0


def test_bloat_uses_the_codecs_own_scale():
    assert replace(H264, codec='hevc').bloat(0.06) > H264.bloat(0.06)
    assert replace(H264, codec='unknown').bloat(0.06) == H264.bloat(0.06)  # unknown codecs count as h264


def test_videos_skips_temp_files_and_the_matcha_dir(tmp_path):
    for rel in ('a.mp4', 'b.txt', 'c.reencoding.mp4', '.matcha/d.mp4', 'sub/e.MKV'):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b'x')
    got = _videos(str(tmp_path))
    assert sorted(os.path.relpath(p, tmp_path) for p in got) == ['a.mp4', 'sub/e.MKV']


def test_info_falls_back_from_video_bitrate_to_file_bitrate_to_size(tmp_path, monkeypatch):
    f = tmp_path / 'v.mp4'
    f.write_bytes(b'x' * 1000)
    base = dict(width=320, height=240, codec='h264', fps=25.0, duration=10.0)
    for probed, want in (
        (Probe(bitrate=900, video_bitrate=800, **base), 800),
        (Probe(bitrate=900, video_bitrate=None, **base), 900),
        (Probe(bitrate=None, video_bitrate=None, **base), 800),  # 1000 bytes * 8 / 10 s
    ):
        monkeypatch.setattr(reencode, 'probe', lambda path, p=probed: p)
        assert _info(str(f)).vbitrate == want


def test_info_is_none_for_missing_file_failed_probe_or_no_video(tmp_path, monkeypatch):
    f = tmp_path / 'v.mp4'
    f.write_bytes(b'x')
    assert _info(str(tmp_path / 'nope.mp4')) is None
    monkeypatch.setattr(reencode, 'probe', lambda path: None)
    assert _info(str(f)) is None
    monkeypatch.setattr(reencode, 'probe', lambda path: Probe(None, None, None, 100))
    assert _info(str(f)) is None
