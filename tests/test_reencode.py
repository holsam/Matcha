
"""Tests for matcha.reencode (hand-built Info values, plus tiny generated clips when ffmpeg is installed)."""

import os, pytest, shutil, subprocess
from dataclasses import replace
from pathlib import Path
from typer.testing import CliRunner

from matcha import reencode
from matcha.cli import app
from matcha.reencode import Info, _info, _candidates, _videos, run_reencode
from matcha.review.media import Probe

# H264: 1080p30 at 8 Mbps for 100 s, about 0.13 bpp
H264 = Info('a.mp4', 105_000_000, 100.0, 1920, 1080, 30.0, 'h264', 8_000_000)

needs_ffmpeg = pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg not installed')

# make_clip: 2 s noisy 320x240 clip at a very high bitrate, so its bpp is far above any target
def make_clip(path: Path, *codec: str) -> Path:
    codec = codec or ('-c:v', 'libx264', '-crf', '10')
    subprocess.run(
        ['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', 'testsrc=s=320x240:r=25:d=2,noise=alls=40:allf=t',
         *codec, str(path)], check=True)
    return path

def test_candidates_apply_both_thresholds_largest_first():
    big = replace(H264, path='big.mp4', vbitrate=16_000_000, size=200_000_000)
    small = replace(H264, path='small.mp4', duration=1.0, size=1_000_000)
    efficient = replace(H264, path='eff.mp4', vbitrate=1_000_000)
    got = _candidates([H264, small, big, efficient], 0.06, min_saving_pct=30, min_saving_mb=10)
    assert [i.path for _, i in got] == ['big.mp4', 'a.mp4']

@needs_ffmpeg
def test_dry_run_changes_nothing(tmp_path):
    clip = make_clip(tmp_path / 'a.mp4')
    before = clip.read_bytes()
    run_reencode(str(tmp_path), min_saving_mb=0, min_saving_pct=0)
    assert clip.read_bytes() == before
    assert list(tmp_path.glob('*.reencoding.*')) == []

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

def test_reencode_command_reports_an_empty_directory(tmp_path):
    result = CliRunner().invoke(app, ['reencode', str(tmp_path)])
    assert result.exit_code == 0 and 'no videos found' in result.output
