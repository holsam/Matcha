
"""Tests for matcha.reencode (hand-built Info values, plus tiny generated clips when ffmpeg is installed)."""

import os, pytest, shutil, subprocess
from dataclasses import replace
from pathlib import Path
from PIL import Image
from typer.testing import CliRunner

from matcha import reencode
from matcha.cli import app
from matcha.db import get_connection, init_schema
from matcha.reencode import Info, _info, _candidates, _label, _preview, _verify, _videos, run_reencode
from matcha.review.media import Probe, probe

# H264: 1080p30 at 8 Mbps for 100 s, about 0.13 bpp
H264 = Info('a.mp4', 105_000_000, 100.0, 1920, 1080, 30.0, 'h264', 8_000_000)

needs_ffmpeg = pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg not installed')

# apply: run_reencode with thresholds low enough for a tiny clip and a fast preset
def apply(directory, **kw) -> None:
    run_reencode(str(directory), apply=True, yes=True, min_saving_mb=0, min_saving_pct=0, preset='ultrafast', **kw)

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

def test_videos_skips_temp_files_matcha_dir_and_exclude(tmp_path):
    for rel in ('a.mp4', 'b.txt', 'c.reencoding.mp4', '.matcha/d.mp4', 'sub/e.MKV', 'keep/f.mp4'):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b'x')
    got = _videos(str(tmp_path), str(tmp_path / 'keep'))
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

@needs_ffmpeg
def test_label_height(tmp_path):
    _label('0.06 bpp (1.0 Mbps, hevc)', str(tmp_path / 'l.png'))
    assert Image.open(tmp_path / 'l.png').height == 36

@needs_ffmpeg
def test_preview_stitches_original_and_encodes_side_by_side(tmp_path):
    info = _info(str(make_clip(tmp_path / 'a.mp4')))
    out = probe(_preview(info, [0.06], str(tmp_path), 0))
    assert (out.width, out.height) == (2 * 480, 360)  # two panels scaled to 360 high

@needs_ffmpeg
def test_preview_flag_writes_comparisons_and_touches_no_originals(tmp_path):
    clip = make_clip(tmp_path / 'a.mp4')
    before = clip.read_bytes()
    run_reencode(str(tmp_path), min_saving_mb=0, min_saving_pct=0, preview=1, bpps='0.06')
    assert len(list((tmp_path / '.matcha' / 'tmp').glob('*_compare.mp4'))) == 1
    assert clip.read_bytes() == before

@needs_ffmpeg
def test_verify_accepts_a_smaller_same_length_hevc_file(tmp_path):
    old = _info(str(make_clip(tmp_path / 'a.mp4')))
    new = make_clip(tmp_path / 'b.mp4', '-c:v', 'libx265', '-crf', '28')
    assert _verify(replace(old, size=10**9), str(new)) is None
    assert 'duration' in _verify(replace(old, size=10**9, duration=old.duration + 5), str(new))
    assert 'smaller' in _verify(replace(old, size=1), str(new))
    assert 'h264' in _verify(replace(old, size=10**9), str(tmp_path / 'a.mp4'))
    assert 'unreadable' in _verify(old, str(tmp_path / 'missing.mp4'))

@needs_ffmpeg
def test_apply_keeps_path_codec_and_mtime_of_an_mp4(tmp_path):
    clip = make_clip(tmp_path / 'a.mp4')
    os.utime(clip, (1_000_000_000, 1_000_000_000))
    before = clip.stat().st_size
    apply(tmp_path)
    assert probe(str(clip)).codec == 'hevc'
    assert clip.stat().st_size < before
    assert int(clip.stat().st_mtime) == 1_000_000_000
    assert list(tmp_path.glob('*.reencoding.*')) == []

@needs_ffmpeg
def test_apply_turns_an_avi_into_mkv_and_repoints_the_index(tmp_path):
    avi = make_clip(tmp_path / 'old.avi', '-c:v', 'mpeg4', '-q:v', '2')
    (tmp_path / '.matcha').mkdir()
    db_path = str(tmp_path / '.matcha' / 'index.db')
    init_schema(db_path)
    conn = get_connection(db_path)
    with conn:
        conn.execute('INSERT INTO videos (id, path, duration, content_key) VALUES (1, ?, 2, ?)', (str(avi), 'k'))
        conn.execute("INSERT INTO frame_hashes (video_id, timestamp, phash) VALUES (1, 0, 'ff')")
    apply(tmp_path)
    assert not avi.exists()
    assert probe(str(tmp_path / 'old.mkv')).codec == 'hevc'
    row = conn.execute('SELECT id, path, content_key FROM videos').fetchone()
    assert tuple(row) == (1, str(tmp_path / 'old.mkv'), None)
    assert conn.execute('SELECT COUNT(*) FROM frame_hashes').fetchone()[0] == 1

@needs_ffmpeg
def test_apply_adopts_paths_from_the_old_move_command_before_repointing(tmp_path):
    (tmp_path / 'duplicates').mkdir()
    held = make_clip(tmp_path / 'duplicates' / 'v.mp4')  # where the old move command left it
    (tmp_path / '.matcha').mkdir()
    db_path = str(tmp_path / '.matcha' / 'index.db')
    init_schema(db_path)
    conn = get_connection(db_path)
    with conn:
        conn.execute('INSERT INTO videos (id, path, duration, moved_to) VALUES (1, ?, 2, ?)',
                     (str(tmp_path / 'v.mp4'), str(held)))
    apply(tmp_path)
    assert tuple(conn.execute('SELECT path, moved_to FROM videos').fetchone()) == (str(held), None)
    assert probe(str(held)).codec == 'hevc'

@needs_ffmpeg
def test_keep_originals_moves_the_original_and_links_the_new_file(tmp_path):
    lib, keep = tmp_path / 'lib', tmp_path / 'keep'
    (lib / 'sub').mkdir(parents=True)
    avi = make_clip(lib / 'sub' / 'old.avi', '-c:v', 'mpeg4', '-q:v', '2')
    mp4 = make_clip(lib / 'a.mp4')
    sizes = {avi.name: avi.stat().st_size, mp4.name: mp4.stat().st_size}
    run_reencode(str(lib), apply=True, min_saving_mb=0, min_saving_pct=0, preset='ultrafast', keep_originals=str(keep))
    assert (keep / 'a.mp4').stat().st_size == sizes['a.mp4']
    assert (keep / 'sub' / 'old.avi').stat().st_size == sizes['old.avi']
    assert os.path.realpath(keep / 'a.reencoded.mp4') == os.path.realpath(mp4)
    assert os.path.realpath(keep / 'sub' / 'old.reencoded.mkv') == os.path.realpath(lib / 'sub' / 'old.mkv')
    assert probe(str(mp4)).codec == 'hevc'

@needs_ffmpeg
def test_keep_dir_inside_the_library_is_not_profiled(tmp_path):
    make_clip(tmp_path / 'a.mp4')
    keep = tmp_path / 'held'
    run_reencode(str(tmp_path), apply=True, min_saving_mb=0, min_saving_pct=0, preset='ultrafast', keep_originals=str(keep))
    before = (keep / 'a.mp4').read_bytes()
    run_reencode(str(tmp_path), apply=True, min_saving_mb=0, min_saving_pct=0, preset='ultrafast', keep_originals=str(keep))
    assert (keep / 'a.mp4').read_bytes() == before
