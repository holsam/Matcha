"""Tests for matcha.borders (pure numpy detector and aggregator on synthetic frames)."""

import numpy as np, pytest
from dataclasses import replace

from matcha.borders import Crop, FrameResult, Params, aggregate, detect_frame, params_hash, snap_even

P = Params()

# make_frame: gradient content with black borders, optional watermark box (y0, y1, x0, x1) at luma 230
def make_frame(h=540, w=960, top=0, bottom=0, left=0, right=0, *, black=16, lo=80, hi=200, watermark=None, seed=0):
    rng = np.random.default_rng(seed)
    f = np.full((h, w), black, dtype=np.float64) + rng.integers(-2, 3, (h, w))
    ch, cw = h - top - bottom, w - left - right
    f[top:h - bottom, left:w - right] = np.tile(np.linspace(lo, hi, cw), (ch, 1)) + rng.integers(-3, 4, (ch, cw))
    if watermark:
        y0, y1, x0, x1 = watermark
        f[y0:y1, x0:x1] = 230
    return np.clip(f, 0, 255).astype(np.uint8)

# margins: detect_frame margins as a tuple, None when the frame is uninformative
def margins(f, p=P):
    r = detect_frame(f, p)
    return None if r is None else (r.margins.top, r.margins.bottom, r.margins.left, r.margins.right)

# fr: FrameResult at 960x540 (scale 1) from a (top, bottom, left, right) tuple, None stays None
def fr(m, shape=(540, 960)):
    return None if m is None else FrameResult(Crop(*m), shape)

# agg: aggregate n frames spread over 5-95% of a 100 s video
def agg(margin_list, *, p=P, source=(960, 540), shape=(540, 960), min_informative=3):
    n = len(margin_list)
    times = list(np.linspace(5, 95, n))
    return aggregate([fr(m, shape) for m in margin_list], times, 100.0, p, source=source, min_informative=min_informative)


@pytest.mark.parametrize('black', [16, 0])
@pytest.mark.parametrize('geom', [(70, 70, 0, 0), (0, 0, 120, 120), (50, 50, 90, 90)])
def test_clean_borders_recovered(geom, black):
    assert margins(make_frame(top=geom[0], bottom=geom[1], left=geom[2], right=geom[3], black=black)) == geom

def test_bright_borderless_is_informative_zero():
    assert margins(make_frame()) == (0, 0, 0, 0)

def test_bright_frame_with_grey_band_not_flagged():
    f = make_frame(lo=100, hi=220)
    f[-54:] = 70  # a 10% grey band is not black, and the reference is capped
    assert margins(f) == (0, 0, 0, 0)

def test_small_watermark_tolerated():
    # 4% of the width, covering every border row
    f = make_frame(top=70, bottom=70, watermark=(0, 70, 100, 138))
    assert margins(f) == (70, 70, 0, 0)

def test_large_watermark_breaks_blank_line():
    # 8% of the width over all border rows: documents the 95% limit
    f = make_frame(top=70, bottom=70, watermark=(0, 70, 100, 178))
    assert margins(f)[0] == 0

def test_dark_scene_discarded():
    assert margins(make_frame(top=70, bottom=70, lo=18, hi=45)) is None

def test_dark_scene_without_border_discarded():
    assert margins(make_frame(lo=18, hi=45)) is None

@pytest.mark.parametrize('scale', [0.15, 0.0])
def test_fade_discarded(scale):
    f = (make_frame(top=70, bottom=70).astype(float) * scale).astype(np.uint8)
    assert margins(f) is None

def test_margin_under_one_percent_ignored():
    assert margins(make_frame(top=4, bottom=4)) == (0, 0, 0, 0)

def test_aggregate_agreement_boundary():
    border = (70, 70, 0, 0)
    odd = [(10 * k, 0, 0, 0) for k in range(1, 5)]  # four frames that agree with nothing
    assert agg([border] * 16 + odd).result == 'border'
    assert agg([border] * 15 + odd + [(300, 0, 0, 0)]).result == 'variable'

def test_aggregate_confidence():
    v = agg([(70, 70, 0, 0)] * 18 + [None, None])
    assert (v.result, v.confidence, v.frames_used) == ('border', 1.0, 18)

def test_aggregate_middle_third_only_is_variable():
    m = [None] * 7 + [(70, 70, 0, 0)] * 6 + [None] * 7
    assert agg(m).result == 'variable'

def test_aggregate_short_windowbox_is_no_border():
    v = agg([(70, 70, 60, 60)] * 2 + [(0, 0, 0, 0)] * 18)
    assert v.result == 'no border' and v.crop is None

def test_aggregate_insufficient():
    assert agg([None] * 20).result == 'insufficient'
    assert agg([(70, 70, 0, 0)] * 2 + [None] * 18).result == 'insufficient'

def test_aggregate_two_px_agree_three_do_not():
    a, b = (140, 140, 0, 0), (142, 140, 0, 0)
    assert agg([a] * 10 + [b] * 10).result == 'border'
    assert agg([a] * 10 + [(143, 140, 0, 0)] * 10).result == 'variable'

def test_aggregate_scales_to_source():
    v = agg([(35, 35, 0, 0)] * 20, source=(1920, 1080), shape=(540, 960))
    assert v.crop == Crop(70, 70, 0, 0)

@pytest.mark.parametrize('crop', [Crop(3, 5, 7, 9), Crop(1, 1, 0, 0), Crop(140, 141, 0, 3), Crop(0, 0, 0, 0)])
@pytest.mark.parametrize('size', [(1920, 1080), (961, 541)])
def test_snap_even(crop, size):
    s = snap_even(crop, size)
    assert s.left % 2 == 0 and s.top % 2 == 0
    assert (size[0] - s.left - s.right) % 2 == 0 and (size[1] - s.top - s.bottom) % 2 == 0
    assert all(abs(a - b) <= 1 for a, b in zip((s.top, s.bottom, s.left, s.right), (crop.top, crop.bottom, crop.left, crop.right)))

def test_params_hash():
    assert params_hash(P) == params_hash(Params())
    for field, value in [('black_tol', 5), ('agree_frac', 0.9), ('span', (0.1, 0.9)), ('max_width', 320)]:
        assert params_hash(replace(P, **{field: value})) != params_hash(P)


# ---- phase 2: sampling, scan, store ----

import json, os, shutil, subprocess, threading
from pathlib import Path
from typer.testing import CliRunner

from matcha import borders
from matcha.borders import Store, Verdict, run_scan, scan_video
from matcha.cli import app

needs_ffmpeg = pytest.mark.skipif(shutil.which('ffmpeg') is None, reason='ffmpeg not installed')

# GRAD: deterministic bright horizontal luma ramp, {d} is the duration
GRAD = 'nullsrc=s=640x360:r=10:d={d},geq=lum=80+120*X/W:cb=128:cr=128'

# clip: encode a tiny test clip from a lavfi graph (-vf style filters on a 640x360 ramp)
def clip(path: Path, vf: str = 'null', *, d: int = 6, src: str = GRAD, extra: tuple[str, ...] = ()) -> Path:
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'lavfi', '-i', src.format(d=d), '-vf', vf,
                    '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', *extra, str(path)], check=True)
    return path

# near: crop within 2 px of (top, bottom, left, right)
def near(crop: Crop | None, want: tuple[int, int, int, int]) -> bool:
    return crop is not None and all(abs(a - b) <= 2 for a, b in zip((crop.top, crop.bottom, crop.left, crop.right), want))

@needs_ffmpeg
def test_scan_video_letterbox(tmp_path):
    v, stage = scan_video(str(clip(tmp_path / 'a.mp4', 'pad=640:480:0:60:black')), P)
    assert v.result == 'border' and near(v.crop, (60, 60, 0, 0)) and stage == 'B'

@needs_ffmpeg
def test_scan_video_pillarbox(tmp_path):
    v, _ = scan_video(str(clip(tmp_path / 'a.mp4', 'pad=800:360:80:0:black')), P)
    assert v.result == 'border' and near(v.crop, (0, 0, 80, 80))

@needs_ffmpeg
def test_scan_video_no_border_stops_at_stage_a(tmp_path):
    v, stage = scan_video(str(clip(tmp_path / 'a.mp4')), P)
    assert (v.result, stage) == ('no border', 'A')

@needs_ffmpeg
def test_scan_video_dark_is_insufficient(tmp_path):
    v, _ = scan_video(str(clip(tmp_path / 'a.mp4', 'pad=640:480:0:60:black', src='color=c=0x181818:s=640x360:r=10:d={d}')), P)
    assert v.result == 'insufficient'

@needs_ffmpeg
def test_scan_video_short_windowbox_segment_is_no_border(tmp_path):
    out = tmp_path / 'a.mp4'
    graph = (GRAD.format(d=18) + '[a];' + GRAD.format(d=2) + ',scale=480:270,pad=640:360:80:45[b];[a][b]concat=n=2:v=1:a=0')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-filter_complex', graph, '-c:v', 'libx264', '-preset', 'ultrafast',
                    '-pix_fmt', 'yuv420p', str(out)], check=True)
    assert scan_video(str(out), P)[0].result == 'no border'

@needs_ffmpeg
def test_scan_video_rotated_crop_is_in_displayed_orientation(tmp_path):
    src = clip(tmp_path / 'src.mp4', 'pad=640:480:0:60:black')
    out = tmp_path / 'rot.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-display_rotation', '90', '-i', str(src), '-c', 'copy', str(out)], check=True)
    v, _ = scan_video(str(out), P)
    assert v.result == 'border' and near(v.crop, (0, 0, 60, 60))

@needs_ffmpeg
def test_scan_video_very_short_clip(tmp_path):
    v, _ = scan_video(str(clip(tmp_path / 'a.mp4', 'pad=640:480:0:60:black', d=1)), P)
    assert v.result in ('border', 'insufficient')  # tiny clips must not crash; border once enough distinct frames exist

# Stub frames for stage logic without ffmpeg
BRIGHT = make_frame()
BORDERED = make_frame(top=70, bottom=70)
DARK = make_frame(lo=18, hi=45)

# stub_sampler: patch _dims and sample_frame, returns the list of sampled times
def stub_sampler(monkeypatch, frame_for, duration=100.0):
    calls: list[float] = []
    monkeypatch.setattr(borders, '_dims', lambda path: (960, 540, duration))
    def fake(path, t, shape):
        calls.append(t)
        return frame_for(t)
    monkeypatch.setattr(borders, 'sample_frame', fake)
    return calls

def test_stage_a_short_circuit(monkeypatch):
    calls = stub_sampler(monkeypatch, lambda t: BRIGHT)
    v, stage = scan_video('x.mp4', P)
    assert (v.result, stage, len(calls)) == ('no border', 'A', P.frames_a)

def test_stage_b_for_bordered_reuses_stage_a_frames(monkeypatch):
    calls = stub_sampler(monkeypatch, lambda t: BORDERED)
    v, stage = scan_video('x.mp4', P)
    assert (v.result, stage) == ('border', 'B')
    assert len(calls) == len(set(calls)) == P.frames_b

def test_dark_video_goes_to_stage_b_then_insufficient(monkeypatch):
    calls = stub_sampler(monkeypatch, lambda t: DARK)
    v, stage = scan_video('x.mp4', P)
    assert (v.result, stage, len(calls)) == ('insufficient', 'B', P.frames_b)

def test_unreadable_probe_is_insufficient(monkeypatch):
    monkeypatch.setattr(borders, '_dims', lambda path: None)
    assert scan_video('x.mp4', P)[0].result == 'insufficient'

def test_times_dedupe_on_very_short_video():
    assert len(borders._times(0.3, P)) < P.frames_b

# library: directory of dummy video files (content is irrelevant when scan_video is stubbed)
@pytest.fixture
def library(tmp_path):
    for n in range(6):
        (tmp_path / f'v{n}.mp4').write_bytes(b'x' * (n + 1))
    return tmp_path

# stub_scan: replace scan_video, records scanned file names
def stub_scan(monkeypatch, result='border', fail=None):
    seen: list[str] = []
    def fake(path, p):
        seen.append(os.path.basename(path))
        if fail and fail(path):
            raise fail.exc
        return Verdict(result, Crop(70, 70, 0, 0) if result == 'border' else None, 1.0, 20), 'B'
    monkeypatch.setattr(borders, 'scan_video', fake)
    return seen

def scan(lib, **kw):
    run_scan(str(lib), workers=2, **kw)

def load(lib):
    return json.loads((lib / '.matcha' / 'borders.json').read_text())['videos']

def test_scan_writes_store(library, monkeypatch):
    stub_scan(monkeypatch)
    scan(library)
    v = load(library)
    assert sorted(v) == [f'v{n}.mp4' for n in range(6)]
    e = v['v0.mp4']
    assert e['status'] == 'complete' and e['result'] == 'border' and e['crop'] == {'top': 70, 'bottom': 70, 'left': 0, 'right': 0}
    assert e['size'] == 1 and e['params_hash'] == params_hash(P) and e['frames_used'] == 20

def test_second_scan_skips_everything(library, monkeypatch):
    seen = stub_scan(monkeypatch)
    scan(library)
    seen.clear()
    scan(library)
    assert seen == []

def test_rescan_flag_scans_everything(library, monkeypatch):
    seen = stub_scan(monkeypatch)
    scan(library)
    seen.clear()
    scan(library, rescan=True)
    assert len(seen) == 6

def test_changed_size_or_mtime_rescans_that_file_only(library, monkeypatch):
    seen = stub_scan(monkeypatch)
    scan(library)
    seen.clear()
    (library / 'v1.mp4').write_bytes(b'longer content')
    os.utime(library / 'v2.mp4', (1, 1))
    scan(library)
    assert sorted(seen) == ['v1.mp4', 'v2.mp4']

def test_changed_param_rescans_all(library, monkeypatch):
    seen = stub_scan(monkeypatch)
    scan(library)
    seen.clear()
    scan(library, params=replace(P, agree_frac=0.9))
    assert len(seen) == 6

def test_workers_do_not_affect_hash(library, monkeypatch):
    seen = stub_scan(monkeypatch)
    run_scan(str(library), workers=1)
    seen.clear()
    run_scan(str(library), workers=3)
    assert seen == []

def test_pending_entries_are_rerun(library, monkeypatch):
    seen = stub_scan(monkeypatch)
    scan(library)
    store = Store(str(library / '.matcha' / 'borders.json'))
    store.videos['v3.mp4']['status'] = 'pending'
    store.dirty = True
    store.flush()
    seen.clear()
    scan(library)
    assert seen == ['v3.mp4']

def test_unreadable_file_recorded_and_scan_continues(library):
    scan(library)  # real scan_video on junk files: ffprobe fails
    v = load(library)
    assert len(v) == 6 and {e['result'] for e in v.values()} == {'insufficient'}

def test_worker_error_leaves_entry_pending(library, monkeypatch):
    def fail(path):
        return path.endswith('v2.mp4')
    fail.exc = RuntimeError('boom')
    stub_scan(monkeypatch, fail=fail)
    scan(library)
    v = load(library)
    assert v['v2.mp4']['status'] == 'pending' and v['v0.mp4']['status'] == 'complete'

def test_interrupt_keeps_store_valid_and_resumes(library, monkeypatch):
    done: list[str] = []
    def fake(path, p):
        name = os.path.basename(path)
        if name == 'v3.mp4':
            raise KeyboardInterrupt
        done.append(name)
        return Verdict('no border', None, 1.0, 20), 'A'
    monkeypatch.setattr(borders, 'scan_video', fake)
    run_scan(str(library), workers=1)
    v = load(library)  # valid JSON after the interrupt
    complete = {n for n, e in v.items() if e['status'] == 'complete'}
    assert 'v0.mp4' in complete and complete <= set(done) and v['v3.mp4']['status'] == 'pending'  # in-flight results may be lost
    seen = stub_scan(monkeypatch)
    scan(library)
    assert not complete & set(seen) and 'v3.mp4' in seen
    assert {e['status'] for e in load(library).values()} == {'complete'}

def test_failed_replace_keeps_original_store(tmp_path, monkeypatch):
    path = tmp_path / '.matcha' / 'borders.json'
    s = Store(str(path))
    s.put('a.mp4', {'status': 'complete'})
    s.flush()
    before = path.read_text()
    s.put('b.mp4', {'status': 'complete'})
    monkeypatch.setattr(os, 'replace', lambda *a: (_ for _ in ()).throw(OSError('crash')))
    with pytest.raises(OSError):
        s.flush()
    monkeypatch.undo()
    assert path.read_text() == before
    (tmp_path / '.matcha' / 'borders.json.tmp').write_text('{half')  # stray temp from a real crash is ignored
    assert list(Store(str(path)).videos) == ['a.mp4']

def test_corrupt_store_moved_aside(library, monkeypatch):
    (library / '.matcha').mkdir()
    (library / '.matcha' / 'borders.json').write_text('{not json')
    stub_scan(monkeypatch)
    scan(library)
    assert (library / '.matcha' / 'borders.json.corrupt').exists() and len(load(library)) == 6

def test_only_main_thread_writes(library, monkeypatch):
    stub_scan(monkeypatch)
    threads = set()
    for name in ('put', 'flush'):
        orig = getattr(Store, name)
        def spy(self, *a, _orig=orig, **k):
            threads.add(threading.current_thread())
            return _orig(self, *a, **k)
        monkeypatch.setattr(Store, name, spy)
    scan(library)
    assert threads == {threading.main_thread()}

def test_prune_forgets_vanished_files(library, monkeypatch):
    stub_scan(monkeypatch)
    scan(library)
    (library / 'v0.mp4').unlink()
    scan(library)
    assert 'v0.mp4' in load(library)
    scan(library, prune=True)
    assert 'v0.mp4' not in load(library)

def test_store_ignores_matcha_dir_and_uses_relative_keys(tmp_path, monkeypatch):
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'sub' / 'a.mp4').write_bytes(b'x')
    (tmp_path / '.matcha').mkdir()
    (tmp_path / '.matcha' / 'skip.mp4').write_bytes(b'x')
    stub_scan(monkeypatch)
    scan(tmp_path)
    assert list(load(tmp_path)) == [os.path.join('sub', 'a.mp4')]

def test_cli_scan(library, monkeypatch):
    stub_scan(monkeypatch)
    r = CliRunner().invoke(app, ['borders', 'scan', str(library), '--workers', '2', '--agree-frac', '0.9'])
    assert r.exit_code == 0, r.output
    assert next(iter(load(library).values()))['params_hash'] == params_hash(replace(P, agree_frac=0.9))

def test_cli_scan_missing_directory(tmp_path):
    assert CliRunner().invoke(app, ['borders', 'scan', str(tmp_path / 'nope')]).exit_code == 1


# ---- phase 3: list ----

from matcha.borders import run_list

# seed_store: write a borders.json from (name, result, crop, confidence, status) rows and make the files exist
def seed_store(lib: Path, rows) -> None:
    videos = {}
    for name, result, crop, conf, *rest in rows:
        (lib / name).parent.mkdir(parents=True, exist_ok=True)
        (lib / name).write_bytes(b'x')
        videos[name] = {'size': 1, 'mtime': 1.0, 'params_hash': 'x', 'status': rest[0] if rest else 'complete', 'result': result,
                        'crop': crop and dict(zip(('top', 'bottom', 'left', 'right'), crop)), 'confidence': conf, 'frames_used': 20}
    (lib / '.matcha').mkdir(exist_ok=True)
    (lib / '.matcha' / 'borders.json').write_text(json.dumps({'version': 1, 'videos': videos}))

ROWS = [
    ('b.mp4', 'border', (70, 70, 0, 0), 0.9),
    ('a.mp4', 'border', (0, 0, 80, 80), 1.0),
    ('c.mp4', 'variable', None, 0.5),
    ('d.mp4', 'no border', None, 1.0),
    ('e.mp4', 'border', (4, 4, 0, 0), 0.85),
    ('f.mp4', 'border', None, 0.0, 'pending'),
]

# listed: run `borders list --json` and return the parsed items
def listed(lib, capsys, **kw):
    code = run_list(str(lib), as_json=True, **kw)
    assert code == 0
    return json.loads(capsys.readouterr().out)

@pytest.fixture
def listing(tmp_path, monkeypatch):
    seed_store(tmp_path, ROWS)
    monkeypatch.setattr(borders, '_dims', lambda path: (960, 540, 100.0))
    return tmp_path

def test_list_only_border_by_default(listing, capsys):
    assert [i['path'] for i in listed(listing, capsys)] == ['a.mp4', 'b.mp4', 'e.mp4']

def test_list_include_variable(listing, capsys):
    items = listed(listing, capsys, include_variable=True)
    assert [i['path'] for i in items] == ['a.mp4', 'b.mp4', 'c.mp4', 'e.mp4']
    assert items[2]['crop'] is None and items[2]['out_width'] is None

def test_list_json_fields(listing, capsys):
    a = listed(listing, capsys)[0]
    assert (a['width'], a['height'], a['out_width'], a['out_height']) == (960, 540, 800, 540)
    assert a['crop'] == {'top': 0, 'bottom': 0, 'left': 80, 'right': 80}

def test_list_vanished_files_omitted(listing, capsys):
    (listing / 'a.mp4').unlink()
    assert [i['path'] for i in listed(listing, capsys)] == ['b.mp4', 'e.mp4']

def test_list_min_confidence(listing, capsys):
    assert [i['path'] for i in listed(listing, capsys, min_confidence=0.9)] == ['a.mp4', 'b.mp4']

def test_list_sort(listing, capsys):
    assert [i['path'] for i in listed(listing, capsys, sort='confidence')] == ['a.mp4', 'b.mp4', 'e.mp4']
    assert [i['path'] for i in listed(listing, capsys, sort='area')] == ['b.mp4', 'a.mp4', 'e.mp4']  # 134400, 86400, 7680 px removed
    assert run_list(str(listing), sort='nope') == 1

def test_list_table_footer_and_markup_escape(tmp_path, monkeypatch, capsys):
    seed_store(tmp_path, [('[red]odd[/red].mp4', 'border', (70, 70, 0, 0), 0.9), ('n.mp4', 'no border', None, 1.0), ('p.mp4', 'border', None, 0, 'pending')])
    monkeypatch.setattr(borders, '_dims', lambda path: (960, 540, 100.0))
    assert run_list(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert '[red]odd[/red]' in out and '3 stored: 1 border, 1 no border, 0 variable, 0 insufficient, 1 pending' in out

def test_list_unprobeable_file_still_listed(listing, capsys, monkeypatch):
    monkeypatch.setattr(borders, '_dims', lambda path: None)
    a = listed(listing, capsys)[0]
    assert a['width'] is None and a['crop'] is not None

def test_list_no_store(tmp_path, capsys):
    assert run_list(str(tmp_path)) == 1
    assert 'borders scan' in capsys.readouterr().out

def test_list_unreadable_store_left_in_place(tmp_path):
    (tmp_path / '.matcha').mkdir()
    (tmp_path / '.matcha' / 'borders.json').write_text('{bad')
    assert run_list(str(tmp_path)) == 1
    assert (tmp_path / '.matcha' / 'borders.json').exists()

def test_cli_list(listing):
    r = CliRunner().invoke(app, ['borders', 'list', str(listing), '--json', '--sort', 'confidence'])
    assert r.exit_code == 0 and [i['path'] for i in json.loads(r.output)] == ['a.mp4', 'b.mp4', 'e.mp4']
