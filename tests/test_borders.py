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
