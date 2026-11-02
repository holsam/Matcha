'''
MATCHA: detect near-black borders (letterbox, pillarbox, window-box) in sampled frames and aggregate them per video.
'''

# Import external dependencies
import hashlib, json
from dataclasses import asdict, astuple, dataclass
from typing import Literal

import numpy as np

# Result: the four per-video outcomes
Result = Literal['border', 'no border', 'variable', 'insufficient']

# Params: detection thresholds, every field is exposed as a CLI option and feeds params_hash
@dataclass(frozen=True)
class Params:
    black_tol: int = 12  # luma above the frame's black reference still counted near-black
    max_black_ref: int = 40  # cap on the black reference, so a bright frame cannot raise the bar
    blank_frac: float = 0.95  # share of near-black pixels that makes a line blank
    min_margin_frac: float = 0.01  # margins under this share of the dimension are ignored
    min_content_luma: float = 40.0  # mean luma of the content area for a frame to be informative
    min_edge_step: float = 20.0  # luma step across a detected edge
    edge_band: int = 4  # lines each side of an edge used for the step
    agree_px: int = 2  # margins agree within this many source pixels
    agree_frac: float = 0.8  # share of informative frames that must agree
    min_informative_a: int = 3  # informative frames needed to conclude after stage A
    min_informative_b: int = 10  # informative frames needed to conclude after stage B
    frames_a: int = 5
    frames_b: int = 20
    span: tuple[float, float] = (0.05, 0.95)  # fraction of the duration frames are sampled across
    max_width: int = 480  # frames are scaled down to this width before analysis

# Crop: margins in pixels, in the frame they were measured on until aggregate scales them to the source
@dataclass(frozen=True)
class Crop:
    top: int
    bottom: int
    left: int
    right: int

# FrameResult: margins found in one informative frame, with the (height, width) of that frame
@dataclass(frozen=True)
class FrameResult:
    margins: Crop
    shape: tuple[int, int]

# Verdict: per-video outcome, crop is only set for 'border', frames_used counts informative frames
@dataclass(frozen=True)
class Verdict:
    result: Result
    crop: Crop | None
    confidence: float
    frames_used: int

# params_hash: short stable hash of the detection params, a change triggers a rescan
def params_hash(p: Params) -> str:
    return hashlib.sha1(json.dumps(asdict(p), sort_keys=True).encode()).hexdigest()[:8]

# detect_frame: margins of one greyscale frame, None when the frame is not informative (dark, fade, no usable edge)
def detect_frame(gray: np.ndarray, p: Params) -> FrameResult | None:
    h, w = gray.shape
    # black reference per frame, so limited (16) and full (0) range both self-calibrate
    ref = np.clip(np.percentile(gray, 1), 0, p.max_black_ref)
    dark = gray <= ref + p.black_tol
    rows = dark.mean(axis=1) < p.blank_frac  # True where a line has content
    cols = dark.mean(axis=0) < p.blank_frac
    if not rows.any() or not cols.any():
        return None
    top, bottom = int(rows.argmax()), int(rows[::-1].argmax())
    left, right = int(cols.argmax()), int(cols[::-1].argmax())
    top, bottom = (m if m >= p.min_margin_frac * h else 0 for m in (top, bottom))
    left, right = (m if m >= p.min_margin_frac * w else 0 for m in (left, right))
    content = gray[top:h - bottom, left:w - right]
    if content.size == 0 or content.mean() < p.min_content_luma:
        return None
    if not _sharp(gray, top, bottom, left, right, p):
        return None
    return FrameResult(Crop(top, bottom, left, right), (h, w))

# aggregate: one verdict from per-frame results (None = uninformative) sampled at times, crop scaled to source (w, h)
def aggregate(
    results: list[FrameResult | None],
    times: list[float],
    duration: float,
    p: Params,
    *,
    source: tuple[int, int],
    min_informative: int,
) -> Verdict:
    pts = [(t, _scale(r, source)) for r, t in zip(results, times) if r is not None]
    n = len(pts)
    if n < min_informative or duration <= 0:
        return Verdict('insufficient', None, 0.0, n)
    # the margins with most neighbours within agree_px win, ties go to the earliest
    best = max(pts, key=lambda a: sum(_close(a[1], b[1], p.agree_px) for b in pts))[1]
    agree = [(t, c) for t, c in pts if _close(c, best, p.agree_px)]
    confidence = len(agree) / n
    if confidence < p.agree_frac:
        return Verdict('variable', None, confidence, n)
    med = np.median([astuple(c) for _, c in agree], axis=0).round().astype(int)
    crop = snap_even(Crop(*(int(m) for m in med)), source)
    if not any(astuple(crop)):
        return Verdict('no border', None, confidence, n)
    # agreeing frames must span the video, otherwise it is a short window-boxed segment
    if not (any(t < duration / 3 for t, _ in agree) and any(t >= 2 * duration / 3 for t, _ in agree)):
        return Verdict('variable', None, confidence, n)
    return Verdict('border', crop, confidence, n)

# snap_even: left and top floored to even, output size forced even by trimming at most 1 px from right or bottom
def snap_even(crop: Crop, size: tuple[int, int]) -> Crop:
    w, h = size
    left, top = crop.left // 2 * 2, crop.top // 2 * 2
    right, bottom = crop.right // 2 * 2, crop.bottom // 2 * 2
    if (w - left - right) % 2 and w - left - right > 1:
        right += 1
    if (h - top - bottom) % 2 and h - top - bottom > 1:
        bottom += 1
    return Crop(top, bottom, left, right)

# _sharp: every non-zero margin has a luma step of at least min_edge_step across its edge
def _sharp(gray: np.ndarray, top: int, bottom: int, left: int, right: int, p: Params) -> bool:
    h, w = gray.shape
    b = p.edge_band
    cols, rows = slice(left, w - right), slice(top, h - bottom)  # span of the content on the other axis
    steps = []
    if top:
        steps.append(gray[top:top + b, cols].mean() - gray[max(0, top - b):top, cols].mean())
    if bottom:
        steps.append(gray[h - bottom - b:h - bottom, cols].mean() - gray[h - bottom:h - bottom + b, cols].mean())
    if left:
        steps.append(gray[rows, left:left + b].mean() - gray[rows, max(0, left - b):left].mean())
    if right:
        steps.append(gray[rows, w - right - b:w - right].mean() - gray[rows, w - right:w - right + b].mean())
    return all(s >= p.min_edge_step for s in steps)

# _scale: frame margins in source pixels
def _scale(r: FrameResult, source: tuple[int, int]) -> Crop:
    sy, sx = source[1] / r.shape[0], source[0] / r.shape[1]
    m = r.margins
    return Crop(round(m.top * sy), round(m.bottom * sy), round(m.left * sx), round(m.right * sx))

# _close: every side of a and b within tol pixels
def _close(a: Crop, b: Crop, tol: int) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(astuple(a), astuple(b)))
