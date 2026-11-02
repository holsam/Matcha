'''
MATCHA: detect near-black borders (letterbox, pillarbox, window-box) in sampled frames and aggregate them per video.
'''

# Import external dependencies
import fnmatch, hashlib, json, os, subprocess, time
from datetime import datetime, timezone
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, astuple, dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageDraw
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.progress import track
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeRemainingColumn

# Import internal matcha objects
from matcha.db import fold_in_moved_paths
from matcha.indexer import find_videos
from matcha.reencode import _CODEC_SCALE, _DEFAULT_SCALE, Info, _encode, _info, _precheck, _swap_in, _target
from matcha.review.media import human_size

# _TIMEOUT: seconds before a slow disk or corrupt file is given up on
_TIMEOUT = 30

# _GRID: sample times are snapped to this many seconds, so very short clips are not sampled twice at one frame
_GRID = 0.04

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

# sample_frame: one greyscale frame at t seconds scaled to shape (height, width), None when ffmpeg cannot deliver it
def sample_frame(path: str, t: float, shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    cmd = [
        'ffmpeg', '-v', 'error', '-threads', '1', '-ss', f'{t:.3f}', '-i', path,  # seek before input, no full decode
        '-frames:v', '1', '-an', '-sn', '-dn', '-vf', f'scale={w}:{h}', '-pix_fmt', 'gray', '-f', 'rawvideo', '-',
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=_TIMEOUT).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    return np.frombuffer(out, np.uint8).reshape(h, w) if len(out) == h * w else None

# scan_video: verdict for one video and the stage that decided it, A (few frames) or B (full sample)
def scan_video(path: str, p: Params) -> tuple[Verdict, Literal['A', 'B']]:
    dims = _dims(path)
    if dims is None or dims[2] <= 0:
        return Verdict('insufficient', None, 0.0, 0), 'A'
    w, h, duration = dims
    fw = min(w, p.max_width)
    shape = (max(1, round(h * fw / w)), fw)
    grid = _times(duration, p)
    results: dict[int, FrameResult | None] = {}

    # decide: sample any frames of idx not yet taken, then aggregate those frames
    def decide(idx: list[int], min_informative: int) -> Verdict:
        for i in idx:
            if i not in results:
                gray = sample_frame(path, grid[i], shape)
                results[i] = None if gray is None else detect_frame(gray, p)
        return aggregate([results[i] for i in idx], [grid[i] for i in idx], duration, p,
                         source=(w, h), min_informative=min_informative)

    a = sorted(set(np.linspace(0, len(grid) - 1, min(p.frames_a, len(grid))).round().astype(int).tolist()))
    verdict = decide(a, p.min_informative_a)
    if verdict.result == 'no border':
        return verdict, 'A'
    return decide(list(range(len(grid))), p.min_informative_b), 'B'

# Store: the borders.json file, entries keyed by path relative to the scanned directory
class Store:
    # __init__: load the file, a corrupt one is moved aside to .corrupt and flagged in recovered
    def __init__(self, path: str) -> None:
        self.path = path
        self.videos: dict[str, dict] = {}
        self.recovered = False
        self.dirty = False
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                videos = json.load(f)['videos']
            if not isinstance(videos, dict):
                raise TypeError('videos is not a mapping')
            self.videos = videos
        except (OSError, ValueError, KeyError, TypeError):
            os.replace(path, path + '.corrupt')
            self.recovered = True

    # put: set one entry, in memory until the next flush
    def put(self, rel: str, entry: dict) -> None:
        self.videos[rel] = entry
        self.dirty = True

    # drop: remove one entry
    def drop(self, rel: str) -> None:
        del self.videos[rel]
        self.dirty = True

    # flush: write to a temp file then rename over the store, so an interrupted write never corrupts it
    def flush(self) -> None:
        if not self.dirty:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + '.tmp'
        try:
            with open(tmp, 'w') as f:
                json.dump({'version': 1, 'videos': self.videos}, f, indent=1)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        self.dirty = False

# run_scan: scan every video under directory, resumable; only this (main) thread touches the store
def run_scan(
    directory: str,
    *,
    params: Params = Params(),
    workers: int = 4,
    rescan: bool = False,
    flush_every: float = 30.0,
    prune: bool = False,
) -> None:
    console = Console()
    root = os.path.abspath(directory)
    keys = {os.path.relpath(p, root): p for p in find_videos(root)}
    store = Store(os.path.join(root, '.matcha', 'borders.json'))
    if store.recovered:
        console.print(f'[yellow]Unreadable store moved to {escape(store.path)}.corrupt, starting again.[/yellow]')
    if prune:
        for rel in [k for k in store.videos if k not in keys]:
            store.drop(rel)
    ph = params_hash(params)
    todo: deque[tuple[str, str, os.stat_result]] = deque()
    for rel, path in keys.items():
        try:
            st = os.stat(path)
        except OSError:
            continue
        if rescan or not _fresh(store.videos.get(rel), st, ph):
            todo.append((rel, path, st))
    skipped = len(keys) - len(todo)
    console.print(f'{len(keys)} videos, {skipped} up to date, {len(todo)} to scan')

    counts: dict[str, int] = {'border': 0, 'no border': 0, 'variable': 0, 'insufficient': 0}
    stage_b = 0
    errors: list[tuple[str, Exception]] = []
    interrupted = False
    last_flush = time.monotonic()
    progress = Progress(
        SpinnerColumn(), BarColumn(), MofNCompleteColumn(), TextColumn('{task.fields[stats]}'), TimeRemainingColumn(),
        console=console,
    )
    task = progress.add_task('Scanning', total=len(todo), stats='')
    ex = ThreadPoolExecutor(max_workers=workers)
    inflight: dict = {}
    try:
        with progress:
            while todo or inflight:
                while todo and len(inflight) < workers * 2:
                    rel, path, st = todo.popleft()
                    store.put(rel, _entry(st, ph))  # pending, so a crash leaves a visible trace
                    inflight[ex.submit(scan_video, path, params)] = (rel, st)
                done, _ = wait(inflight, timeout=1.0, return_when=FIRST_COMPLETED)
                for fut in done:
                    rel, st = inflight.pop(fut)
                    try:
                        verdict, stage = fut.result()
                    except Exception as exc:  # leave the entry pending so a re-run retries it
                        errors.append((rel, exc))
                    else:
                        store.put(rel, _entry(st, ph, verdict))
                        counts[verdict.result] += 1
                        stage_b += stage == 'B'
                    progress.update(
                        task, advance=1,
                        stats=f'border {counts["border"]} | variable {counts["variable"]} | stage B {stage_b}',
                    )
                if time.monotonic() - last_flush >= flush_every:
                    store.flush()
                    last_flush = time.monotonic()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
        store.flush()
    if interrupted:
        console.print('\nStopped early. Progress saved, re-run `matcha borders scan` to resume.')
    console.print(', '.join(f'{n} {r}' for r, n in counts.items()) + f'; {stage_b} needed stage B')
    for rel, exc in errors[:5]:
        console.print(f'[red]failed[/red] {escape(rel)}: {escape(repr(exc))}')
    if errors:
        console.print(f'{len(errors)} videos failed and stay pending')

# run_list: print flagged videos with their crop boxes, returns the exit code
def run_list(
    directory: str,
    *,
    include_variable: bool = False,
    min_confidence: float = 0.0,
    sort: str = 'path',
    as_json: bool = False,
) -> int:
    console = Console()
    root = os.path.abspath(directory)
    path = os.path.join(root, '.matcha', 'borders.json')
    if sort not in ('path', 'confidence', 'area'):
        console.print(f'Unknown sort {escape(repr(sort))}, use path, confidence or area.')
        return 1
    try:
        with open(path) as f:
            videos = json.load(f)['videos']
    except FileNotFoundError:
        console.print('No border store found. Run `matcha borders scan` first.')
        return 1
    except (OSError, ValueError, KeyError, TypeError):  # read-only: scan is the command that moves a bad store aside
        console.print(f'Unreadable store {escape(path)}. Run `matcha borders scan` to rebuild it.')
        return 1
    wanted = {'border', 'variable'} if include_variable else {'border'}
    rows = [
        (rel, e) for rel, e in videos.items()
        if e.get('status') == 'complete' and e.get('result') in wanted
        and (e.get('confidence') or 0) >= min_confidence and os.path.exists(os.path.join(root, rel))
    ]
    with ThreadPoolExecutor(max_workers=4) as ex:  # only flagged rows are probed, the store holds no dimensions
        dims = list(ex.map(lambda r: _dims(os.path.join(root, r[0])), rows))
    items = [_item(rel, e, d) for (rel, e), d in zip(rows, dims)]
    items.sort(key={
        'path': lambda i: i['path'],
        'confidence': lambda i: (-i['confidence'], i['path']),
        'area': lambda i: (-i['removed_pixels'], i['path']),
    }[sort])
    if as_json:
        print(json.dumps(items, indent=1))
        return 0
    table = Table(title='Videos with borders')
    for col, right in (('Path', False), ('Result', False), ('Source', True), ('Crop (t b l r)', True), ('Output', True), ('Conf.', True)):
        table.add_column(col, justify='right' if right else 'left')
    for i in items:
        c = i['crop']
        table.add_row(
            escape(i['path']), i['result'],
            f'{i["width"]}x{i["height"]}' if i['width'] else '?',
            ' '.join(str(c[k]) for k in ('top', 'bottom', 'left', 'right')) if c else '-',
            f'{i["out_width"]}x{i["out_height"]}' if i['out_width'] else '-',
            f'{i["confidence"]:.2f}',
        )
    console.print(table)
    done = [e for e in videos.values() if e.get('status') == 'complete']
    pending = len(videos) - len(done)
    tally = ', '.join(f'{sum(e.get("result") == r for e in done)} {r}' for r in ('border', 'no border', 'variable', 'insufficient'))
    console.print(f'{len(videos)} stored: {tally}, {pending} pending')
    return 0

# run_preview: save a frame with the proposed crop box drawn on it, returns the exit code
def run_preview(path: str, *, at: float | None = None, out: str | None = None, detect: bool = False) -> int:
    console = Console()
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        console.print(f'Not a file: {escape(path)}')
        return 1
    dims = _dims(path)
    if dims is None or dims[2] <= 0:
        console.print(f'Cannot read video details from {escape(path)}')
        return 1
    w, h, duration = dims
    root, videos = _find_store(path)
    entry = videos.get(os.path.relpath(path, root)) if videos else None
    st = os.stat(path)
    if entry and entry.get('status') == 'complete' and (entry.get('size'), entry.get('mtime')) != (st.st_size, st.st_mtime):
        console.print('Stored result is stale (file changed), detecting afresh.')
        entry = None
    if detect or not entry or entry.get('status') != 'complete':
        verdict, _ = scan_video(path, Params())
        result, crop = verdict.result, verdict.crop and asdict(verdict.crop)
        console.print('Detected afresh, nothing is written to the store.')
    else:
        result, crop = entry['result'], entry.get('crop')
    box = Crop(**crop) if crop else None
    if at is None:
        at = _pick_time(path, duration, (w, h), box) if box else duration / 2
    out = out or os.path.join(root, '.matcha', 'tmp', f'borders_{os.path.splitext(os.path.basename(path))[0]}.png')
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    if not _grab(path, at, out):
        console.print(f'Could not read a frame at {at:.1f}s')
        return 1
    if box:
        with Image.open(out) as im:
            img = im.convert('RGB')
        iw, ih = img.size
        sx, sy = iw / w, ih / h  # the grabbed frame should match the probe, scale in case it does not
        ImageDraw.Draw(img).rectangle(
            [round(box.left * sx), round(box.top * sy), iw - 1 - round(box.right * sx), ih - 1 - round(box.bottom * sy)],
            outline=(255, 0, 0), width=max(2, iw // 300),
        )
        img.save(out)
        console.print(f'{result}: top {box.top}, bottom {box.bottom}, left {box.left}, right {box.right}, '
                      f'output {w - box.left - box.right}x{h - box.top - box.bottom}')
    else:
        console.print(f'Result is {result!r}, so no crop box is drawn.')
    print(out)
    return 0

# run_apply: crop every stored 'border' video in place to HEVC, deleting (or holding) the original, returns the exit code
def run_apply(
    directory: str,
    *,
    min_confidence: float = 0.0,
    only: str | None = None,
    preset: str = 'fast',
    keep_originals: str | None = None,
    flush_every: float = 30.0,
) -> int:
    console = Console()
    root = os.path.abspath(directory)
    keep = os.path.abspath(keep_originals) if keep_originals else None
    store = Store(os.path.join(root, '.matcha', 'borders.json'))
    todo = []
    for rel, e in sorted(store.videos.items()):
        if e.get('status') != 'complete' or e.get('result') != 'border' or not e.get('crop'):
            continue
        if (e.get('confidence') or 0) < min_confidence or (only and not fnmatch.fnmatch(rel, only)):
            continue
        path = os.path.join(root, rel)
        try:
            st = os.stat(path)
        except OSError:
            console.print(escape(f'skipped {rel}: file is missing'), style='yellow')
            continue
        if (e.get('size'), e.get('mtime')) != (st.st_size, st.st_mtime):
            console.print(escape(f'skipped {rel}: changed since the scan, scan again'), style='yellow')
            continue
        todo.append((rel, path, e))
    if not todo:
        console.print('Nothing to crop. `matcha borders list` shows what the store holds.')
        return 0
    db = os.path.join(root, '.matcha', 'index.db')
    db_path = db if os.path.exists(db) else None
    if db_path:
        fold_in_moved_paths(db_path)
    before = after = done = 0
    last_flush = time.monotonic()
    try:
        for rel, path, e in track(todo, description='Cropping', console=console):
            held = os.path.join(keep, rel) if keep else None
            new_path, sizes, reason = _crop_one(path, Crop(**e['crop']), preset, held, db_path)
            if reason:
                console.print(escape(f'skipped {rel}: {reason}'), style='yellow')
                continue
            new_rel = os.path.relpath(new_path, root)
            st = os.stat(new_path)
            if new_rel != rel:
                store.drop(rel)
            store.put(new_rel, {
                **e, 'size': st.st_size, 'mtime': st.st_mtime, 'result': 'no border', 'crop': None,
                'applied': {'crop': e['crop'], 'at': datetime.now(timezone.utc).isoformat(), 'original_size': sizes[0]},
            })
            before += sizes[0]
            after += sizes[1]
            done += 1
            if time.monotonic() - last_flush >= flush_every:
                store.flush()
                last_flush = time.monotonic()
    finally:
        store.flush()
    console.print(f'[bold green]{done}[/]/{len(todo)} cropped, {human_size(before)} to {human_size(after)}')
    return 0

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

# _dims: displayed (width, height, duration) of the first video stream, rotation applied, None when ffprobe fails
def _dims(path: str) -> tuple[int, int, float] | None:
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,duration:stream_side_data=rotation:stream_tags=rotate:format=duration',
        '-of', 'json', path,
    ]
    try:
        data = json.loads(subprocess.run(cmd, capture_output=True, timeout=_TIMEOUT, check=True).stdout)
    except (subprocess.SubprocessError, OSError, ValueError):
        return None
    stream = (data.get('streams') or [{}])[0]
    w, h = stream.get('width'), stream.get('height')
    if not (w and h):
        return None
    rotation = next((s['rotation'] for s in stream.get('side_data_list') or [] if 'rotation' in s),
                    (stream.get('tags') or {}).get('rotate', 0))
    try:
        if round(float(rotation)) % 180 == 90:  # ffmpeg autorotates, so frames come out turned
            w, h = h, w
    except (TypeError, ValueError):
        pass
    for raw in ((data.get('format') or {}).get('duration'), stream.get('duration')):
        try:
            return w, h, float(raw)
        except (TypeError, ValueError):
            continue
    return w, h, 0.0

# _times: stage B sample times, evenly across the span and snapped to _GRID
def _times(duration: float, p: Params) -> list[float]:
    lo, hi = p.span
    return sorted({round(t / _GRID) * _GRID for t in duration * np.linspace(lo, hi, p.frames_b)})

# _entry: store entry for a file, pending until a verdict is given
def _entry(st: os.stat_result, ph: str, v: Verdict | None = None) -> dict:
    return {
        'size': st.st_size,
        'mtime': st.st_mtime,
        'params_hash': ph,
        'status': 'pending' if v is None else 'complete',
        'result': v and v.result,
        'crop': v and v.crop and asdict(v.crop),
        'confidence': v and round(v.confidence, 3),
        'frames_used': v and v.frames_used,
    }

# _fresh: entry is complete and still matches the file and the params
def _fresh(entry: dict | None, st: os.stat_result, ph: str) -> bool:
    return bool(
        entry and entry.get('status') == 'complete' and entry.get('size') == st.st_size
        and entry.get('mtime') == st.st_mtime and entry.get('params_hash') == ph
    )

# _item: list/JSON record for one stored video, dimensions are None when the file cannot be probed
def _item(rel: str, e: dict, dims: tuple[int, int, float] | None) -> dict:
    crop = e.get('crop')
    w, h = (dims[0], dims[1]) if dims else (None, None)
    out = (w - crop['left'] - crop['right'], h - crop['top'] - crop['bottom']) if crop and dims else (None, None)
    removed = w * h - out[0] * out[1] if crop and dims else 0
    return {
        'path': rel, 'result': e['result'], 'crop': crop, 'confidence': e.get('confidence') or 0.0,
        'width': w, 'height': h, 'out_width': out[0], 'out_height': out[1], 'removed_pixels': removed,
    }

# _find_store: (directory, videos) of the nearest borders.json above path, or (the video's directory, None) when there is none
def _find_store(path: str) -> tuple[str, dict | None]:
    here = os.path.dirname(path)
    root = here
    while True:
        store = os.path.join(root, '.matcha', 'borders.json')
        if os.path.exists(store):
            try:
                with open(store) as f:
                    return root, json.load(f)['videos']
            except (OSError, ValueError, KeyError, TypeError):
                return here, None
        parent = os.path.dirname(root)
        if parent == root:
            return here, None
        root = parent

# _pick_time: sample time nearest the middle whose frame shows the stored crop, so the preview is a representative frame
def _pick_time(path: str, duration: float, source: tuple[int, int], crop: Crop) -> float:
    p = Params()
    shape = (max(1, round(source[1] * min(source[0], p.max_width) / source[0])), min(source[0], p.max_width))
    for t in sorted(_times(duration, p), key=lambda t: abs(t - duration / 2)):
        gray = sample_frame(path, t, shape)
        r = None if gray is None else detect_frame(gray, p)
        if r is not None and _close(_scale(r, source), crop, 2 * p.agree_px):
            return t
    return duration / 2

# _grab: save the full-size frame at t seconds as an image, False when ffmpeg cannot deliver it
def _grab(path: str, t: float, out: str) -> bool:
    cmd = ['ffmpeg', '-v', 'error', '-y', '-ss', f'{t:.3f}', '-i', path, '-frames:v', '1', out]
    try:
        subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=_TIMEOUT, check=True)
    except (subprocess.SubprocessError, OSError):
        return False
    return os.path.exists(out)

# _crop_one: crop and re-encode one file beside the original, verify, swap it in; returns (new path, (size before, after), reason skipped)
def _crop_one(path: str, crop: Crop, preset: str, held: str | None, db_path: str | None) -> tuple[str, tuple[int, int], str]:
    fail = (path, (0, 0))
    i, dims = _info(path), _dims(path)
    if i is None or dims is None:
        return *fail, 'unreadable'
    ow, oh = dims[0] - crop.left - crop.right, dims[1] - crop.top - crop.bottom
    if ow <= 0 or oh <= 0 or ow % 2 or oh % 2 or crop.left % 2 or crop.top % 2:
        return *fail, f'crop box {crop} does not give an even {dims[0]}x{dims[1]} output'
    if not i.bpp:
        return *fail, 'bitrate unknown'
    # hevc needs fewer bits than the source codec for the same quality, so scale the source's bits per pixel down
    rate = int(ow * oh * i.fps * i.bpp / _CODEC_SCALE.get(i.codec, _DEFAULT_SCALE))
    dst = _target(path)
    tmp = f'{os.path.splitext(path)[0]}.reencoding{os.path.splitext(dst)[1]}'
    try:
        reason = (_precheck(i, dst, held) or _encode(i, tmp, 0, preset, vf=f'crop={ow}:{oh}:{crop.left}:{crop.top}', rate=rate)
                  or _verify_crop(i, tmp, ow, oh))
        if reason:
            return *fail, reason
        size = os.path.getsize(tmp)
        _swap_in(path, tmp, dst, held, db_path)
        return dst, (i.size, size), ''
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

# _verify_crop: why the cropped file cannot replace the original, or None when it is hevc, the same length and the expected size
def _verify_crop(old: Info, tmp: str, ow: int, oh: int) -> str | None:
    new = _info(tmp)
    if new is None:
        return 'output unreadable'
    if new.codec != 'hevc':
        return f'output is {new.codec}'
    if abs(new.duration - old.duration) > 1:
        return f'duration {new.duration:.1f}s vs {old.duration:.1f}s'
    if (new.width, new.height) != (ow, oh):
        return f'output is {new.width}x{new.height}, expected {ow}x{oh}'
    return None
