'''
MATCHA: profile a library by codec and bitrate, and re-encode bloated videos to HEVC in place.
'''

# Import external dependencies
import os, random, shlex, shutil, subprocess, tempfile, time
from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable
import psutil
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TaskProgressColumn, TextColumn, TimeRemainingColumn, track
from rich.table import Table

# Import internal matcha objects
from matcha.db import fold_in_moved_paths, repoint_video
from matcha.indexer import find_videos
from matcha.review.media import human_duration, human_size, probe

# _HEVC_OK: containers that hold HEVC and keep their extension, anything else becomes mkv
_HEVC_OK = {'.mp4', '.mkv', '.mov'}

# _APPLE_TAG: containers that need the hvc1 tag for Apple players
_APPLE_TAG = {'.mp4', '.mov'}

# _ENCODERS: HEVC encoders --encoder accepts, x265 is software and videotoolbox is Apple's hardware encoder
_ENCODERS = ('x265', 'videotoolbox')

# _VT_RATE_SCALE: videotoolbox needs more bits than x265 for the same quality, so its bitrate is scaled up by this
_VT_RATE_SCALE = 1.2

# _BLOAT: a file is reported as bloated above this multiple of the bpp its codec needs
_BLOAT = 1.5

# _CODEC_SCALE: bpp a codec needs relative to hevc, lower is more efficient
_CODEC_SCALE = {'av1': 0.75, 'hevc': 1.0, 'vp9': 1.1, 'h264': 1.5, 'mpeg4': 2.0, 'mpeg2video': 2.5}

# _DEFAULT_SCALE: scale for codecs missing from _CODEC_SCALE, treated like h264
_DEFAULT_SCALE = 1.5

# Info: probed facts for one file
@dataclass(frozen=True)
class Info:
    path: str
    size: int
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    vbitrate: int  # video stream bits per second

    # bpp: bits per pixel per frame, the codec-agnostic density measure
    @property
    def bpp(self) -> float:
        px = self.width * self.height * self.fps
        return self.vbitrate / px if px else 0.0

    # bloat: bpp as a multiple of what this file's own codec needs at target_bpp, above 1 is over-allocated
    def bloat(self, target_bpp: float) -> float:
        return self.bpp / (target_bpp * _CODEC_SCALE.get(self.codec, _DEFAULT_SCALE))

    # saving: estimated bytes saved re-encoding to hevc at target_bpp, audio is unchanged
    def saving(self, target_bpp: float) -> int:
        if not self.bpp:
            return 0
        video_bytes = min(self.size, self.vbitrate * self.duration / 8)
        new_bytes = self.width * self.height * self.fps * target_bpp * self.duration / 8
        return int(max(0, video_bytes - new_bytes))

# _info: stat and probe one file, None when it is unreadable or has no video stream
def _info(path: str) -> Info | None:
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    p = probe(path)
    if p is None or not (p.width and p.height and p.codec):
        return None
    duration = p.duration or 0.0
    vbitrate = p.video_bitrate or p.bitrate or (int(size * 8 / duration) if duration else 0)
    return Info(path, size, duration, p.width, p.height, p.fps or 25.0, p.codec, vbitrate)

# _videos: video paths under directory, minus temp files from an interrupted run and anything inside exclude
def _videos(directory: str, exclude: str | None = None) -> list[str]:
    return [p for p in find_videos(os.path.abspath(directory))
            if '.reencoding.' not in os.path.basename(p) and not (exclude and p.startswith(exclude + os.sep))]

# _probe_all: probe paths in parallel, dropping files ffprobe cannot read
def _probe_all(paths: list[str], workers: int) -> list[Info]:
    with ThreadPoolExecutor(workers) as ex:
        return [i for i in track(ex.map(_info, paths), total=len(paths), description='Probing') if i]

# _bucket: resolution label by height
def _bucket(height: int) -> str:
    for limit, label in ((300, '<=240p'), (500, '480p'), (800, '720p'), (1200, '1080p'), (1800, '1440p')):
        if height <= limit:
            return label
    return '2160p+'

# _candidates: files whose estimated saving clears both thresholds, largest saving first
def _candidates(infos: list[Info], bpp: float, *, min_saving_pct: float, min_saving_mb: float) -> list[tuple[int, Info]]:
    scored = [(i.saving(bpp), i) for i in infos]
    keep = [(s, i) for s, i in scored if s >= min_saving_mb * 2**20 and s >= i.size * min_saving_pct / 100]
    return sorted(keep, key=lambda c: -c[0])

# _report: by-codec and by-resolution tables plus the bloated-file count
def _report(infos: list[Info], savings: dict[str, int], bpp: float, console: Console) -> None:
    for title, key in (('Codec', lambda i: i.codec), ('Resolution', lambda i: _bucket(i.height))):
        groups: dict[str, list[Info]] = defaultdict(list)
        for i in infos:
            groups[key(i)].append(i)
        table = Table(title=f'By {title.lower()}')
        for col in (title, 'Files', 'Size', 'Avg bpp', 'Est. saving'):
            table.add_column(col, justify='left' if col == title else 'right')
        for k, v in sorted(groups.items(), key=lambda kv: -sum(i.size for i in kv[1])):
            table.add_row(k, str(len(v)), human_size(sum(i.size for i in v)), f'{sum(i.bpp for i in v) / len(v):.3f}',
                          human_size(sum(savings[i.path] for i in v)))
        console.print(table)
    bloated = [i for i in infos if i.bloat(bpp) > _BLOAT and savings[i.path]]
    console.print(f'{len(bloated)} bloated files (>{_BLOAT}x the bpp their codec needs), '
                  f'{human_size(sum(savings[i.path] for i in bloated))} of the estimate')

# _ffmpeg: run ffmpeg quietly, overwriting outputs, raising on failure
def _ffmpeg(args: list[str]) -> None:
    subprocess.run(['ffmpeg', '-v', 'error', '-y', *args], check=True)

# _label: render a caption PNG, as ffmpeg builds often lack drawtext
def _label(text: str, path: str) -> None:
    font = ImageFont.load_default(size=22)
    img = Image.new('RGB', (int(font.getlength(text)) + 16, 36), 'black')
    ImageDraw.Draw(img).text((8, 5), text, fill='white', font=font)
    img.save(path)

# _preview: cut a 10s clip, re-encode it at each bpp, stitch original and encodes side by side, return the output path
def _preview(i: Info, bpps: list[float], out_dir: str, n: int) -> str:
    stem = f'{n:02d}_{os.path.splitext(os.path.basename(i.path))[0]}'[:60]
    clip = os.path.join(out_dir, f'{stem}_orig.mkv')
    _ffmpeg(['-ss', str(max(0, i.duration / 2 - 5)), '-i', i.path, '-t', '10', '-map', '0:v:0', '-c', 'copy', clip])
    panels = [(clip, f'original ({i.vbitrate / 1e6:.1f} Mbps, {i.codec})')]
    for bpp in bpps:
        rate = int(i.width * i.height * i.fps * bpp)
        enc = os.path.join(out_dir, f'{stem}_{bpp}.mp4')
        _ffmpeg(['-i', clip, '-c:v', 'libx265', '-tag:v', 'hvc1', '-x265-params', 'log-level=error',
                 '-b:v', str(rate), '-an', enc])
        panels.append((enc, f'{bpp} bpp ({rate / 1e6:.1f} Mbps, hevc)'))
    inputs: list[str] = []
    graph = []
    for k, (path, text) in enumerate(panels):
        png = os.path.join(out_dir, f'{stem}_label{k}.png')
        _label(text, png)
        inputs += ['-i', path, '-i', png]
        graph.append(f'[{2 * k}:v]scale=-2:360,setsar=1[v{k}];[v{k}][{2 * k + 1}:v]overlay=0:0[p{k}]')
    graph.append(''.join(f'[p{k}]' for k in range(len(panels))) + f'hstack=inputs={len(panels)}[out]')
    out = os.path.join(out_dir, f'{stem}_compare.mp4')
    _ffmpeg([*inputs, '-filter_complex', ';'.join(graph), '-map', '[out]', '-c:v', 'libx264', '-crf', '18', '-an', out])
    return out

# _previews: build a comparison clip for n random candidates and print where they are
def _previews(cands: list[Info], n: int, bpps: list[float], out_dir: str, console: Console) -> None:
    pool = [i for i in cands if i.duration > 1]
    os.makedirs(out_dir, exist_ok=True)
    picks = random.sample(pool, min(n, len(pool)))
    outs = [_preview(i, bpps, out_dir, k) for k, i in enumerate(track(picks, description='Previewing'))]
    console.print('\n'.join(['Previews:', *(os.path.abspath(o) for o in outs)]))

# _target: path the re-encoded file gets, the original path unless its container cannot hold hevc
def _target(path: str) -> str:
    root, ext = os.path.splitext(path)
    return path if ext.lower() in _HEVC_OK else root + '.mkv'

# Watch: optional hooks into a running encode
@dataclass(frozen=True)
class Watch:
    on_progress: Callable[[float, float, float, str], None] | None = None  # seconds done, total seconds, fps, speed
    echo: Callable[[str], None] | None = None  # receives the ffmpeg command line

# _parse_block: one ffmpeg -progress block as (seconds encoded, fps, speed), None before the first frame is out
def _parse_block(block: dict[str, str]) -> tuple[float, float, str] | None:
    try:
        done = max(0.0, int(block['out_time_us']) / 1e6)
    except (KeyError, ValueError):
        return None  # 'N/A' until the first frame is written
    try:
        fps = float(block.get('fps', ''))
    except ValueError:
        fps = 0.0
    return done, fps, block.get('speed', 'N/A').strip()

# _progress: live display with one overall row and one row for the file being encoded
def _progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(), TextColumn('{task.description}'), BarColumn(), TaskProgressColumn(), TextColumn('{task.fields[stats]}'),
        TimeRemainingColumn(), console=console,
    )

# _file_task: add a row for one file, returns its id and the Watch that feeds it
# verbose also prints the ffmpeg command above the display
def _file_task(progress: Progress, label: str, verbose: bool) -> tuple[TaskID, Watch]:
    task = progress.add_task(escape(label[-40:]).ljust(40), total=None, stats='')
    def update(done: float, total: float, fps: float, speed: str) -> None:
        progress.update(task, total=total or None, completed=min(done, total) if total else 0, stats=f'{fps:.0f} fps {speed}')
    def echo(cmd: str) -> None:
        progress.console.print(escape(cmd), style='dim')
    return task, Watch(update, echo if verbose else None)

# _CPU_BUSY: cpu percent above which no further job is started
_CPU_BUSY = 75.0

# _MEM_FLOOR: available bytes below which no further job is started
_MEM_FLOOR = 2 * 2**30

# adaptive_map: run fn over items with up to max_jobs threads, yielding (item, result) as each finishes
# starts at one job and adds another every settle seconds while cpu and memory have headroom, so a saturating x265 stays at one
def adaptive_map[T, R](items: list[T], fn: Callable[[T], R], max_jobs: int, *, settle: float = 15.0) -> Iterator[tuple[T, R]]:
    queue = list(reversed(items))
    limit, last_change = 1, time.monotonic()
    psutil.cpu_percent()  # first call only primes the counter
    with ThreadPoolExecutor(max(1, max_jobs)) as ex:
        running: dict[Future[R], T] = {}
        while queue or running:
            while queue and len(running) < limit:
                item = queue.pop()
                running[ex.submit(fn, item)] = item
            done, _ = wait(running, timeout=1.0, return_when=FIRST_COMPLETED)
            for f in done:
                yield running.pop(f), f.result()
            if queue and limit < max_jobs and len(running) >= limit and time.monotonic() - last_change >= settle:
                if psutil.cpu_percent() < _CPU_BUSY and psutil.virtual_memory().available > _MEM_FLOOR:
                    limit += 1
                last_change = time.monotonic()

# _summary: one result line for a finished file
def _summary(name: str, before: int, after: int, seconds: float) -> str:
    pct = (1 - after / before) * 100 if before else 0.0
    return f'{name}: {human_size(before)} to {human_size(after)} (-{pct:.0f}%) in {human_duration(seconds)}'

# check_encoder: exit with a clear message when the encoder is unknown or this ffmpeg lacks it
def check_encoder(encoder: str) -> None:
    if encoder not in _ENCODERS:
        raise SystemExit(f'unknown encoder {encoder!r}, choose from {", ".join(_ENCODERS)}')
    if encoder == 'videotoolbox':
        out = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
        if 'hevc_videotoolbox' not in out:
            raise SystemExit('this ffmpeg has no hevc_videotoolbox encoder')

# _video_args: the ffmpeg options that select the video encoder and its rate
# videotoolbox has no presets and takes a scaled-up rate, and decodes on the hardware where it can (ffmpeg falls back to software)
def _video_args(encoder: str, preset: str, rate: int) -> tuple[list[str], list[str]]:
    if encoder == 'videotoolbox':
        return ['-hwaccel', 'videotoolbox'], ['-c:v', 'hevc_videotoolbox', '-b:v', str(int(rate * _VT_RATE_SCALE))]
    return [], ['-c:v', 'libx265', '-preset', preset, '-b:v', str(rate), '-x265-params', 'log-level=error']

# _encode: re-encode to tmp at bpp keeping audio and subtitles, returns ffmpeg's last error line or None on success
# vf adds a video filter (such as a crop) and rate overrides the bitrate derived from bpp
def _encode(i: Info, tmp: str, bpp: float, preset: str, *, vf: str | None = None, rate: int | None = None, watch: Watch | None = None, encoder: str = 'x265') -> str | None:
    rate = rate or int(i.width * i.height * i.fps * bpp)
    pre, enc = _video_args(encoder, preset, rate)
    cmd = ['ffmpeg', '-v', 'error', '-y', '-progress', 'pipe:1', '-nostats', *pre, '-i', i.path, '-map', '0:v:0', '-map', '0:a?', '-map', '0:s?',
           *(['-vf', vf] if vf else []),
           '-c', 'copy', *enc]
    if os.path.splitext(tmp)[1].lower() in _APPLE_TAG:
        cmd += ['-tag:v', 'hvc1']
    cmd.append(tmp)
    if watch and watch.echo:
        watch.echo(shlex.join(cmd))
    with tempfile.TemporaryFile('w+') as err:  # a file, so a chatty stderr can never block the progress pipe
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err, text=True)
        block: dict[str, str] = {}
        for line in proc.stdout:
            key, _, value = line.strip().partition('=')
            block[key] = value
            if key == 'progress':  # last line of each block
                parsed, block = _parse_block(block), {}
                if parsed and watch and watch.on_progress:
                    watch.on_progress(parsed[0], i.duration, parsed[1], parsed[2])
        proc.wait()
        err.seek(0)
        lines = err.read().strip().splitlines()
    return None if proc.returncode == 0 else (lines or ['ffmpeg failed'])[-1]

# _verify: why the new file cannot replace the original, or None when it is hevc, the same length and 10% smaller
def _verify(old: Info, tmp: str) -> str | None:
    new = _info(tmp)
    if new is None:
        return 'output unreadable'
    if new.codec != 'hevc':
        return f'output is {new.codec}'
    if abs(new.duration - old.duration) > 1:
        return f'duration {new.duration:.1f}s vs {old.duration:.1f}s'
    if new.size >= old.size * 0.9:
        return 'output not at least 10% smaller'
    return None

# _reencode_one: encode beside the original, verify, then swap it in; returns (bytes saved, reason skipped)
def _reencode_one(i: Info, db_path: str | None, bpp: float, preset: str, held: str | None, watch: Watch | None = None, *, encoder: str = 'x265') -> tuple[int, str]:
    dst = _target(i.path)
    tmp = f'{os.path.splitext(i.path)[0]}.reencoding{os.path.splitext(dst)[1]}'
    try:
        reason = _precheck(i, dst, held) or _encode(i, tmp, bpp, preset, watch=watch, encoder=encoder) or _verify(i, tmp)
        if reason:
            return 0, reason
        saved = i.size - os.path.getsize(tmp)
        _swap_in(i.path, tmp, dst, held, db_path)
        return saved, ''
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

# _swap_in: replace the file at src with tmp (as dst), keeping its times; the original is held or deleted, the index follows
def _swap_in(src: str, tmp: str, dst: str, held: str | None, db_path: str | None) -> None:
    st = os.stat(src)
    if held:
        os.makedirs(os.path.dirname(held), exist_ok=True)
        shutil.move(src, held)  # copies across filesystems
    try:
        os.replace(tmp, dst)
    except OSError:
        if held:
            shutil.move(held, src)  # put the original back
        raise
    if held:
        os.symlink(dst, f'{os.path.splitext(held)[0]}.reencoded{os.path.splitext(dst)[1]}')
    elif dst != src:
        os.remove(src)
    os.utime(dst, (st.st_atime, st.st_mtime))
    if db_path:
        # same path too: content_key hashes the old size
        repoint_video(db_path, src, dst)

# _precheck: why this file must be skipped before any work is done, or None
def _precheck(i: Info, dst: str, held: str | None) -> str | None:
    if dst != i.path and os.path.lexists(dst):
        return f'{dst!r} already exists'
    if held and os.path.lexists(held):
        return f'{held!r} already exists in the keep directory'
    streams = _video_streams(i.path)
    if streams > 1:
        return f'{streams} video streams, only the first would be kept'
    return None

# _video_streams: number of video streams in a file
def _video_streams(path: str) -> int:
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v', '-show_entries', 'stream=index', '-of', 'csv=p=0', path],
        capture_output=True, text=True,
    ).stdout
    return len(out.split())

# run_reencode: profile the library, then optionally build preview clips or re-encode the candidates in place
def run_reencode(
    directory: str,
    *,
    bpp: float = 0.06,
    min_saving_pct: float = 30.0,
    min_saving_mb: float = 50.0,
    probe_workers: int = 4,
    preview: int = 0,
    bpps: str = '0.03,0.06,0.10',
    preview_dir: str | None = None,
    apply: bool = False,
    preset: str = 'fast',
    yes: bool = False,
    keep_originals: str | None = None,
    verbose: bool = False,
    encoder: str = 'x265',
    jobs: int = 1,
) -> None:
    check_encoder(encoder)
    directory = os.path.abspath(directory)
    keep = os.path.abspath(keep_originals) if keep_originals else None
    console = Console()
    paths = _videos(directory, keep)
    infos = _probe_all(paths, probe_workers)
    if not infos:
        console.print('no videos found')
        return
    savings = {i.path: i.saving(bpp) for i in infos}
    console.print(f'{len(infos)}/{len(paths)} probed, {human_size(sum(i.size for i in infos))} total')
    _report(infos, savings, bpp, console)
    cands = _candidates(infos, bpp, min_saving_pct=min_saving_pct, min_saving_mb=min_saving_mb)
    est = sum(s for s, _ in cands)
    console.print(f'{len(cands)} of {len(infos)} files qualify, est. saving [bold green]{human_size(est)}[/] at {bpp} bpp')
    if preview:
        out_dir = preview_dir or os.path.join(directory, '.matcha', 'tmp')
        _previews([i for _, i in cands], preview, [float(b) for b in bpps.split(',')], out_dir, console)
        return
    if not apply:
        for s, i in cands[:20]:
            console.print(escape(f'  {human_size(s):>10}  {i.codec:<6} {i.path}'))
        console.print('dry run: pass --apply to re-encode' + ('' if keep else ' (originals are deleted)'))
        return
    if not (keep or yes) and input(f'Permanently delete {len(cands)} originals? Type yes to continue: ').strip() != 'yes':
        raise SystemExit('aborted')
    db = os.path.join(directory, '.matcha', 'index.db')
    db_path = db if os.path.exists(db) else None
    if db_path:
        fold_in_moved_paths(db_path)  # videos from the old `move` command sit at moved_to, not path
    saved = done = 0
    with _progress(console) as progress:
        overall = progress.add_task('Re-encoding'.ljust(40), total=len(cands), stats='')
        def work(c: tuple[int, Info]) -> tuple[int, str, float]:
            i = c[1]
            held = os.path.join(keep, os.path.relpath(i.path, directory)) if keep else None
            task, watch = _file_task(progress, os.path.relpath(i.path, directory), verbose)
            start = time.monotonic()
            try:
                return *_reencode_one(i, db_path, bpp, preset, held, watch, encoder=encoder), time.monotonic() - start
            finally:
                progress.remove_task(task)
        for n, ((_, i), (gained, reason, secs)) in enumerate(adaptive_map(cands, work, jobs)):
            progress.update(overall, advance=1, stats=f'{n + 1}/{len(cands)} files')
            saved += gained
            done += bool(gained)
            if reason:
                progress.console.print(escape(f'skipped {i.path}: {reason}'), style='yellow')
            else:
                progress.console.print(escape(_summary(os.path.relpath(i.path, directory), i.size, i.size - gained, secs)))
    console.print(f'[bold green]{human_size(saved)}[/] saved across {done}/{len(cands)} files')
