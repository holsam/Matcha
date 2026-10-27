'''
MATCHA: step through match groups one at a time with timelines, file details and a recommended keep.
'''

# Import external dependencies
import json, math, numpy as np, os, questionary, time, sqlite3, subprocess
from dataclasses import dataclass
from numpy.lib.stride_tricks import sliding_window_view
from rich.console import Console
from rich.progress import track
from rich.text import Text

# Import internal matcha objects
from matcha.groups import build_groups

# _BAR_WIDTH: widest bar, shrunk on narrow terminals
_BAR_WIDTH = 50

# _MIN_BAR, _MIN_NAME: smallest bar and file name shown before the row would wrap
_MIN_BAR = 10
_MIN_NAME = 20

# _FULL_RATIO: a member at least this fraction of the source length counts as a duplicate, else a subclip (as in matcha)
_FULL_RATIO = 0.95

# _CODEC_EFFICIENCY: rough quality-per-bit relative to h264, used to compare bitrates across codecs
_CODEC_EFFICIENCY = {
    'av1': 2.0, 'hevc': 1.6, 'vp9': 1.5, 'h264': 1.0,
    'vp8': 0.8, 'mpeg4': 0.6, 'msmpeg4v3': 0.5, 'wmv3': 0.5, 'mpeg2video': 0.5, 'mpeg1video': 0.4,
}

# _STYLE: checkbox look, no reverse-video highlight; the cursor row is bold and ticked rows are green
_STYLE = questionary.Style([
    ('pointer', 'bold fg:ansicyan'),
    ('highlighted', 'bold noreverse'),
    ('selected', 'fg:ansigreen noreverse'),
    ('text', ''),
])

# _BPP_KNEE: codec-weighted bits per pixel per frame at which quality is about 63% of its ceiling (~0.1 is good h264)
_BPP_KNEE = 0.1

# Info: probed facts about one file
@dataclass(frozen=True)
class Info:
    id: int
    path: str
    size: int
    duration: float
    width: int
    height: int
    bitrate: int  # bits per second, 0 when unknown
    fps: float
    codec: str

# _clock: seconds as mm:ss (or h:mm:ss)
def _clock(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m:02d}:{s:02d}'

# _fmt: human readable byte count
def _fmt(n: float) -> str:
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.2f} TB'

# _probe: ffprobe one file, None when it is missing or unreadable
def _probe(vid: int, path: str, stored_duration: float) -> Info | None:
    try:
        size = os.path.getsize(path)
        out = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-print_format', 'json',
             '-show_entries', 'stream=width,height,codec_name,bit_rate,avg_frame_rate:format=duration,bit_rate', path],
            capture_output=True, text=True,
        ).stdout
        data = json.loads(out)
    except (OSError, ValueError):
        return None
    stream = (data.get('streams') or [{}])[0]
    fmt = data.get('format', {})
    duration = float(fmt.get('duration') or stored_duration or 0)
    bitrate = int(stream.get('bit_rate') or fmt.get('bit_rate') or (size * 8 / duration if duration else 0))
    num, _, den = (stream.get('avg_frame_rate') or '0/1').partition('/')
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return Info(vid, path, size, duration, int(stream.get('width') or 0), int(stream.get('height') or 0),
                bitrate, fps or 30.0, stream.get('codec_name') or '?')

# _quality: pixels scaled by how well the bitrate feeds them, so a starved 1080p can lose to a well fed 720p
def _quality(i: Info) -> float:
    pixels = i.width * i.height
    if not pixels:
        return 0.0
    bpp = i.bitrate * _CODEC_EFFICIENCY.get(i.codec, 1.0) / (pixels * i.fps)  # unknown codecs count as h264
    return pixels * (1 - math.exp(-bpp / _BPP_KNEE))

# _score: higher is better; quality first, then size as the tiebreak
def _score(i: Info) -> tuple[float, int]:
    return (_quality(i), i.size)

# _identical: files indistinguishable on everything shown, so no basis to prefer one
def _identical(i: Info) -> tuple:
    return (i.width, i.height, i.codec, i.bitrate, i.size, round(i.duration, 1))

# _hashes: (timestamps, uint64 phashes) for one video
def _hashes(conn: sqlite3.Connection, video_id: int) -> tuple[np.ndarray, np.ndarray]:
    rows = conn.execute(
        'SELECT timestamp, phash FROM frame_hashes WHERE video_id = ? ORDER BY timestamp', (video_id,)
    ).fetchall()
    stamps = np.array([r[0] for r in rows])
    hashes = np.frombuffer(bytes.fromhex(''.join(r[1] for r in rows)), dtype='>u8').astype(np.uint64)
    return stamps, hashes

# _locate: start time in the source where the clip best aligns, plus the fraction of frames matching
def _locate(conn: sqlite3.Connection, clip_id: int, source_id: int, threshold: int) -> tuple[float, float]:
    _, clip = _hashes(conn, clip_id)
    stamps, source = _hashes(conn, source_id)
    if len(clip) == 0 or len(source) < len(clip):
        return 0.0, 0.0
    xor = np.bitwise_xor(sliding_window_view(source, len(clip)), clip)
    hits = (np.bitwise_count(xor) <= threshold).sum(axis=1)
    best = int(hits.argmax())
    return float(stamps[best]), float(hits[best]) / len(clip)

# _bar: one row of the diagram, the clip drawn from start to end on a scale of total seconds
def _bar(start: float, end: float, total: float, width: int) -> str:
    a = min(int(start / total * width), width - 1)
    b = min(width, max(a + 1, int(end / total * width)))
    return ' ' * a + '-' * (b - a) + ' ' * (width - b)

# _show: print one group, probing only its own files
def _show(
    console: Console, conn: sqlite3.Connection, videos: dict[int, tuple[str, float]],
    members: set[int], index: int, count: int, threshold: int,
) -> tuple[list[Info], set[int]]:
    infos = [i for i in (_probe(v, *videos[v]) for v in members) if i]
    console.print(f'[bold]Group {index + 1}/{count}[/]  {len(members)} files'
                  + (f'  [dim]({len(members) - len(infos)} missing on disk)[/]' if len(infos) < len(members) else ''))
    if len(infos) < 2:
        console.print('  [dim]fewer than two files on disk, nothing to compare[/]\n')
        return [], set()

    source = max(infos, key=lambda i: (i.duration, _score(i)))
    full = [i for i in infos if i.duration >= _FULL_RATIO * source.duration]
    keep = max(full, key=_score)
    tied = [i for i in full if _identical(i) == _identical(keep)]
    tied = tied if len(tied) > 1 else []  # several identical top files: keep any one, so none is pre-picked
    saving = sum(i.size for i in full if i is not keep)

    total = max(source.duration, 1e-9)
    ordered = sorted(sorted(infos, key=_score, reverse=True), key=lambda i: i not in full)
    rows: list[tuple[str, float, float, str, Text]] = []  # (name, bar start, bar end, bar style, details right of the bar)
    for i in ordered:
        is_full = i in full
        start, frac = (0.0, 1.0) if i is source else _locate(conn, i.id, source.id, threshold)
        if is_full:
            start = 0.0  # a duplicate spans the whole source, so draw it from the start
        tag = Text('KEEP ONE', style='green') if i in tied else Text('KEEP', style='green') if i is keep else (
            Text('remove', style='dim') if is_full else Text(f'subclip @ {_clock(start)}-{_clock(start + i.duration)} ({frac:.0%})', style='yellow'))
        detail = Text(f'  {_clock(i.duration)}  {i.height}p  {i.codec}  {i.bitrate / 1e6:.1f} Mbps  {_fmt(i.size):>9}  ')
        detail.append_text(tag)
        rows.append((os.path.basename(i.path), start, start + i.duration,
                     'green' if i is keep or i in tied else 'cyan' if is_full else 'yellow', detail))

    # fit every row to the terminal: names get up to their full length, the bar absorbs what is left
    detail_w = max(d.cell_len for *_, d in rows)
    longest = max(len(r[0]) for r in rows)
    bar_w = min(_BAR_WIDTH, max(_MIN_BAR, console.width - 4 - 2 - detail_w - _MIN_NAME))
    name_w = min(longest, max(_MIN_NAME, console.width - 4 - 2 - detail_w - bar_w))
    for name, start, end, style, detail in rows:
        shown = name if len(name) <= name_w else name[:name_w - 3] + '...'
        line = Text(f'  {shown:<{name_w}}  ')
        line.append(_bar(start, end, total, bar_w), style=style)
        line.append_text(detail)
        console.print(line, soft_wrap=True)
    console.print(f'  [bold green]{_fmt(saving)}[/] reclaimable by keeping '
                  + (f'one of {len(tied)} identical files' if tied else os.path.basename(keep.path))
                  + ('  [dim](subclips left alone)[/]' if len(full) < len(infos) else '') + '\n')
    return ordered, set() if tied else {keep.id}  # the recommended keep starts ticked, unless it is a tie

# _ask_name: prompt for a file's new name without its extension, which is added back; None when cancelled
def _ask_name(path: str) -> str | None:
    folder, old = os.path.split(path)
    stem, ext = os.path.splitext(old)

    def check(name: str) -> bool | str:
        if not name.strip() or '/' in name:
            return 'enter a file name without slashes'
        if name.strip() + ext != old and os.path.exists(os.path.join(folder, name.strip() + ext)):
            return 'a file with that name already exists'
        return True

    answer = questionary.text(f'Name for {old} (extension {ext or "none"} kept)', default=stem, validate=check, style=_STYLE).ask()
    return answer.strip() + ext if answer is not None else None

# _load_plan: previously decided groups keyed by their sorted video ids, empty when there is no plan yet
def _load_plan(path: str) -> dict[str, dict]:
    try:
        with open(path) as f:
            return {g['key']: g for g in json.load(f)['groups']}
    except (OSError, ValueError, KeyError):
        return {}

# _save_plan: write the plan atomically so an interrupted run never leaves a half-written file
def _save_plan(path: str, entries: dict[str, dict]) -> None:
    tmp = f'{path}.tmp'
    with open(tmp, 'w') as f:
        json.dump({'version': 1, 'updated_at': time.time(), 'groups': list(entries.values())}, f, indent=2)
    os.replace(tmp, path)

# run_plan: build groups from the db, then step through them
def run_plan(directory, *, min_confidence=0.8, threshold=10, no_pause=False, plan=None, restart=False) -> None:
    db_path = os.path.join(os.path.abspath(directory), '.matcha', 'index.db')
    if not os.path.exists(db_path):
        raise SystemExit(f'no index at {db_path!r}')
    conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)  # read only
    query = 'SELECT video_a_id, video_b_id FROM matches WHERE confidence >= ?'
    if 'dismissed' in {r[1] for r in conn.execute('PRAGMA table_info(matches)')}:
        query += ' AND dismissed = 0'
    pairs = conn.execute(query, (min_confidence,)).fetchall()
    videos = {r[0]: (r[1], r[2] or 0) for r in conn.execute('SELECT id, path, duration FROM videos')}

    # order groups by estimated saving: full-length files only (subclips are never removed), all but the largest,
    # from stored durations and cheap stat calls so no ffprobe is needed up front
    def estimate(group: set[int]) -> int:
        present = [v for v in group if os.path.exists(videos[v][0])]
        longest = max((videos[v][1] for v in present), default=0)
        sizes = sorted((os.path.getsize(videos[v][0]) for v in present if videos[v][1] >= _FULL_RATIO * longest), reverse=True)
        return sum(sizes[1:])
    groups = sorted(track(build_groups(pairs), description='Sizing groups'), key=estimate, reverse=True)

    plan_path = plan or os.path.join(os.path.abspath(directory), '.matcha', 'review_plan.json')
    plan = {} if restart or no_pause else _load_plan(plan_path)  # --no-pause decides nothing, so never touches the plan
    console = Console(highlight=False)
    saved = sum(i['size'] for g in plan.values() for i in g['files'] if i['action'] == 'delete')
    deleted = sum(i['action'] == 'delete' for g in plan.values() for i in g['files'])  # bytes and files marked for deletion so far
    for n, group in enumerate(groups):
        key = ','.join(map(str, sorted(group)))
        if key in plan:
            continue  # already decided in an earlier run
        if not no_pause:
            console.clear()
            console.print(f'[bold green]{_fmt(saved)}[/] saved so far ({deleted} files marked for deletion)\n')
        ordered, ticked = _show(console, conn, videos, group, n, len(groups), threshold)
        if no_pause or not ordered:
            continue
        # Ctrl-C returns None and ends the review; ticked files are kept, the rest are marked for deletion
        picks = questionary.checkbox(
            'Tick files to keep (space toggles, Enter confirms, Ctrl-C finishes)',
            choices=[questionary.Choice(os.path.basename(i.path), value=i.path, checked=i.id in ticked) for i in ordered],
            style=_STYLE,
        ).ask()
        if picks is None:
            break
        renames = {}
        for path in picks:
            name = _ask_name(path)
            if name is None:
                picks = None
                break
            renames[path] = name
        if picks is None:
            break
        console.print()
        gone = [i for i in ordered if i.path not in picks]
        saved += sum(i.size for i in gone)
        deleted += len(gone)
        plan[key] = {
            'key': key,
            'decided_at': time.time(),
            'files': [
                {
                    'id': i.id, 'path': i.path, 'size': i.size,
                    **({'action': 'delete'} if i.path not in picks else
                       {'action': 'rename', 'new_path': os.path.join(os.path.dirname(i.path), renames[i.path])}
                       if renames[i.path] != os.path.basename(i.path) else {'action': 'keep'}),
                }
                for i in ordered
            ],
        }
        _save_plan(plan_path, plan)
        for i in ordered:
            old = os.path.basename(i.path)
            if i.path not in picks:
                console.print(f'  [red]delete[/]  {old}', markup=True)
            elif renames[i.path] != old:
                console.print(f'  [yellow]rename[/]  {old} -> {renames[i.path]}', markup=True)
            else:
                console.print(f'  [green]keep[/]    {old}', markup=True)
        console.print()

    if not no_pause:
        console.print(f'[bold green]{_fmt(saved)}[/] saved in total ({deleted} files marked for deletion)\nplan: {plan_path}')
