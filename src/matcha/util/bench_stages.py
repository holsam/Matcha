'''
MATCHA: per-step timing benchmarks for the index and match commands, with sweeps and plots.

Usage:
    uv run python src/matcha/util/bench_stages.py run --label baseline
    uv run python src/matcha/util/bench_stages.py run --label phase1 --match-videos 1000 4000 --workers 1 4 8
    uv run python src/matcha/util/bench_stages.py plot
    uv run python src/matcha/util/bench_stages.py scaling --label m1 --seconds 60

run saves .local/bench/<label>.json; plot overlays every json in that folder (label = linestyle).

Steps are timed by wrapping module-level functions by name, so the same script works
across code versions. Steps a version lacks are simply absent from its results.
'''
# Import external dependencies
import argparse, contextlib, functools, inspect, io, json, os, resource, shutil, statistics, subprocess, sys, tempfile, threading, time
import numpy as np
from collections import defaultdict
from pathlib import Path

# Import internal matcha objects
import matcha.indexer as indexer
import matcha.matcher as matcher
from matcha.db import get_connection, init_schema

# _OUT_DIR: where result files and plots are written, relative to the working directory
_OUT_DIR = Path('.local/bench')

# _MATCH_TARGETS: (module attribute, step name) wrapped while run_match executes, in pipeline order
_MATCH_TARGETS = [
    ('load_videos', 'load videos'),
    ('update_index', 'faiss build/update'),
    ('maybe_rebuild_index', 'faiss rebuild check'),
    ('find_candidate_pairs', 'pass 1 candidates'),
    ('load_hashes_for', 'pass 2 load hashes'),
]

# _MATCH_REST: remainder of run_match wall time, i.e. pass 2 comparison and recording
_MATCH_REST = 'pass 2 verify'

# _INDEX_TARGETS: wrapped while run_index executes; per-video steps report mean seconds per call
_INDEX_TARGETS = [
    ('find_videos', 'scan files'),
    ('register_videos', 'register videos'),
    ('get_video_duration', 'ffprobe duration'),
    ('extract_frame_hashes', 'extract + hash frames'),
    ('get_audio_fingerprint', 'audio fingerprint'),
]


# _Timer: thread-safe accumulator of per-call durations by step name
class _Timer:
    def __init__(self) -> None:
        self.calls: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def add(self, name: str, secs: float) -> None:
        with self._lock:
            self.calls[name].append(secs)

# _instrument: temporarily wrap module functions (when they exist) so calls are timed
@contextlib.contextmanager
def _instrument(module, targets: list[tuple[str, str]], timer: _Timer):
    originals = {}
    for attr, name in targets:
        orig = getattr(module, attr, None)
        if orig is None:
            continue
        originals[attr] = orig

        def wrapper(*a, _orig=orig, _name=name, **k):
            start = time.perf_counter()
            try:
                return _orig(*a, **k)
            finally:
                timer.add(_name, time.perf_counter() - start)
        setattr(module, attr, wrapper)
    try:
        yield
    finally:
        for attr, orig in originals.items():
            setattr(module, attr, orig)

# _quiet: silence stdout/stderr and detach stdin so progress bars and key watchers stay out of the way
@contextlib.contextmanager
def _quiet():
    devnull = open(os.devnull, 'r+')
    old = sys.stdout, sys.stderr, sys.stdin
    sys.stdout = sys.stderr = io.TextIOWrapper(io.BytesIO())
    sys.stdin = devnull
    try:
        yield
    finally:
        sys.stdout, sys.stderr, sys.stdin = old
        devnull.close()

# _populate: fill a fresh DB with random hashes plus planted near-duplicate subclips
def _populate(db_path: str, n_videos: int, seed: int) -> int:
    rng = np.random.default_rng(seed)
    init_schema(db_path)
    conn = get_connection(db_path)
    lengths = rng.integers(20, 200, n_videos)
    hashes = [rng.integers(0, 2**63, int(n), dtype=np.uint64) for n in lengths]
    # every 10th video is a subclip of an earlier one with one flipped bit per frame
    for i in range(10, n_videos, 10):
        src = hashes[int(rng.integers(0, i))]
        n = min(len(hashes[i]), len(src))
        start = int(rng.integers(0, len(src) - n + 1))
        flips = np.uint64(1) << rng.integers(0, 64, n).astype(np.uint64)
        hashes[i] = src[start:start + n] ^ flips
    now = time.time()
    with conn:
        for i, h in enumerate(hashes, start=1):
            conn.execute(
                'INSERT INTO videos (id, path, duration, fingerprinted_at) VALUES (?, ?, ?, ?)',
                (i, f'/synthetic/{i}.mp4', float(len(h)), now),
            )
            conn.executemany(
                'INSERT INTO frame_hashes (video_id, timestamp, phash) VALUES (?, ?, ?)',
                [(i, float(t), f'{int(v):016x}') for t, v in enumerate(h)],
            )
    return int(lengths.sum())

# _match_once: one fresh synthetic project, one run_match, returns (steps, rows, frames)
def _match_once(n_videos: int, workers: int, seed: int) -> tuple[dict[str, float], dict[str, int], int]:
    timer = _Timer()
    with tempfile.TemporaryDirectory() as tmp:
        matcha_dir = os.path.join(tmp, '.matcha')
        os.makedirs(matcha_dir)
        db_path = os.path.join(matcha_dir, 'index.db')
        frames = _populate(db_path, n_videos, seed)
        with _quiet(), _instrument(matcher, _MATCH_TARGETS, timer):
            start = time.perf_counter()
            matcher.run_match(tmp, workers=workers)
            wall = time.perf_counter() - start
        conn = get_connection(db_path)
        rows = {t: conn.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]
                for t in ('candidate_pairs', 'comparisons', 'matches')}
    steps = {name: sum(timer.calls[name]) for _, name in _MATCH_TARGETS if name in timer.calls}
    steps[_MATCH_REST] = max(0.0, wall - sum(steps.values()))
    return steps, rows, frames

# _make_clips: write n_clips distinct test videos (H.264 video + sine audio) into directory; noise adds realistic bitrate
def _make_clips(directory: Path, n_clips: int, seconds: int, *, size: str = '320x240', noise: bool = False) -> list[Path]:
    clips = []
    for i in range(n_clips):
        path = directory / f'src_{i}.mp4'
        subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error',
             '-f', 'lavfi', '-i', f'testsrc2=size={size}:rate=30:duration={seconds}',
             '-f', 'lavfi', '-i', f'sine=frequency={220 + 55 * i}:duration={seconds}',
             *(['-vf', 'noise=alls=20:allf=t'] if noise else []),
             '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
             '-shortest', '-pix_fmt', 'yuv420p', str(path)],
            check=True,
        )
        clips.append(path)
    return clips

# _index_once: hard-link n_videos copies of the source clips, run_index once, returns (steps, wall)
def _index_once(
    clips: list[Path], n_videos: int, workers: int, *, no_audio: bool, hwaccel: bool = False
) -> tuple[dict[str, float], float]:
    timer = _Timer()
    with tempfile.TemporaryDirectory(dir=clips[0].parent) as tmp:
        for i in range(n_videos):
            os.link(clips[i % len(clips)], os.path.join(tmp, f'video_{i:05d}.mp4'))
        with _quiet(), _instrument(indexer, _INDEX_TARGETS, timer):
            start = time.perf_counter()
            # hard-linked copies would all be reused, so switch that off where this version has it
            extra = {'reuse_duplicates': False} if 'reuse_duplicates' in inspect.signature(indexer.run_index).parameters else {}
            indexer.run_index(tmp, fps=1.0, workers=workers, no_audio=no_audio, hwaccel=hwaccel, **extra)
            wall = time.perf_counter() - start
    steps = {name: statistics.fmean(timer.calls[name]) for _, name in _INDEX_TARGETS if name in timer.calls}
    return steps, wall

# _cpu_seconds: CPU time used by this process and its finished children (ffmpeg, ffprobe)
def _cpu_seconds() -> float:
    own, kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + kids.ru_utime + kids.ru_stime

# _scaling_once: one run_index with ffmpeg threads capped, returns (wall seconds, cpu seconds)
def _scaling_once(
    clips: list[Path], n_videos: int, workers: int, ffmpeg_threads: int | None, hwaccel: bool
) -> tuple[float, float]:
    original = indexer.extract_frame_hashes
    indexer.extract_frame_hashes = functools.partial(original, threads=ffmpeg_threads)
    try:
        cpu_start = _cpu_seconds()
        _, wall = _index_once(clips, n_videos, workers, no_audio=True, hwaccel=hwaccel)
        return wall, _cpu_seconds() - cpu_start
    finally:
        indexer.extract_frame_hashes = original

# _run_scaling: index wall time, speedup and CPU use over workers x ffmpeg threads, plus a PNG
def _run_scaling(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if shutil.which('ffmpeg') is None:
        sys.exit('ffmpeg not found on PATH')
    cores = os.cpu_count() or 1
    rows = []
    with tempfile.TemporaryDirectory() as clip_dir:
        clips = _make_clips(Path(clip_dir), args.clips, args.seconds, size=args.size, noise=args.noise)
        modes = {'off': [False], 'on': [True], 'both': [False, True]}[args.hwaccel]
        for hw, t in ((hw, t) for hw in modes for t in args.ffmpeg_threads):
            for w in args.workers:
                reps = [_scaling_once(clips, args.videos, w, t or None, hw) for _ in range(args.repeats)]
                wall = statistics.median(r[0] for r in reps)
                cpu = statistics.median(r[1] for r in reps)
                rows.append({'hwaccel': hw, 'ffmpeg_threads': t, 'workers': w, 'wall': wall, 'cpu': cpu,
                             'cpu_util': cpu / (wall * cores)})
                print(f'hwaccel={"on" if hw else "off":<4} ffmpeg-threads={t or "default":<8} workers={w:<3} wall={wall:6.2f}s  '
                      f'cpu={cpu:6.2f}s  cpu use={rows[-1]["cpu_util"]:5.0%} of {cores} cores', flush=True)
    out_dir = _OUT_DIR / 'scaling'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{args.label}.json').write_text(json.dumps(
        {'label': args.label, 'cores': cores, 'videos': args.videos, 'seconds': args.seconds, 'rows': rows}, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for i, (hw, t) in enumerate((hw, t) for hw in modes for t in args.ffmpeg_threads):
        mine = sorted((r for r in rows if r['ffmpeg_threads'] == t and r['hwaccel'] == hw), key=lambda r: r['workers'])
        xs = [r['workers'] for r in mine]
        name = f'hwaccel {"on" if hw else "off"}, threads: {t or "default"}'
        axes[0].plot(xs, [r['wall'] for r in mine], marker='o', color=f'C{i}', label=name)
        axes[1].plot(xs, [mine[0]['wall'] / r['wall'] for r in mine], marker='o', color=f'C{i}', label=name)
        axes[2].plot(xs, [r['cpu_util'] * 100 for r in mine], marker='o', color=f'C{i}', label=name)
    axes[1].plot(args.workers, args.workers, color='grey', ls=':', label='ideal')
    axes[0].set(title='wall time', ylabel='seconds')
    axes[1].set(title='speedup vs 1 worker (same ffmpeg threads)', ylabel='x')
    axes[2].set(title=f'CPU use ({cores} cores)', ylabel='% of all cores')
    axes[2].axhline(100, color='grey', ls=':')
    for ax in axes:
        ax.set(xlabel='workers')
        ax.set_xscale('log', base=2)
        ax.set_xticks(args.workers, [str(w) for w in args.workers])
        ax.minorticks_off()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=11)
    fig.suptitle(f'index scaling: {args.videos} videos of {args.seconds}s at {args.size}, {cores} cores')
    out = out_dir / f'{args.label}.png'
    fig.savefig(out, dpi=120)
    print(f'\nwritten to {out}')

# _median_steps: per-step median across repeats
def _median_steps(runs: list[dict[str, float]]) -> dict[str, float]:
    names = dict.fromkeys(n for r in runs for n in r)  # keeps pipeline order
    return {n: statistics.median(r[n] for r in runs if n in r) for n in names}

# _save: write one result file per label holding the runs of each command
def _save(label: str, commands: dict[str, list[dict]]) -> Path:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = _OUT_DIR / f'{label}.json'
    out.write_text(json.dumps({'label': label, **commands}, indent=2))
    return out

# _run_match: sweep videos x workers
def _run_match(args: argparse.Namespace) -> list[dict]:
    runs = []
    for n in args.match_videos:
        for w in args.workers:
            reps = [_match_once(n, w, args.seed) for _ in range(args.repeats)]
            steps = _median_steps([r[0] for r in reps])
            run = {'videos': n, 'workers': w, 'frames': reps[0][2], 'steps': steps,
                   'total': sum(steps.values()), 'rows': reps[0][1]}
            runs.append(run)
            print(f'match  videos={n:<6} workers={w:<3} total={run["total"]:7.2f}s  '
                  + '  '.join(f'{k}={v:.2f}' for k, v in steps.items()), flush=True)
    return runs

# _run_index: sweep videos x workers
def _run_index(args: argparse.Namespace) -> list[dict]:
    if shutil.which('ffmpeg') is None:
        sys.exit('ffmpeg not found on PATH')
    runs = []
    with tempfile.TemporaryDirectory() as clip_dir:
        clips = _make_clips(Path(clip_dir), args.clips, args.seconds)
        for n in args.index_videos:
            for w in args.workers:
                reps = [_index_once(clips, n, w, no_audio=args.no_audio) for _ in range(args.repeats)]
                steps = _median_steps([r[0] for r in reps])
                wall = statistics.median(r[1] for r in reps)
                runs.append({'videos': n, 'workers': w, 'steps': steps, 'total': wall,
                             'videos_per_s': n / wall})
                print(f'index  videos={n:<6} workers={w:<3} wall={wall:7.2f}s  {n / wall:5.2f} videos/s  '
                      + '  '.join(f'{k}={v:.2f}' for k, v in steps.items()), flush=True)
    return runs

# _load: read result files, one entry per (file, command)
def _load(paths: list[Path]) -> list[dict]:
    results = []
    for path in paths:
        data = json.loads(path.read_text())
        results += [{'command': c, 'label': data['label'], 'runs': data[c]}
                    for c in ('match', 'index') if c in data]
    return results

# _step_names: union of step names across result sets, in first-seen order
def _step_names(results: list[dict]) -> list[str]:
    seen: dict[str, None] = {}
    for res in results:
        for run in res['runs']:
            seen.update(dict.fromkeys(run['steps']))
    return list(seen)

# _plot: one figure per command; colour = label only, steps are separate panels
def _plot(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    results = _load(sorted(_OUT_DIR.glob('*.json')))
    if not results:
        sys.exit(f'no result files in {_OUT_DIR}, run the benchmark first')
    out_dir = _OUT_DIR / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = list(dict.fromkeys(r['label'] for r in results))
    colour = {label: f'C{i}' for i, label in enumerate(labels)}

    for command in sorted({r['command'] for r in results}):
        group = [r for r in results if r['command'] == command]
        steps = _step_names(group)
        unit = 'seconds' if command == 'match' else 'seconds per video'
        all_runs = [run for r in group for run in r['runs']]
        max_videos = max(run['videos'] for run in all_runs)
        max_workers = max(run['workers'] for run in all_runs)
        ncols = max(3, len(steps))

        fig = plt.figure(figsize=(3.4 * ncols, 11), constrained_layout=True)
        grid = fig.add_gridspec(3, ncols)

        # summary row: where the time goes, then total against videos and workers
        ax = fig.add_subplot(grid[0, 0])
        # viridis for steps so they never share a colour with a run label
        step_cols = {s: plt.cm.viridis(i / max(1, len(steps) - 1)) for i, s in enumerate(steps)}
        bars = [(res['label'], run) for res in group for run in res['runs']
                if run['videos'] == max_videos and run['workers'] == max_workers]  # labels run at the largest config
        for yi, (_, run) in enumerate(bars):
            left = 0.0
            for step in steps:
                width = run['steps'].get(step, 0.0)
                ax.barh(yi, width, left=left, color=step_cols[step], label=step if yi == 0 else None)
                left += width
        ax.set_yticks(range(len(bars)), [label for label, _ in bars])
        ax.set(title=f'where time goes ({max_videos:,} videos, {max_workers} workers)', xlabel='seconds')
        if bars:
            ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)  # below the bars so it hides nothing

        ax = fig.add_subplot(grid[0, 1])
        for res in group:
            pts = sorted((r['videos'], r['total']) for r in res['runs'] if r['workers'] == max_workers)
            ax.plot(*zip(*pts), marker='o', color=colour[res['label']], label=res['label'])
        ax.set(title=f'total vs videos ({max_workers} workers)', xlabel='videos', ylabel='wall seconds')
        ax.legend(fontsize=11)

        ax = fig.add_subplot(grid[0, 2])
        for res in group:
            pts = sorted((r['workers'], r['total']) for r in res['runs'] if r['videos'] == max_videos)
            ax.plot(*zip(*pts), marker='o', color=colour[res['label']], label=res['label'])
        first = next((sorted((r['workers'], r['total']) for r in res['runs'] if r['videos'] == max_videos)
                      for res in group if any(r['videos'] == max_videos for r in res['runs'])), [])
        ax.plot([p[0] for p in first], [first[0][1] * first[0][0] / p[0] for p in first],
                color='grey', ls=':', label='ideal')
        ax.set(title=f'total vs workers ({max_videos:,} videos)', xlabel='workers', ylabel='wall seconds')
        ax.legend(fontsize=11)

        # one small panel per step: row 1 against videos, row 2 against workers
        for ci, step in enumerate(steps):
            for row, x_key, fixed_key, fixed in ((1, 'videos', 'workers', max_workers),
                                                 (2, 'workers', 'videos', max_videos)):
                ax = fig.add_subplot(grid[row, ci])
                for res in group:
                    pts = sorted((r[x_key], r['steps'][step]) for r in res['runs']
                                 if r[fixed_key] == fixed and step in r['steps'])
                    if pts:
                        ax.plot(*zip(*pts), marker='o', color=colour[res['label']])
                ax.set(title=step, xlabel=x_key, ylabel=unit if ci == 0 else None)
                ax.set_ylim(bottom=0)
        for ax in fig.axes:
            ax.grid(alpha=0.3)
            if ax.get_xlabel() == 'workers':
                ax.set_xscale('log', base=2)
                ticks = sorted({r['workers'] for r in all_runs})
                ax.set_xticks(ticks, [str(t) for t in ticks])
                ax.minorticks_off()
        fig.suptitle(f'{command}: row 2 = steps vs videos at {max_workers} workers, '
                     f'row 3 = steps vs workers at {max_videos:,} videos (colour = run label)')
        out = out_dir / f'{command}.png'
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f'written to {out}')

def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)

    p_run = sub.add_parser('run', help='run the match and index benchmarks, save to .local/bench/<label>.json')
    p_run.add_argument('--label', default='run')
    p_run.add_argument('--only', choices=['match', 'index'], help='run just one command')
    p_run.add_argument('--match-videos', type=int, nargs='+', default=[1000, 2000, 4000, 8000])
    p_run.add_argument('--index-videos', type=int, nargs='+', default=[8, 32])
    p_run.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4, 8])
    p_run.add_argument('--repeats', type=int, default=1, help='median of N runs per config')
    p_run.add_argument('--seed', type=int, default=0)
    p_run.add_argument('--clips', type=int, default=4, help='distinct source clips, reused via hard links')
    p_run.add_argument('--seconds', type=int, default=20, help='clip length')
    p_run.add_argument('--no-audio', action='store_true')

    p_scale = sub.add_parser('scaling', help='index wall time and CPU use over workers x ffmpeg threads, with a plot')
    p_scale.add_argument('--label', default='scaling')
    p_scale.add_argument('--videos', type=int, default=32)
    p_scale.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4, 8])
    p_scale.add_argument('--ffmpeg-threads', type=int, nargs='+', default=[0, 1, 2, 4], help='0 = ffmpeg default')
    p_scale.add_argument('--repeats', type=int, default=1)
    p_scale.add_argument('--clips', type=int, default=4)
    p_scale.add_argument('--seconds', type=int, default=60, help='clip length; longer clips show decode-bound scaling')
    p_scale.add_argument('--hwaccel', choices=['off', 'on', 'both'], default='off', help='pass -hwaccel auto to ffmpeg')
    p_scale.add_argument('--size', default='320x240', help='source resolution, e.g. 1920x1080')
    p_scale.add_argument('--noise', action='store_true', help='add noise so the bitrate looks like real footage')

    sub.add_parser('plot', help='plot every result file in .local/bench together')

    args = parser.parse_args()
    if args.cmd == 'plot':
        _plot(args)
        return
    if args.cmd == 'scaling':
        _run_scaling(args)
        return
    commands = {}
    if args.only != 'index':
        commands['match'] = _run_match(args)
    if args.only != 'match':
        commands['index'] = _run_index(args)
    print(f'\nwritten to {_save(args.label, commands)}')

if __name__ == '__main__':
    main()
