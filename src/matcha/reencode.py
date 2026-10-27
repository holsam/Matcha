'''
MATCHA: profile a library by codec and bitrate, and re-encode bloated videos to HEVC in place.
'''

# Import external dependencies
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from rich.console import Console
from rich.markup import escape
from rich.progress import track
from rich.table import Table

# Import internal matcha objects
from matcha.indexer import find_videos
from matcha.review.media import human_size, probe

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

# _videos: video paths under directory, minus temp files from an interrupted run
def _videos(directory: str) -> list[str]:
    return [p for p in find_videos(os.path.abspath(directory)) if '.reencoding.' not in os.path.basename(p)]

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

# run_reencode: profile the library and list the files worth re-encoding, without touching anything
def run_reencode(
    directory: str,
    *,
    bpp: float = 0.06,
    min_saving_pct: float = 30.0,
    min_saving_mb: float = 50.0,
    probe_workers: int = 4,
) -> None:
    console = Console()
    paths = _videos(directory)
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
    for s, i in cands[:20]:
        console.print(escape(f'  {human_size(s):>10}  {i.codec:<6} {i.path}'))
