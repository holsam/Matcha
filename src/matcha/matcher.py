import itertools, os, threading, time, typer
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from numpy.lib.stride_tricks import sliding_window_view
from datetime import datetime, timezone
from rich import print
from tqdm import tqdm

from matcha.config import save_run_config
from matcha.db import get_connection
from matcha.faiss_index import update_index, maybe_rebuild_index, find_candidate_pairs
from matcha.interactive import watch_for_quit

@dataclass
class VideoRecord:
    id: int
    path: str
    duration: float
    has_audio: bool

def _print_message(stage: str, msg: str):
    ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
    tab = stage.count('.') + 1
    print_msg = f'({ts})'+'\t'*tab+f'{msg}'
    print(f'[dim]{print_msg}[/dim]')


def load_videos(db_path: str) -> list[VideoRecord]:
    conn = get_connection(db_path)
    audio_ids = {
        row["video_id"]
        for row in conn.execute("SELECT video_id FROM audio_fingerprints").fetchall()
    }
    rows = conn.execute(
        "SELECT id, path, duration FROM videos WHERE fingerprinted_at IS NOT NULL"
    ).fetchall()
    return [
        VideoRecord(
            id=row["id"],
            path=row["path"],
            duration=row["duration"] or 0.0,
            has_audio=row["id"] in audio_ids,
        )
        for row in rows
    ]


# load_hashes_for: one streaming pass over frame_hashes, returns uint64 arrays for the wanted videos only
def load_hashes_for(db_path: str, video_ids: set[int]) -> dict[int, np.ndarray]:
    conn = get_connection(db_path)
    cursor = conn.execute(
        "SELECT video_id, phash FROM frame_hashes ORDER BY video_id, timestamp"
    )
    hashes: dict[int, np.ndarray] = {}
    for vid, group in itertools.groupby(cursor, key=lambda r: r["video_id"]):
        if vid in video_ids:
            hexes = ''.join(r["phash"] for r in group)
            # big-endian parse equals int(hex, 16) per 16-char hash
            hashes[vid] = np.frombuffer(bytes.fromhex(hexes), dtype='>u8').astype(np.uint64)
    return hashes


def load_frame_hashes(db_path: str, video_id: int) -> list[str]:
    """Fetch frame hashes for a single video. Called per-pair inside the worker."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT phash FROM frame_hashes WHERE video_id = ? ORDER BY timestamp",
        (video_id,),
    ).fetchall()
    return [row["phash"] for row in rows]


def generate_pairs(
    videos: list[VideoRecord],
    filter_length: bool,
) -> list[tuple[VideoRecord, VideoRecord]]:
    pairs = []
    for a, b in itertools.combinations(videos, 2):
        if filter_length and a.duration == b.duration:
            continue
        short, long = (a, b) if a.duration <= b.duration else (b, a)
        pairs.append((short, long))
    return pairs


def get_compared_pairs(db_path: str) -> set[tuple[int, int]]:
    conn = get_connection(db_path)
    rows = conn.execute("SELECT video_a_id, video_b_id FROM comparisons").fetchall()
    return {(row["video_a_id"], row["video_b_id"]) for row in rows}


# _FLUSH_EVERY: comparisons buffered before a DB write
_FLUSH_EVERY = 500

# _flush_results: write buffered comparisons and matches in one transaction
def _flush_results(
    db_path: str,
    comparisons: list[tuple[int, int]],
    matches: list[tuple[int, int, str, float]],
) -> None:
    if not comparisons and not matches:
        return
    conn = get_connection(db_path)
    now = time.time()
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO comparisons (video_a_id, video_b_id) VALUES (?, ?)",
            [(min(a, b), max(a, b)) for a, b in comparisons],
        )
        conn.executemany(
            """
            INSERT INTO matches (video_a_id, video_b_id, match_type, confidence, found_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(a, b, t, c, now) for a, b, t, c in matches],
        )
    comparisons.clear()
    matches.clear()

# _WINDOW_CHUNK_ELEMS: cap on window positions x frames per XOR block (~16 MB of uint64)
_WINDOW_CHUNK_ELEMS = 2_000_000

def sliding_window_match_numpy(
    short_hashes: np.ndarray,
    long_hashes: np.ndarray,
    frame_step: int,
    threshold: int,
) -> float:
    """
    Vectorised sliding window over uint64 pHash arrays. XOR + np.bitwise_count
    gives Hamming distances for every window position, chunked to bound memory.
    """
    n = len(short_hashes)
    m = len(long_hashes)
    if n == 0 or m < n:
        return 0.0
    windows = sliding_window_view(long_hashes, n)[::frame_step]
    chunk = max(1, _WINDOW_CHUNK_ELEMS // n)
    best = 0
    for start in range(0, len(windows), chunk):
        xor = np.bitwise_xor(windows[start:start + chunk], short_hashes)
        best = max(best, int((np.bitwise_count(xor) <= threshold).sum(axis=1).max()))
        if best == n:
            break  # can't improve further
    return best / n

def determine_match_type(short: VideoRecord, long: VideoRecord) -> str:
    if long.duration == 0:
        return "duplicate"
    return "duplicate" if (short.duration / long.duration) >= 0.95 else "subclip"


def _compare_pair(args: tuple) -> tuple[int, int, float]:
    """Worker — runs the sliding window on preloaded uint64 hash arrays."""
    short_id, long_id, short_hashes, long_hashes, frame_step, threshold = args
    confidence = sliding_window_match_numpy(short_hashes, long_hashes, frame_step, threshold)
    return short_id, long_id, confidence

def run_match(
    directory: str,
    filter_length: bool = False,
    window: float = 10.0,
    frame_step: int = 3,
    threshold: int = 10,
    min_confidence: float = 0.8,
    workers: int = 4,
    nprobe: int = 32,
):
    """Main entry point for the match subcommand."""
    directory = os.path.abspath(directory)
    matcha_dir = os.path.join(directory, ".matcha")
    save_run_config(matcha_dir, "match", {
        "filter_length": filter_length,
        "window": window,
        "frame_step": frame_step,
        "threshold": threshold,
        "min_confidence": min_confidence,
        "workers": workers,
        "nprobe": nprobe,
    })

    index_dir = os.path.join(directory, ".matcha")
    db_path = os.path.join(index_dir, "index.db")

    if not os.path.exists(db_path):
        typer.echo("No index found. Run `matcha index` first.")
        raise SystemExit(1)

    print(f"\n:tea: [bold green]Matcha[/bold green]")
    print(f"Matching videos in [cyan]{directory}[/cyan] by perceptual hashes...")
    _print_message('1', 'Loading index...')
    videos = load_videos(db_path)
    if not videos:
        _print_message('1', 'No indexed videos found. Run `matcha index` first.')
        return
    video_map: dict[int, VideoRecord] = {v.id: v for v in videos}
    # Pass 1
    _print_message('2', 'Starting Pass 1 (candidate generation)...')
    _print_message('2.1', 'Checking FAISS index state...')
    changed = update_index(db_path, index_dir, nprobe)
    if not changed:
        _print_message('2.1', 'FAISS index up to date.')
    if maybe_rebuild_index(db_path, index_dir, nprobe):
        _print_message('2.1', 'FAISS index rebuilt to restore query speed.')
    conn = get_connection(db_path)
    _print_message('2.3', 'Querying FAISS index for candidate pairs...')
    new_candidates, candidates_stopped_early = find_candidate_pairs(
        db_path,
        index_dir,
        threshold=threshold,
        nprobe=nprobe,
        batch_size=10_000,
        workers=workers,
    )
    if candidates_stopped_early:
        typer.echo("\nStopped early during candidate search. Progress has been saved — resume with `matcha match`.")
        return
    all_candidates: set[tuple[int, int]] = {
        (row['video_a_id'], row['video_b_id'])
        for row in conn.execute('SELECT video_a_id, video_b_id FROM candidate_pairs').fetchall()
    }
    _print_message('2', f'{len(all_candidates):,} candidate pairs identified ({len(new_candidates):,} new this run).')
    # Pass 2
    _print_message('3', 'Starting Pass 2 (candidate comparisons)...')
    already_compared = get_compared_pairs(db_path)
    pairs_to_run: list[tuple[VideoRecord, VideoRecord]] = []
    too_short: list[tuple[VideoRecord, VideoRecord]] = []
    skipped = 0
    _print_message('3.1', 'Verifying candidates...')
    for a_id, b_id in all_candidates:
        if (a_id, b_id) in already_compared or (b_id, a_id) in already_compared:
            skipped += 1
            continue
        a = video_map.get(a_id)
        b = video_map.get(b_id)
        if a is None or b is None:
            continue
        if filter_length and a.duration == b.duration:
            continue
        short, long = (a, b) if a.duration <= b.duration else (b, a)
        if short.duration < window:
            too_short.append((short, long))
        else:
            pairs_to_run.append((short, long))
    _print_message('3.1', 'All candidates verified.')
    _print_message('3.2', 'Pass 2 pairs:')
    _print_message('3.2.1', f'Pairs to verify: {len(pairs_to_run)}')
    _print_message('3.2.2', f'Already compared: {skipped}')
    _print_message('3.3.3', f'Too short to check: {len(too_short)}')
    _flush_results(db_path, [(short.id, long.id) for short, long in too_short], [])
    if not pairs_to_run:
        _print_message('3.3', 'No eligible pairs to verify')
        return
    needed = {v.id for pair in pairs_to_run for v in pair}
    _print_message('3.4', f'Loading frame hashes for {len(needed):,} video(s)...')
    hashes = load_hashes_for(db_path, needed)
    worker_args = [
        (s.id, l.id, hashes[s.id], hashes[l.id], frame_step, threshold)
        for s, l in pairs_to_run
    ]
    if filter_length:
        typer.echo("Length filter: on")
    typer.echo("Press 'q' to stop matching early.\n")
    matches_found = 0
    stopped_early = False
    stop_event = threading.Event()
    quit_thread = threading.Thread(target=watch_for_quit, args=(stop_event,), daemon=True)
    quit_thread.start()
    pending_comparisons: list[tuple[int, int]] = []
    pending_matches: list[tuple[int, int, str, float]] = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_compare_pair, arg): arg for arg in worker_args}
            with tqdm(total=len(futures), unit="pair", dynamic_ncols=True) as bar:
                for future in as_completed(futures):
                    if stop_event.is_set():
                        # Cancel all queued futures — in-flight ones finish but
                        # their results are not consumed, so they remain unrecorded
                        for f in futures:
                            f.cancel()
                        stopped_early = True
                        break
                    short_id, long_id, confidence = future.result()
                    short = video_map[short_id]
                    long = video_map[long_id]
                    pending_comparisons.append((short_id, long_id))
                    if confidence >= min_confidence:
                        match_type = determine_match_type(short, long)
                        pending_matches.append((short_id, long_id, match_type, confidence))
                        matches_found += 1
                        bar.write(
                            f"  MATCH  {match_type:<10}  {confidence:.0%}  "
                            f"{os.path.basename(short.path)}  ←  {os.path.basename(long.path)}"
                        )
                    if len(pending_comparisons) >= _FLUSH_EVERY:
                        _flush_results(db_path, pending_comparisons, pending_matches)
                    bar.update(1)
    finally:
        # flush what was consumed, including on quit or error
        _flush_results(db_path, pending_comparisons, pending_matches)
    stop_event.set()  # signal quit thread to exit if matching finished normally
    if stopped_early:
        typer.echo("\nStopped early. Progress has been saved — resume with `matcha match`.")
    else:
        typer.echo(f"\nDone. {matches_found} match(es) found from {len(pairs_to_run)} comparison(s).")
        if matches_found:
            typer.echo("Run `matcha move` to organise matched files into duplicates/.")