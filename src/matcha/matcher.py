import imagehash, io, itertools, os, sys, termios, threading, time, tty, typer
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from rich import print
from tqdm import tqdm

from .db import get_connection
from .faiss_index import build_index, find_candidate_pairs

@dataclass
class VideoRecord:
    id: int
    path: str
    duration: float
    frame_hashes: list[str]
    has_audio: bool

def _print_message(stage: str, msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    tab = stage.count('.') + 1
    print_msg = f'({ts})'+'\t'*tab+f'{msg}'
    print(f'[dim]{print_msg}[/dim]')


def _hex_to_uint64(hex_str: str) -> int:
    return int(hex_str, 16)

def _hamming_uint64(a: int, b: int) -> int:
    return bin(a ^ b).count("1")

def load_videos(db_path: str) -> list[VideoRecord]:
    conn = get_connection(db_path)
    audio_ids = {
        row["video_id"]
        for row in conn.execute("SELECT video_id FROM audio_fingerprints").fetchall()
    }
    rows = conn.execute(
        "SELECT id, path, duration FROM videos WHERE fingerprinted_at IS NOT NULL"
    ).fetchall()
    videos = []
    for row in rows:
        hash_rows = conn.execute(
            "SELECT phash FROM frame_hashes WHERE video_id = ? ORDER BY timestamp",
            (row["id"],),
        ).fetchall()
        videos.append(VideoRecord(
            id=row["id"],
            path=row["path"],
            duration=row["duration"] or 0.0,
            frame_hashes=[r["phash"] for r in hash_rows],
            has_audio=row["id"] in audio_ids,
        ))
    return videos


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


def record_comparison(db_path: str, id_a: int, id_b: int):
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO comparisons (video_a_id, video_b_id) VALUES (?, ?)",
            (min(id_a, id_b), max(id_a, id_b)),
        )

def record_match(
    db_path: str,
    short: VideoRecord,
    long: VideoRecord,
    match_type: str,
    confidence: float,
):
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO matches (video_a_id, video_b_id, match_type, confidence, found_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (short.id, long.id, match_type, confidence, time.time()),
        )

def hamming_distance(hash_a: str, hash_b: str) -> int:
    return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)

def sliding_window_match(
    short_hashes: list[str],
    long_hashes: list[str],
    frame_step: int,
    threshold: int,
) -> float:
    n = len(short_hashes)
    m = len(long_hashes)
    if n == 0 or m < n:
        return 0.0
    best_ratio = 0.0
    for start in range(0, m - n + 1, frame_step):
        matches = sum(
            1
            for j in range(n)
            if hamming_distance(short_hashes[j], long_hashes[start + j]) <= threshold
        )
        ratio = matches / n
        if ratio > best_ratio:
            best_ratio = ratio
    return best_ratio

def sliding_window_match_numpy(
    short_hashes: list[str],
    long_hashes: list[str],
    frame_step: int,
    threshold: int,
) -> float:
    """
    Vectorised sliding window using NumPy, converts hex pHash strings to uint64 integers once, then computes Hamming distances across all window positions in batch using XOR + popcount.
    """
    n = len(short_hashes)
    m = len(long_hashes)
    if n == 0 or m < n:
        return 0.0
    # Convert hex strings → uint64 arrays once
    short_ints = np.array([_hex_to_uint64(h) for h in short_hashes], dtype=np.uint64)
    long_ints  = np.array([_hex_to_uint64(h) for h in long_hashes],  dtype=np.uint64)
    # Precompute popcount lookup table for uint8 values (0–255)
    popcount_table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        popcount_table[i] = bin(i).count("1")
    def popcount_array(arr: np.ndarray) -> np.ndarray:
        """Popcount each uint64 element via byte decomposition."""
        # View as uint8 → 8 bytes per element → sum popcount per group of 8
        as_bytes = arr.view(np.uint8).reshape(-1, 8)
        return popcount_table[as_bytes].sum(axis=1).astype(np.int32)
    best_ratio = 0.0
    positions = range(0, m - n + 1, frame_step)
    for start in positions:
        window = long_ints[start : start + n]
        xor = np.bitwise_xor(short_ints, window)
        distances = popcount_array(xor)
        match_count = int(np.sum(distances <= threshold))
        ratio = match_count / n
        if ratio > best_ratio:
            best_ratio = ratio
            if best_ratio == 1.0:
                break  # can't improve further
    return best_ratio

def determine_match_type(short: VideoRecord, long: VideoRecord) -> str:
    if long.duration == 0:
        return "duplicate"
    return "duplicate" if (short.duration / long.duration) >= 0.95 else "subclip"


def _compare_pair(args: tuple) -> tuple[int, int, float]:
    """
    Worker — fetches frame hashes from the DB and runs the sliding window.
    Hash lists are not passed in; they are loaded here and discarded after,
    keeping peak memory to one pair at a time per thread.
    """
    short_id, long_id, short_hashes, long_hashes, frame_step, threshold = args
    confidence = sliding_window_match_numpy(short_hashes, long_hashes, frame_step, threshold)
    return short_id, long_id, confidence


def _watch_for_quit(stop_event: threading.Event):
    """
    Background thread that sets stop_event when 'q' is pressed.

    Puts stdin into raw (unbuffered, no-echo) mode so keypresses are
    received immediately without the user pressing Enter. Restores the
    original terminal settings on exit regardless of how it ends.
    """
    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
    except (termios.error, io.UnsupportedOperation): 
        # stdin is not a tty (e.g. in tests or piped input) — skip listener
        return
    try:
        tty.setraw(fd)
        while not stop_event.is_set():
            # os.read is non-blocking after setraw; use select to avoid busy-wait
            import select
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if readable:
                ch = os.read(fd, 1)
                if ch in (b"q", b"Q"):
                    stop_event.set()
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

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
    rebuilt = build_index(db_path, index_dir, nprobe)
    if not rebuilt:
        _print_message('2.1', 'FAISS index up to date.')
    conn = get_connection(db_path)
    existing_candidates: set[tuple[int, int]] = {
        (row['video_a_id'], row['video_b_id']) for row in conn.execute('SELECT video_a_id, video_b_id FROM candidate_pairs').fetchall()
    }
    _print_message('2.3', 'Querying FAISS index for candidate pairs...')
    new_candidates = find_candidate_pairs(db_path, index_dir, threshold, nprobe, workers)
    all_candidates = existing_candidates | new_candidates
    new_to_write = new_candidates - existing_candidates
    if new_to_write:
        _print_message('2.4', 'Writing new candidates to index...')
        with conn:
            conn.executemany(
                'INSERT OR IGNORE INTO candidate_pairs (video_a_id, video_b_id) VALUES (?, ?)',
                list(new_to_write),
            )
        _print_message('2', f'{len(all_candidates):,} candidate pairs identified.')
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
    for short, long in too_short:
        record_comparison(db_path, short.id, long.id)
    if not pairs_to_run:
        _print_message('3.3', 'No eligible pairs to verify')
        return
    worker_args = [
        (s.id, l.id, s.frame_hashes, l.frame_hashes, frame_step, threshold)
        for s, l in pairs_to_run
    ]
    if filter_length:
        typer.echo("Length filter: on")
    typer.echo("Press 'q' to stop matching early.\n")
    matches_found = 0
    stopped_early = False
    stop_event = threading.Event()
    quit_thread = threading.Thread(target=_watch_for_quit, args=(stop_event,), daemon=True)
    quit_thread.start()
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
                record_comparison(db_path, short_id, long_id)
                if confidence >= min_confidence:
                    match_type = determine_match_type(short, long)
                    record_match(db_path, short, long, match_type, confidence)
                    matches_found += 1
                    bar.write(
                        f"  MATCH  {match_type:<10}  {confidence:.0%}  "
                        f"{os.path.basename(short.path)}  ←  {os.path.basename(long.path)}"
                    )
                bar.update(1)
    stop_event.set()  # signal quit thread to exit if matching finished normally
    if stopped_early:
        typer.echo("\nStopped early. Progress has been saved — resume with `matcha match`.")
    else:
        typer.echo(f"\nDone. {matches_found} match(es) found from {len(pairs_to_run)} comparison(s).")
        if matches_found:
            typer.echo("Run `matcha move` to organise matched files into duplicates/.")