import imagehash, io, itertools, os, sys, termios, threading, time, tty, typer
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from tqdm import tqdm

from .db import get_connection


@dataclass
class VideoRecord:
    id: int
    path: str
    duration: float
    has_audio: bool

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
    short_id, long_id, db_path, frame_step, threshold = args
    short_hashes = load_frame_hashes(db_path, short_id)
    long_hashes = load_frame_hashes(db_path, long_id)
    confidence = sliding_window_match(short_hashes, long_hashes, frame_step, threshold)
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
):
    """Main entry point for the match subcommand."""
    directory = os.path.abspath(directory)
    db_path = os.path.join(directory, ".matcha", "index.db")

    if not os.path.exists(db_path):
        typer.echo("No index found. Run `matcha index` first.")
        raise SystemExit(1)

    typer.echo("Loading index...")
    videos = load_videos(db_path)
    if not videos:
        typer.echo("No indexed videos found. Run `matcha index` first.")
        return

    video_map: dict[int, VideoRecord] = {v.id: v for v in videos}

    all_pairs = generate_pairs(videos, filter_length)
    already_compared = get_compared_pairs(db_path)

    pairs_to_run = [
        (s, l) for s, l in all_pairs
        if (min(s.id, l.id), max(s.id, l.id)) not in already_compared
    ]

    eligible, too_short = [], []
    for short, long in pairs_to_run:
        if short.duration < window:
            too_short.append((short, long))
        else:
            eligible.append((short, long))

    skipped_checkpoint = len(all_pairs) - len(pairs_to_run)

    typer.echo(f"Videos indexed:    {len(videos)}")
    typer.echo(f"Pairs to compare:  {len(eligible)}  "
               f"({skipped_checkpoint} already done, {len(too_short)} too short)")
    if filter_length:
        typer.echo("Length filter: on")
    typer.echo("Press 'q' to stop matching early.\n")

    for short, long in too_short:
        record_comparison(db_path, short.id, long.id)

    if not eligible:
        typer.echo("No eligible pairs to compare.")
        return

    worker_args = [
        (s.id, l.id, db_path, frame_step, threshold)
        for s, l in eligible
    ]

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
        typer.echo(f"\nDone. {matches_found} match(es) found from {len(eligible)} comparison(s).")
        if matches_found:
            typer.echo("Run `matcha move` to organise matched files into duplicates/.")