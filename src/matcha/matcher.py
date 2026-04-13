import imagehash, itertools, os, time
from dataclasses import dataclass

from .db import get_connection

@dataclass
class VideoRecord:
    id: int
    path: str
    duration: float
    frame_hashes: list[str]   # ordered by timestamp
    has_audio: bool


def load_videos(db_path: str) -> list[VideoRecord]:
    """Load all fully indexed videos and their frame hashes from the DB."""
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


def generate_pairs(
    videos: list[VideoRecord],
    filter_length: bool,
) -> list[tuple[VideoRecord, VideoRecord]]:
    """
    Generate (shorter, longer) pairs to compare.
    If filter_length is True, skip pairs where both videos have identical durations.
    """
    pairs = []
    for a, b in itertools.combinations(videos, 2):
        if filter_length and a.duration == b.duration:
            continue
        short, long = (a, b) if a.duration <= b.duration else (b, a)
        pairs.append((short, long))
    return pairs


def get_compared_pairs(db_path: str) -> set[tuple[int, int]]:
    """Return the set of (lower_id, higher_id) pairs already compared."""
    conn = get_connection(db_path)
    rows = conn.execute("SELECT video_a_id, video_b_id FROM comparisons").fetchall()
    return {(row["video_a_id"], row["video_b_id"]) for row in rows}


def record_comparison(db_path: str, id_a: int, id_b: int):
    """Mark a pair as compared, keyed as (lower_id, higher_id)."""
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
    """Compute the Hamming distance between two pHash hex strings."""
    return imagehash.hex_to_hash(hash_a) - imagehash.hex_to_hash(hash_b)


def sliding_window_match(
    short_hashes: list[str],
    long_hashes: list[str],
    frame_step: int,
    threshold: int,
) -> float:
    """
    Slide a window the size of short_hashes across long_hashes.
    Returns the best match ratio found (0.0–1.0).
    """
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
    """
    Classify a match as 'duplicate' or 'subclip'.
    If the shorter video is ≥95% of the longer one's duration, treat as duplicate.
    """
    if long.duration == 0:
        return "duplicate"
    return "duplicate" if (short.duration / long.duration) >= 0.95 else "subclip"


def run_match(
    directory: str,
    filter_length: bool = False,
    window: float = 10.0,
    frame_step: int = 3,
    threshold: int = 10,
    min_confidence: float = 0.8,
):
    """Main entry point for the match subcommand."""
    directory = os.path.abspath(directory)
    db_path = os.path.join(directory, ".matcha", "index.db")

    if not os.path.exists(db_path):
        print("No index found. Run `matcha index` first.")
        raise SystemExit(1)

    videos = load_videos(db_path)
    if not videos:
        print("No indexed videos found. Run `matcha index` first.")
        return

    all_pairs = generate_pairs(videos, filter_length)
    already_compared = get_compared_pairs(db_path)

    pairs_to_run = [
        (s, l) for s, l in all_pairs
        if (min(s.id, l.id), max(s.id, l.id)) not in already_compared
    ]

    total = len(pairs_to_run)
    skipped = len(all_pairs) - total
    matches_found = 0

    print(f"Videos indexed: {len(videos)}")
    print(f"Pairs to compare: {total}  (skipped {skipped} already compared)")
    if filter_length:
        print("Length filter: on — skipping identical-duration pairs")
    print()

    for i, (short, long) in enumerate(pairs_to_run, 1):
        label = f"[{i}/{total}]"

        # Skip pairs where the shorter video is below the minimum window duration
        if short.duration < window:
            record_comparison(db_path, short.id, long.id)
            print(f"  {label} SKIP (too short)  {os.path.basename(short.path)}")
            continue

        confidence = sliding_window_match(
            short.frame_hashes,
            long.frame_hashes,
            frame_step=frame_step,
            threshold=threshold,
        )

        record_comparison(db_path, short.id, long.id)

        if confidence >= min_confidence:
            match_type = determine_match_type(short, long)
            record_match(db_path, short, long, match_type, confidence)
            matches_found += 1
            print(
                f"  {label} MATCH  {match_type:<10}  {confidence:.0%}  "
                f"{os.path.basename(short.path)}  ←  {os.path.basename(long.path)}"
            )
        else:
            print(
                f"  {label} no match  ({confidence:.0%})  "
                f"{os.path.basename(short.path)}  vs  {os.path.basename(long.path)}"
            )

    print()
    print(f"Done. {matches_found} match(es) found from {total} comparison(s).")
    if matches_found:
        print("Run `matcha move` to organise matched files into duplicates/.")