import click, os, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .db import get_connection, init_schema
from .fingerprint import (
    VIDEO_EXTENSIONS,
    extract_frame_hashes,
    get_audio_fingerprint,
    get_video_duration,
)

# Thread-local DB connections — sqlite3 connections must not be shared across threads
_local = threading.local()


def _get_local_conn(db_path: str):
    if not hasattr(_local, "conn"):
        _local.conn = get_connection(db_path)
    return _local.conn


def find_videos(root: str) -> list[str]:
    """Recursively find all video files under root, excluding .matcha/."""
    videos = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".matcha"]
        for fname in filenames:
            if Path(fname).suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(os.path.join(dirpath, fname))
    return sorted(videos)


def register_videos(db_path: str, paths: list[str]):
    """Insert any new video paths into the videos table. Existing rows are ignored."""
    conn = get_connection(db_path)
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO videos (path) VALUES (?)",
            [(p,) for p in paths]
        )


def get_unprocessed(db_path: str) -> list[tuple[int, str]]:
    """Return (id, path) for all videos not yet fingerprinted."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT id, path FROM videos WHERE fingerprinted_at IS NULL"
    ).fetchall()
    return [(row["id"], row["path"]) for row in rows]


def process_video(video_id: int, video_path: str, db_path: str, fps: float) -> str:
    """
    Fingerprint a single video and write results to the DB.
    Returns the video path on completion (used for progress reporting).
    Errors are caught and logged per-file so one bad file does not halt the run.
    """
    conn = _get_local_conn(db_path)

    try:
        duration = get_video_duration(video_path)
        frame_hashes = extract_frame_hashes(video_path, fps=fps)
        audio = get_audio_fingerprint(video_path)

        with conn:
            conn.execute(
                "UPDATE videos SET duration = ? WHERE id = ?",
                (duration, video_id)
            )
            conn.executemany(
                "INSERT INTO frame_hashes (video_id, timestamp, phash) VALUES (?, ?, ?)",
                [(video_id, ts, ph) for ts, ph in frame_hashes]
            )
            if audio is not None:
                audio_duration, fingerprint = audio
                conn.execute(
                    """
                    INSERT OR REPLACE INTO audio_fingerprints
                        (video_id, duration, fingerprint)
                    VALUES (?, ?, ?)
                    """,
                    (video_id, audio_duration, fingerprint)
                )
            conn.execute(
                "UPDATE videos SET fingerprinted_at = ? WHERE id = ?",
                (time.time(), video_id)
            )

    except Exception as e:
        click.echo(f"\n  [SKIP] {video_path}: {e}", err=True)

    return video_path


def run_index(directory: str, fps: float = 1.0, workers: int = 4):
    """Main entry point for the index subcommand."""
    directory = os.path.abspath(directory)
    db_dir = os.path.join(directory, ".matcha")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "index.db")

    init_schema(db_path)

    click.echo(f"Scanning {directory} for videos...")
    all_videos = find_videos(directory)
    click.echo(f"Found {len(all_videos)} video(s).")

    register_videos(db_path, all_videos)

    to_process = get_unprocessed(db_path)
    if not to_process:
        click.echo("All videos already indexed. Nothing to do.")
        return

    click.echo(
        f"Indexing {len(to_process)} unprocessed video(s) "
        f"with {workers} worker(s) at {fps}fps..."
    )

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_video, vid_id, path, db_path, fps): path
            for vid_id, path in to_process
        }
        for future in as_completed(futures):
            future.result()
            completed += 1
            click.echo(f"  [{completed}/{len(to_process)}] done")

    click.echo("Indexing complete.")