"""
matcha/indexer.py — Phase 1: walk, fingerprint, store.

Uses ProcessPoolExecutor so that multiple ffmpeg subprocesses run in parallel
without GIL contention. Each worker opens its own SQLite connection — connections
are not safe to share across processes.
"""

import os, time, typer
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

from .db import get_connection, init_schema
from .fingerprint import (
    VIDEO_EXTENSIONS,
    extract_frame_hashes,
    get_audio_fingerprint,
    get_video_duration,
)


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
            [(p,) for p in paths],
        )


def get_unprocessed(db_path: str) -> list[tuple[int, str]]:
    """Return (id, path) for all videos not yet fingerprinted."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT id, path FROM videos WHERE fingerprinted_at IS NULL"
    ).fetchall()
    return [(row["id"], row["path"]) for row in rows]


def process_video(args: tuple) -> tuple[str, str | None]:
    """
    Worker function — runs in a subprocess.

    Opens its own DB connection (connections cannot cross process boundaries).
    Returns (video_path, error_message) where error_message is None on success.
    """
    video_id, video_path, db_path, fps = args
    conn = get_connection(db_path)

    try:
        duration = get_video_duration(video_path)
        frame_hashes = extract_frame_hashes(video_path, fps=fps)
        audio = get_audio_fingerprint(video_path)

        with conn:
            conn.execute(
                "UPDATE videos SET duration = ? WHERE id = ?",
                (duration, video_id),
            )
            conn.executemany(
                "INSERT INTO frame_hashes (video_id, timestamp, phash) VALUES (?, ?, ?)",
                [(video_id, ts, ph) for ts, ph in frame_hashes],
            )
            if audio is not None:
                audio_duration, fingerprint = audio
                conn.execute(
                    """
                    INSERT OR REPLACE INTO audio_fingerprints
                        (video_id, duration, fingerprint)
                    VALUES (?, ?, ?)
                    """,
                    (video_id, audio_duration, fingerprint),
                )
            conn.execute(
                "UPDATE videos SET fingerprinted_at = ? WHERE id = ?",
                (time.time(), video_id),
            )
        return video_path, None

    except Exception as e:
        return video_path, str(e)


def run_index(directory: str, fps: float = 1.0, workers: int = 4):
    """Main entry point for the index subcommand."""
    directory = os.path.abspath(directory)
    db_dir = os.path.join(directory, ".matcha")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "index.db")

    init_schema(db_path)

    typer.echo(f"Scanning {directory} for videos...")
    all_videos = find_videos(directory)
    typer.echo(f"Found {len(all_videos)} video(s).")

    register_videos(db_path, all_videos)

    to_process = get_unprocessed(db_path)
    if not to_process:
        typer.echo("All videos already indexed. Nothing to do.")
        return

    typer.echo(f"Indexing {len(to_process)} video(s) with {workers} worker(s) at {fps}fps...")

    args = [(vid_id, path, db_path, fps) for vid_id, path in to_process]
    errors: list[str] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_video, arg): arg for arg in args}
        with tqdm(total=len(futures), unit="video", dynamic_ncols=True) as bar:
            for future in as_completed(futures):
                path, error = future.result()
                bar.update(1)
                bar.set_postfix_str(os.path.basename(path), refresh=False)
                if error:
                    errors.append(f"{path}: {error}")

    if errors:
        typer.echo(f"\n{len(errors)} video(s) failed:", err=True)
        for msg in errors:
            typer.echo(f"  [SKIP] {msg}", err=True)

    typer.echo("Indexing complete.")