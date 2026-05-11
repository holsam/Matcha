import os, typer

from .db import get_connection

def load_all_videos(db_path: str) -> list[dict]:
    """Return all video records as dicts with id, path, moved_to."""
    conn = get_connection(db_path)
    rows = conn.execute("SELECT id, path, moved_to FROM videos").fetchall()
    return [{"id": row["id"], "path": row["path"], "moved_to": row["moved_to"]} for row in rows]


def delete_video_record(db_path: str, video_id: int):
    """Remove a video and all associated data from the DB."""
    conn = get_connection(db_path)
    with conn:
        conn.execute("DELETE FROM frame_hashes WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM audio_fingerprints WHERE video_id = ?", (video_id,))
        conn.execute(
            "DELETE FROM matches WHERE video_a_id = ? OR video_b_id = ?",
            (video_id, video_id),
        )
        conn.execute(
            "DELETE FROM comparisons WHERE video_a_id = ? OR video_b_id = ?",
            (video_id, video_id),
        )
        conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))


def run_sync(directory: str, dry_run: bool = False):
    """Main entry point for the sync subcommand."""
    directory = os.path.abspath(directory)
    db_path = os.path.join(directory, ".matcha", "index.db")

    if not os.path.exists(db_path):
        typer.echo("No index found. Run `matcha index` first.")
        raise SystemExit(1)

    videos = load_all_videos(db_path)
    if not videos:
        typer.echo("Index is empty. Nothing to sync.")
        return

    missing = []
    for video in videos:
        on_disk = (
            os.path.exists(video["path"])
            or (video["moved_to"] and os.path.exists(video["moved_to"]))
        )
        if not on_disk:
            missing.append(video)

    if not missing:
        typer.echo(f"All {len(videos)} indexed file(s) accounted for. Nothing to remove.")
        return

    if dry_run:
        typer.echo(f"[dry-run] {len(missing)} file(s) would be removed from the index:")
        for v in missing:
            typer.echo(f"  {v['path']}")
        return

    for video in missing:
        delete_video_record(db_path, video["id"])

    typer.echo(
        f"Removed {len(missing)} missing file(s) from the index "
        f"({len(videos) - len(missing)} remaining)."
    )