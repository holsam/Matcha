import os, typer

from .db import delete_video_record, get_connection, load_all_videos


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