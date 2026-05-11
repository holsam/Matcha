import os, shutil, typer

from .db import get_connection

def get_group_records(db_path: str, group_dir: str) -> list[dict]:
    """
    Return all video records whose moved_to path is inside group_dir.
    Each record is a dict with keys: id, path, moved_to.
    """
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT id, path, moved_to FROM videos WHERE moved_to LIKE ?",
        (group_dir.rstrip("/") + "/%",),
    ).fetchall()
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


def clear_moved_to(db_path: str, video_id: int):
    """Clear moved_to after a file has been returned to its original location."""
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            "UPDATE videos SET moved_to = NULL WHERE id = ?",
            (video_id,),
        )


def reset_match_moved_flag(db_path: str, video_id: int):
    """Reset moved=0 for matches involving this video so it can be re-moved."""
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            "UPDATE matches SET moved = 0 WHERE video_a_id = ? OR video_b_id = ?",
            (video_id, video_id),
        )

def process_group_dir(db_path: str, group_dir: str) -> tuple[int, int]:
    """
    Inspect one duplicates/N/ subdirectory.

    Returns (deleted_count, returned_count).
    """
    records = get_group_records(db_path, group_dir)
    if not records:
        return 0, 0

    present = [r for r in records if r["moved_to"] and os.path.exists(r["moved_to"])]
    deleted = [r for r in records if not r["moved_to"] or not os.path.exists(r["moved_to"])]

    # Nothing deleted — skip
    if not deleted:
        return 0, 0

    deleted_count = len(deleted)
    returned_count = 0

    # Remove DB entries for deleted files
    for record in deleted:
        delete_video_record(db_path, record["id"])

    # If exactly one file survives, return it to its original location
    if len(present) == 1:
        survivor = present[0]
        src = survivor["moved_to"]
        dst = survivor["path"]

        dst_dir = os.path.dirname(dst)
        os.makedirs(dst_dir, exist_ok=True)

        # Avoid overwriting if something already exists at the original path
        if os.path.exists(dst):
            name, ext = os.path.splitext(os.path.basename(dst))
            dst = os.path.join(dst_dir, f"{name}_returned{ext}")

        shutil.move(src, dst)
        clear_moved_to(db_path, survivor["id"])
        reset_match_moved_flag(db_path, survivor["id"])
        returned_count = 1

        # Clean up the now-empty group directory if nothing else remains
        try:
            if not os.listdir(group_dir):
                os.rmdir(group_dir)
        except OSError:
            pass

    return deleted_count, returned_count


def run_cleanup(directory: str):
    """Main entry point for the cleanup subcommand."""
    directory = os.path.abspath(directory)
    db_path = os.path.join(directory, ".matcha", "index.db")

    if not os.path.exists(db_path):
        typer.echo("No index found. Run `matcha index` first.")
        raise SystemExit(1)

    duplicates_dir = os.path.join(directory, "duplicates")
    if not os.path.isdir(duplicates_dir):
        typer.echo("No duplicates/ directory found. Nothing to clean up.")
        return

    group_dirs = sorted(
        os.path.join(duplicates_dir, d)
        for d in os.listdir(duplicates_dir)
        if d.isdigit() and os.path.isdir(os.path.join(duplicates_dir, d))
    )

    if not group_dirs:
        typer.echo("No group subdirectories found. Nothing to clean up.")
        return

    total_deleted = 0
    total_returned = 0
    groups_processed = 0

    for group_dir in group_dirs:
        deleted, returned = process_group_dir(db_path, group_dir)
        if deleted > 0:
            groups_processed += 1
            total_deleted += deleted
            total_returned += returned
            label = os.path.basename(group_dir)
            if returned:
                typer.echo(
                    f"  {label}/  {deleted} removed from index, "
                    f"1 file returned to original location"
                )
            else:
                typer.echo(f"  {label}/  {deleted} removed from index")

    if groups_processed == 0:
        typer.echo("No changes detected. All groups are intact.")
    else:
        typer.echo(
            f"\nDone. {total_deleted} file(s) removed from index across "
            f"{groups_processed} group(s); {total_returned} returned to original location."
        )