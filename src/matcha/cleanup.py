import os, shutil, typer

from .db import get_connection

def get_group_records(db_path: str, group_dir: str) -> list[dict]:
    """All video records whose moved_to path is inside group_dir."""
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT id, path, moved_to FROM videos WHERE moved_to LIKE ?",
        (group_dir.rstrip("/") + "/%",),
    ).fetchall()
    return [{"id": r["id"], "path": r["path"], "moved_to": r["moved_to"]} for r in rows]


def load_all_videos(db_path: str) -> list[dict]:
    """Every video record as a dict with id, path, moved_to."""
    conn = get_connection(db_path)
    rows = conn.execute("SELECT id, path, moved_to FROM videos").fetchall()
    return [{"id": r["id"], "path": r["path"], "moved_to": r["moved_to"]} for r in rows]


def delete_video_record(db_path: str, video_id: int):
    """Remove a video and all associated data from the DB."""
    conn = get_connection(db_path)
    with conn:
        conn.execute("DELETE FROM frame_hashes WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM audio_fingerprints WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM matches WHERE video_a_id = ? OR video_b_id = ?", (video_id, video_id))
        conn.execute("DELETE FROM comparisons WHERE video_a_id = ? OR video_b_id = ?", (video_id, video_id))
        conn.execute("DELETE FROM videos WHERE id = ?", (video_id,))


def clear_moved_to(db_path: str, video_id: int):
    conn = get_connection(db_path)
    with conn:
        conn.execute("UPDATE videos SET moved_to = NULL WHERE id = ?", (video_id,))


def reset_match_moved_flag(db_path: str, video_id: int):
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            "UPDATE matches SET moved = 0 WHERE video_a_id = ? OR video_b_id = ?",
            (video_id, video_id),
        )


def _plan_group(group_dir: str, records: list[dict]) -> tuple[list[int], dict | None]:
    """
    Decide what to do with one duplicates/N/ group without changing anything.
    Returns (deleted_ids, survivor) where survivor is the lone surviving record
    to return to its original location, or None.
    """
    present = [r for r in records if r["moved_to"] and os.path.exists(r["moved_to"])]
    deleted = [r for r in records if not r["moved_to"] or not os.path.exists(r["moved_to"])]
    if not deleted:
        return [], None
    survivor = present[0] if len(present) == 1 else None
    return [r["id"] for r in deleted], survivor


def _return_survivor(db_path: str, survivor: dict, group_dir: str):
    """Move a lone survivor back to its original path and tidy the group dir."""
    src = survivor["moved_to"]
    dst = survivor["path"]
    dst_dir = os.path.dirname(dst)
    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(dst):
        name, ext = os.path.splitext(os.path.basename(dst))
        dst = os.path.join(dst_dir, f"{name}_returned{ext}")
    shutil.move(src, dst)
    clear_moved_to(db_path, survivor["id"])
    reset_match_moved_flag(db_path, survivor["id"])
    try:
        if not os.listdir(group_dir):
            os.rmdir(group_dir)
    except OSError:
        pass


def run_cleanup(directory: str, dry_run: bool = False):
    """Reconcile the index with disk, and clean the FAISS index to match."""
    directory = os.path.abspath(directory)
    matcha_dir = os.path.join(directory, ".matcha")
    db_path = os.path.join(matcha_dir, "index.db")

    if not os.path.exists(db_path):
        typer.echo("No index found. Run `matcha index` first.")
        raise SystemExit(1)

    purge_ids: set[int] = set()
    survivors: list[tuple[dict, str]] = []
    reports: list[str] = []

    # Pass 1 — duplicates/ groups (the old `cleanup` behaviour)
    duplicates_dir = os.path.join(directory, "duplicates")
    if os.path.isdir(duplicates_dir):
        group_dirs = sorted(
            os.path.join(duplicates_dir, d)
            for d in os.listdir(duplicates_dir)
            if d.isdigit() and os.path.isdir(os.path.join(duplicates_dir, d))
        )
        for group_dir in group_dirs:
            records = get_group_records(db_path, group_dir)
            if not records:
                continue
            deleted_ids, survivor = _plan_group(group_dir, records)
            if not deleted_ids:
                continue
            purge_ids.update(deleted_ids)
            label = os.path.basename(group_dir)
            if survivor is not None:
                survivors.append((survivor, group_dir))
                reports.append(f"  {label}/  {len(deleted_ids)} from index, 1 returned to original location")
            else:
                reports.append(f"  {label}/  {len(deleted_ids)} from index")

    # Pass 2 — anything else missing from disk (the old `sync` behaviour)
    for video in load_all_videos(db_path):
        if video["id"] in purge_ids:
            continue
        on_disk = os.path.exists(video["path"]) or (
            video["moved_to"] and os.path.exists(video["moved_to"])
        )
        if not on_disk:
            purge_ids.add(video["id"])
            reports.append(f"  missing  {os.path.basename(video['path'])} from index")

    if not purge_ids:
        typer.echo("Nothing to clean up. Index matches disk.")
        return

    for line in reports:
        typer.echo(line)

    if dry_run:
        typer.echo(
            f"\n[dry-run] {len(purge_ids)} file(s) would be removed from the index; "
            f"{len(survivors)} would be returned to their original location. "
            f"No changes made."
        )
        return

    # Apply: database first, then survivors, then the FAISS index.
    for vid in purge_ids:
        delete_video_record(db_path, vid)
    for survivor, group_dir in survivors:
        _return_survivor(db_path, survivor, group_dir)

    from .faiss_index import remove_videos_from_index
    removed_vectors = remove_videos_from_index(matcha_dir, purge_ids)

    typer.echo(
        f"\nDone. {len(purge_ids)} file(s) removed from the index "
        f"({len(survivors)} returned to original location); "
        f"{removed_vectors:,} vector(s) removed from the FAISS index."
    )