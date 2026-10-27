'''
MATCHA: apply review decisions (delete, rename, keep one, dismiss) to disk and the index together.
'''

# Import external dependencies
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from send2trash import send2trash

# Import internal matcha objects
from matcha.db import delete_video_record, dismiss_matches, get_connection, init_schema


# DeleteResult: which ids were removed and why any were not
@dataclass(frozen=True)
class DeleteResult:
    deleted: list[int] = field(default_factory=list)
    failed: dict[int, str] = field(default_factory=dict)


# delete_videos: trash the files, drop their records, then drop their vectors from FAISS in one pass
def delete_videos(directory: str, video_ids: list[int]) -> DeleteResult:
    matcha_dir, db_path = _paths(directory)
    conn = get_connection(db_path)
    deleted: list[int] = []
    failed: dict[int, str] = {}
    for vid in video_ids:
        row = conn.execute('SELECT path FROM videos WHERE id = ?', (vid,)).fetchone()
        if row is None:
            failed[vid] = 'not in index'
            continue
        try:
            # a file already gone is still purged, so the index heals itself
            if os.path.lexists(row['path']):
                send2trash(row['path'])
        except Exception as exc:
            failed[vid] = f'could not trash {row["path"]!r}: {exc}'
            continue
        # filesystem first, database second: a stale row is recoverable, a stale file is not
        delete_video_record(db_path, vid)
        deleted.append(vid)
    _remove_vectors(matcha_dir, deleted)
    return DeleteResult(deleted, failed)


# keep_one: keep a single video of a group and delete the rest
def keep_one(directory: str, keep_id: int, group_ids: list[int]) -> DeleteResult:
    if keep_id not in group_ids:
        raise ValueError(f'video {keep_id!r} is not in the group')
    return delete_videos(directory, [v for v in group_ids if v != keep_id])


# rename_video: rename the file in place and update its path, returning the new path
def rename_video(directory: str, video_id: int, new_name: str) -> str:
    _, db_path = _paths(directory)
    conn = get_connection(db_path)
    row = conn.execute('SELECT path FROM videos WHERE id = ?', (video_id,)).fetchone()
    if row is None:
        raise ValueError(f'video {video_id!r} is not in the index')
    src = row['path']
    if not os.path.exists(src):
        raise ValueError(f'file is missing: {src!r}')

    new_name = new_name.strip()
    if not new_name or new_name in ('.', '..') or os.path.basename(new_name) != new_name:
        raise ValueError(f'invalid file name {new_name!r}')
    if not os.path.splitext(new_name)[1]:
        new_name += os.path.splitext(src)[1]  # keep the original extension by default

    dst = os.path.join(os.path.dirname(src), new_name)
    if dst == src:
        return src
    taken = conn.execute('SELECT 1 FROM videos WHERE path = ?', (dst,)).fetchone()
    if taken or os.path.lexists(dst):
        raise ValueError(f'{dst!r} already exists')

    os.rename(src, dst)
    try:
        with conn:
            conn.execute('UPDATE videos SET path = ? WHERE id = ?', (dst, video_id))
    except Exception:
        os.rename(dst, src)  # keep disk and index in step
        raise
    return dst


# dismiss_group: mark every match inside the group as not-duplicates so it stops showing
def dismiss_group(directory: str, video_ids: list[int]) -> int:
    _, db_path = _paths(directory)
    return dismiss_matches(get_connection(db_path), video_ids)


# MissingFile: an indexed video whose file is no longer on disk
@dataclass(frozen=True)
class MissingFile:
    id: int
    path: str


# ScanResult: how many indexed videos were checked and which of them are gone
@dataclass(frozen=True)
class ScanResult:
    total: int
    missing: list[MissingFile]


# find_missing: report indexed videos gone from disk, listing each folder once rather than stat-ing every file, calling progress(done, total) as it goes
def find_missing(directory: str, *, progress: Callable[[int, int], None] | None = None) -> ScanResult:
    directory = os.path.abspath(directory)
    # an unmounted drive would make every file look missing, so refuse rather than report it
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'library directory {directory!r} is not available')
    _, db_path = _paths(directory)
    rows = get_connection(db_path).execute('SELECT id, path FROM videos').fetchall()
    by_folder: dict[str, list[tuple[int, str]]] = {}
    for row in rows:
        folder, name = os.path.split(row['path'])
        by_folder.setdefault(folder, []).append((row['id'], name))

    missing: list[MissingFile] = []
    done = reported = 0
    for folder, files in by_folder.items():
        try:
            with os.scandir(folder) as entries:
                present = {entry.name for entry in entries}
        except OSError:
            present = None  # unreadable folder, so check its files one by one instead of trusting a listing
        for vid, name in files:
            path = os.path.join(folder, name)
            # a listing can miss names a stat finds (case or unicode form), purge_missing re-checks before removing
            if (name not in present) if present is not None else not os.path.lexists(path):
                missing.append(MissingFile(vid, path))
        done += len(files)
        if progress and (done - reported >= 100 or done == len(rows)):
            progress(done, len(rows))
            reported = done
    missing.sort(key=lambda m: m.id)
    return ScanResult(len(rows), missing)


# purge_missing: drop index records for videos still missing, returning the ids purged (a file that reappeared is kept)
def purge_missing(directory: str, video_ids: list[int]) -> list[int]:
    matcha_dir, db_path = _paths(directory)
    conn = get_connection(db_path)
    purged: list[int] = []
    for vid in video_ids:
        row = conn.execute('SELECT path FROM videos WHERE id = ?', (vid,)).fetchone()
        if row is None or os.path.lexists(row['path']):
            continue  # already gone from the index, or back on disk since the scan
        delete_video_record(db_path, vid)
        purged.append(vid)
    _remove_vectors(matcha_dir, purged)
    return purged


# restore_dismissed: bring every dismissed group back into the review list, returning the matches restored
def restore_dismissed(directory: str) -> int:
    _, db_path = _paths(directory)
    conn = get_connection(db_path)
    with conn:
        cur = conn.execute('UPDATE matches SET dismissed = 0 WHERE dismissed = 1')
    return cur.rowcount


# _remove_vectors: drop the videos' FAISS vectors in one pass
def _remove_vectors(matcha_dir: str, video_ids: list[int]) -> None:
    if video_ids:
        # lazy import, faiss is slow to load and only needed here
        from matcha.faiss_index import remove_videos_from_index
        remove_videos_from_index(matcha_dir, video_ids)


# _paths: (.matcha dir, index.db path) for a library, with the schema brought up to date
def _paths(directory: str) -> tuple[str, str]:
    matcha_dir = os.path.join(os.path.abspath(directory), '.matcha')
    db_path = os.path.join(matcha_dir, 'index.db')
    init_schema(db_path)
    return matcha_dir, db_path
