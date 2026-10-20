'''
MATCHA: apply review decisions (delete, rename, keep one, dismiss) to disk and the index together.
'''

# Import external dependencies
import os
from dataclasses import dataclass, field
from send2trash import send2trash

# Import internal matcha objects
from matcha.db import delete_video_record, get_connection, init_schema


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
    if deleted:
        # lazy import, faiss is slow to load and only needed here
        from matcha.faiss_index import remove_videos_from_index
        remove_videos_from_index(matcha_dir, deleted)
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
    conn = get_connection(db_path)
    marks = ','.join('?' * len(video_ids))
    with conn:
        cur = conn.execute(
            f'UPDATE matches SET dismissed = 1 WHERE video_a_id IN ({marks}) AND video_b_id IN ({marks})',
            [*video_ids, *video_ids],
        )
    return cur.rowcount


# _paths: (.matcha dir, index.db path) for a library, with the schema brought up to date
def _paths(directory: str) -> tuple[str, str]:
    matcha_dir = os.path.join(os.path.abspath(directory), '.matcha')
    db_path = os.path.join(matcha_dir, 'index.db')
    init_schema(db_path)
    return matcha_dir, db_path
