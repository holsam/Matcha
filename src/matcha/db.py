import sqlite3
import threading

# One connection per (db_path, thread) — avoids exhausting file descriptors by reusing connections rather than opening a new one on every call.
_local = threading.local()


def get_connection(db_path: str) -> sqlite3.Connection:
    """
    Return a cached SQLite connection for the current thread and db_path.

    Opening a new connection on every call leaks file descriptors — with
    hundreds of pairs being compared this quickly hits the OS limit. Caching
    per thread keeps the descriptor count bounded to (workers x db_count).
    """
    if not hasattr(_local, "conns"):
        _local.conns = {}

    if db_path not in _local.conns:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")  # safe with WAL, avoids an fsync per commit
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-65536")  # 64 MB page cache
        _local.conns[db_path] = conn

    return _local.conns[db_path]


def init_schema(db_path: str):
    conn = get_connection(db_path)
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS videos (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                path             TEXT UNIQUE NOT NULL,
                duration         REAL,
                fingerprinted_at REAL,
                moved_to         TEXT,
                content_key      TEXT                -- size + sampled hash, for reusing hashes of identical files
            );

            CREATE TABLE IF NOT EXISTS frame_hashes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id    INTEGER NOT NULL REFERENCES videos(id),
                timestamp   REAL NOT NULL,
                phash       TEXT NOT NULL
            );

            DROP INDEX IF EXISTS idx_frame_hashes_video_id;  -- prefix of the covering index below

            CREATE INDEX IF NOT EXISTS idx_frame_hashes_video_ts
                ON frame_hashes(video_id, timestamp, phash);

            CREATE TABLE IF NOT EXISTS audio_fingerprints (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id    INTEGER NOT NULL REFERENCES videos(id) UNIQUE,
                duration    REAL NOT NULL,
                fingerprint TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS comparisons (
                video_a_id  INTEGER NOT NULL REFERENCES videos(id),
                video_b_id  INTEGER NOT NULL REFERENCES videos(id),
                PRIMARY KEY (video_a_id, video_b_id)
            );

            CREATE TABLE IF NOT EXISTS matches (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                video_a_id  INTEGER NOT NULL REFERENCES videos(id),
                video_b_id  INTEGER NOT NULL REFERENCES videos(id),
                match_type  TEXT NOT NULL,
                confidence  REAL NOT NULL,
                found_at    REAL NOT NULL,
                moved       INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS faiss_index_meta (
                id              INTEGER PRIMARY KEY CHECK (id = 1),  -- enforce single row
                built_at        REAL,                                -- unix timestamp
                vector_count    INTEGER                              -- total hashes indexed
            );

            CREATE TABLE IF NOT EXISTS candidate_pairs (
                video_a_id  INTEGER NOT NULL REFERENCES videos(id),
                video_b_id  INTEGER NOT NULL REFERENCES videos(id),
                generated   INTEGER NOT NULL DEFAULT 0,   -- 1 once Pass 1 has emitted this pair
                PRIMARY KEY (video_a_id, video_b_id)
            );

            CREATE TABLE IF NOT EXISTS candidate_search_progress (
                video_id    INTEGER PRIMARY KEY REFERENCES videos(id),
                queried_at  REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS candidate_search_params (
                id          INTEGER PRIMARY KEY CHECK (id = 1),  -- enforce single row
                threshold   INTEGER NOT NULL,
                nprobe      INTEGER NOT NULL,
                min_votes   INTEGER NOT NULL DEFAULT 1
            );
        """)
        # databases created before these columns existed (moved_to: <1.5.3, content_key: <2.1)
        cols = {row['name'] for row in conn.execute("PRAGMA table_info(videos)")}
        for col in ('moved_to', 'content_key'):
            if col not in cols:
                conn.execute(f"ALTER TABLE videos ADD COLUMN {col} TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_videos_content_key ON videos(content_key)")
        # the composite primary keys only cover video_a_id, so without these every video delete (and its foreign key check) scans the whole table
        for table in ('comparisons', 'candidate_pairs', 'matches'):
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_b ON {table}(video_b_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_matches_a ON matches(video_a_id)")
        # databases created before the offset-vote filter lack min_votes (1 = no filtering, the old behaviour)
        param_cols = {row['name'] for row in conn.execute("PRAGMA table_info(candidate_search_params)")}
        if 'min_votes' not in param_cols:
            conn.execute("ALTER TABLE candidate_search_params ADD COLUMN min_votes INTEGER NOT NULL DEFAULT 1")
        # databases created before review-ui lack the dismissed flag on matches
        match_cols = {row['name'] for row in conn.execute("PRAGMA table_info(matches)")}
        if 'dismissed' not in match_cols:
            conn.execute("ALTER TABLE matches ADD COLUMN dismissed INTEGER NOT NULL DEFAULT 0")

def get_faiss_meta(db_path: str) -> dict | None:
    conn = get_connection(db_path)
    row = conn.execute("SELECT * FROM faiss_index_meta WHERE id = 1").fetchone()
    return dict(row) if row else None

def set_faiss_meta(db_path: str, vector_count: int):
    import time
    conn = get_connection(db_path)
    with conn:
        conn.execute("""
            INSERT INTO faiss_index_meta (id, built_at, vector_count)
            VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET built_at = excluded.built_at,
                                          vector_count = excluded.vector_count
        """, (time.time(), vector_count))

# load_all_videos: every video record as a dict with id, path, moved_to
def load_all_videos(db_path: str) -> list[dict]:
    conn = get_connection(db_path)
    rows = conn.execute("SELECT id, path, moved_to FROM videos").fetchall()
    return [{"id": r["id"], "path": r["path"], "moved_to": r["moved_to"]} for r in rows]

# purge_video: remove a video and every row that references it on an open connection, returning the rows deleted
def purge_video(conn: sqlite3.Connection, video_id: int) -> int:
    rows = 0
    with conn:
        # child rows first, foreign_keys is ON
        for table in ('frame_hashes', 'audio_fingerprints', 'candidate_search_progress'):
            rows += conn.execute(f"DELETE FROM {table} WHERE video_id = ?", (video_id,)).rowcount
        for table in ('matches', 'comparisons', 'candidate_pairs'):
            rows += conn.execute(f"DELETE FROM {table} WHERE video_a_id = ? OR video_b_id = ?", (video_id, video_id)).rowcount
        rows += conn.execute("DELETE FROM videos WHERE id = ?", (video_id,)).rowcount
    return rows

# delete_video_record: remove a video and every row that references it
def delete_video_record(db_path: str, video_id: int) -> None:
    purge_video(get_connection(db_path), video_id)

# repoint_video: point a video's row at a new path and clear content_key, which hashes the old file's size
def repoint_video(db_path: str, old_path: str, new_path: str) -> None:
    conn = get_connection(db_path)
    with conn:
        conn.execute('UPDATE videos SET path = ?, content_key = NULL WHERE path = ?', (new_path, old_path))

# dismiss_matches: mark every match among the given videos as not-duplicates, returning the matches marked
def dismiss_matches(conn: sqlite3.Connection, video_ids: list[int]) -> int:
    marks = ','.join('?' * len(video_ids))
    with conn:
        return conn.execute(
            f'UPDATE matches SET dismissed = 1 WHERE video_a_id IN ({marks}) AND video_b_id IN ({marks})',
            [*video_ids, *video_ids],
        ).rowcount

# fold_in_moved_paths: adopt the legacy duplicates/ location as the real path so the review UI works in place
def fold_in_moved_paths(db_path: str) -> int:
    conn = get_connection(db_path)
    with conn:
        cur = conn.execute(
            "UPDATE videos SET path = moved_to, moved_to = NULL WHERE moved_to IS NOT NULL"
        )
    return cur.rowcount
