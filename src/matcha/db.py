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
                moved_to         TEXT
            );

            CREATE TABLE IF NOT EXISTS frame_hashes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id    INTEGER NOT NULL REFERENCES videos(id),
                timestamp   REAL NOT NULL,
                phash       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_frame_hashes_video_id
                ON frame_hashes(video_id);

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
        """)

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