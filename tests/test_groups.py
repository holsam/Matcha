"""Tests for matcha.groups, matcha.db migration helpers (synthetic data)."""

from matcha.db import fold_in_moved_paths, get_connection, init_schema
from matcha.groups import build_groups, find_groups


def _db(tmp_path):
    (tmp_path / ".matcha").mkdir()
    db_path = str(tmp_path / ".matcha" / "index.db")
    init_schema(db_path)
    return db_path


def _seed(db_path, videos, matches):
    conn = get_connection(db_path)
    with conn:
        for vid, dur in videos.items():
            conn.execute("INSERT INTO videos (id, path, duration) VALUES (?, ?, ?)", (vid, f"/x/{vid}.mp4", dur))
        for a, b, kind, conf in matches:
            conn.execute(
                "INSERT INTO matches (video_a_id, video_b_id, match_type, confidence, found_at) VALUES (?, ?, ?, ?, 0)",
                (a, b, kind, conf),
            )


def test_build_groups_merges_chains():
    groups = build_groups([(1, 2), (2, 3), (4, 5)])
    assert sorted(map(sorted, groups)) == [[1, 2, 3], [4, 5]]


def test_find_groups_summary_and_order(tmp_path):
    db = _db(tmp_path)
    _seed(db, {1: 10, 2: 10, 3: 5, 4: 20, 5: 20},
          [(1, 2, "duplicate", 0.95), (3, 2, "subclip", 0.85), (4, 5, "duplicate", 0.99)])
    big, small = find_groups(db)
    assert big.video_ids == (1, 2, 3) and big.key == 1
    assert big.max_confidence == 0.95 and big.has_subclip
    assert big.total_duration == 25
    assert small.video_ids == (4, 5) and not small.has_subclip


def test_find_groups_skips_dismissed(tmp_path):
    db = _db(tmp_path)
    _seed(db, {1: 1, 2: 1}, [(1, 2, "duplicate", 0.9)])
    conn = get_connection(db)
    with conn:
        conn.execute("UPDATE matches SET dismissed = 1")
    assert find_groups(db) == []


def test_dismissed_column_added_to_old_db(tmp_path):
    db = _db(tmp_path)
    conn = get_connection(db)
    with conn:
        conn.executescript("DROP TABLE matches; CREATE TABLE matches (id INTEGER PRIMARY KEY, video_a_id INT, video_b_id INT, match_type TEXT, confidence REAL, found_at REAL, moved INT DEFAULT 0);")
    init_schema(db)
    assert "dismissed" in {r["name"] for r in conn.execute("PRAGMA table_info(matches)")}


def test_fold_in_moved_paths(tmp_path):
    db = _db(tmp_path)
    conn = get_connection(db)
    with conn:
        conn.execute("INSERT INTO videos (id, path, moved_to) VALUES (1, '/a.mp4', '/duplicates/1/a.mp4')")
        conn.execute("INSERT INTO videos (id, path) VALUES (2, '/b.mp4')")
    assert fold_in_moved_paths(db) == 1
    rows = {r["id"]: (r["path"], r["moved_to"]) for r in conn.execute("SELECT * FROM videos")}
    assert rows == {1: ("/duplicates/1/a.mp4", None), 2: ("/b.mp4", None)}
