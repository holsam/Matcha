import pytest

from matcha.db import get_connection
from matcha.indexer import run_index


class TestIndexCreation:
    def test_creates_db_file(self, indexed_dir):
        db = indexed_dir["dir"] / ".matcha" / "index.db"
        assert db.exists()

    def test_registers_all_videos(self, indexed_dir):
        db_path = str(indexed_dir["dir"] / ".matcha" / "index.db")
        conn = get_connection(db_path)
        count = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        assert count == 30

    def test_skips_matcha_directory(self, indexed_dir):
        db_path = str(indexed_dir["dir"] / ".matcha" / "index.db")
        conn = get_connection(db_path)
        paths = [r["path"] for r in conn.execute("SELECT path FROM videos").fetchall()]
        assert not any(".matcha" in p for p in paths)


class TestFingerprinting:
    def test_all_videos_fingerprinted(self, indexed_dir):
        db_path = str(indexed_dir["dir"] / ".matcha" / "index.db")
        conn = get_connection(db_path)
        unprocessed = conn.execute(
            "SELECT COUNT(*) FROM videos WHERE fingerprinted_at IS NULL"
        ).fetchone()[0]
        assert unprocessed == 0

    def test_frame_hashes_created(self, indexed_dir):
        db_path = str(indexed_dir["dir"] / ".matcha" / "index.db")
        conn = get_connection(db_path)
        count = conn.execute("SELECT COUNT(*) FROM frame_hashes").fetchone()[0]
        assert count > 0

    def test_frame_hash_count_matches_duration(self, indexed_dir):
        """At 1fps, each video should have ~duration-in-seconds frames (±1)."""
        db_path = str(indexed_dir["dir"] / ".matcha" / "index.db")
        conn = get_connection(db_path)
        rows = conn.execute(
            """
            SELECT v.duration, COUNT(f.id) as hash_count
            FROM videos v JOIN frame_hashes f ON f.video_id = v.id
            GROUP BY v.id
            """
        ).fetchall()
        for row in rows:
            assert abs(row["hash_count"] - int(row["duration"])) <= 1

    def test_durations_stored(self, indexed_dir):
        db_path = str(indexed_dir["dir"] / ".matcha" / "index.db")
        conn = get_connection(db_path)
        for row in conn.execute("SELECT duration FROM videos").fetchall():
            assert row["duration"] is not None and row["duration"] > 0

    def test_all_videos_fingerprinted_hwaccel(self, indexed_dir_hwaccel):
        db_path = str(indexed_dir_hwaccel["dir"] / ".matcha" / "index.db")
        conn = get_connection(db_path)
        unprocessed = conn.execute(
            "SELECT COUNT(*) FROM videos WHERE fingerprinted_at IS NULL"
        ).fetchone()[0]
        assert unprocessed == 0

    def test_frame_hashes_created_hwaccel(self, indexed_dir_hwaccel):
        db_path = str(indexed_dir_hwaccel["dir"] / ".matcha" / "index.db")
        conn = get_connection(db_path)
        count = conn.execute("SELECT COUNT(*) FROM frame_hashes").fetchone()[0]
        assert count > 0

    def test_frame_hash_count_matches_duration_hwaccel(self, indexed_dir_hwaccel):
        """At 1fps, each video should have ~duration-in-seconds frames (±1)."""
        db_path = str(indexed_dir_hwaccel["dir"] / ".matcha" / "index.db")
        conn = get_connection(db_path)
        rows = conn.execute(
            """
            SELECT v.duration, COUNT(f.id) as hash_count
            FROM videos v JOIN frame_hashes f ON f.video_id = v.id
            GROUP BY v.id
            """
        ).fetchall()
        for row in rows:
            assert abs(row["hash_count"] - int(row["duration"])) <= 1

    def test_durations_stored_hwaccel(self, indexed_dir_hwaccel):
        db_path = str(indexed_dir_hwaccel["dir"] / ".matcha" / "index.db")
        conn = get_connection(db_path)
        for row in conn.execute("SELECT duration FROM videos").fetchall():
            assert row["duration"] is not None and row["duration"] > 0
    
    def test_all_videos_fingerprinted_noaudio(self, indexed_dir_no_audio):
        db_path = str(indexed_dir_no_audio["dir"] / ".matcha" / "index.db")
        conn = get_connection(db_path)
        unprocessed = conn.execute(
            "SELECT COUNT(*) FROM videos WHERE fingerprinted_at IS NULL"
        ).fetchone()[0]
        assert unprocessed == 0

    def test_frame_hashes_created_noaudio(self, indexed_dir_no_audio):
        db_path = str(indexed_dir_no_audio["dir"] / ".matcha" / "index.db")
        conn = get_connection(db_path)
        count = conn.execute("SELECT COUNT(*) FROM frame_hashes").fetchone()[0]
        assert count > 0

    def test_frame_hash_count_matches_duration_noaudio(self, indexed_dir_no_audio):
        """At 1fps, each video should have ~duration-in-seconds frames (±1)."""
        db_path = str(indexed_dir_no_audio["dir"] / ".matcha" / "index.db")
        conn = get_connection(db_path)
        rows = conn.execute(
            """
            SELECT v.duration, COUNT(f.id) as hash_count
            FROM videos v JOIN frame_hashes f ON f.video_id = v.id
            GROUP BY v.id
            """
        ).fetchall()
        for row in rows:
            assert abs(row["hash_count"] - int(row["duration"])) <= 1

    def test_durations_stored_noaudio(self, indexed_dir_no_audio):
        db_path = str(indexed_dir_no_audio["dir"] / ".matcha" / "index.db")
        conn = get_connection(db_path)
        for row in conn.execute("SELECT duration FROM videos").fetchall():
            assert row["duration"] is not None and row["duration"] > 0

class TestCheckpointing:
    def test_idempotent(self, indexed_dir):
        """Running index twice does not change fingerprinted_at timestamps."""
        db_path = str(indexed_dir["dir"] / ".matcha" / "index.db")
        conn = get_connection(db_path)
        before = {
            r["path"]: r["fingerprinted_at"]
            for r in conn.execute("SELECT path, fingerprinted_at FROM videos").fetchall()
        }
        run_index(str(indexed_dir["dir"]), fps=1.0, workers=2)
        after = {
            r["path"]: r["fingerprinted_at"]
            for r in conn.execute("SELECT path, fingerprinted_at FROM videos").fetchall()
        }
        assert before == after