from pathlib import Path
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

class TestReuseDuplicates:
    def _setup(self, video_dir, tmp_path):
        import shutil
        src = sorted(video_dir["dir"].glob("*.mp4"))
        shutil.copy(src[0], tmp_path / "a.mp4")
        shutil.copy(src[0], tmp_path / "a_copy.mp4")  # byte-identical
        shutil.copy(src[1], tmp_path / "b.mp4")
        return tmp_path

    def test_identical_file_reuses_hashes(self, video_dir, tmp_path, monkeypatch):
        import matcha.indexer as indexer
        directory = self._setup(video_dir, tmp_path)
        calls = []
        real = indexer.extract_frame_hashes
        monkeypatch.setattr(indexer, "extract_frame_hashes", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
        run_index(str(directory), fps=1.0, workers=1, no_audio=True)  # one worker so the copy sees the original
        assert len(calls) == 2  # a and b decoded, the copy reused
        conn = get_connection(str(directory / ".matcha" / "index.db"))
        hashes = {
            Path(r["path"]).name: [x["phash"] for x in conn.execute(
                "SELECT phash FROM frame_hashes WHERE video_id = ? ORDER BY timestamp", (r["id"],))]
            for r in conn.execute("SELECT id, path FROM videos")
        }
        assert hashes["a.mp4"] == hashes["a_copy.mp4"] and hashes["a.mp4"]
        assert hashes["a.mp4"] != hashes["b.mp4"]

    def test_disabled_decodes_every_file(self, video_dir, tmp_path, monkeypatch):
        import matcha.indexer as indexer
        directory = self._setup(video_dir, tmp_path)
        calls = []
        real = indexer.extract_frame_hashes
        monkeypatch.setattr(indexer, "extract_frame_hashes", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
        run_index(str(directory), fps=1.0, workers=1, no_audio=True, reuse_duplicates=False)
        assert len(calls) == 3


class TestBackfillContentKeys:
    def test_keys_existing_videos_and_enables_reuse(self, video_dir, tmp_path, monkeypatch):
        import shutil
        import matcha.indexer as indexer
        src = sorted(video_dir["dir"].glob("*.mp4"))[0]
        shutil.copy(src, tmp_path / "a.mp4")
        run_index(str(tmp_path), fps=1.0, workers=1, no_audio=True, reuse_duplicates=False)  # no keys stored
        db_path = str(tmp_path / ".matcha" / "index.db")
        conn = get_connection(db_path)
        assert conn.execute("SELECT content_key FROM videos").fetchone()[0] is None

        shutil.copy(src, tmp_path / "a_copy.mp4")
        calls = []
        real = indexer.extract_frame_hashes
        monkeypatch.setattr(indexer, "extract_frame_hashes", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
        run_index(str(tmp_path), fps=1.0, workers=1, no_audio=True)
        assert calls == []  # the copy reused the backfilled original
        assert conn.execute("SELECT COUNT(*) FROM videos WHERE content_key IS NULL").fetchone()[0] == 0
