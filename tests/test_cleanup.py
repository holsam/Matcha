"""
Tests for matcha.cleanup

Covers duplicates reconciliation, the global missing-file purge, dry-run, and removal of the matching vectors from the FAISS index. Uses synthetic data and dummy files rather than the generated test videos.
"""

import os
import pytest

pytest.importorskip("faiss")

from matcha.db import get_connection, init_schema
from matcha.cleanup import run_cleanup


def _phash(n: int) -> str:
    v = (n * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    return f"{v:016x}"


class _World:
    """A throwaway library directory with a .matcha DB and helpers."""

    def __init__(self, tmp_path):
        self.root = tmp_path / "library"
        self.matcha_dir = self.root / ".matcha"
        self.matcha_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.matcha_dir / "index.db")
        init_schema(self.db_path)
        self._counter = 0

    def add_video(self, vid, path, moved_to=None, on_disk=True,
                  moved_on_disk=True, n_frames=6):
        conn = get_connection(self.db_path)
        with conn:
            conn.execute(
                "INSERT INTO videos (id, path, duration, fingerprinted_at, moved_to) "
                "VALUES (?, ?, ?, ?, ?)",
                (vid, str(path), float(n_frames), 1.0,
                 str(moved_to) if moved_to else None),
            )
            for t in range(n_frames):
                conn.execute(
                    "INSERT INTO frame_hashes (video_id, timestamp, phash) VALUES (?, ?, ?)",
                    (vid, float(t), _phash(self._counter)),
                )
                self._counter += 1
        if on_disk:
            self._touch(path)
        if moved_to and moved_on_disk:
            self._touch(moved_to)

    @staticmethod
    def _touch(path):
        path = str(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "a").close()

    def video_ids(self):
        conn = get_connection(self.db_path)
        return [r["id"] for r in conn.execute("SELECT id FROM videos").fetchall()]

    def count(self):
        conn = get_connection(self.db_path)
        return conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]


def _make_group(w, n=3):
    """Create a duplicates/1 group of n videos whose originals were moved."""
    grp = w.root / "duplicates" / "1"
    for i in range(n):
        w.add_video(
            i + 1,
            w.root / f"v{i}.mp4",            # original location (moved away)
            moved_to=grp / f"v{i}.mp4",      # current location in duplicates/
            on_disk=False,
            moved_on_disk=True,
        )
    return grp


class TestDuplicatesReconciliation:
    def test_deleted_members_removed_survivor_returned(self, tmp_path):
        w = _World(tmp_path)
        grp = _make_group(w, 3)
        (grp / "v1.mp4").unlink()   # user deletes videos 2 and 3
        (grp / "v2.mp4").unlink()

        run_cleanup(str(w.root))

        assert w.count() == 1
        conn = get_connection(w.db_path)
        row = conn.execute("SELECT path, moved_to FROM videos").fetchone()
        assert row["moved_to"] is None              # moved_to cleared
        assert os.path.exists(row["path"])          # returned to original
        assert not grp.exists()                     # empty group removed

    def test_group_with_no_deletions_untouched(self, tmp_path):
        w = _World(tmp_path)
        _make_group(w, 3)
        run_cleanup(str(w.root))
        assert w.count() == 3


class TestGlobalMissing:
    def test_missing_source_removed(self, tmp_path):
        w = _World(tmp_path)
        w.add_video(1, w.root / "present.mp4", on_disk=True)
        w.add_video(2, w.root / "gone.mp4", on_disk=False)
        run_cleanup(str(w.root))
        assert set(w.video_ids()) == {1}

    def test_moved_file_kept(self, tmp_path):
        w = _World(tmp_path)
        w.add_video(
            1, w.root / "a.mp4",
            moved_to=w.root / "elsewhere" / "a.mp4",
            on_disk=False, moved_on_disk=True,
        )
        run_cleanup(str(w.root))
        assert set(w.video_ids()) == {1}


class TestDryRun:
    def test_dry_run_changes_nothing(self, tmp_path):
        w = _World(tmp_path)
        w.add_video(1, w.root / "present.mp4", on_disk=True)
        w.add_video(2, w.root / "gone.mp4", on_disk=False)
        run_cleanup(str(w.root), dry_run=True)
        assert set(w.video_ids()) == {1, 2}

    def test_dry_run_does_not_move_survivor(self, tmp_path):
        w = _World(tmp_path)
        grp = _make_group(w, 3)
        (grp / "v1.mp4").unlink()
        (grp / "v2.mp4").unlink()
        run_cleanup(str(w.root), dry_run=True)
        assert (grp / "v0.mp4").exists()             # survivor untouched
        assert not (w.root / "v0.mp4").exists()      # not returned
        assert w.count() == 3


class TestIndexRemoval:
    def test_purged_vectors_removed_from_index(self, tmp_path):
        from matcha.faiss_index import update_index, load_index

        w = _World(tmp_path)
        w.add_video(1, w.root / "a.mp4", on_disk=True, n_frames=20)
        w.add_video(2, w.root / "gone.mp4", on_disk=False, n_frames=15)
        w.add_video(3, w.root / "c.mp4", on_disk=True, n_frames=18)

        update_index(w.db_path, str(w.matcha_dir))
        index, _ = load_index(str(w.matcha_dir))
        assert index.ntotal == 53

        run_cleanup(str(w.root))   # purges video 2 (missing) from DB and index

        assert set(w.video_ids()) == {1, 3}
        index, id_map = load_index(str(w.matcha_dir))
        assert index.ntotal == 38                        # 20 + 18
        assert int((id_map[:, 0] == -1).sum()) == 15     # video 2 tombstoned


class TestNoIndex:
    def test_cleanup_without_index(self, tmp_path):
        w = _World(tmp_path)
        w.add_video(1, w.root / "gone.mp4", on_disk=False)
        run_cleanup(str(w.root))   # no FAISS index built; should still purge DB
        assert w.video_ids() == []