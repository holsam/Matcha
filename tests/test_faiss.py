"""
Tests for matcha.faiss_index: incremental index building in-place vector removal.

These exercise the index helpers directly against a small synthetic database, so they do not need the generated test videos. faiss is required.
"""

import types

import numpy as np
import pytest

pytest.importorskip("faiss")

from matcha.db import get_connection, init_schema
from matcha.faiss_index import (
    update_index,
    remove_videos_from_index,
    rebuild_recommended,
    maybe_rebuild_index,
    load_index,
    _load_new_hashes,
)


def _spread(n: int) -> str:
    """Deterministic, unique 16-hex (64-bit) pHash for a global counter."""
    v = (n * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    return f"{v:016x}"


class _Builder:
    """Build a synthetic .matcha dir with videos and frame hashes."""

    def __init__(self, tmp_path):
        self.matcha_dir = tmp_path / ".matcha"
        self.matcha_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = str(self.matcha_dir / "index.db")
        init_schema(self.db_path)
        self._counter = 0

    @property
    def index_dir(self) -> str:
        return str(self.matcha_dir)

    def add_video(self, vid: int, n_frames: int = 20):
        conn = get_connection(self.db_path)
        with conn:
            conn.execute(
                "INSERT INTO videos (id, path, duration, fingerprinted_at) VALUES (?, ?, ?, ?)",
                (vid, f"/videos/{vid}.mp4", float(n_frames), 1.0),
            )
            for t in range(n_frames):
                conn.execute(
                    "INSERT INTO frame_hashes (video_id, timestamp, phash) VALUES (?, ?, ?)",
                    (vid, float(t), _spread(self._counter)),
                )
                self._counter += 1

    def delete_video(self, vid: int):
        conn = get_connection(self.db_path)
        with conn:
            conn.execute("DELETE FROM frame_hashes WHERE video_id = ?", (vid,))
            conn.execute("DELETE FROM videos WHERE id = ?", (vid,))

    def first_phash(self, vid: int) -> str:
        conn = get_connection(self.db_path)
        row = conn.execute(
            "SELECT phash FROM frame_hashes WHERE video_id = ? ORDER BY timestamp LIMIT 1",
            (vid,),
        ).fetchone()
        return row["phash"]


def _vec(phash: str) -> np.ndarray:
    return np.array([list(bytes.fromhex(phash))], dtype=np.uint8)


class TestFullBuild:
    def test_first_run_builds_and_returns_true(self, tmp_path):
        b = _Builder(tmp_path)
        b.add_video(1, 20)
        b.add_video(2, 20)
        assert update_index(b.db_path, b.index_dir) is True
        index, id_map = load_index(b.index_dir)
        assert index.ntotal == 40
        assert id_map.shape[0] == 40

    def test_empty_db_writes_nothing(self, tmp_path):
        b = _Builder(tmp_path)
        assert update_index(b.db_path, b.index_dir) is False
        assert not (b.matcha_dir / "frame_index.faiss").exists()


class TestIncrementalAppend:
    def test_no_change_when_nothing_new(self, tmp_path):
        b = _Builder(tmp_path)
        b.add_video(1, 20)
        update_index(b.db_path, b.index_dir)
        assert update_index(b.db_path, b.index_dir) is False
        index, _ = load_index(b.index_dir)
        assert index.ntotal == 20

    def test_new_video_appends(self, tmp_path):
        b = _Builder(tmp_path)
        b.add_video(1, 20)
        update_index(b.db_path, b.index_dir)
        b.add_video(2, 15)
        assert update_index(b.db_path, b.index_dir) is True
        index, id_map = load_index(b.index_dir)
        assert index.ntotal == 35
        assert id_map.shape[0] == 35
        assert {int(v) for v in id_map[20:, 0]} == {2}

    def test_appended_vectors_are_searchable(self, tmp_path):
        b = _Builder(tmp_path)
        b.add_video(1, 20)
        update_index(b.db_path, b.index_dir)
        b.add_video(2, 15)
        update_index(b.db_path, b.index_dir)
        index, id_map = load_index(b.index_dir)
        index.nprobe = 32
        dist, labels = index.search(_vec(b.first_phash(2)), 1)
        label = int(labels[0, 0])
        assert label >= 0
        assert int(id_map[label, 0]) == 2
        assert int(dist[0, 0]) == 0  # exact self match


class TestLoadNewHashes:
    def test_filters_known_videos(self, tmp_path):
        b = _Builder(tmp_path)
        b.add_video(1, 10)
        b.add_video(2, 8)
        vectors, id_map = _load_new_hashes(b.db_path, {1})
        assert len(vectors) == 8
        assert {int(v) for v in id_map[:, 0]} == {2}

    def test_empty_when_all_known(self, tmp_path):
        b = _Builder(tmp_path)
        b.add_video(1, 10)
        vectors, _ = _load_new_hashes(b.db_path, {1})
        assert len(vectors) == 0


class TestRemoveVideos:
    def test_no_index_returns_zero(self, tmp_path):
        b = _Builder(tmp_path)
        assert remove_videos_from_index(b.index_dir, {1}) == 0

    def test_empty_ids_returns_zero(self, tmp_path):
        b = _Builder(tmp_path)
        b.add_video(1, 10)
        update_index(b.db_path, b.index_dir)
        assert remove_videos_from_index(b.index_dir, set()) == 0

    def test_removes_and_tombstones(self, tmp_path):
        b = _Builder(tmp_path)
        b.add_video(1, 20)
        b.add_video(2, 15)
        b.add_video(3, 18)
        update_index(b.db_path, b.index_dir)
        assert remove_videos_from_index(b.index_dir, {2}) == 15
        index, id_map = load_index(b.index_dir)
        assert index.ntotal == 38  # 20 + 18
        assert id_map.shape[0] == 53  # rows kept, not dropped
        assert int((id_map[:, 0] == -1).sum()) == 15
        assert 2 not in {int(v) for v in id_map[:, 0]}

    def test_removed_video_not_returned_by_search(self, tmp_path):
        b = _Builder(tmp_path)
        b.add_video(1, 20)
        b.add_video(2, 15)
        update_index(b.db_path, b.index_dir)
        query = _vec(b.first_phash(2))
        remove_videos_from_index(b.index_dir, {2})
        index, id_map = load_index(b.index_dir)
        index.nprobe = 32
        _, labels = index.search(query, 5)
        returned = {int(id_map[l, 0]) for l in labels[0] if l >= 0}
        assert 2 not in returned

class TestRemoveThenAppend:
    def test_append_after_remove_has_no_id_collision(self, tmp_path):
        b = _Builder(tmp_path)
        b.add_video(1, 20)
        b.add_video(2, 15)
        b.add_video(3, 18)
        update_index(b.db_path, b.index_dir)

        # Mirror cleanup: remove video 2 from the index AND the database,
        # then add a brand-new video and append.
        remove_videos_from_index(b.index_dir, {2})
        b.delete_video(2)
        b.add_video(4, 12)
        assert update_index(b.db_path, b.index_dir) is True

        index, id_map = load_index(b.index_dir)
        index.nprobe = 32

        live = int((id_map[:, 0] >= 0).sum())
        assert index.ntotal == live == 50  # 20 + 18 survivors + 12 new

        # Video 4's own frame resolves to video 4, proving its IDs did not
        # collide with the survivors that kept their original IDs.
        dist, labels = index.search(_vec(b.first_phash(4)), 1)
        assert int(dist[0, 0]) == 0
        assert int(id_map[int(labels[0, 0]), 0]) == 4


class TestRebuildDecision:
    """Unit tests for the rebuild thresholds, using a stub index."""

    @staticmethod
    def _idmap(live: int, total: int) -> np.ndarray:
        col0 = np.array([1] * live + [-1] * (total - live), dtype=np.int64)
        return np.column_stack([col0, np.zeros(total, dtype=np.int64)])

    def test_healthy_index_not_rebuilt(self):
        idx = types.SimpleNamespace(ntotal=100, nlist=10)
        should, _ = rebuild_recommended(idx, self._idmap(100, 100))
        assert should is False

    def test_nlist_drift_triggers(self):
        idx = types.SimpleNamespace(ntotal=100, nlist=4)
        should, _ = rebuild_recommended(idx, self._idmap(100, 100))
        assert should is True

    def test_tombstones_trigger(self):
        idx = types.SimpleNamespace(ntotal=40, nlist=6)
        should, _ = rebuild_recommended(idx, self._idmap(40, 60))
        assert should is True

    def test_empty_map(self):
        idx = types.SimpleNamespace(ntotal=0, nlist=1)
        should, _ = rebuild_recommended(idx, np.empty((0, 2), dtype=np.int64))
        assert should is False


class TestMaybeRebuild:
    """Integration tests against real indexes."""

    def test_no_index_no_rebuild(self, tmp_path):
        b = _Builder(tmp_path)
        assert maybe_rebuild_index(b.db_path, b.index_dir) is False

    def test_healthy_index_not_rebuilt(self, tmp_path):
        b = _Builder(tmp_path)
        b.add_video(1, 100)
        update_index(b.db_path, b.index_dir)
        before, _ = load_index(b.index_dir)
        assert maybe_rebuild_index(b.db_path, b.index_dir) is False
        after, _ = load_index(b.index_dir)
        assert after.ntotal == before.ntotal == 100

    def test_nlist_drift_rebuilds(self, tmp_path):
        b = _Builder(tmp_path)
        b.add_video(1, 16)                 # trained nlist = 4
        update_index(b.db_path, b.index_dir)
        idx, _ = load_index(b.index_dir)
        assert idx.nlist == 4

        b.add_video(2, 84)                 # grow to 100 live vectors
        update_index(b.db_path, b.index_dir)

        assert maybe_rebuild_index(b.db_path, b.index_dir) is True
        idx, id_map = load_index(b.index_dir)
        assert idx.nlist == 10             # retrained for the new size
        assert idx.ntotal == 100
        assert int((id_map[:, 0] == -1).sum()) == 0

    def test_tombstones_compacted(self, tmp_path):
        b = _Builder(tmp_path)
        b.add_video(1, 20)
        b.add_video(2, 20)
        b.add_video(3, 20)
        update_index(b.db_path, b.index_dir)

        # Remove one video from index and DB (mirroring cleanup): 20/60 = 33%.
        remove_videos_from_index(b.index_dir, {2})
        b.delete_video(2)

        assert maybe_rebuild_index(b.db_path, b.index_dir) is True
        idx, id_map = load_index(b.index_dir)
        assert idx.ntotal == 40
        assert id_map.shape[0] == 40                   # tombstones compacted away
        assert int((id_map[:, 0] == -1).sum()) == 0