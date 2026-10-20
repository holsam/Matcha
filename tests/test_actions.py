"""Tests for matcha.actions (synthetic library, trash and FAISS removal mocked)."""

import pytest

from matcha import actions
from matcha.db import get_connection, init_schema


@pytest.fixture()
def lib(tmp_path, monkeypatch):
    (tmp_path / ".matcha").mkdir()
    db_path = str(tmp_path / ".matcha" / "index.db")
    init_schema(db_path)
    conn = get_connection(db_path)
    with conn:
        for vid in (1, 2, 3):
            f = tmp_path / f"v{vid}.mp4"
            f.write_text("x")
            conn.execute("INSERT INTO videos (id, path, duration) VALUES (?, ?, 1)", (vid, str(f)))
            conn.execute("INSERT INTO frame_hashes (video_id, timestamp, phash) VALUES (?, 0, 'ff')", (vid,))
        conn.execute("INSERT INTO matches (video_a_id, video_b_id, match_type, confidence, found_at) VALUES (1, 2, 'duplicate', 0.9, 0)")
        conn.execute("INSERT INTO matches (video_a_id, video_b_id, match_type, confidence, found_at) VALUES (2, 3, 'duplicate', 0.9, 0)")
        # rows that would block the videos delete under foreign_keys=ON
        conn.execute("INSERT INTO candidate_pairs (video_a_id, video_b_id) VALUES (1, 3)")
        conn.execute("INSERT INTO candidate_search_progress (video_id, queried_at) VALUES (1, 0)")

    trashed: list[str] = []
    removed: list[list[int]] = []

    def fake_trash(path):
        trashed.append(path)
        import os
        os.remove(path)

    monkeypatch.setattr(actions, "send2trash", fake_trash)
    monkeypatch.setattr("matcha.faiss_index.remove_videos_from_index", lambda d, ids: removed.append(list(ids)) or 0)
    return {"dir": str(tmp_path), "root": tmp_path, "conn": conn, "trashed": trashed, "removed": removed}


def _ids(lib):
    return {r["id"] for r in lib["conn"].execute("SELECT id FROM videos")}


class TestDelete:
    def test_trashes_file_and_purges_records(self, lib):
        res = actions.delete_videos(lib["dir"], [1])
        assert res.deleted == [1] and not res.failed
        assert lib["trashed"] == [str(lib["root"] / "v1.mp4")]
        assert _ids(lib) == {2, 3}
        assert lib["conn"].execute("SELECT COUNT(*) FROM frame_hashes WHERE video_id = 1").fetchone()[0] == 0
        assert lib["conn"].execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1

    def test_faiss_removal_batched_once(self, lib):
        actions.delete_videos(lib["dir"], [1, 2])
        assert lib["removed"] == [[1, 2]]

    def test_missing_file_still_purged(self, lib):
        (lib["root"] / "v1.mp4").unlink()
        res = actions.delete_videos(lib["dir"], [1])
        assert res.deleted == [1] and lib["trashed"] == []

    def test_trash_failure_keeps_record(self, lib, monkeypatch):
        def boom(path):
            raise OSError("denied")
        monkeypatch.setattr(actions, "send2trash", boom)
        res = actions.delete_videos(lib["dir"], [1])
        assert 1 in res.failed and not res.deleted
        assert 1 in _ids(lib) and lib["removed"] == []

    def test_unknown_id_reported(self, lib):
        assert 99 in actions.delete_videos(lib["dir"], [99]).failed


class TestKeepOne:
    def test_deletes_others_only(self, lib):
        actions.keep_one(lib["dir"], 2, [1, 2, 3])
        assert _ids(lib) == {2}

    def test_rejects_keep_outside_group(self, lib):
        with pytest.raises(ValueError):
            actions.keep_one(lib["dir"], 9, [1, 2, 3])


class TestRename:
    def test_renames_file_and_db(self, lib):
        new = actions.rename_video(lib["dir"], 1, "better.mp4")
        assert (lib["root"] / "better.mp4").exists() and not (lib["root"] / "v1.mp4").exists()
        assert lib["conn"].execute("SELECT path FROM videos WHERE id = 1").fetchone()[0] == new

    def test_keeps_extension_when_omitted(self, lib):
        assert actions.rename_video(lib["dir"], 1, "better").endswith("better.mp4")

    @pytest.mark.parametrize("bad", ["", "..", "a/b.mp4", "v2.mp4"])
    def test_rejects_invalid_or_colliding(self, lib, bad):
        with pytest.raises(ValueError):
            actions.rename_video(lib["dir"], 1, bad)
        assert (lib["root"] / "v1.mp4").exists()


class TestDismiss:
    def test_marks_group_matches(self, lib):
        assert actions.dismiss_group(lib["dir"], [1, 2, 3]) == 2
        assert lib["conn"].execute("SELECT COUNT(*) FROM matches WHERE dismissed = 0").fetchone()[0] == 0
