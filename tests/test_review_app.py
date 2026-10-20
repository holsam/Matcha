"""Tests for the review UI list view and its pure helpers (synthetic data)."""

import pytest
from pathlib import Path
from streamlit.testing.v1 import AppTest

from matcha.db import get_connection, init_schema
from matcha.groups import Group, paginate, sort_groups

APP = str(Path(__file__).parent.parent / "src" / "matcha" / "review" / "app.py")


def _g(key, n, conf=0.9, dur=10.0):
    return Group(key, tuple(range(key, key + n)), conf, False, dur)


class TestHelpers:
    def test_paginate_clamps_and_counts(self):
        items = list(range(53))
        assert paginate(items, 0, 25) == (items[:25], 0, 3)
        assert paginate(items, 2, 25)[0] == items[50:]
        assert paginate(items, 99, 25)[1] == 2
        assert paginate([], 0, 25) == ([], 0, 1)

    def test_sort_groups(self):
        a, b, c = _g(1, 2, 0.99, 5), _g(10, 4, 0.8, 1), _g(20, 3, 0.9, 50)
        assert sort_groups([a, b, c], "size") == [b, c, a]
        assert sort_groups([a, b, c], "confidence") == [a, c, b]
        assert sort_groups([a, b, c], "duration") == [c, a, b]


@pytest.fixture()
def library(tmp_path, monkeypatch):
    (tmp_path / ".matcha").mkdir()
    db_path = str(tmp_path / ".matcha" / "index.db")
    init_schema(db_path)
    conn = get_connection(db_path)
    with conn:
        for vid in range(1, 61):  # 30 pair groups
            conn.execute("INSERT INTO videos (id, path, duration) VALUES (?, ?, 60)", (vid, f"/lib/v{vid}.mp4"))
        for vid in range(1, 61, 2):
            conn.execute(
                "INSERT INTO matches (video_a_id, video_b_id, match_type, confidence, found_at) VALUES (?, ?, 'duplicate', 0.9, 0)",
                (vid, vid + 1),
            )
    monkeypatch.setenv("MATCHA_REVIEW_DIR", str(tmp_path))
    return tmp_path


def _run():
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception
    return at


def _click(at, label):
    next(b for b in at.button if b.label == label).click()
    at.run()
    assert not at.exception


class TestListView:
    def test_first_page_only(self, library):
        at = _run()
        assert len([b for b in at.button if b.label == "Open"]) == 25
        assert any("30 groups" in c.value and "page 1 of 2" in c.value for c in at.caption)

    def test_next_page(self, library):
        at = _run()
        _click(at, "Next")
        assert len([b for b in at.button if b.label == "Open"]) == 5

    def test_open_and_back(self, library):
        at = _run()
        _click(at, "Open")
        assert at.session_state.open_group is not None
        _click(at, "Back")
        assert at.session_state.open_group is None

    def test_empty_library(self, library):
        conn = get_connection(str(library / ".matcha" / "index.db"))
        with conn:
            conn.execute("DELETE FROM matches")
        assert any("No duplicate groups" in i.value for i in _run().info)


@pytest.fixture()
def real_library(tmp_path, monkeypatch):
    """Three real (non-video) files forming one group: 1~2, 2~3."""
    (tmp_path / ".matcha").mkdir()
    db_path = str(tmp_path / ".matcha" / "index.db")
    init_schema(db_path)
    conn = get_connection(db_path)
    with conn:
        for vid in (1, 2, 3):
            f = tmp_path / f"v{vid}.mp4"
            f.write_text("x" * vid)
            conn.execute("INSERT INTO videos (id, path, duration) VALUES (?, ?, 60)", (vid, str(f)))
        conn.execute("INSERT INTO matches (video_a_id, video_b_id, match_type, confidence, found_at) VALUES (1, 2, 'duplicate', 0.95, 0)")
        conn.execute("INSERT INTO matches (video_a_id, video_b_id, match_type, confidence, found_at) VALUES (2, 3, 'subclip', 0.85, 0)")
    monkeypatch.setenv("MATCHA_REVIEW_DIR", str(tmp_path))
    trashed = []
    monkeypatch.setattr("matcha.actions.send2trash", lambda p: (trashed.append(p), __import__("os").remove(p)))
    monkeypatch.setattr("matcha.faiss_index.remove_videos_from_index", lambda d, ids: 0)
    return {"root": tmp_path, "conn": conn, "trashed": trashed}


def _open(at):
    _click(at, "Open")


class TestDetailView:
    def test_shows_every_video_and_pairs(self, real_library):
        at = _run()
        _open(at)
        assert len(at.checkbox) == 3
        pairs = at.dataframe[0].value
        assert sorted(pairs["Type"]) == ["duplicate", "subclip"]
        assert sorted(pairs["Similarity"]) == ["85%", "95%"]

    def test_missing_file_flagged(self, real_library):
        (real_library["root"] / "v3.mp4").unlink()
        at = _run()
        _open(at)
        assert any("missing from disk" in w.value for w in at.warning)

    def test_delete_selected_requires_confirmation(self, real_library):
        at = _run()
        _open(at)
        at.checkbox(key="sel-1").check().run()
        _click(at, "Delete selected (1)")
        assert real_library["trashed"] == []  # nothing yet
        assert any("1 video(s) to the trash" in w.value for w in at.warning)
        _click(at, "Move to trash")
        assert [p.rsplit("/", 1)[-1] for p in real_library["trashed"]] == ["v1.mp4"]
        ids = {r[0] for r in real_library["conn"].execute("SELECT id FROM videos")}
        assert ids == {2, 3}
        assert len(at.checkbox) == 2  # group re-resolved, still open

    def test_cancel_leaves_everything(self, real_library):
        at = _run()
        _open(at)
        at.checkbox(key="sel-1").check().run()
        _click(at, "Delete selected (1)")
        _click(at, "Cancel")
        assert real_library["trashed"] == []

    def test_keep_only_this(self, real_library):
        at = _run()
        _open(at)
        at.button(key="keep-2").click().run()
        _click(at, "Move to trash")
        ids = {r[0] for r in real_library["conn"].execute("SELECT id FROM videos")}
        assert ids == {2}
        assert at.session_state.open_group is None  # no group left, back to list

    def test_rename(self, real_library):
        at = _run()
        _open(at)
        at.text_input(key="name-1").set_value("fresh.mp4").run()
        at.button(key="rename-1").click().run()
        assert (real_library["root"] / "fresh.mp4").exists()
        assert not (real_library["root"] / "v1.mp4").exists()

    def test_dismiss_returns_to_empty_list(self, real_library):
        at = _run()
        _open(at)
        _click(at, "Not duplicates, dismiss")
        assert at.session_state.open_group is None
        assert any("No duplicate groups" in i.value for i in at.info)
