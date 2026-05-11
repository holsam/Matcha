"""Tests for matcha.mover (matcha move subcommand)."""

import pytest, shutil
from pathlib import Path
from matcha.db import get_connection
from matcha.mover import run_move


@pytest.fixture()
def move_dir(matched_dir, tmp_path):
    """
    Fresh copy of the matched directory for each test.
    Re-maps video paths in the DB to point at the new location.
    """
    fresh = tmp_path / "videos"
    shutil.copytree(matched_dir["dir"], fresh)
    db_path = str(fresh / ".matcha" / "index.db")
    conn = get_connection(db_path)
    rows = conn.execute("SELECT id, path FROM videos").fetchall()
    with conn:
        for row in rows:
            new = fresh / Path(row["path"]).name
            conn.execute("UPDATE videos SET path = ? WHERE id = ?", (str(new), row["id"]))
    return {**matched_dir, "dir": fresh, "db_path": db_path}


class TestDryRun:
    def test_dry_run_prints_summary(self, move_dir, capsys):
        run_move(str(move_dir["dir"]), dry_run=True)
        out = capsys.readouterr().out
        assert "dry-run" in out and "would be moved" in out

    def test_dry_run_does_not_move_files(self, move_dir):
        run_move(str(move_dir["dir"]), dry_run=True)
        assert not (move_dir["dir"] / "duplicates").exists()

    def test_dry_run_does_not_update_db(self, move_dir):
        conn = get_connection(move_dir["db_path"])
        before = conn.execute("SELECT COUNT(*) FROM matches WHERE moved = 0").fetchone()[0]
        run_move(str(move_dir["dir"]), dry_run=True)
        after = conn.execute("SELECT COUNT(*) FROM matches WHERE moved = 0").fetchone()[0]
        assert before == after


class TestMoving:
    def test_creates_duplicates_directory(self, move_dir):
        run_move(str(move_dir["dir"]))
        assert (move_dir["dir"] / "duplicates").is_dir()

    def test_matched_files_moved_into_duplicates(self, move_dir):
        original, copy = move_dir["exact_pair"]
        original = move_dir["dir"] / original.name
        copy = move_dir["dir"] / copy.name
        run_move(str(move_dir["dir"]))
        duplicates = move_dir["dir"] / "duplicates"
        moved_names = {f.name for f in duplicates.rglob("*.mp4")}
        assert original.name in moved_names
        assert copy.name in moved_names

    def test_same_group_in_same_directory(self, move_dir):
        original, copy = move_dir["exact_pair"]
        original = move_dir["dir"] / original.name
        copy = move_dir["dir"] / copy.name
        run_move(str(move_dir["dir"]))
        duplicates = move_dir["dir"] / "duplicates"
        orig_loc = next(duplicates.rglob(original.name), None)
        copy_loc = next(duplicates.rglob(copy.name), None)
        assert orig_loc is not None and copy_loc is not None
        assert orig_loc.parent == copy_loc.parent

    def test_subclip_group_in_same_directory(self, move_dir):
        core, container = move_dir["partial_pair"]
        core = move_dir["dir"] / core.name
        container = move_dir["dir"] / container.name
        run_move(str(move_dir["dir"]))
        duplicates = move_dir["dir"] / "duplicates"
        core_loc = next(duplicates.rglob(core.name), None)
        container_loc = next(duplicates.rglob(container.name), None)
        assert core_loc is not None and container_loc is not None
        assert core_loc.parent == container_loc.parent

    def test_independent_videos_not_moved(self, move_dir):
        run_move(str(move_dir["dir"]))
        for path in move_dir["independent"]:
            assert (move_dir["dir"] / path.name).exists()

    def test_groups_in_separate_directories(self, move_dir):
        run_move(str(move_dir["dir"]))
        duplicates = move_dir["dir"] / "duplicates"
        original, _ = move_dir["exact_pair"]
        core, _ = move_dir["partial_pair"]
        orig_loc = next(duplicates.rglob(original.name), None)
        core_loc = next(duplicates.rglob(core.name), None)
        assert orig_loc is not None and core_loc is not None
        assert orig_loc.parent != core_loc.parent


class TestDBUpdates:
    def test_moved_to_paths_updated_in_db(self, move_dir):
        run_move(str(move_dir["dir"]))
        conn = get_connection(move_dir["db_path"])
        moved_files = [str(file) for file in (move_dir["dir"] / "duplicates").rglob("*") if file.is_file()]
        moved_paths = {r["moved_to"] for r in conn.execute("SELECT moved_to FROM videos").fetchall() if r["moved_to"] is not None}
        assert sorted(moved_files) == sorted(moved_paths)

    def test_matches_marked_as_moved(self, move_dir):
        run_move(str(move_dir["dir"]))
        conn = get_connection(move_dir["db_path"])
        pending = conn.execute("SELECT COUNT(*) FROM matches WHERE moved = 0").fetchone()[0]
        assert pending == 0


class TestCheckpointing:
    def test_second_run_does_nothing(self, move_dir, capsys):
        run_move(str(move_dir["dir"]))
        capsys.readouterr()
        run_move(str(move_dir["dir"]))
        assert "No pending matches" in capsys.readouterr().out

    def test_sequential_numbering_no_gaps(self, move_dir):
        run_move(str(move_dir["dir"]))
        duplicates = move_dir["dir"] / "duplicates"
        numbers = sorted(
            int(d.name) for d in duplicates.iterdir()
            if d.is_dir() and d.name.isdigit()
        )
        assert numbers == list(range(min(numbers), max(numbers) + 1))