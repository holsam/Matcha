"""Tests for matcha.matcher (matcha match subcommand)."""

import pytest, shutil

from matcha.db import get_connection
from matcha.matcher import run_match
from .conftest import get_match, get_video_id


class TestMatchDetection:
    def test_exact_match_detected(self, matched_dir):
        d = matched_dir["dir"]
        db_path = str(d / ".matcha" / "index.db")
        original, copy = matched_dir["exact_pair"]
        match = get_match(db_path, get_video_id(db_path, original), get_video_id(db_path, copy))
        assert match is not None, "Exact match pair not detected"
        assert match["match_type"] == "duplicate"
        assert match["confidence"] > 0.95

    def test_subclip_detected(self, matched_dir):
        d = matched_dir["dir"]
        db_path = str(d / ".matcha" / "index.db")
        core, container = matched_dir["partial_pair"]
        match = get_match(db_path, get_video_id(db_path, core), get_video_id(db_path, container))
        assert match is not None, "Subclip not detected"
        assert match["match_type"] == "subclip"

    def test_independent_videos_not_matched(self, matched_dir):
        d = matched_dir["dir"]
        db_path = str(d / ".matcha" / "index.db")
        ids = [get_video_id(db_path, p) for p in matched_dir["independent"]]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                assert get_match(db_path, ids[i], ids[j]) is None, (
                    f"False match between independent videos {ids[i]} and {ids[j]}"
                )

    def test_independent_not_matched_against_pairs(self, matched_dir):
        d = matched_dir["dir"]
        db_path = str(d / ".matcha" / "index.db")
        ind_ids = [get_video_id(db_path, p) for p in matched_dir["independent"]]
        original, _ = matched_dir["exact_pair"]
        core, container = matched_dir["partial_pair"]
        pair_ids = [get_video_id(db_path, p) for p in [original, core, container]]
        for ind_id in ind_ids:
            for pair_id in pair_ids:
                assert get_match(db_path, ind_id, pair_id) is None, (
                    f"False match between independent {ind_id} and pair video {pair_id}"
                )


class TestCheckpointing:
    def test_second_run_adds_no_new_comparisons(self, matched_dir):
        d = matched_dir["dir"]
        db_path = str(d / ".matcha" / "index.db")
        conn = get_connection(db_path)
        before = conn.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0]
        run_match(str(d), window=5.0, frame_step=1, threshold=10, min_confidence=0.8)
        after = conn.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0]
        assert before == after


class TestFilterLength:
    def test_filter_reduces_or_equals_comparison_count(self, indexed_dir, tmp_path):
        fresh = tmp_path / "videos"
        shutil.copytree(indexed_dir["dir"], fresh)
        db_path = str(fresh / ".matcha" / "index.db")
        conn = get_connection(db_path)

        conn.execute("DELETE FROM comparisons")
        conn.execute("DELETE FROM matches")
        conn.commit()
        run_match(str(fresh), filter_length=True, window=5.0, frame_step=1,
                  threshold=10, min_confidence=0.8)
        count_filtered = conn.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0]

        conn.execute("DELETE FROM comparisons")
        conn.execute("DELETE FROM matches")
        conn.commit()
        run_match(str(fresh), filter_length=False, window=5.0, frame_step=1,
                  threshold=10, min_confidence=0.8)
        count_unfiltered = conn.execute("SELECT COUNT(*) FROM comparisons").fetchone()[0]

        assert count_filtered <= count_unfiltered