"""
Shared fixtures and video generation helpers for Matcha tests.
"""

import math, pytest, shutil, subprocess
from pathlib import Path

from matcha.db import get_connection
from matcha.indexer import run_index
from matcha.matcher import run_match


@pytest.fixture(scope="session")
def video_dir():
    """
    Use pre-generated OpenCV videos from output_videos/.

    Assumes manifest.csv defines relationships.
    """
    d = Path("tests/output_videos").resolve()

    assert d.exists(), "output_videos directory not found"
    manifest = d / "manifest.csv"
    assert manifest.exists(), "manifest.csv not found"

    # Load manifest
    import csv

    exact_pairs = []
    partial_pairs = []
    all_videos = []
    paired = set()

    with open(manifest) as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = d / row["video_a"]
            b = d / row["video_b"]
            rel = row["relationship"]

            all_videos.extend([a, b])

            if rel == "exact":
                exact_pairs.append((a, b))
                paired.update([a, b])
            elif rel == "partial":
                partial_pairs.append((a, b))
                paired.update([a, b])

    # Deduplicate
    all_videos = list(set(all_videos))

    # Independent videos belong to no relationship at all
    independent = [v for v in all_videos if v not in paired][:3]

    return {
        "dir": d,
        "independent": independent,
        # First pair of each kind, as a 2-element list so tests can unpack it
        "exact_pair": list(exact_pairs[0]) if exact_pairs else [],
        "partial_pair": list(partial_pairs[0]) if partial_pairs else [],
        # All pairs, for any test that wants to iterate
        "exact_pairs": exact_pairs,
        "partial_pairs": partial_pairs,
    }



@pytest.fixture(scope="module")
def indexed_dir(video_dir):
    run_index(str(video_dir["dir"]), fps=1.0, workers=2)
    return video_dir

@pytest.fixture(scope="module")
def indexed_dir_hwaccel(video_dir):
    run_index(str(video_dir["dir"]), fps=1.0, workers=2, hwaccel=True)
    return video_dir

@pytest.fixture(scope="module")
def indexed_dir_no_audio(video_dir):
    run_index(str(video_dir["dir"]), fps=1.0, workers=2, hwaccel=True)
    return video_dir

@pytest.fixture(scope="module")
def matched_dir(indexed_dir):
    run_match(
        str(indexed_dir["dir"]),
        window=5.0,
        frame_step=1,
        threshold=10,
        min_confidence=0.8,
    )
    return indexed_dir


def get_video_id(db_path: str, path: Path) -> int:
    conn = get_connection(db_path)
    row = conn.execute(
        "SELECT id FROM videos WHERE path = ?", (str(path),)
    ).fetchone()
    assert row is not None, f"Video not found in DB: {path}"
    return row["id"]


def get_match(db_path: str, id_a: int, id_b: int):
    conn = get_connection(db_path)
    return conn.execute(
        """
        SELECT * FROM matches
        WHERE (video_a_id = ? AND video_b_id = ?)
           OR (video_a_id = ? AND video_b_id = ?)
        """,
        (id_a, id_b, id_b, id_a),
    ).fetchone()