import faiss, os, threading, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from rich import print
from tqdm import tqdm

from matcha.db import get_connection, set_faiss_meta
from matcha.interactive import watch_for_quit

# Number of IVF cells. Rule of thumb: sqrt(N) where N is total vector count.
# This is recalculated at build time; this is just a fallback default.
_DEFAULT_NLIST = 100

# How many IVF cells to probe at query time (higher = more accurate but slower).
DEFAULT_NPROBE = 32

# Rebuild guard variables. A full rebuild is worthwhile when the trained IVF cell count has fallen well behind the ideal for the current size, or when tombstones from removals have built up. Both slow queries down; a rebuild restores the ideal shape and compacts the ID map.
_REBUILD_NLIST_GROWTH = 2.0
_REBUILD_TOMBSTONE_FRACTION = 0.25

def _print_message(stage: str, msg: str):
    ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
    tab = stage.count('.') + 1
    print_msg = f'({ts})'+'\t'*tab+f'{msg}'
    print(f'[dim]{print_msg}[/dim]')

def _hex_to_bytes(hex_str: str) -> bytes:
    """Convert a 16-character hex pHash string to 8 packed bytes."""
    return bytes.fromhex(hex_str)

def _load_all_hashes(db_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read every frame hash from the DB.
    """
    conn = get_connection(db_path)
    rows = conn.execute("""
        SELECT video_id, phash
        FROM frame_hashes
        ORDER BY video_id, timestamp
    """).fetchall()
    if not rows:
        return np.empty((0, 8), dtype=np.uint8), np.empty((0, 2), dtype=np.int64)
    vectors = np.array([list(_hex_to_bytes(r["phash"])) for r in rows], dtype=np.uint8)
    frame_counter: dict[int, int] = {}
    id_map_rows = []
    for r in rows:
        vid = r["video_id"]
        frame_counter[vid] = frame_counter.get(vid, 0)
        id_map_rows.append([vid, frame_counter[vid]])
        frame_counter[vid] += 1
    id_map = np.array(id_map_rows, dtype=np.int64)
    return vectors, id_map

def _query_batch(args: tuple) -> tuple[set[tuple[int, int]], set[int]]:
    """Process a single batch of queries. Returns (candidate pairs, video_ids covered) for this batch."""
    batch, batch_vids, index, id_map, k, threshold = args
    distances, labels = index.search(batch, k)
    pairs = set()
    for i, (dists, lbls) in enumerate(zip(distances, labels)):
        query_vid = int(batch_vids[i])
        for dist, lbl in zip(dists, lbls):
            if lbl < 0:
                continue
            if dist > threshold:
                continue
            candidate_vid = int(id_map[lbl, 0])
            if candidate_vid < 0:
                continue  # tombstoned
            if candidate_vid == query_vid:
                continue
            pair = (min(query_vid, candidate_vid), max(query_vid, candidate_vid))
            pairs.add(pair)
    covered_vids = {int(v) for v in batch_vids}
    return pairs, covered_vids

def _index_paths(index_dir: str) -> tuple[str, str]:
    """Return (faiss_path, map_path) for a given index directory."""
    return (
        os.path.join(index_dir, "frame_index.faiss"),
        os.path.join(index_dir, "frame_index_map.npy"),
    )

def _ensure_progress_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candidate_search_progress (
            video_id INTEGER PRIMARY KEY,
            queried_at REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candidate_search_params (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            threshold INTEGER NOT NULL,
            nprobe INTEGER NOT NULL
        )
    """)
    conn.commit()

def _reset_progress_if_params_changed(conn, threshold: int, nprobe: int):
    """
    Candidate-search progress is only valid for the threshold/nprobe it was
    recorded under — both affect which neighbours pass the filter. If either
    has changed since the last run, drop all progress so every video gets
    re-queried under the new parameters.
    """
    row = conn.execute(
        "SELECT threshold, nprobe FROM candidate_search_params WHERE id = 1"
    ).fetchone()
    if row is not None and (row["threshold"] != threshold or row["nprobe"] != nprobe):
        with conn:
            conn.execute("DELETE FROM candidate_search_progress")
            conn.execute("DELETE FROM candidate_search_params WHERE id = 1")
        row = None
    if row is None:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO candidate_search_params (id, threshold, nprobe) VALUES (1, ?, ?)",
                (threshold, nprobe),
            )


def get_queried_video_ids(db_path: str) -> set[int]:
    conn = get_connection(db_path)
    _ensure_progress_tables(conn)
    rows = conn.execute("SELECT video_id FROM candidate_search_progress").fetchall()
    return {row["video_id"] for row in rows}


def mark_videos_queried(db_path: str, video_ids) -> None:
    if not video_ids:
        return
    conn = get_connection(db_path)
    now = time.time()
    with conn:
        conn.executemany(
            "INSERT OR REPLACE INTO candidate_search_progress (video_id, queried_at) VALUES (?, ?)",
            [(vid, now) for vid in video_ids],
        )


def _video_aligned_batch_bounds(query_video_ids: np.ndarray, batch_size: int) -> list[tuple[int, int]]:
    """
    Return (start, end) index bounds into the flat vector/id arrays such
    that each batch holds at least batch_size frames, but a single video's
    frames are never split across two batches. Rows must already be ordered
    by (video_id, timestamp). This keeps "mark this video as queried"
    accurate at the batch level — a video is only ever in one batch.
    """
    bounds = []
    n = len(query_video_ids)
    start = 0
    while start < n:
        end = min(start + batch_size, n)
        while end < n and query_video_ids[end] == query_video_ids[end - 1]:
            end += 1
        bounds.append((start, end))
        start = end
    return bounds

def _load_new_hashes(
    db_path: str, known_video_ids: set[int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load frame hashes for videos that are not yet in the index.

    Rows are read in the same (video_id, timestamp) order used everywhere
    else, so the appended ID-map rows line up with the order FAISS assigns to
    the new vectors.
    """
    conn = get_connection(db_path)
    rows = conn.execute(
        """
        SELECT video_id, phash
        FROM frame_hashes
        ORDER BY video_id, timestamp
        """
    ).fetchall()
    rows = [r for r in rows if r["video_id"] not in known_video_ids]
    if not rows:
        return np.empty((0, 8), dtype=np.uint8), np.empty((0, 2), dtype=np.int64)
    vectors = np.array(
        [list(_hex_to_bytes(r["phash"])) for r in rows], dtype=np.uint8
    )
    frame_counter: dict[int, int] = {}
    id_map_rows = []
    for r in rows:
        vid = r["video_id"]
        frame_counter[vid] = frame_counter.get(vid, 0)
        id_map_rows.append([vid, frame_counter[vid]])
        frame_counter[vid] += 1
    id_map = np.array(id_map_rows, dtype=np.int64)
    return vectors, id_map

def _full_build(db_path: str, index_dir: str, nprobe: int = DEFAULT_NPROBE) -> bool:
    """Train and build the index from scratch over all frame hashes."""
    conn = get_connection(db_path)
    current_count = conn.execute("SELECT COUNT(*) FROM frame_hashes").fetchone()[0]
    if current_count == 0:
        return False  # nothing to index yet

    print(f"Building FAISS index over {current_count:,} frame hashes...")
    vectors, id_map = _load_all_hashes(db_path)
    n = len(vectors)
    nlist = max(1, min(_DEFAULT_NLIST, int(n ** 0.5)))
    d = 64  # 64-bit pHash -> 64 binary dimensions
    quantiser = faiss.IndexBinaryFlat(d)
    index = faiss.IndexBinaryIVF(quantiser, d, nlist)
    index.nprobe = nprobe
    index.train(vectors)
    index.add_with_ids(vectors, np.arange(n, dtype=np.int64))

    faiss_path, map_path = _index_paths(index_dir)
    faiss.write_index_binary(index, faiss_path)
    np.save(map_path, id_map)
    set_faiss_meta(db_path, n)
    print(f"FAISS index saved ({n:,} vectors, {nlist} IVF cells).")
    return True

def update_index(db_path: str, index_dir: str, nprobe: int = DEFAULT_NPROBE) -> bool:
    """
    Bring the FAISS index up to date with the database.

    First run (no index on disk): trains and builds from all frame hashes.
    Later runs: appends only the vectors for videos not already in the index,
    with no retraining.

    Videos deleted from the database are not removed here. Their vectors stay
    in the index and are filtered out by the matcher, which skips any candidate
    whose video_id no longer exists. Returns True if the index changed on disk.
    """
    faiss_path, map_path = _index_paths(index_dir)

    # First run, or a forced rebuild (files were removed) -> full build.
    if not os.path.exists(faiss_path) or not os.path.exists(map_path):
        return _full_build(db_path, index_dir, nprobe)

    index = faiss.read_index_binary(faiss_path)
    id_map = np.load(map_path)
    known_vids = (
        {int(v) for v in id_map[:, 0].tolist() if int(v) >= 0} if id_map.size else set()
    )

    new_vectors, new_id_map = _load_new_hashes(db_path, known_vids)
    if len(new_vectors) == 0:
        return False  # already up to date

    print(f"Appending {len(new_vectors):,} new frame hashes to the FAISS index...")
    start = int(id_map.shape[0]) if id_map.size else 0
    ids = np.arange(start, start + len(new_vectors), dtype=np.int64)
    index.add_with_ids(new_vectors, ids)
    id_map = np.vstack([id_map, new_id_map]) if id_map.size else new_id_map

    faiss.write_index_binary(index, faiss_path)
    np.save(map_path, id_map)
    set_faiss_meta(db_path, len(id_map))
    print(f"FAISS index updated ({len(id_map):,} vectors total).")
    return True

def load_index(index_dir: str) -> tuple[faiss.IndexBinaryIVF, np.ndarray]:
    """Load a previously built index and its ID map from disk."""
    faiss_path = os.path.join(index_dir, "frame_index.faiss")
    map_path = os.path.join(index_dir, "frame_index_map.npy")
    if not os.path.exists(faiss_path) or not os.path.exists(map_path):
        raise FileNotFoundError(
            "FAISS index not found. Run the match command to build it first."
        )
    index = faiss.read_index_binary(faiss_path)
    id_map = np.load(map_path)
    return index, id_map

def find_candidate_pairs(
    db_path: str,
    index_dir: str,
    threshold: int = 10,
    nprobe: int = 32,
    batch_size: int = 10_000,
    workers: int = 4,
) -> tuple[set[tuple[int, int]], bool]:
    """
    Query the FAISS index to find candidate pairs using multiple threads.

    Progress is persisted incrementally: each completed batch's candidate
    pairs are written to candidate_pairs immediately, and its videos are
    marked as queried in candidate_search_progress. This means an
    interrupted run resumes where it left off, and a later run with no new
    videos skips straight past videos already searched. Press 'q' during
    the search to stop early — anything already saved stays saved.

    Returns (candidate_pairs_found_this_run, stopped_early). Pairs found in
    earlier runs are already in the candidate_pairs table and are not
    re-returned here — callers wanting the full set should read the table.
    """
    conn = get_connection(db_path)
    _ensure_progress_tables(conn)
    _reset_progress_if_params_changed(conn, threshold, nprobe)

    _print_message('2.3.1', 'Loading index...')
    index, id_map = load_index(index_dir)
    index.nprobe = nprobe

    already_queried = get_queried_video_ids(db_path)

    _print_message('2.3.2', 'Retrieved videos and perceptual hashes...')
    if already_queried:
        placeholders = ",".join(["?"] * len(already_queried))
        all_hashes_rows = conn.execute(
            f"""
            SELECT video_id, phash
            FROM frame_hashes
            WHERE video_id NOT IN ({placeholders})
            ORDER BY video_id, timestamp
            """,
            tuple(already_queried),
        ).fetchall()
    else:
        all_hashes_rows = conn.execute("""
            SELECT video_id, phash
            FROM frame_hashes
            ORDER BY video_id, timestamp
        """).fetchall()

    if not all_hashes_rows:
        _print_message('2.3.2', 'No new videos to query — all already searched.')
        return set(), False

    vectors = np.array(
        [list(_hex_to_bytes(r["phash"])) for r in all_hashes_rows], dtype=np.uint8
    )
    query_video_ids = np.array([r["video_id"] for r in all_hashes_rows], dtype=np.int64)
    k = 16  # number of nearest neighbours per query frame

    _print_message('2.3.3', 'Defining batches...')
    bounds = _video_aligned_batch_bounds(query_video_ids, batch_size)
    batch_args = [
        (vectors[s:e], query_video_ids[s:e], index, id_map, k, threshold)
        for s, e in bounds
    ]

    candidate_pairs: set[tuple[int, int]] = set()
    stopped_early = False
    stop_event = threading.Event()
    quit_thread = threading.Thread(target=watch_for_quit, args=(stop_event,), daemon=True)
    quit_thread.start()

    _print_message('2.3.4', 'Starting pair queries... (press q to stop early)')
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_query_batch, arg): arg for arg in batch_args}
            with tqdm(total=len(futures), desc="Querying FAISS index", unit="batch", dynamic_ncols=True) as bar:
                for future in as_completed(futures):
                    if stop_event.is_set():
                        # Cancel queued futures — in-flight ones finish but
                        # their results are discarded, same as Pass 2's pattern
                        for f in futures:
                            f.cancel()
                        stopped_early = True
                        break
                    batch_pairs, batch_vids = future.result()
                    candidate_pairs.update(batch_pairs)
                    if batch_pairs:
                        with conn:
                            conn.executemany(
                                'INSERT OR IGNORE INTO candidate_pairs (video_a_id, video_b_id) VALUES (?, ?)',
                                list(batch_pairs),
                            )
                    mark_videos_queried(db_path, batch_vids)
                    bar.update(1)
    finally:
        stop_event.set()  # signal quit thread to exit if search finished normally

    if stopped_early:
        _print_message('2.3.4', 'Stopped early. Progress saved — re-run to continue.')

    return candidate_pairs, stopped_early

def remove_videos_from_index(index_dir: str, video_ids) -> int:
    """
    Remove every vector belonging to the given video_ids from the FAISS index, in place, and tombstone those rows in the ID map. Returns the number of vectors removed. No-op (returns 0) if there is no index yet.
    """
    faiss_path, map_path = _index_paths(index_dir)
    if not os.path.exists(faiss_path) or not os.path.exists(map_path):
        return 0

    wanted = {int(v) for v in video_ids}
    if not wanted:
        return 0

    index = faiss.read_index_binary(faiss_path)
    id_map = np.load(map_path)
    if id_map.size == 0:
        return 0

    mask = np.isin(id_map[:, 0], list(wanted)) & (id_map[:, 0] >= 0)
    labels = np.where(mask)[0].astype(np.int64)
    if labels.size == 0:
        return 0

    try:
        selector = faiss.IDSelectorBatch(labels)
    except TypeError:  # older faiss constructor signature
        selector = faiss.IDSelectorBatch(len(labels), faiss.swig_ptr(labels))
    n_removed = index.remove_ids(selector)

    id_map[mask, 0] = -1  # tombstone; keeps positions aligned with FAISS IDs

    faiss.write_index_binary(index, faiss_path)
    np.save(map_path, id_map)
    db_path = os.path.join(index_dir, "index.db")
    set_faiss_meta(db_path, int(index.ntotal))

    conn = get_connection(db_path)
    _ensure_progress_tables(conn)
    placeholders = ",".join(["?"] * len(wanted))
    with conn:
        conn.execute(
            f"DELETE FROM candidate_search_progress WHERE video_id IN ({placeholders})",
            tuple(wanted),
        )

    return int(n_removed)

def rebuild_recommended(index, id_map) -> tuple[bool, str]:
    """Return (should_rebuild, human_reason) for the given index and ID map."""
    live = int(index.ntotal)
    total_rows = int(id_map.shape[0]) if id_map.size else 0
    if total_rows == 0 or live == 0:
        return False, ""

    trained_nlist = int(index.nlist)
    ideal_nlist = max(1, min(_DEFAULT_NLIST, int(live ** 0.5)))
    if ideal_nlist > trained_nlist and ideal_nlist >= _REBUILD_NLIST_GROWTH * trained_nlist:
        return True, f"IVF cells {trained_nlist} -> {ideal_nlist} for {live:,} live vectors"

    tombstone_fraction = (total_rows - live) / total_rows
    if tombstone_fraction >= _REBUILD_TOMBSTONE_FRACTION:
        return True, f"{tombstone_fraction:.0%} of the index is tombstoned"

    return False, ""

def maybe_rebuild_index(db_path: str, index_dir: str, nprobe: int = DEFAULT_NPROBE) -> bool:
    """
    Rebuild the index from scratch if it has drifted from its ideal shape.

    A no-op in the common case; it only fires after a lot of growth or a lot of
    removals. Reads the current frame hashes from the DB, so deleted videos
    (whose rows are gone) drop out and their tombstones are compacted away.
    Returns True if a rebuild happened.
    """
    faiss_path, map_path = _index_paths(index_dir)
    if not os.path.exists(faiss_path) or not os.path.exists(map_path):
        return False

    index = faiss.read_index_binary(faiss_path)
    id_map = np.load(map_path)
    should, reason = rebuild_recommended(index, id_map)
    if not should:
        return False

    print(f"Rebuilding FAISS index ({reason})...")
    return _full_build(db_path, index_dir, nprobe)