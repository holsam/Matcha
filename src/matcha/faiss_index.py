import faiss, os, threading, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from rich import print
from tqdm import tqdm

from matcha.db import get_connection, init_schema, set_faiss_meta
from matcha.interactive import watch_for_quit

# Number of IVF cells. Rule of thumb: sqrt(N) where N is total vector count.
# This is recalculated at build time; this is just a fallback default.
_NLIST_MULTIPLIER = 1.5
_MIN_NLIST = 100

# How many IVF cells to probe at query time (higher = more accurate but slower).
DEFAULT_NPROBE = 32

# Rebuild guard variables. A full rebuild is worthwhile when the trained IVF cell count has fallen well behind the ideal for the current size, or when tombstones from removals have built up. Both slow queries down; a rebuild restores the ideal shape and compacts the ID map.
_REBUILD_NLIST_GROWTH = 2.0
_REBUILD_TOMBSTONE_FRACTION = 0.25

def _target_nlist(n: int) -> int:
    return min(n, max(_MIN_NLIST, int(_NLIST_MULTIPLIER * (n ** 0.5))))  # FAISS needs n >= nlist

def _print_message(stage: str, msg: str):
    ts = datetime.now(timezone.utc).strftime('%H:%M:%S')
    tab = stage.count('.') + 1
    print_msg = f'({ts})'+'\t'*tab+f'{msg}'
    print(f'[dim]{print_msg}[/dim]')

# _train_sample_size: IVF training needs roughly 40+ points per cell, not every vector
def _train_sample_size(nlist: int) -> int:
    return max(nlist * 40, 100_000)

# _hashes_to_vectors: 16-char hex pHashes to an (n, 8) uint8 array for FAISS
def _hashes_to_vectors(phashes: list[str]) -> np.ndarray:
    if not phashes:
        return np.empty((0, 8), dtype=np.uint8)
    # bytearray so the array is writeable and contiguous
    return np.frombuffer(bytearray(bytes.fromhex(''.join(phashes))), dtype=np.uint8).reshape(-1, 8)

# _build_id_map: (video_id, frame number) per row; rows must be ordered so each video is contiguous
def _build_id_map(video_ids: np.ndarray) -> np.ndarray:
    if len(video_ids) == 0:
        return np.empty((0, 2), dtype=np.int64)
    run_start = np.flatnonzero(np.r_[True, video_ids[1:] != video_ids[:-1]])
    run_len = np.diff(np.r_[run_start, len(video_ids)])
    frame_no = np.arange(len(video_ids)) - np.repeat(run_start, run_len)
    return np.column_stack([video_ids, frame_no]).astype(np.int64)

# _split_rows: (video_id, phash) rows to (video id array, FAISS vectors)
def _split_rows(rows) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.empty(0, dtype=np.int64), np.empty((0, 8), dtype=np.uint8)
    video_ids, phashes = zip(*rows)
    return np.array(video_ids, dtype=np.int64), _hashes_to_vectors(list(phashes))

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
    video_ids, vectors = _split_rows(rows)
    return vectors, _build_id_map(video_ids)

def _query_batch(args: tuple) -> tuple[set[tuple[int, int]], set[int]]:
    """Process a single batch of queries. Returns (candidate pairs, video_ids covered) for this batch."""
    batch, batch_vids, index, id_map, k, threshold = args
    distances, labels = index.search(batch, k)
    valid = (labels >= 0) & (distances <= threshold)
    query_vids = np.broadcast_to(batch_vids[:, None], labels.shape)[valid]
    candidate_vids = id_map[labels[valid], 0]
    keep = (candidate_vids >= 0) & (candidate_vids != query_vids)  # drop tombstoned and same-video hits
    lo = np.minimum(query_vids[keep], candidate_vids[keep])
    hi = np.maximum(query_vids[keep], candidate_vids[keep])
    # pack each pair into one int so np.unique dedupes it; video ids fit in 32 bits
    packed = np.unique((lo << 32) | hi)
    pairs = set(zip((packed >> 32).tolist(), (packed & 0xFFFFFFFF).tolist()))
    return pairs, set(np.unique(batch_vids).tolist())

def _index_paths(index_dir: str) -> tuple[str, str]:
    """Return (faiss_path, map_path) for a given index directory."""
    return (
        os.path.join(index_dir, "frame_index.faiss"),
        os.path.join(index_dir, "frame_index_map.npy"),
    )

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
    the new vectors. Known videos are filtered out in SQL via a temp table.
    """
    conn = get_connection(db_path)
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS known_videos (video_id INTEGER PRIMARY KEY)")
    conn.execute("DELETE FROM known_videos")
    conn.executemany("INSERT INTO known_videos VALUES (?)", [(v,) for v in known_video_ids])
    rows = conn.execute(
        """
        SELECT fh.video_id, fh.phash
        FROM frame_hashes fh
        LEFT JOIN known_videos k ON k.video_id = fh.video_id
        WHERE k.video_id IS NULL
        ORDER BY fh.video_id, fh.timestamp
        """
    ).fetchall()
    video_ids, vectors = _split_rows(rows)
    return vectors, _build_id_map(video_ids)

def _full_build(db_path: str, index_dir: str, nprobe: int = DEFAULT_NPROBE) -> bool:
    """Train and build the index from scratch over all frame hashes."""
    conn = get_connection(db_path)
    current_count = conn.execute("SELECT COUNT(*) FROM frame_hashes").fetchone()[0]
    if current_count == 0:
        return False  # nothing to index yet

    _print_message('2.2', f"Building FAISS index over {current_count:,} frame hashes...")
    vectors, id_map = _load_all_hashes(db_path)
    n = len(vectors)
    nlist = _target_nlist(n)
    d = 64  # 64-bit pHash -> 64 binary dimensions
    quantiser = faiss.IndexBinaryFlat(d)
    index = faiss.IndexBinaryIVF(quantiser, d, nlist)
    index.nprobe = nprobe
    train_n = _train_sample_size(nlist)
    if train_n < n:
        # train on a random sample, then add every vector
        sample = np.sort(np.random.default_rng(0).choice(n, size=train_n, replace=False))
        index.train(vectors[sample])
    else:
        index.train(vectors)
    index.add_with_ids(vectors, np.arange(n, dtype=np.int64))

    faiss_path, map_path = _index_paths(index_dir)
    faiss.write_index_binary(index, faiss_path)
    np.save(map_path, id_map)
    set_faiss_meta(db_path, n)
    _print_message('2.2', f"FAISS index saved ({n:,} vectors, {nlist} IVF cells).")
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

    _print_message('2.2', f"Appending {len(new_vectors):,} new frame hashes to the FAISS index...")
    start = int(id_map.shape[0]) if id_map.size else 0
    ids = np.arange(start, start + len(new_vectors), dtype=np.int64)
    index.add_with_ids(new_vectors, ids)
    id_map = np.vstack([id_map, new_id_map]) if id_map.size else new_id_map

    faiss.write_index_binary(index, faiss_path)
    np.save(map_path, id_map)
    set_faiss_meta(db_path, len(id_map))
    _print_message('2.2', f"FAISS index updated ({len(id_map):,} vectors total).")
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
    init_schema(db_path)  # DBs predating the progress tables lack them
    _reset_progress_if_params_changed(conn, threshold, nprobe)

    _print_message('2.3.1', 'Loading index...')
    index, id_map = load_index(index_dir)
    index.nprobe = nprobe

    _print_message('2.3.2', 'Retrieving videos and perceptual hashes...')

    all_hashes_rows = conn.execute("""
        SELECT fh.video_id, fh.phash
        FROM frame_hashes fh
        LEFT JOIN candidate_search_progress p ON p.video_id = fh.video_id
        WHERE p.video_id IS NULL
        ORDER BY fh.video_id, fh.timestamp
    """).fetchall()

    if not all_hashes_rows:
        _print_message('2.3.2', 'No new videos to query — all already searched.')
        return set(), False

    query_video_ids, vectors = _split_rows(all_hashes_rows)
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

    init_schema(db_path)  # DBs predating the progress tables lack them
    conn = get_connection(db_path)
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
    ideal_nlist = _target_nlist(live)
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

    _print_message('2.2', f"Rebuilding FAISS index ({reason})...")
    return _full_build(db_path, index_dir, nprobe)