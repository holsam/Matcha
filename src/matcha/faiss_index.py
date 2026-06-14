import faiss, os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from tqdm import tqdm

from .db import get_connection, set_faiss_meta

# Number of IVF cells. Rule of thumb: sqrt(N) where N is total vector count.
# This is recalculated at build time; this is just a fallback default.
_DEFAULT_NLIST = 100

# How many IVF cells to probe at query time (higher = more accurate but slower).
DEFAULT_NPROBE = 32

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

def _query_batch(args: tuple) -> set[tuple[int, int]]:
    """Process a single batch of queries. Returns candidate pairs found in this batch."""
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
            if candidate_vid == query_vid:
                continue
            pair = (min(query_vid, candidate_vid), max(query_vid, candidate_vid))
            pairs.add(pair)
    return pairs

def _index_paths(index_dir: str) -> tuple[str, str]:
    """Return (faiss_path, map_path) for a given index directory."""
    return (
        os.path.join(index_dir, "frame_index.faiss"),
        os.path.join(index_dir, "frame_index_map.npy"),
    )

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
    index.add(vectors)

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
    known_vids = {int(v) for v in id_map[:, 0].tolist()} if id_map.size else set()

    new_vectors, new_id_map = _load_new_hashes(db_path, known_vids)
    if len(new_vectors) == 0:
        return False  # already up to date

    print(f"Appending {len(new_vectors):,} new frame hashes to the FAISS index...")
    index.add(new_vectors)
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
) -> set[tuple[int, int]]:
    """
    Query the FAISS index to find candidate pairs using multiple threads.
    """
    _print_message('2.3.1','Loading index...')
    index, id_map = load_index(index_dir)
    index.nprobe = nprobe
    conn = get_connection(db_path)
    all_hashes_rows = conn.execute("""
        SELECT video_id, phash
        FROM frame_hashes
        ORDER BY video_id, timestamp
    """).fetchall()
    if not all_hashes_rows:
        return set()
    _print_message('2.3.2','Retrieved videos and perceptual hashes...')
    vectors = np.array(
        [list(_hex_to_bytes(r["phash"])) for r in all_hashes_rows], dtype=np.uint8
    )
    query_video_ids = np.array([r["video_id"] for r in all_hashes_rows], dtype=np.int64)
    k = 16  # number of nearest neighbours per query frame
    # Prepare batch arguments
    _print_message('2.3.3','Defining batches...')
    batch_args = []
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start : start + batch_size]
        batch_vids = query_video_ids[start : start + batch_size]
        batch_args.append((batch, batch_vids, index, id_map, k, threshold))
    candidate_pairs: set[tuple[int, int]] = set()
    _print_message('2.3.4','Starting pair queries...')
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for batch_result in tqdm(
            executor.map(_query_batch, batch_args),
            total=len(batch_args),
            desc="Querying FAISS index",
            unit="batch",
            dynamic_ncols=True,
        ):
            candidate_pairs.update(batch_result)
    return candidate_pairs