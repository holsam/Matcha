import faiss, os, typer
from datetime import datetime
import numpy as np

from .db import get_connection, get_faiss_meta, set_faiss_meta

# Number of IVF cells. Rule of thumb: sqrt(N) where N is total vector count.
# This is recalculated at build time; this is just a fallback default.
_DEFAULT_NLIST = 100

# How many IVF cells to probe at query time (higher = more accurate but slower).
DEFAULT_NPROBE = 32


def _print_message(stage: str, msg: str):
    ts = datetime.now().strftime('%H:%M:%S')
    tab = stage.count('.')
    print_msg = f'\t'*tab+f'[{stage}] ({ts}) {msg}'
    typer.echo(print_msg)

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


def build_index(db_path: str, index_dir: str, nprobe: int = DEFAULT_NPROBE) -> bool:
    """
    Build (or rebuild) the FAISS index from all frame hashes currently in the DB.
    """
    conn = get_connection(db_path)
    current_count = conn.execute("SELECT COUNT(*) FROM frame_hashes").fetchone()[0]
    meta = get_faiss_meta(db_path)
    if meta and meta["vector_count"] == current_count:
        return False  # Nothing new — skip rebuild
    print(f"Building FAISS index over {current_count:,} frame hashes...")
    vectors, id_map = _load_all_hashes(db_path)
    n = len(vectors)
    nlist = max(1, min(_DEFAULT_NLIST, int(n ** 0.5)))
    d = 64  # 64-bit pHash → 64 binary dimensions
    quantiser = faiss.IndexBinaryFlat(d)
    index = faiss.IndexBinaryIVF(quantiser, d, nlist)
    index.nprobe = nprobe
    index.train(vectors)
    index.add(vectors)
    faiss_path = os.path.join(index_dir, "frame_index.faiss")
    map_path = os.path.join(index_dir, "frame_index_map.npy")
    faiss.write_index_binary(index, faiss_path)
    np.save(map_path, id_map)
    set_faiss_meta(db_path, current_count)
    print(f"FAISS index saved ({n:,} vectors, {nlist} IVF cells).")
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
    nprobe: int = DEFAULT_NPROBE,
    batch_size: int = 10_000,
) -> set[tuple[int, int]]:
    """
    Query the FAISS index to find (video_a_id, video_b_id) candidate pairs whose frame hashes are within `threshold` Hamming distance of each other. Uses batched querying so memory usage stays bounded regardless of dataset size.
    """
    index, id_map = load_index(index_dir)
    _print_message('2.3.1', 'Loaded FAISS index.')
    index.nprobe = nprobe
    conn = get_connection(db_path)
    all_hashes_rows = conn.execute("""
        SELECT video_id, phash
        FROM frame_hashes
        ORDER BY video_id, timestamp
    """).fetchall()
    _print_message('2.3.2', 'Retrieved video ids and phashes.')
    if not all_hashes_rows:
        return set()
    vectors = np.array(
        [list(_hex_to_bytes(r["phash"])) for r in all_hashes_rows], dtype=np.uint8
    )
    _print_message('2.3.3', 'Created vector list array.')
    query_video_ids = np.array([r["video_id"] for r in all_hashes_rows], dtype=np.int64)
    _print_message('2.3.4', 'Created video id array.')
    candidate_pairs: set[tuple[int, int]] = set()
    _print_message('2.3.5', 'Created candidate pair set.')
    _print_message('2.3.6', f'{len(vectors)} vectors to search using batch size {batch_size}')
    k = 16  # number of nearest neighbours to retrieve per query frame
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start : start + batch_size]
        batch_vids = query_video_ids[start : start + batch_size]
        # distances shape: (batch, k), labels shape: (batch, k)
        distances, labels = index.search(batch, k)
        for i, (dists, lbls) in enumerate(zip(distances, labels)):
            query_vid = int(batch_vids[i])
            for dist, lbl in zip(dists, lbls):
                if lbl < 0:
                    continue  # FAISS returns -1 for unfilled slots
                if dist > threshold:
                    continue
                candidate_vid = int(id_map[lbl, 0])
                if candidate_vid == query_vid:
                    continue  # skip self
                pair = (min(query_vid, candidate_vid), max(query_vid, candidate_vid))
                candidate_pairs.add(pair)
        _print_message('2.3.7', f'Vectors {start+batch_size}/{len(vectors)} queried.')
    return candidate_pairs