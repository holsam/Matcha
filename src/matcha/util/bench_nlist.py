"""
Standalone benchmark: times training, building, and searching a binary IVF
index at a given nlist against your real frame-hash data. Does not touch
the live index files on disk, so it's safe to run before committing to a
full rebuild.

Usage:
    python bench_nlist.py /path/to/project/directory --nlist 65536
    python bench_nlist.py /path/to/project/directory --nlist 32768 --nprobe 32
"""
import argparse, os, threading, time
import numpy as np
import faiss

from matcha.faiss_index import _load_all_hashes


def _heartbeat(stop_event: threading.Event, label: str, interval: float = 30.0):
    """Prints a progress line every `interval` seconds so a long-running
    step never looks frozen, regardless of whether FAISS itself is verbose."""
    start = time.time()
    while not stop_event.wait(interval):
        print(f"  ...still {label} ({time.time() - start:.0f}s elapsed)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="Project directory containing .matcha/")
    parser.add_argument("--nlist", type=int, default=65536)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument(
        "--niter",
        type=int,
        default=25,
        help="k-means iterations for training. FAISS default is 25. "
             "Drop this to e.g. 5 for a fast, rough throughput read "
             "before committing to a full training run.",
    )
    args = parser.parse_args()

    matcha_dir = os.path.join(os.path.abspath(args.directory), ".matcha")
    db_path = os.path.join(matcha_dir, "index.db")

    print(f"Loading frame hashes from {db_path}...")
    t0 = time.time()
    vectors, id_map = _load_all_hashes(db_path)
    n = len(vectors)
    print(f"Loaded {n:,} vectors in {time.time() - t0:.1f}s")

    d = 64
    nlist = args.nlist
    print(f"\nTraining IndexBinaryIVF with nlist={nlist:,} (flat quantizer)...")

    quantizer = faiss.IndexBinaryFlat(d)
    index = faiss.IndexBinaryIVF(quantizer, d, nlist)

    # Try to reduce k-means iterations and enable FAISS's own verbose
    # logging. Wrapped defensively since the exact attribute isn't
    # confirmed to exist on IndexBinaryIVF across all FAISS versions —
    # if it's not there, we still get the heartbeat below regardless.
    try:
        index.cp.niter = args.niter
        index.cp.verbose = True
        index.verbose = True
    except AttributeError:
        print("(Could not set FAISS-native verbose/niter controls on this "
              "build — relying on the heartbeat below instead.)")

    # FAISS wants roughly 30x-256x nlist training vectors. Cap the sample
    # so training itself doesn't become the bottleneck for very large nlist.
    train_target = min(n, max(nlist * 40, 100_000))
    if train_target < n:
        rng = np.random.default_rng(0)
        sample_idx = rng.choice(n, size=train_target, replace=False)
        train_vectors = vectors[sample_idx]
    else:
        train_vectors = vectors
    print(f"Training on {len(train_vectors):,} sampled vectors, niter={args.niter}...")

    stop_event = threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(stop_event, "training"), daemon=True)
    hb.start()
    t0 = time.time()
    index.train(train_vectors)
    train_time = time.time() - t0
    stop_event.set()
    hb.join()
    print(f"Trained in {train_time:.1f}s")

    print(f"\nAdding all {n:,} vectors...")
    stop_event = threading.Event()
    hb = threading.Thread(target=_heartbeat, args=(stop_event, "adding vectors"), daemon=True)
    hb.start()
    t0 = time.time()
    index.add_with_ids(vectors, np.arange(n, dtype=np.int64))
    add_time = time.time() - t0
    stop_event.set()
    hb.join()
    print(f"Added in {add_time:.1f}s")

    build_time = train_time + add_time
    print(f"\nTotal build time: {build_time:.1f}s ({build_time / 60:.1f} min)")

    # Search timing: a sample of real queries at the candidate nprobe, to
    # project total time for a full Pass 1 search at this nlist.
    nprobe = args.nprobe
    index.nprobe = nprobe
    k = args.k
    sample_size = min(10_000, n)
    rng = np.random.default_rng(1)
    query_idx = rng.choice(n, size=sample_size, replace=False)
    queries = vectors[query_idx]

    print(f"\nTiming search: {sample_size:,} queries, nprobe={nprobe}, k={k}...")
    t0 = time.time()
    index.search(queries, k)
    search_time = time.time() - t0
    qps = sample_size / search_time
    print(f"Searched {sample_size:,} queries in {search_time:.1f}s ({qps:.0f} queries/sec)")
    print(
        f"Projected time for all {n:,} queries: "
        f"{n / qps / 60:.1f} min ({n / qps / 3600:.1f} hours)"
    )


if __name__ == "__main__":
    main()