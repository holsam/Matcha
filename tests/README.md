# Testing

## Test file overview
File | What it covers
------|---------------
`test_indexer.py` | DB creation, video registration, frame hashing, duration storage, idempotency of repeated indexing
`test_matcher.py` | Exact match detection, subclip detection, false-positive rejection, match checkpointing, `--filter-length` behaviour
`test_mover.py` | Dry-run output, file movement, group directory assignment, DB path updates, `moved` flag, sequential numbering
`test_continuer.py` | Config saving and loading, atomic writes, staleness detection, FAISS invalidation, `run_continue` dispatch, prompting behaviour

## Running tests
Test videos should be generated before the session using `generate_test_videos.py` then reused across all tests.
```bash
# Generate test videos
cd tests
uv run generate_test_videos.py --count 30 --exact-matches 2 --partial-matches 2

# Run all tests
uv run pytest ./ -v

# Run only indexer tests
uv run pytest test_indexer.py -v

# Run only matcher tests
uv run pytest test_matcher.py -v

# Run only continuer tests
uv run pytest test_continuer.py -v

# Stop on first failure
uv run pytest ./ -x
```
## How test videos work
Independent videos use a **Lissajous ball animation**: a coloured ball following a figure-8 path on a contrasting background. The colour, ball size, and motion path are derived from a seed using golden-angle hue spacing and prime-modulo frequency selection, so each seed produces visually distinct content.
```
seed 0 → orange ball, slow wide orbit
seed 1 → teal ball, fast narrow orbit
seed 2 → purple ball, medium asymmetric orbit
...
```
Exact match pairs are byte-identical copies of the same source. Partial match pairs embed a core video between a prefix and suffix of different seeds:
```
container = [prefix 9s (seed=5)] + [core 15s (seed=4)] + [suffix 6s (seed=6)]
```
The prefix is exactly 9 seconds so the core starts at frame index 9, which is hit exactly when `frame_step=3` checks positions 0, 3, 6, **9**, 12...

### generate_test_videos.py
A `--seed` option controls reproducibility.

```bash
# Generate 30 videos with 2 exact and 2 partial match pairs
uv run python generate_test_videos.py --count 30 --exact-matches 2 --partial-matches 2

# Same output every time
uv run python generate_test_videos.py --count 30 --exact-matches 2 --partial-matches 2 --seed 42
```


## Fixture hierarchy
```
video_dir (session)        — generates video files once per test session
    └── indexed_dir (module)   — runs matcha index
            └── matched_dir (module)  — runs matcha match
```
Tests that need an isolated DB (e.g. `test_filter_length`) copy the indexed directory into `tmp_path` and manipulate the DB directly.


## `test_continuer.py`

`test_continuer.py` is largely unit-level and does not require the pre-generated test videos. Most tests build a minimal SQLite database or config file directly in `tmp_path`, which keeps them fast and isolated from the session-scoped video fixtures.

The four test classes map to the four units under test:

- **`TestConfigSaving`** — verifies that `run_index` and `run_match` write a correctly structured JSON config, that args round-trip through `load_run_config`, and that writes are atomic (temp file then `os.replace`).
- **`TestLoadConfig`** — verifies graceful handling of missing files and malformed JSON.
- **`TestGetAvailableConfigs`** — verifies that only valid configs are returned and that malformed files are silently skipped.
- **`TestStaleIndexCheck`** — verifies staleness detection by inserting videos with known `fingerprinted_at` timestamps relative to a synthetic `match.json` `last_run` value.
- **`TestInvalidateFaissIndex`** — verifies that the `.faiss` and `.npy` files are deleted and the `faiss_index_meta` table is cleared, with no error if the files are already absent.
- **`TestRunContinue`** — integration-level tests that mock `run_index` and `run_match` to verify dispatch, arg forwarding, prompting behaviour, and FAISS invalidation on a stale index.

Tests that exercise `check_index_stale` construct their own SQLite `videos` table rather than reusing the session fixture, because the staleness logic depends on precise timestamp ordering that would be fragile to control in a shared database.