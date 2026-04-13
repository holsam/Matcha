# Testing
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
## Fixture hierarchy
```
video_dir (session)        — generates video files once per test session
    └── indexed_dir (module)   — runs matcha index
            └── matched_dir (module)  — runs matcha match
```
Tests that need an isolated DB (e.g. `test_filter_length`) copy the indexed directory into `tmp_path` and manipulate the DB directly.

## generate_test_videos.py
A `--seed` option controls reproducibility.

```bash
# Generate 30 videos with 2 exact and 2 partial match pairs
uv run python generate_test_videos.py --count 30 --exact-matches 2 --partial-matches 2

# Same output every time
uv run python generate_test_videos.py --count 30 --exact-matches 2 --partial-matches 2 --seed 42
```