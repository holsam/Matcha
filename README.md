<!-- shields -->
<div align="right">

![Version][version-shield]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]

</div>

<br>

<!-- logo -->
<div align="center">
    <img src="images/logo.png" alt="Matcha Logo" width="240" height="240">
  </a>
</div>

# Matcha
Matcha is a CLI tool for finding duplicate and near-duplicate videos in a directory, including sub-clip detection across different qualities and lengths. It can be installed using the Python `uv` package manager as described in [Installation](#installation).

Matcha provides three subcommands:
```sh
matcha index <directory>   Fingerprint all videos and populate the index
matcha match <directory>   Compare indexed videos and record matches
matcha move  <directory>   Move matched videos into duplicates/ subdirectories
```
All three subcommands are checkpointed — if interrupted, they pick up where they left off on the next run. `match` and `move` are deliberately separate so you can review what Matcha found before anything is touched on disk. For more information on using the subcommands, see the relevant section of [Subcommand Usage](#subcommand-usage).

## Installation
### Dependencies
#### System dependencies (accessible on `$PATH`):
- `ffmpeg`
- `Python` (version ≥3.14)
- `uv`
- (optional) `chromaprint`
#### Python dependencies:
- `imagehash`
- `Pillow`
- `pyacoustid`
- `tqdm`
- `typer`
These can be installed using: `uv add imagehash Pillow pyacoustid tqdm typer`.

### Installation
Matcha can be downloaded from this repository using: `git clone https://github.com/holsam/Matcha.git`.

Then either run using: `uvx matcha --help` 
OR to fully install Matcha use the following commands:
```sh
# Fully install Matcha
cd Matcha
uv tool install .
# Use Matcha
matcha --help
```

## Subcommand Usage
### `matcha index`
#### CLI options and usage
| Option  | Default | Description |
|--------|---------|-------------|
| `--fps`  | `1.0` | Frame sample rate for pHash extraction |
| `--workers` | `4` | Parallel indexing workers |
| `--no-audio` | n/a | Skip audio fingerprinting |
| `--hwaccel` | n/a | Use GPU/hardware-accelerated decoding | 

```sh
# Index a directory with defaults (1fps, 4 workers)
uv run matcha index /path/to/Videos

# Index with custom settings
uv run matcha index /path/to/Videos --fps 0.5 --workers 8

# Interrupt with Ctrl+C and resume — already-indexed files are skipped automatically
uv run matcha index /path/to/Videos
```
#### Explanation
1. Walks `<directory>` recursively for video files (.mp4, .mkv, .avi, .mov, .wmv, .flv, .webm), skipping the .matcha/ directory itself
2. Registers each file in the videos table (skips if already present)
3. Skips files that already have fingerprinted_at set
4. For each unprocessed file:
    1. Extracts frames at a configurable rate (default: 1fps) into a temp directory on disk, scaling them from high-resolution to reduce the decode overhead
    2. Computes a perceptual hash (pHash) for each frame immediately after writing, then deletes the frame file
    3. Attempts audio fingerprinting via Chromaprint; skips gracefully if no audio track is found
    4. Writes frame hashes and audio fingerprint to the DB
    5. Sets fingerprinted_at to mark the file as complete
5. Processes files in parallel using ThreadPoolExecutor

Frames are written to a temp directory on disk rather than held in RAM. Peak memory per worker is one frame at a time (~1–5 MB depending on resolution), regardless of video length. With 4 workers running in parallel, frame data contributes roughly 20 MB total — negligible. The temp directory for each video is cleaned up automatically after processing, even if an error occurs.

To improve speed, the `--no-audio` and `--hwaccel` flags can be used. The former will cause the audio fingerprinting used for validation be skipped, while the latter will offload frame decoding to the local machine's GPU (on Mac, this is via VideoToolbox).

### `matcha match`
#### CLI options and usage
| Option | Default | Description |
|--------|---------|-------------|
| `--filter-length` | off | Skip pairs with identical durations |
| `--window` |  `10.0` | Minimum duration (seconds) for a video to be compared |
| `--frame-step` |  `3` | Step size when sliding the comparison window |
| `--threshold` |  `10` | Max Hamming distance to count a frame pair as matching (0–64) |
| `--min-confidence` |  `0.8` | Minimum match ratio to record a result (0.0–1.0) |
| `--workers` |  `4` | Parallel comparison workers |

```sh
# Compare all indexed videos with defaults
uv run matcha match /path/to/Videos

# Only compare pairs where durations differ
uv run matcha match /path/to/Videos --filter-length

# Stricter threshold, larger window step
uv run matcha match /path/to/Videos --threshold 6 --frame-step 5

# Resume a previous run — already-compared pairs are skipped automatically
uv run matcha match /path/to/Videos
```

#### Explanation
1. Loads all fully indexed videos and their frame hashes from the DB
2. Generates all pairs to compare — by default every combination; with --filter-length, only pairs where one video is strictly longer than the other
3. Skips pairs already recorded in the comparisons table (checkpoint)
4. Separates remaining pairs into eligible (duration ≥ --window) and too-short; marks too-short pairs as compared immediately
5. Submits eligible pairs to a ProcessPoolExecutor — the CPU-bound sliding window runs in parallel across workers
6. As results come back, the main process records comparisons and writes any matches to the DB, printing matched pairs above the progress bar via tqdm.write
7. Prints a summary on completion

The shorter video's frame hashes are slid across the longer video's hash sequence in steps of --frame-step. At each position, every frame pair is compared using Hamming distance. The proportion of frames below --threshold is the match ratio for that window position. The best ratio across all positions is the confidence score for the pair.

A match is classified as:
- duplicate: the shorter video is ≥95% the duration of the longer one
- subclip: the shorter video is <95% the duration of the longer one

### `matcha move`
#### CLI options and usage
| Option | Default | Description |
|--------- | --- |-------------|
| `--dry-run` | off | Preview without moving any files |
```sh
# Preview what would be moved
uv run matcha move /path/to/Videos --dry-run

# Move confirmed matches
uv run matcha move /path/to/Videos

# Safe to run again — already-moved matches are skipped
uv run matcha move /path/to/Videos
```
#### Explanation
1. Loads all matches where moved = 0 from the DB
2. Forms groups using union-find — if A matches B and A matches C, all three end up in the same group
3. In --dry-run mode, prints a summary and exits without touching anything on disk
4. Otherwise, creates `<directory>/duplicates/` and a sequentially numbered subdirectory for each group
5. Moves each video in the group into its subdirectory, avoiding filename collisions by appending the video ID if needed
6. Updates the path column in the videos table for every moved file
7. Sets moved = 1 on all resolved match rows

Numbering continues from where it left off — if duplicates/1/ and duplicates/2/ already exist, the next run starts at duplicates/3/.

`matcha move` uses union-find as matches are stored as pairs, but groups can be larger. Union-find computes the transitive closure efficiently: it processes each pair in O(α(n)) time (effectively constant), so even a large match table resolves instantly.

## Getting Help & Contributing
If you come across any bugs/issues while using Matcha, or if you have a feature request, please open an issue [here][issues-url].

Any contributions to this project are also very welcome! To contribute, please fork the repo, commit any changes, and then open a pull request. More information on the tests is provided in the `README.md` file under `tests/`.

## License

This repository is distributed under the GPL-3.0 license. See [LICENSE][license-url] for more information.

## Database Schema
The SQLite database lives at `<target_dir>/.matcha/index.db`. It is created automatically on the first run of matcha index.
```sql
-- One row per video file
CREATE TABLE IF NOT EXISTS videos (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    path             TEXT UNIQUE NOT NULL,   -- absolute path
    duration         REAL,                   -- seconds
    fingerprinted_at REAL                    -- unix timestamp; NULL = not yet done
);

-- One row per sampled frame
CREATE TABLE IF NOT EXISTS frame_hashes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    INTEGER NOT NULL REFERENCES videos(id),
    timestamp   REAL NOT NULL,              -- seconds from start of video
    phash       TEXT NOT NULL               -- 64-bit hex pHash string
);

-- One row per video that has a detectable audio track
CREATE TABLE IF NOT EXISTS audio_fingerprints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id    INTEGER NOT NULL REFERENCES videos(id) UNIQUE,
    duration    REAL NOT NULL,
    fingerprint TEXT NOT NULL               -- raw Chromaprint fingerprint string
);

-- Tracks which pairs have already been compared (for checkpointing)
CREATE TABLE IF NOT EXISTS comparisons (
    video_a_id  INTEGER NOT NULL REFERENCES videos(id),
    video_b_id  INTEGER NOT NULL REFERENCES videos(id),
    PRIMARY KEY (video_a_id, video_b_id)    -- always stored as (lower_id, higher_id)
);

-- Confirmed matches written by `matcha match`
CREATE TABLE IF NOT EXISTS matches (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    video_a_id  INTEGER NOT NULL REFERENCES videos(id),
    video_b_id  INTEGER NOT NULL REFERENCES videos(id),
    match_type  TEXT NOT NULL,              -- 'duplicate' | 'subclip'
    confidence  REAL NOT NULL,              -- 0.0–1.0
    found_at    REAL NOT NULL,              -- unix timestamp
    moved       INTEGER NOT NULL DEFAULT 0  -- 0 = pending, 1 = moved
);
```
<br>

---
<p align="right"><a href="#matcha">^ Back to top</a></p>

<!-- MARKDOWN LINKS & IMAGES -->
[version-shield]: https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/holsam/Matcha/refs/heads/main/pyproject.toml&query=$.project.version&style=for-the-badge&label=Current%20version&color=important
[issues-shield]: https://img.shields.io/github/issues/holsam/Matcha.svg?style=for-the-badge&color=critical
[issues-url]: https://github.com/holsam/Matcha/issues
[license-shield]: https://img.shields.io/github/license/holsam/Matcha.svg?style=for-the-badge&color=informational
[license-url]: https://github.com/holsam/Matcha/blob/main/LICENSE
