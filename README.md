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

Matcha provides several subcommands. `review` (browser) and `plan` + `apply` (terminal) are two ways of acting on the matches once `match` has run:
```sh
matcha index    <directory>            Fingerprint all videos and populate the index
matcha match    <directory>            Compare indexed videos and record matches
matcha review   <directory>            Open a browser UI to review duplicate groups, then delete, rename or dismiss
matcha plan     <directory>            Step through duplicate groups in the terminal and write a review plan
matcha apply    <directory>            Apply a review plan: delete and rename files and update the index (dry run by default)
matcha continue [directory] [command]  Re-run the last index or match command with the same configuration
```

All subcommands are checkpointed — if interrupted, they pick up where they left off on the next run. However `continue` is provided as a utility to continue the last `index` and `match` commands, and will continue using the same parameters as previously used. 

For more information on using the subcommands, see the relevant section of [Subcommand Usage](#subcommand-usage).

## Installation
### Dependencies
#### System dependencies (accessible on `$PATH`):
- `ffmpeg`
- `Python` (version ≥3.14)
- `uv`
- (optional) `chromaprint`
#### Python dependencies:
- `faiss-cpu`
- `imagehash`
- `numpy`
- `Pillow`
- `pyacoustid`
- `questionary`
- `rich`
- `send2trash`
- `streamlit`
- `tqdm`
- `typer`
These can be installed using: `uv add faiss-cpu imagehash numpy Pillow pyacoustid questionary rich send2trash streamlit tqdm typer`.

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
| `--nprobe` | `32` | FAISS IVF cells to probe at query time (higher = more accurate but slower) |
| `--min-votes` | `2` | Frame hits that must agree on one offset between two videos for the pair to go on to Pass 2 (`1` keeps every near-identical frame hit) |

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
2. Pass 1 (candidate generation): builds or updates a FAISS binary index over every frame pHash, then queries it to find videos that share near-identical frames. Only these candidate pairs go on to Pass 2, which avoids comparing every possible combination. New candidate pairs are checkpointed in the candidate_pairs table.
3. Pass 2 (verification): skips pairs already in the comparisons table; with `--filter-length`, also skips pairs of identical duration; separates the rest into eligible (duration ≥ `--window`) and too-short, marking too-short pairs as compared immediately
4. Runs the sliding-window comparison on eligible pairs in parallel using a ThreadPoolExecutor
5. Records comparisons and writes any matches to the DB, printing matched pairs above the progress bar via tqdm.write
6. Prints a summary on completion

The shorter video's frame hashes are slid across the longer video's hash sequence in steps of `--frame-step`. At each position, every frame pair is compared using Hamming distance. The proportion of frames below `--threshold` is the match ratio for that window position. The best ratio across all positions is the confidence score for the pair.

A match is classified as:
- duplicate: the shorter video is ≥95% the duration of the longer one
- subclip: the shorter video is <95% the duration of the longer one

#### How the FAISS index is maintained
The FAISS binary index lives at `<directory>/.matcha/frame_index.faiss`, with an ID map alongside it at `frame_index_map.npy`. It is built on the first `match` run. On later runs only the frames of newly indexed videos are appended, so the index is not rebuilt from scratch each time. Deleting a video in `review` removes its vectors in place. Occasionally, when the index has grown well past the size it was first trained for or has accumulated many removals, `match` rebuilds it once automatically to keep queries fast. `continue match` no longer forces a rebuild; new videos are picked up by the incremental update.

### `matcha review`
#### CLI options and usage
| Option | Default | Description |
|--------|---------|-------------|
| `--port` | `8501` | Port to serve the review UI on |

```sh
uv run matcha review /path/to/Videos
```
#### Explanation
Opens a Streamlit app in the browser. Files stay where they are; nothing is moved into a `duplicates/` folder.

- **Group list:** duplicate groups (connected components of the recorded matches, so A~B and B~C form one group), sortable by video count, best confidence or total duration, 25 per page. The list reads only the database, so it stays fast on very large libraries.
- **Group detail:** opening a group shows each video's thumbnail, name, path, size, duration and modified date, with optional ffprobe details (resolution, codec, bitrate) and an in-page preview. A pairwise table shows the similarity and match type (`duplicate` or `subclip`) for every matched pair.
- **Actions:** delete selected videos, keep only one video, rename a video, or dismiss the group as not duplicates. Deleting sends files to the OS trash after a confirmation, then removes the video's records and its FAISS vectors. A file that has vanished from disk is flagged and can be deleted to drop it from the index.
- **Maintenance:** a section at the top of the list. "Scan for missing files" checks every indexed path against the disk, shows what is gone, and "Remove from index" drops those records and their FAISS vectors (it never touches files, and skips any that have reappeared). It refuses to run if the library directory is unavailable and warns when over half the files are missing, since that usually means an unmounted drive. "Restore dismissed groups" brings back every group you dismissed.
- **Dismissed groups** are recorded on the match rows (`matches.dismissed`), so they stay hidden. Matches found by later `match` runs still appear.

Thumbnails are generated on first view and cached in `.matcha/thumbs/`. Libraries organised with the old `move` command are adopted on launch: each video's `duplicates/` location becomes its path.

The grouping uses union-find, which processes each pair in O(α(n)) time (effectively constant), so even a large match table resolves instantly.


### `matcha plan`
#### CLI options and usage
| Option | Default | Description |
|--------|---------|-------------|
| `--min-confidence` | `0.8` | Minimum match confidence to group on |
| `--threshold` | `10` | Max Hamming distance per frame when locating subclips (as in `match`) |
| `--no-pause` | off | Print every group without prompting; reads and writes no plan |
| `--plan` | `.matcha/review_plan.json` | Plan file to write and resume from |
| `--restart` | off | Ignore any existing plan and start again |

```sh
# Step through groups, largest potential saving first
uv run matcha plan /path/to/Videos

# Preview every group without deciding anything
uv run matcha plan /path/to/Videos --no-pause

# Throw away earlier decisions and begin again
uv run matcha plan /path/to/Videos --restart
```
#### Explanation
Shows one duplicate group at a time (the same groups as `review`, dismissed matches excluded), ordered by how much space each could free. Each group is drawn as a timeline against the longest file, with duration, resolution, codec, bitrate and size per file. Subclips are placed at the offset where they best align with the source.

- **Recommended keep:** among the full-length files (at least 95% of the longest), the one with the best quality, judged by pixels weighted by how well the bitrate feeds them and by codec efficiency, with file size as the tiebreak. It starts ticked. If the top files are indistinguishable, nothing is pre-ticked.
- **Your choice:** tick the files to keep (space toggles, Enter confirms). Anything unticked is marked for deletion. Each kept file then prompts for a name, defaulting to its current one, so a changed name becomes a rename. Subclips are never the recommended keep, so they start unticked: tick any you want to keep, or they are marked for deletion like the other unticked files. The saving figure shown for a group counts full-length duplicates only.
- **Ctrl-C** ends the session. Groups decided so far are already saved.

`plan` never touches your files or the index. It only writes the plan, atomically after every group, so an interrupted session resumes at the first undecided group. Groups are keyed by their video ids, so decisions stay valid across runs.

### `matcha apply`
#### CLI options and usage
| Option | Default | Description |
|--------|---------|-------------|
| `--plan` | `.matcha/review_plan.json` | Plan file to apply |
| `--apply` | off | Make the changes. Without it, `apply` is a dry run |
| `--yes` | off | Skip the confirmation prompt |
| `--quiet` | off | Only print the summary and failures |

```sh
# Dry run: show what would happen
uv run matcha apply /path/to/Videos

# Do it (asks for confirmation)
uv run matcha apply /path/to/Videos --apply
```
#### Explanation
Carries out a plan group by group, keeping the index consistent with the disk:

1. **Deletes first**, so a freed file name can be reused by a rename. The file is removed, then its rows are purged from every table. The database step only runs once the file step succeeded.
2. **Renames**: the file is renamed, then `videos.path` is updated. If the update fails the rename is undone.
3. **Dismissal:** when two or more files in a group survive, the matches between them are marked dismissed so the group does not come back.

Safety checks:
- It never deletes the last copy: if no surviving file in a group exists on disk, the group's deletes are skipped.
- A file whose stored path differs from the plan is skipped as stale.
- Every step is recorded in the plan as it completes, so an interrupted run can be re-run and picks up where it stopped.
- A dry run is the default, and `--apply` asks you to type `yes` unless `--yes` is given.

**Deletes are permanent** (`os.remove`), unlike `review`, which sends files to the OS trash. Stop `index`, `match` and `review` while applying.

#### Plan then apply
```sh
matcha index  /path/to/Videos
matcha match  /path/to/Videos
matcha plan   /path/to/Videos            # decide group by group, Ctrl-C any time
matcha apply  /path/to/Videos            # dry run: check the summary
matcha apply  /path/to/Videos --apply    # make the changes
```
Separating the decisions from the file changes lets you review a whole library over several sittings, inspect the plan (`.matcha/review_plan.json`) before anything is touched, and re-run `apply` safely after a failure.

### `matcha continue`
#### CLI options and usage
| Option | Default | Description |
|--------|---------|-------------|
| `directory` | `.` | Directory to continue from |
| `command` | (prompted) | Command to continue: `index` or `match` |

```sh
# Re-run the last command (prompts if both index and match configs exist)
uv run matcha continue /path/to/Videos

# Re-run specifically the last index
uv run matcha continue /path/to/Videos index

# Re-run specifically the last match
uv run matcha continue /path/to/Videos match

# Use the current directory (default)
uv run matcha continue
```

#### Explanation
`matcha continue` re-runs the last `index` or `match` command using the configuration that was saved automatically when that command was last run. This means you never have to remember which flags you used.

Each run of `matcha index` or `matcha match` saves its arguments to `.matcha/index.json` or `.matcha/match.json`. `matcha continue` reads the relevant file and dispatches with the same arguments. If only one config file exists, `continue` runs that command immediately. If both exist, it lists them and prompts for a choice.

When continuing a `match` run, `continue` checks whether any videos have been fingerprinted since the match config was last saved and notes it. Those videos are added to the FAISS index incrementally on the next match run, so nothing is silently missed and no full rebuild is forced.

`continue` exits with an error if no database is found (i.e. `matcha index` has never been run) or if no saved config exists for the requested command.


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
    path             TEXT UNIQUE NOT NULL,   -- current absolute path (updated by rename in `review`)
    duration         REAL,                   -- seconds
    fingerprinted_at REAL,                   -- unix timestamp; NULL = not yet done
    moved_to         TEXT                    -- legacy, unused since `move` was removed; NULL
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
    moved       INTEGER NOT NULL DEFAULT 0, -- legacy, unused since `move` was removed
    dismissed   INTEGER NOT NULL DEFAULT 0  -- 1 = marked not-duplicates in `review`
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
