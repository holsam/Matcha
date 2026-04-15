"""
generate_test_videos_opencv.py

OpenCV-based test video generator with multiprocessing + tqdm.
"""

import cv2, csv, math, random, shutil, typer
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

app = typer.Typer()
OUTPUT_DIR = Path("output_videos")


# ---------------------------------------------------------------------------
# Core video generation (pure function for multiprocessing)
# ---------------------------------------------------------------------------

def render_video(path: str, duration: int, seed: int):
    random.seed(seed)

    width, height = 320, 240
    fps = 30
    frames = duration * fps

    # colours
    hue = (seed * 137) % 360
    r = int(128 + 127 * math.sin(math.radians(hue)))
    g = int(128 + 127 * math.sin(math.radians(hue + 120)))
    b = int(128 + 127 * math.sin(math.radians(hue + 240)))
    ball_color = (b, g, r)
    bg_color = (255 - b, 255 - g, 255 - r)

    # motion
    fx = 0.3 + (seed % 13) * 0.1
    fy = 0.3 + (seed % 11) * 0.12
    radius = 10 + (seed % 6) * 4

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for i in range(frames):
        t = i / fps
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        cx = int(width / 2 + (width / 3) * math.sin(fx * t))
        cy = int(height / 2 + (height / 3) * math.cos(fy * t))

        cv2.circle(frame, (cx, cy), radius, ball_color, -1)
        out.write(frame)

    out.release()


def random_duration():
    return random.randint(50, 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def generate(
    count: int = typer.Option(30),
    exact_matches: int = typer.Option(5),
    partial_matches: int = typer.Option(5),
    seed: int = typer.Option(42),
    workers: int = typer.Option(4, help="Number of parallel workers"),
):
    if exact_matches + partial_matches > count // 2:
        raise typer.BadParameter("Too many match pairs for the total count.")

    random.seed(seed)

    OUTPUT_DIR.mkdir(exist_ok=True)
    temp_dir = OUTPUT_DIR / "_tmp"
    temp_dir.mkdir(exist_ok=True)

    videos = []  # (path, seed)
    relationships = []
    jobs = []
    next_seed = 0

    # -----------------------------------------------------------------------
    # Plan jobs (no execution yet)
    # -----------------------------------------------------------------------

    # Exact matches
    for i in range(exact_matches):
        s = next_seed
        next_seed += 1

        base = temp_dir / f"exact_base_{i}.mp4"
        copy_ = temp_dir / f"exact_copy_{i}.mp4"

        jobs.append((base, random_duration(), s))
        videos.extend([(base, s), (copy_, s)])
        relationships.append((base.name, copy_.name, "exact"))

    # Partial matches
    partial_specs = []
    for i in range(partial_matches):
        core_seed = next_seed
        prefix_seed = next_seed + 1
        suffix_seed = next_seed + 2
        next_seed += 3

        core = temp_dir / f"partial_core_{i}.mp4"
        prefix = temp_dir / f"partial_prefix_{i}.mp4"
        suffix = temp_dir / f"partial_suffix_{i}.mp4"
        container = temp_dir / f"partial_container_{i}.mp4"

        jobs.append((core, random_duration(), core_seed))
        jobs.append((prefix, random_duration() // 3, prefix_seed))
        jobs.append((suffix, random_duration() // 3, suffix_seed))

        partial_specs.append((prefix, core, suffix, container))
        videos.extend([(core, core_seed), (container, None)])
        relationships.append((core.name, container.name, "partial"))

    # Independent
    remaining = count - len(videos)
    for i in range(remaining):
        s = next_seed
        next_seed += 1
        vid = temp_dir / f"random_{i}.mp4"
        jobs.append((vid, random_duration(), s))
        videos.append((vid, s))

    # -----------------------------------------------------------------------
    # Run generation in parallel
    # -----------------------------------------------------------------------

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(render_video, str(path), dur, seed)
            for path, dur, seed in jobs
        ]

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Rendering"):
            pass

    # -----------------------------------------------------------------------
    # Post-processing (single-threaded, fast)
    # -----------------------------------------------------------------------

    # Exact copies
    for path, seed_used in videos:
        if "exact_copy" in path.name:
            base = temp_dir / path.name.replace("copy", "base")
            shutil.copy(base, path)

    # Partial concatenation
    for prefix, core, suffix, container in partial_specs:
        clips = []
        for p in [prefix, core, suffix]:
            cap = cv2.VideoCapture(str(p))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                clips.append(frame)
            cap.release()

        h, w, _ = clips[0].shape
        out = cv2.VideoWriter(
            str(container),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (w, h),
        )

        for f in clips:
            out.write(f)
        out.release()

        prefix.unlink()
        suffix.unlink()

    # -----------------------------------------------------------------------
    # Shuffle + rename
    # -----------------------------------------------------------------------

    random.shuffle(videos)

    name_map = {}
    for idx, (src, _) in enumerate(videos):
        dst = OUTPUT_DIR / f"video_{idx:03d}.mp4"
        shutil.move(src, dst)
        name_map[src.name] = dst.name

    # -----------------------------------------------------------------------
    # Manifest
    # -----------------------------------------------------------------------

    manifest_path = OUTPUT_DIR / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_a", "video_b", "relationship"])
        for a, b, rel in relationships:
            writer.writerow([name_map[a], name_map[b], rel])

    shutil.rmtree(temp_dir, ignore_errors=True)

    typer.echo(f"Generated {count} videos in {OUTPUT_DIR}/")
    typer.echo(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    app()