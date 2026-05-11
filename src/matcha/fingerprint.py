import acoustid, imagehash, os, subprocess, tempfile
from pathlib import Path
from PIL import Image

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}


def extract_frame_hashes(
    video_path: str,
    fps: float = 1.0,
    hwaccel: bool = False,
) -> list[tuple[float, str]]:
    """
    Extract frames from a video at `fps` frames per second.
    Returns a list of (timestamp_seconds, phash_hex) tuples.

    Frames are written to a temporary directory and deleted immediately
    after hashing, keeping peak memory to roughly one frame at a time.

    If hwaccel=True, passes -hwaccel auto to ffmpeg (on Mac this uses
    VideoToolbox). Falls back silently to software decoding if unavailable.

    The scale filter (160:120) reduces decode work for high-resolution
    videos — pHash only needs a small image, so full-resolution frames
    are unnecessary.
    """
    hashes = []

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_pattern = os.path.join(tmpdir, "frame_%07d.png")
        cmd = ["ffmpeg"]

        if hwaccel:
            cmd += ["-hwaccel", "auto"]
        cmd += [
            "-i", video_path,
            "-vf", f"fps={fps},scale=160:120",
            "-vsync", "vfr",
            "-f", "image2",
            frame_pattern,
            "-loglevel", "error",
        ]

        result = subprocess.run(cmd, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for {video_path}: {result.stderr.decode()}"
            )

        frame_files = sorted(Path(tmpdir).glob("frame_*.png"))
        for i, frame_path in enumerate(frame_files):
            timestamp = i / fps
            img = Image.open(frame_path).convert("L")  # greyscale — faster and sufficient for pHash
            phash = str(imagehash.phash(img))
            hashes.append((timestamp, phash))
            frame_path.unlink()

    return hashes


def get_video_duration(video_path: str) -> float:
    """Return video duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        return float(result.stdout.decode().strip())
    except ValueError:
        return 0.0


def get_audio_fingerprint(video_path: str) -> tuple[float, str] | None:
    """
    Return (duration, fingerprint_string) for a video's audio track,
    or None if the video has no audio or fingerprinting fails.
    """
    try:
        duration, fingerprint = acoustid.fingerprint_file(video_path)
        return duration, fingerprint
    except Exception:
        return None