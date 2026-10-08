import acoustid, hashlib, imagehash, os, subprocess, tempfile
from PIL import Image

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm"}


# _SAMPLE_BYTES: bytes read from the head, middle and tail of a file for its content key
_SAMPLE_BYTES = 256 * 1024


# content_key: size plus a hash of three samples; equal keys mean the files are almost certainly byte-identical
def content_key(path: str) -> str | None:
    try:
        size = os.path.getsize(path)
        digest = hashlib.blake2b(digest_size=16)
        with open(path, "rb") as f:
            for offset in (0, max(0, size // 2 - _SAMPLE_BYTES // 2), max(0, size - _SAMPLE_BYTES)):
                f.seek(offset)
                digest.update(f.read(_SAMPLE_BYTES))
        return f"{size}:{digest.hexdigest()}"
    except OSError:
        return None


# _FRAME_SIZE: (width, height) of frames piped from ffmpeg, matching the scale filter below
_FRAME_SIZE = (160, 120)


def extract_frame_hashes(
    video_path: str,
    fps: float = 1.0,
    hwaccel: bool = False,
    threads: int | None = None,
) -> list[tuple[float, str]]:
    """
    Extract frames from a video at `fps` frames per second.
    Returns a list of (timestamp_seconds, phash_hex) tuples.

    ffmpeg writes raw RGB frames to a pipe and each frame is hashed as it
    arrives, so nothing touches disk and peak memory is one frame. RGB (not
    greyscale) is piped so PIL does the greyscale conversion, which keeps
    hashes identical to the earlier PNG-on-disk approach.

    If hwaccel=True, passes -hwaccel auto to ffmpeg (on Mac this uses
    VideoToolbox). Falls back silently to software decoding if unavailable.

    threads caps ffmpeg's decode and filter threads (None = ffmpeg's default,
    which uses every core per process; lower it when running many workers).

    The scale filter (160:120) reduces decode work for high-resolution
    videos — pHash only needs a small image, so full-resolution frames
    are unnecessary.
    """
    hashes = []
    frame_bytes = _FRAME_SIZE[0] * _FRAME_SIZE[1] * 3
    cmd = ["ffmpeg"]
    if hwaccel:
        cmd += ["-hwaccel", "auto"]
    if threads is not None:
        cmd += ["-threads", str(threads), "-filter_threads", str(threads)]
    cmd += [
        "-i", video_path,
        "-vf", f"fps={fps},scale={_FRAME_SIZE[0]}:{_FRAME_SIZE[1]}",
        "-fps_mode", "vfr",
        "-pix_fmt", "rgb24",
        "-f", "rawvideo",
        "-loglevel", "error",
        "pipe:1",
    ]

    # stderr goes to a file so a full pipe can never block ffmpeg while we read frames
    with tempfile.TemporaryFile() as stderr:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=stderr)
        try:
            while len(chunk := proc.stdout.read(frame_bytes)) == frame_bytes:
                img = Image.frombytes("RGB", _FRAME_SIZE, chunk).convert("L")  # greyscale — faster and sufficient for pHash
                hashes.append((len(hashes) / fps, str(imagehash.phash(img))))
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
            returncode = proc.wait()
        if returncode != 0:
            stderr.seek(0)
            raise RuntimeError(
                f"ffmpeg failed for {video_path}: {stderr.read().decode()}"
            )

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