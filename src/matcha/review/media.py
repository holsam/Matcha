'''
MATCHA: lazy file facts for the review UI (stat, ffprobe, thumbnails), computed only for videos on screen.
'''

# Import external dependencies
import json, os, subprocess
from dataclasses import dataclass


# FileInfo: size and modification time of a file on disk
@dataclass(frozen=True)
class FileInfo:
    size: int
    mtime: float


# Probe: stream facts from ffprobe, each None when the file does not report it
@dataclass(frozen=True)
class Probe:
    width: int | None
    height: int | None
    codec: str | None
    bitrate: int | None


# THUMB_WIDTH: pixel width of cached thumbnails
THUMB_WIDTH = 320
# _TIMEOUT: seconds before a slow disk or corrupt file is given up on
_TIMEOUT = 30


# file_info: stat a file, or None if it is missing
def file_info(path: str) -> FileInfo | None:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return FileInfo(st.st_size, st.st_mtime)


# human_size: bytes as a short string such as 1.4 GB
def human_size(size: float) -> str:
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024 or unit == 'GB':
            return f'{size:.0f} {unit}' if unit == 'B' else f'{size:.1f} {unit}'
        size /= 1024
    raise AssertionError('unreachable')


# human_duration: seconds as h:mm:ss or m:ss
def human_duration(seconds: float) -> str:
    h, rest = divmod(int(seconds), 3600)
    m, s = divmod(rest, 60)
    return f'{h}:{m:02d}:{s:02d}' if h else f'{m}:{s:02d}'


# probe: first video stream facts via ffprobe, or None if ffprobe fails
def probe(path: str) -> Probe | None:
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height:format=bit_rate',
        '-of', 'json', path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=_TIMEOUT, check=True).stdout
        data = json.loads(out)
    except (subprocess.SubprocessError, OSError, ValueError):
        return None
    stream = (data.get('streams') or [{}])[0]
    bitrate = (data.get('format') or {}).get('bit_rate')
    return Probe(
        stream.get('width'),
        stream.get('height'),
        stream.get('codec_name'),
        int(bitrate) if bitrate and bitrate.isdigit() else None,
    )


# thumbnail: path of a cached mid-video frame, generated on first use, or None if ffmpeg cannot read the file
def thumbnail(matcha_dir: str, video_id: int, path: str, duration: float) -> str | None:
    out = os.path.join(matcha_dir, 'thumbs', f'{video_id}.jpg')
    if os.path.exists(out):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + '.tmp.jpg'  # written aside so a killed ffmpeg never leaves a half-image in the cache
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-ss', f'{max(duration, 0) / 2:.2f}',  # seek before -i, so the disk is not read end to end
        '-i', path, '-frames:v', '1', '-vf', f'scale={THUMB_WIDTH}:-2', tmp,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=_TIMEOUT, check=True)
        os.replace(tmp, out)
    except (subprocess.SubprocessError, OSError):
        if os.path.exists(tmp):
            os.remove(tmp)
        return None
    return out
