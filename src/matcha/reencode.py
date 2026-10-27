'''
MATCHA: profile a library by codec and bitrate, and re-encode bloated videos to HEVC in place.
'''

# Import external dependencies
import os
from dataclasses import dataclass

# Import internal matcha objects
from matcha.indexer import find_videos
from matcha.review.media import probe

# _CODEC_SCALE: bpp a codec needs relative to hevc, lower is more efficient
_CODEC_SCALE = {'av1': 0.75, 'hevc': 1.0, 'vp9': 1.1, 'h264': 1.5, 'mpeg4': 2.0, 'mpeg2video': 2.5}

# _DEFAULT_SCALE: scale for codecs missing from _CODEC_SCALE, treated like h264
_DEFAULT_SCALE = 1.5

# Info: probed facts for one file
@dataclass(frozen=True)
class Info:
    path: str
    size: int
    duration: float
    width: int
    height: int
    fps: float
    codec: str
    vbitrate: int  # video stream bits per second

    # bpp: bits per pixel per frame, the codec-agnostic density measure
    @property
    def bpp(self) -> float:
        px = self.width * self.height * self.fps
        return self.vbitrate / px if px else 0.0

    # bloat: bpp as a multiple of what this file's own codec needs at target_bpp, above 1 is over-allocated
    def bloat(self, target_bpp: float) -> float:
        return self.bpp / (target_bpp * _CODEC_SCALE.get(self.codec, _DEFAULT_SCALE))

    # saving: estimated bytes saved re-encoding to hevc at target_bpp, audio is unchanged
    def saving(self, target_bpp: float) -> int:
        if not self.bpp:
            return 0
        video_bytes = min(self.size, self.vbitrate * self.duration / 8)
        new_bytes = self.width * self.height * self.fps * target_bpp * self.duration / 8
        return int(max(0, video_bytes - new_bytes))

# _info: stat and probe one file, None when it is unreadable or has no video stream
def _info(path: str) -> Info | None:
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    p = probe(path)
    if p is None or not (p.width and p.height and p.codec):
        return None
    duration = p.duration or 0.0
    vbitrate = p.video_bitrate or p.bitrate or (int(size * 8 / duration) if duration else 0)
    return Info(path, size, duration, p.width, p.height, p.fps or 25.0, p.codec, vbitrate)

# _videos: video paths under directory, minus temp files from an interrupted run
def _videos(directory: str) -> list[str]:
    return [p for p in find_videos(os.path.abspath(directory)) if '.reencoding.' not in os.path.basename(p)]
