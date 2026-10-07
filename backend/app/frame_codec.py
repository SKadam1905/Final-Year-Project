"""Pack a burst of frames (as JPEGs) plus metadata into one blob for the Redis queue.

Layout: b"KBF1" | 4-byte header length | JSON header (metadata + JPEG sizes) | JPEGs
"""

from __future__ import annotations

import json
import struct
from typing import Sequence

import cv2
import numpy as np

MAGIC = b"KBF1"
_LENGTH = struct.Struct(">I")


def encode_burst(frames: Sequence[np.ndarray], meta: dict, quality: int = 90) -> bytes:
    jpegs = []
    for frame in frames:
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise ValueError("JPEG encoding failed")
        jpegs.append(buffer.tobytes())
    header = json.dumps({**meta, "sizes": [len(j) for j in jpegs]}).encode()
    return MAGIC + _LENGTH.pack(len(header)) + header + b"".join(jpegs)


def decode_burst(blob: bytes) -> tuple[list[np.ndarray], dict]:
    if blob[:4] != MAGIC or len(blob) < 8:
        raise ValueError("not a frame burst")
    (header_len,) = _LENGTH.unpack_from(blob, 4)
    meta = json.loads(blob[8 : 8 + header_len])
    offset = 8 + header_len
    frames = []
    for size in meta.pop("sizes"):
        if offset + size > len(blob):
            raise ValueError("truncated frame burst")
        frame = cv2.imdecode(np.frombuffer(blob, np.uint8, count=size, offset=offset), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("corrupt JPEG in frame burst")
        frames.append(frame)
        offset += size
    return frames, meta
