"""PCM s16le → WAV for browser playback."""
from __future__ import annotations

import struct


def pcm_s16le_to_wav(pcm: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    if not pcm:
        return b""
    bits = 16
    block_align = channels * bits // 8
    byte_rate = sample_rate * block_align
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(pcm),
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits,
        b"data",
        len(pcm),
    )
    return header + pcm
