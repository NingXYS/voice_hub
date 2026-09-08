"""Short local beep. Same 叮 as voice_control_online (880 Hz, 120 ms)."""
from __future__ import annotations

import math
import os
import struct
import subprocess
import tempfile
import wave
from pathlib import Path

TONE_DURATION_SEC = 0.12
TONE_FREQUENCY_HZ = 880
TONE_SAMPLE_RATE = 16000


def render_prompt_tone_pcm(
    duration_sec: float = TONE_DURATION_SEC,
    frequency_hz: float = TONE_FREQUENCY_HZ,
    sample_rate: int = TONE_SAMPLE_RATE,
) -> bytes:
    if duration_sec <= 0 or frequency_hz <= 0 or sample_rate <= 0:
        return b""
    frames = int(duration_sec * sample_rate)
    parts: list[bytes] = []
    for i in range(frames):
        envelope = min(1.0, i / max(1, frames * 0.1), (frames - i) / max(1, frames * 0.1))
        value = int(8000 * envelope * math.sin(2 * math.pi * frequency_hz * i / sample_rate))
        parts.append(struct.pack("<h", value))
    return b"".join(parts)


def write_prompt_tone_wav(
    path: str | Path,
    duration_sec: float = TONE_DURATION_SEC,
    frequency_hz: float = TONE_FREQUENCY_HZ,
    sample_rate: int = TONE_SAMPLE_RATE,
) -> Path:
    pcm = render_prompt_tone_pcm(duration_sec, frequency_hz, sample_rate)
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return dest


def play_prompt_tone(
    duration_sec: float = TONE_DURATION_SEC,
    frequency_hz: float = TONE_FREQUENCY_HZ,
    sample_rate: int = TONE_SAMPLE_RATE,
) -> bool:
    pcm = render_prompt_tone_pcm(duration_sec, frequency_hz, sample_rate)
    if not pcm:
        return False
    path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(pcm)
        device = os.environ.get("APLAY_DEVICE", "default") or "default"
        result = subprocess.run(
            ["aplay", "-q", "-D", device, path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0
    except Exception as exc:
        print(f"wake tone failed: {type(exc).__name__}: {exc}", flush=True)
        return False
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
