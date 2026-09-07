"""Short local beep. Not proof that the turn succeeded."""
from __future__ import annotations

import math
import os
import struct
import subprocess
import tempfile
import wave


def play_prompt_tone(duration_sec: float = 0.12, frequency_hz: float = 880, sample_rate: int = 16000) -> bool:
    if duration_sec <= 0 or frequency_hz <= 0 or sample_rate <= 0:
        return False
    path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name
        frames = int(duration_sec * sample_rate)
        with wave.open(path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            for i in range(frames):
                envelope = min(1.0, i / max(1, frames * 0.1), (frames - i) / max(1, frames * 0.1))
                value = int(8000 * envelope * math.sin(2 * math.pi * frequency_hz * i / sample_rate))
                wav.writeframesraw(struct.pack("<h", value))
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
