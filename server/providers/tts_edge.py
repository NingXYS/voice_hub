from __future__ import annotations

import asyncio
import io

from server import config


class EdgeTTS:
    """Microsoft Edge online TTS. No local voice model. Swap-in via TTS_PROVIDER=edge."""

    name = "edge"
    sample_rate = 16000

    def synthesize(self, text: str) -> bytes:
        if not isinstance(text, str) or not text.strip():
            return b""
        try:
            import edge_tts
        except ImportError as exc:
            raise RuntimeError("TTS_PROVIDER=edge requires the edge-tts package") from exc

        async def _run() -> bytes:
            communicate = edge_tts.Communicate(text.strip(), config.tts_voice())
            raw = b""
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio":
                    raw += chunk.get("data") or b""
            return raw

        mp3 = asyncio.run(_run())
        return _mp3_to_pcm16(mp3, self.sample_rate)


def _mp3_to_pcm16(mp3: bytes, sample_rate: int) -> bytes:
    if not mp3:
        return b""
    try:
        import soundfile as sf
        import numpy as np
    except ImportError:
        return mp3
    try:
        samples, src_rate = sf.read(io.BytesIO(mp3), dtype="float32")
    except Exception:
        return b""
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if src_rate != sample_rate and len(samples) > 1:
        count = max(1, int(len(samples) * sample_rate / src_rate))
        samples = np.interp(
            np.linspace(0, len(samples) - 1, count),
            np.arange(len(samples)),
            samples,
        ).astype(np.float32)
    pcm = np.clip(samples, -1.0, 1.0)
    return (pcm * 32767.0).astype("<i2").tobytes()
