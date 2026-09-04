from __future__ import annotations


class DummyTTS:
    """No audio bytes. Used in tests and when no TTS model is installed."""

    name = "dummy"
    sample_rate = 16000

    def synthesize(self, text: str) -> bytes:
        return b""

    def synthesize_iter(self, text: str):
        pcm = self.synthesize(text)
        if pcm:
            yield pcm
