from __future__ import annotations

from pathlib import Path

import numpy as np


class LocalKokoroTTS:
    """On-PC Kokoro via sherpa-onnx. Jetson never loads this model."""

    name = "local"

    def __init__(self, model_dir: Path, speaker_id: int | None = None):
        from server import config

        self.model_dir = Path(model_dir)
        self.speaker_id = speaker_id if speaker_id is not None else config.tts_speaker_id()
        self._tts = None
        self.sample_rate = 24000

    def _ensure(self):
        if self._tts is not None:
            return
        import sherpa_onnx

        model_file = self.model_dir / "model.onnx"
        tokens = self.model_dir / "tokens.txt"
        voices = self.model_dir / "voices.bin"
        data_dir = self.model_dir / "espeak-ng-data"
        if not model_file.is_file() or not tokens.is_file():
            raise FileNotFoundError(f"Kokoro TTS missing under {self.model_dir}")
        lexicon = ",".join(
            str(self.model_dir / name)
            for name in ("lexicon-us-en.txt", "lexicon-zh.txt")
            if (self.model_dir / name).is_file()
        )
        kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
            model=str(model_file),
            voices=str(voices) if voices.is_file() else "",
            tokens=str(tokens),
            lexicon=lexicon,
            data_dir=str(data_dir) if data_dir.is_dir() else "",
        )
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(kokoro=kokoro, num_threads=4),
        )
        self._tts = sherpa_onnx.OfflineTts(tts_config)

    def synthesize(self, text: str) -> bytes:
        if not isinstance(text, str) or not text.strip():
            return b""
        self._ensure()
        audio = self._tts.generate(text.strip(), sid=self.speaker_id, speed=1.0)
        if audio is None or not getattr(audio, "samples", None):
            return b""
        self.sample_rate = int(getattr(audio, "sample_rate", self.sample_rate) or self.sample_rate)
        pcm = np.clip(np.asarray(audio.samples, dtype=np.float32), -1.0, 1.0)
        return (pcm * 32767.0).astype("<i2").tobytes()
