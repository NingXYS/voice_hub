from __future__ import annotations

from pathlib import Path

import numpy as np


class NullASR:
    """Text-only mode: binary audio is ignored. Swap this out for a real ASR."""

    name = "none"

    def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> str:
        return ""


class LocalParaformerASR:
    name = "sherpa_local"

    def __init__(self, model_dir: Path, num_threads: int = 4):
        self.model_dir = Path(model_dir)
        self.num_threads = num_threads
        self._model = None

    def _ensure(self):
        if self._model is not None:
            return
        import sherpa_onnx

        model_file = self.model_dir / "model.int8.onnx"
        tokens = self.model_dir / "tokens.txt"
        if not model_file.is_file() or not tokens.is_file():
            raise FileNotFoundError(
                f"Paraformer model missing under {self.model_dir}. "
                "Set ASR_MODEL_DIR or ASR_PROVIDER=none for text-only."
            )
        self._model = sherpa_onnx.OfflineRecognizer.from_paraformer(
            paraformer=str(model_file),
            tokens=str(tokens),
            num_threads=self.num_threads,
            sample_rate=16000,
        )

    def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> str:
        if not pcm:
            return ""
        self._ensure()
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        if samples.size == 0:
            return ""
        stream = self._model.create_stream()
        stream.accept_waveform(int(sample_rate), samples)
        self._model.decode_stream(stream)
        return str(getattr(getattr(stream, "result", None), "text", "") or "").strip()
