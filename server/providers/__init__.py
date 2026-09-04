from __future__ import annotations

from server import config
from server.providers.asr_doubao import DoubaoASR
from server.providers.asr_local import LocalParaformerASR, NullASR
from server.providers.asr_online import OnlineASR
from server.providers.llm_openai import OpenAICompatLLM
from server.providers.search_tavily import web_search
from server.providers.tts_doubao import DoubaoTTS
from server.providers.tts_dummy import DummyTTS
from server.providers.tts_edge import EdgeTTS
from server.providers.tts_local import LocalKokoroTTS


def build_asr():
    name = config.asr_provider()
    if name in {"none", "null", "off"}:
        return NullASR()
    if name in {"doubao", "volc", "volcengine", "online", "cloud"}:
        return DoubaoASR()
    model_dir = config.asr_model_dir()
    if not (model_dir / "model.int8.onnx").is_file():
        print(f"⚠️ ASR model not found at {model_dir}; text-only until the file is present")
        return NullASR()
    return LocalParaformerASR(model_dir)


def build_tts():
    name = config.tts_provider()
    if name in {"dummy", "none", "off"}:
        return DummyTTS()
    if name in {"doubao", "volc", "volcengine", "online", "cloud"}:
        return DoubaoTTS()
    if name == "edge":
        return EdgeTTS()
    model_dir = config.tts_model_dir()
    if not (model_dir / "model.onnx").is_file():
        print(f"⚠️ TTS model not found at {model_dir}; dummy TTS until the file is present")
        return DummyTTS()
    try:
        return LocalKokoroTTS(model_dir)
    except Exception as exc:
        print(f"⚠️ local TTS unavailable ({exc}); falling back to dummy")
        return DummyTTS()


def build_llm():
    return OpenAICompatLLM()
