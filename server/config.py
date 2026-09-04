from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def host() -> str:
    return env("VOICE_HUB_HOST", "0.0.0.0") or "0.0.0.0"


def port() -> int:
    try:
        return int(env("VOICE_HUB_PORT", "8765") or "8765")
    except ValueError:
        return 8765


def adapter_dir() -> Path:
    configured = env("VOICE_HUB_ADAPTER_DIR")
    if configured:
        return Path(configured)
    return ROOT / "adapters"


def llm_base_url() -> str:
    # OpenAI-compatible root (no /chat/completions suffix).
    # 百炼: https://dashscope.aliyuncs.com/compatible-mode/v1
    # 智谱: https://open.bigmodel.cn/api/paas/v4
    return env("CHAT_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")


def llm_api_key() -> str:
    return env("CHAT_LLM_API_KEY") or env("ZHIPU_API_KEY") or env("DASHSCOPE_API_KEY")


def llm_model() -> str:
    return env("CHAT_LLM_MODEL")


def llm_timeout() -> float:
    try:
        return max(5.0, float(env("CHAT_LLM_TIMEOUT_SEC", "30") or "30"))
    except ValueError:
        return 30.0


def llm_max_tokens() -> int:
    try:
        return max(32, min(1024, int(env("CHAT_LLM_MAX_TOKENS", "256") or "256")))
    except ValueError:
        return 256


def asr_provider() -> str:
    """doubao | sherpa_local | none"""
    return env("ASR_PROVIDER", "doubao") or "doubao"


def asr_model_dir() -> Path:
    raw = env("ASR_MODEL_DIR")
    if raw:
        return Path(raw)
    return ROOT.parent / "lyz/src/voice_control_online/models/sherpa-onnx-paraformer-zh-2023-09-14"


def asr_online_url() -> str:
    return env("ASR_ONLINE_URL")


def asr_online_api_key() -> str:
    return env("ASR_ONLINE_API_KEY")


def doubao_asr_api_key() -> str:
    return env("DOUBAO_ASR_API_KEY") or doubao_api_key()


def doubao_asr_resource_id() -> str:
    return env("DOUBAO_ASR_RESOURCE_ID", "volc.seedasr.sauc.duration") or "volc.seedasr.sauc.duration"


def doubao_asr_url() -> str:
    return env(
        "DOUBAO_ASR_URL",
        "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async",
    ) or "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel_async"


def tts_provider() -> str:
    """doubao | local | edge | dummy"""
    return env("TTS_PROVIDER", "doubao") or "doubao"


def tts_model_dir() -> Path:
    raw = env("TTS_MODEL_DIR")
    if raw:
        return Path(raw)
    return ROOT.parent / "lyz/src/voice_control_online/models/tts/kokoro-multi-lang-v1_0"


def tts_voice() -> str:
    return env("TTS_EDGE_VOICE", "zh-CN-XiaoxiaoNeural") or "zh-CN-XiaoxiaoNeural"


def tts_sample_rate() -> int:
    try:
        return int(env("TTS_SAMPLE_RATE", "16000") or "16000")
    except ValueError:
        return 16000


def tts_speaker_id() -> int:
    try:
        return int(env("TTS_SPEAKER_ID", "47") or "47")
    except ValueError:
        return 47


def doubao_api_key() -> str:
    return env("DOUBAO_TTS_API_KEY") or env("VOLC_TTS_API_KEY")


def doubao_appid() -> str:
    return env("DOUBAO_TTS_APPID") or env("VOLC_TTS_APPID")


def doubao_access_token() -> str:
    return env("DOUBAO_TTS_ACCESS_TOKEN") or env("VOLC_TTS_ACCESS_TOKEN")


def doubao_resource_id() -> str:
    return env("DOUBAO_TTS_RESOURCE_ID", "seed-tts-1.0") or "seed-tts-1.0"


def doubao_voice() -> str:
    return env("DOUBAO_TTS_VOICE") or env("TTS_VOICE")


def doubao_cluster() -> str:
    return env("DOUBAO_TTS_CLUSTER", "volcano_tts") or "volcano_tts"


def doubao_v3_url() -> str:
    return env(
        "DOUBAO_TTS_V3_URL",
        "https://openspeech.bytedance.com/api/v3/tts/unidirectional",
    ) or "https://openspeech.bytedance.com/api/v3/tts/unidirectional"


def doubao_v1_url() -> str:
    return env("DOUBAO_TTS_V1_URL", "https://openspeech.bytedance.com/api/v1/tts") or (
        "https://openspeech.bytedance.com/api/v1/tts"
    )


def search_provider() -> str:
    return env("WEB_SEARCH_PROVIDER", "metaso").strip().lower() or "metaso"


def search_api_key() -> str:
    provider = search_provider()
    if provider == "tavily":
        return env("TAVILY_API_KEY")
    return env("METASO_API_KEY") or env("TAVILY_API_KEY")


def memory_dir() -> Path:
    raw = env("VOICE_HUB_MEMORY_DIR")
    if raw:
        return Path(raw)
    return Path.home() / ".local/share/voice_hub/memory"


def dialog_provider() -> str:
    return env("DIALOG_PROVIDER", "doubao").strip().lower() or "doubao"


def dialog_configured() -> bool:
    if dialog_provider() in {"off", "none", "0", "false"}:
        return False
    return bool((dialog_appid() and dialog_access_token()) or dialog_api_key())


def dialog_url() -> str:
    return env(
        "DIALOG_URL",
        "wss://openspeech.bytedance.com/api/v3/realtime/dialogue",
    ) or "wss://openspeech.bytedance.com/api/v3/realtime/dialogue"


def dialog_appid() -> str:
    return env("DIALOG_APPID") or doubao_appid()


def dialog_access_token() -> str:
    return env("DIALOG_ACCESS_TOKEN") or doubao_access_token()


def dialog_api_key() -> str:
    # New-console X-Api-Key is shared across 豆包 ASR/TTS/dialog if the product is opened.
    return env("DIALOG_API_KEY") or doubao_api_key() or doubao_asr_api_key()


def dialog_resource_id() -> str:
    return env("DIALOG_RESOURCE_ID", "volc.speech.dialog") or "volc.speech.dialog"


def dialog_app_key() -> str:
    return env("DIALOG_APP_KEY", "PlgvMymc7f3tQnJ6") or "PlgvMymc7f3tQnJ6"


def dialog_speaker() -> str:
    raw = env("DIALOG_SPEAKER") or env("DOUBAO_TTS_VOICE") or "zh_female_vv_jupiter_bigtts"
    lowered = raw.lower()
    if "jupiter" in lowered or lowered.startswith(("icl_", "s_", "saturn_")):
        return raw
    # seed-tts voices (uranus/moon/...) are not valid on realtime dialog.
    return "zh_female_vv_jupiter_bigtts"


def dialog_output_rate() -> int:
    try:
        return int(env("DIALOG_TTS_SAMPLE_RATE", "16000") or "16000")
    except ValueError:
        return 16000
