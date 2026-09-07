#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Thin device client. Never load host cloud keys.
unset DOUBAO_TTS_API_KEY DOUBAO_TTS_APPID DOUBAO_TTS_ACCESS_TOKEN \
  DOUBAO_ASR_API_KEY DIALOG_API_KEY DIALOG_APPID DIALOG_ACCESS_TOKEN \
  METASO_API_KEY TAVILY_API_KEY CHAT_LLM_API_KEY \
  ASR_PROVIDER TTS_PROVIDER DIALOG_PROVIDER WEB_SEARCH_PROVIDER || true

# Ignore leftover host env. client.env may turn live mode on after this.
export VOICE_ENABLE_ROBOT_COMMAND=0
export VOICE_ENABLE_MASSAGE=0
export VOICE_SPEAKER_VERIFY_MODE=off

if [ -f /etc/voice_hub_client.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/voice_hub_client.env
  set +a
elif [ -f "${VOICE_HUB_CLIENT_ENV:-$HOME/.config/voice_hub/client.env}" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${VOICE_HUB_CLIENT_ENV:-$HOME/.config/voice_hub/client.env}"
  set +a
fi

unset DOUBAO_TTS_API_KEY DOUBAO_TTS_APPID DOUBAO_TTS_ACCESS_TOKEN \
  DOUBAO_ASR_API_KEY DIALOG_API_KEY DIALOG_APPID DIALOG_ACCESS_TOKEN \
  METASO_API_KEY TAVILY_API_KEY CHAT_LLM_API_KEY || true

if [ -z "${VOICE_HUB_BRIDGE_PATH:-}" ]; then
  if [ -d "$HOME/tianlian_voice_bridge" ]; then
    export VOICE_HUB_BRIDGE_PATH="$HOME/tianlian_voice_bridge"
  elif [ -d /home/ning/model/lyz/src/tianlian_voice_bridge ]; then
    export VOICE_HUB_BRIDGE_PATH=/home/ning/model/lyz/src/tianlian_voice_bridge
  fi
fi
if [ -z "${VOICE_KWS_PYTHONPATH:-}" ]; then
  for cand in "$HOME/VoiceOnline/.venv_online/lib"/python3.*/site-packages "$ROOT/.venv/lib"/python3.*/site-packages; do
    if [ -d "$cand/sherpa_onnx" ]; then
      export VOICE_KWS_PYTHONPATH="$cand"
      break
    fi
  done
fi
if [ -z "${VOICE_KWS_MODEL_DIR:-}" ] && [ -f "$HOME/VoiceOnline/models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/encoder.onnx" ]; then
  export VOICE_KWS_MODEL_DIR="$HOME/VoiceOnline/models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
fi
if [ -z "${VOICE_ONLINE_ROOT:-}" ] && [ -d "$HOME/VoiceOnline" ]; then
  export VOICE_ONLINE_ROOT="$HOME/VoiceOnline"
fi
export PYTHONPATH="$ROOT${VOICE_HUB_BRIDGE_PATH:+:${VOICE_HUB_BRIDGE_PATH}}:${PYTHONPATH:-}"
if [ -n "${VOICE_KWS_PYTHONPATH:-}" ]; then
  export PYTHONPATH="${VOICE_KWS_PYTHONPATH}:${PYTHONPATH}"
fi

PYTHON_BIN=python3
if [ -x "$ROOT/.venv/bin/python" ]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
fi
exec "$PYTHON_BIN" "$ROOT/jetson_client/voice_client.py" "$@"
