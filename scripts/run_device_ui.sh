#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Device UI is the arm-side client. Never load host cloud keys.
unset DOUBAO_TTS_API_KEY DOUBAO_TTS_APPID DOUBAO_TTS_ACCESS_TOKEN \
  DOUBAO_ASR_API_KEY DIALOG_API_KEY DIALOG_APPID DIALOG_ACCESS_TOKEN \
  METASO_API_KEY TAVILY_API_KEY CHAT_LLM_API_KEY \
  ASR_PROVIDER TTS_PROVIDER DIALOG_PROVIDER WEB_SEARCH_PROVIDER || true

# Ignore leftover host env. client.env may turn live mode on after this.
export VOICE_ENABLE_ROBOT_COMMAND=0
export VOICE_ENABLE_MASSAGE=0
export VOICE_SPEAKER_VERIFY_MODE=off

# Device-local settings only. Do not fall back to the host .env.
CLIENT_ENV="${VOICE_HUB_CLIENT_ENV:-$HOME/.config/voice_hub/client.env}"
if [ -f "$CLIENT_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$CLIENT_ENV"
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
if [ -z "${VOICE_KWS_MODEL_DIR:-}" ] && [ -f "$HOME/VoiceOnline/models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20/encoder.onnx" ]; then
  export VOICE_KWS_MODEL_DIR="$HOME/VoiceOnline/models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
fi
if [ -z "${VOICE_ONLINE_ROOT:-}" ] && [ -d "$HOME/VoiceOnline" ]; then
  export VOICE_ONLINE_ROOT="$HOME/VoiceOnline"
fi

pick_python() {
  local candidate
  for candidate in "$ROOT/.venv/bin/python" python3; do
    if [ "$candidate" = "python3" ] || [ -x "$candidate" ]; then
      if "$candidate" -c "import starlette, uvicorn" >/dev/null 2>&1; then
        echo "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON_BIN="$(pick_python || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "设备端缺 starlette/uvicorn。当前脚本会优先用 $ROOT/.venv，请先安装：" >&2
  echo "  $ROOT/.venv/bin/pip install -r $ROOT/requirements.txt" >&2
  echo "或在当前 Python 里： pip install -r $ROOT/requirements.txt" >&2
  exit 1
fi

PY_TAG="$("$PYTHON_BIN" -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
if [ -z "${VOICE_KWS_PYTHONPATH:-}" ]; then
  for cand in \
    "$HOME/VoiceOnline/.venv_online/lib/$PY_TAG/site-packages" \
    "$ROOT/.venv/lib/$PY_TAG/site-packages" \
    "$HOME/.local/lib/$PY_TAG/site-packages"
  do
    if [ -d "$cand/sherpa_onnx" ]; then
      export VOICE_KWS_PYTHONPATH="$cand"
      break
    fi
  done
fi
export PYTHONPATH="$ROOT${VOICE_HUB_BRIDGE_PATH:+:${VOICE_HUB_BRIDGE_PATH}}:${PYTHONPATH:-}"
if [ -n "${VOICE_KWS_PYTHONPATH:-}" ]; then
  export PYTHONPATH="${VOICE_KWS_PYTHONPATH}:${PYTHONPATH}"
fi
exec "$PYTHON_BIN" -m device_ui
