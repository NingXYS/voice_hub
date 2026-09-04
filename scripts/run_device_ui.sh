#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:/home/ning/model/lyz/src/tianlian_voice_bridge:${PYTHONPATH:-}"
# Device UI must not load the host .env (cloud keys stay on the server).
CLIENT_ENV="${VOICE_HUB_CLIENT_ENV:-$HOME/.config/voice_hub/client.env}"
if [ -f "$CLIENT_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$CLIENT_ENV"
  set +a
fi
exec python3 -m device_ui
