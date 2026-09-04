#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT:/home/ning/model/lyz/src/tianlian_voice_bridge:${PYTHONPATH:-}"
if [ -f /etc/voice_hub_client.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/voice_hub_client.env
  set +a
elif [ -f "$ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi
exec python3 "$ROOT/jetson_client/voice_client.py" "$@"
