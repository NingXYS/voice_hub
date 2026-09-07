#!/usr/bin/env bash
# Copy the device client (not the host brain, not cloud keys) to the arm PC.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${1:-li@10.101.67.221}"
DEST="${2:-}"

ssh -o BatchMode=yes -o ConnectTimeout=8 "$REMOTE" 'true' >/dev/null

if [ -z "$DEST" ]; then
  DEST="$(ssh -o BatchMode=yes "$REMOTE" 'for d in "$HOME/voice_hub" "$HOME/model/voice_hub"; do [ -d "$d" ] && echo "$d" && exit 0; done; echo "$HOME/voice_hub"')"
fi

echo "sync $ROOT -> $REMOTE:$DEST" >&2
ssh -o BatchMode=yes "$REMOTE" "mkdir -p '$DEST/scripts'"

rsync -az --omit-dir-times \
  --exclude '.env' \
  --exclude '.venv' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.git/' \
  "$ROOT/jetson_client/" "$REMOTE:$DEST/jetson_client/"

rsync -az --omit-dir-times \
  --exclude '__pycache__/' --exclude '*.pyc' \
  "$ROOT/device_ui/" "$REMOTE:$DEST/device_ui/"

rsync -az --omit-dir-times \
  --exclude '__pycache__/' --exclude '*.pyc' \
  "$ROOT/bridges/" "$REMOTE:$DEST/bridges/"

rsync -az \
  "$ROOT/scripts/run_client.sh" \
  "$ROOT/scripts/run_device_ui.sh" \
  "$REMOTE:$DEST/scripts/"

if [ -f "$ROOT/requirements.txt" ]; then
  rsync -az "$ROOT/requirements.txt" "$REMOTE:$DEST/requirements.txt"
fi

echo "synced client trees. did not copy .env / .venv / server / adapters / KWS models." >&2
echo "remote dest=$DEST" >&2
