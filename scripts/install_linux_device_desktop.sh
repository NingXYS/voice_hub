#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APPDIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APPDIR"
chmod +x "$ROOT/scripts/run_device_ui.sh"
sed "s|@ROOT@|$ROOT|g" "$ROOT/packaging/voice-hub-device.desktop" > "$APPDIR/voice-hub-device.desktop"
chmod 644 "$APPDIR/voice-hub-device.desktop"
if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APPDIR" >/dev/null 2>&1 || true
fi
echo "已安装桌面图标: $APPDIR/voice-hub-device.desktop"
echo "启动: $ROOT/scripts/run_device_ui.sh"
echo "页面: http://127.0.0.1:8780"
