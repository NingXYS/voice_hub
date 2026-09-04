"""Device-local client settings. Never stores host cloud API keys."""
from __future__ import annotations

import os
from pathlib import Path

ALLOWED = frozenset({"VOICE_HUB_URL", "ROBOT_ID", "ROBOT_SCHEMA_VERSION", "APLAY_DEVICE"})
FORBIDDEN_HINTS = (
    "CHAT_LLM",
    "DASHSCOPE",
    "DOUBAO",
    "VOLC",
    "METASO",
    "TAVILY",
    "ASR_",
    "TTS_",
)
DEFAULTS = {
    "VOICE_HUB_URL": "",
    "ROBOT_ID": "tianlian_dual",
    "ROBOT_SCHEMA_VERSION": "1",
    "APLAY_DEVICE": "default",
}


class SettingsError(ValueError):
    pass


def settings_path() -> Path:
    override = os.environ.get("VOICE_HUB_CLIENT_ENV", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".config" / "voice_hub" / "client.env"


def load() -> dict[str, str]:
    data = dict(DEFAULTS)
    for key in ALLOWED:
        env_val = os.environ.get(key)
        if env_val:
            data[key] = env_val.strip()
    path = settings_path()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            if key in ALLOWED:
                data[key] = value.strip().strip('"').strip("'")
    return data


def public_view(data: dict[str, str] | None = None) -> dict[str, str | int]:
    data = data or load()
    try:
        schema = int(data.get("ROBOT_SCHEMA_VERSION") or "1")
    except ValueError:
        schema = 1
    url = data.get("VOICE_HUB_URL") or ""
    warning = ""
    if not url:
        warning = "请填写主机的局域网地址，例如 ws://10.101.67.111:8765。"
    elif "198.18." in url:
        warning = "地址像 Clash 假 IP，其它设备连这个会失败。"
    elif "127.0.0.1" in url or "://localhost:" in url:
        warning = "127.0.0.1 只指向这台设备自己。连另一台主机时请填主机的局域网 IP。"
    return {
        "url": url,
        "robot_id": data.get("ROBOT_ID") or DEFAULTS["ROBOT_ID"],
        "schema_version": schema,
        "warning": warning,
    }


def save(updates: dict) -> dict[str, str | int]:
    if not isinstance(updates, dict):
        raise SettingsError("配置必须是对象。")
    incoming: dict[str, str] = {}
    for raw_key, raw_value in updates.items():
        key = str(raw_key).strip()
        if key in {"url", "VOICE_HUB_URL"}:
            incoming["VOICE_HUB_URL"] = str(raw_value or "").strip()
        elif key in {"robot_id", "ROBOT_ID"}:
            incoming["ROBOT_ID"] = str(raw_value or "").strip()
        elif key in {"schema_version", "ROBOT_SCHEMA_VERSION"}:
            incoming["ROBOT_SCHEMA_VERSION"] = str(raw_value).strip()
        elif key == "APLAY_DEVICE":
            incoming["APLAY_DEVICE"] = str(raw_value or "").strip()
        elif key in ALLOWED:
            incoming[key] = str(raw_value or "").strip()
        else:
            _reject_key(key)
    current = load()
    current.update({k: v for k, v in incoming.items() if v != "" or k == "VOICE_HUB_URL"})
    url = current.get("VOICE_HUB_URL") or ""
    if url and not (url.startswith("ws://") or url.startswith("wss://")):
        raise SettingsError("主机地址必须以 ws:// 或 wss:// 开头。")
    try:
        int(current.get("ROBOT_SCHEMA_VERSION") or "1")
    except ValueError as exc:
        raise SettingsError("schema_version 必须是整数。") from exc
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# voice_hub device client — no cloud API keys",
        f"VOICE_HUB_URL={current.get('VOICE_HUB_URL') or ''}",
        f"ROBOT_ID={current.get('ROBOT_ID') or 'tianlian_dual'}",
        f"ROBOT_SCHEMA_VERSION={current.get('ROBOT_SCHEMA_VERSION') or '1'}",
        f"APLAY_DEVICE={current.get('APLAY_DEVICE') or 'default'}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return public_view(current)


def _reject_key(key: str) -> None:
    upper = key.upper()
    if any(hint in upper for hint in FORBIDDEN_HINTS) or key not in ALLOWED:
        raise SettingsError("设备端不能保存主机云密钥或未知配置项。")
    raise SettingsError(f"不支持的配置项：{key}")
