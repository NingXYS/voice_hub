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
SWITCHES = ("VOICE_ENABLE_ROBOT_COMMAND", "VOICE_ENABLE_MASSAGE")
SPEAKER_MODES = frozenset({"off", "shadow", "enforce"})


def _truthy(value: str) -> bool:
    return (value or "0").strip().lower() in {"1", "true", "yes", "on"}


def load_switches() -> dict[str, str]:
    """Read live/demo switches from client.env only. Default off."""
    flags = {key: "0" for key in SWITCHES}
    path = settings_path()
    if not path.is_file():
        return flags
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key in flags:
            flags[key] = "1" if _truthy(value.strip().strip('"').strip("'")) else "0"
    return flags


def load_speaker_mode() -> str:
    """Read speaker gate mode from client.env only. Default off."""
    mode = "off"
    path = settings_path()
    if not path.is_file():
        return mode
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() != "VOICE_SPEAKER_VERIFY_MODE":
            continue
        raw = value.strip().strip('"').strip("'").lower()
        if raw in SPEAKER_MODES:
            mode = raw
    return mode


def set_speaker_mode(mode: str) -> str:
    raw = (mode or "").strip().lower()
    if raw not in SPEAKER_MODES:
        raise SettingsError("声纹模式必须是 off、shadow 或 enforce。")
    upsert_extra("VOICE_SPEAKER_VERIFY_MODE", raw)
    os.environ["VOICE_SPEAKER_VERIFY_MODE"] = raw
    return raw


def upsert_extra(key: str, value: str) -> None:
    """Insert or replace a preserved extra key in client.env."""
    if key in ALLOWED:
        raise SettingsError(f"不支持的配置项：{key}")
    upper = key.upper()
    if any(hint in upper for hint in FORBIDDEN_HINTS):
        raise SettingsError("设备端不能保存主机云密钥或未知配置项。")
    path = settings_path()
    lines: list[str] = []
    if path.is_file():
        lines = path.read_text(encoding="utf-8").splitlines()
    found = False
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            existing = stripped.split("=", 1)[0].strip()
            if existing == key:
                out.append(f"{key}={value}")
                found = True
                continue
        out.append(line)
    if not found:
        if out and out[-1].strip():
            out.append("")
        out.append(f"{key}={value}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def apply_device_flags() -> None:
    """Honor client.env switches. Ignore leftover host process env."""
    flags = load_switches()
    os.environ["VOICE_ENABLE_ROBOT_COMMAND"] = flags["VOICE_ENABLE_ROBOT_COMMAND"]
    os.environ["VOICE_ENABLE_MASSAGE"] = flags["VOICE_ENABLE_MASSAGE"]
    os.environ["VOICE_SPEAKER_VERIFY_MODE"] = load_speaker_mode()


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
        warning = "本机联调可以用 ws://127.0.0.1:8765。如果主机在另一台电脑，请改成那台电脑的局域网 IP。"
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
    extras = _preserved_lines(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# voice_hub device client — no cloud API keys",
        f"VOICE_HUB_URL={current.get('VOICE_HUB_URL') or ''}",
        f"ROBOT_ID={current.get('ROBOT_ID') or 'tianlian_dual'}",
        f"ROBOT_SCHEMA_VERSION={current.get('ROBOT_SCHEMA_VERSION') or '1'}",
        f"APLAY_DEVICE={current.get('APLAY_DEVICE') or 'default'}",
    ]
    if extras:
        lines.append("")
        lines.extend(extras)
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return public_view(current)


def _preserved_lines(path: Path) -> list[str]:
    """Keep local device keys (KWS path, bridge, wake word) when saving URL/robot."""
    if not path.is_file():
        return []
    kept: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key or key in ALLOWED or key in seen:
            continue
        upper = key.upper()
        if any(hint in upper for hint in FORBIDDEN_HINTS):
            continue
        seen.add(key)
        kept.append(f"{key}={value.strip()}")
    return kept


def _reject_key(key: str) -> None:
    upper = key.upper()
    if any(hint in upper for hint in FORBIDDEN_HINTS) or key not in ALLOWED:
        raise SettingsError("设备端不能保存主机云密钥或未知配置项。")
    raise SettingsError(f"不支持的配置项：{key}")
