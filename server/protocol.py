"""JSON text frames + binary PCM. Robot commands are opaque JSON lists."""
from __future__ import annotations

from typing import Any


PROTOCOL_PATH = "/v1"
AUDIO_FORMAT = {
    "format": "pcm_s16le",
    "sample_rate": 16000,
    "channels": 1,
}


def hello_ok(robot_id: str, schema_version: int, adapter_name: str) -> dict[str, Any]:
    return {
        "type": "hello_ok",
        "robot_id": robot_id,
        "schema_version": schema_version,
        "adapter": adapter_name,
        "audio_params": AUDIO_FORMAT,
    }


def hello_reject(reason: str) -> dict[str, Any]:
    return {"type": "hello_reject", "reason": reason}


def stt(text: str) -> dict[str, Any]:
    return {"type": "stt", "text": text}


def tts_text(text: str) -> dict[str, Any]:
    return {"type": "tts_text", "text": text}


def robot_command(commands: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "robot_command", "commands": commands}


def error(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}


def turn_end() -> dict[str, Any]:
    return {"type": "turn_end"}


def memory_state(city: str = "", name: str = "", turns: list | None = None, path: str = "") -> dict[str, Any]:
    return {
        "type": "memory",
        "city": city or "",
        "name": name or "",
        "turns": list(turns or [])[-16:],
        "path": path or "",
    }
