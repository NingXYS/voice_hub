"""Demo single-arm bridge with a JSON format that tianlian must reject."""
from __future__ import annotations

from typing import Any

ROBOT_ID = "demo_single"
SCHEMA_VERSION = 1


def validate_command(command: Any) -> dict | None:
    if not isinstance(command, dict):
        return None
    if set(command) != {"cmd", "axis", "mm"}:
        return None
    if command.get("cmd") != "jog":
        return None
    if command.get("axis") not in {"x", "y", "z"}:
        return None
    mm = command.get("mm")
    if not isinstance(mm, (int, float)) or isinstance(mm, bool):
        return None
    return dict(command)


def validate_commands(commands: Any) -> list[dict] | None:
    if not isinstance(commands, list) or not commands:
        return None
    checked = []
    for command in commands:
        valid = validate_command(command)
        if valid is None:
            return None
        checked.append(valid)
    return checked


def execute_commands(commands: list[dict], publisher=None) -> str:
    checked = validate_commands(commands)
    if checked is None:
        return "rejected"
    return "dry_run"
