from __future__ import annotations

import importlib
import os
from types import ModuleType

DEFAULT_MODULES = {
    "tianlian_dual": "tianlian_voice_bridge",
    "demo_single": "bridges.demo_single",
}


def load_bridge(robot_id: str, module_name: str | None = None) -> ModuleType:
    name = (module_name or os.environ.get("ROBOT_BRIDGE_MODULE") or DEFAULT_MODULES.get(robot_id) or "").strip()
    if not name:
        raise RuntimeError(
            f"No robot bridge module for robot_id={robot_id!r}. "
            "Set ROBOT_BRIDGE_MODULE or install the arm's voice_bridge package."
        )
    module = importlib.import_module(name)
    if not hasattr(module, "validate_commands") or not hasattr(module, "execute_commands"):
        raise RuntimeError(f"{name} must export validate_commands and execute_commands")
    return module
