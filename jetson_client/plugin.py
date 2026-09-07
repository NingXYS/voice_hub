from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

DEFAULT_MODULES = {
    "tianlian_dual": "tianlian_voice_bridge",
    "demo_single": "bridges.demo_single",
}


def ensure_bridge_on_path() -> None:
    """Add VOICE_HUB_BRIDGE_PATH to sys.path if set. No hardcoded home directories."""
    raw = (os.environ.get("VOICE_HUB_BRIDGE_PATH") or "").strip()
    if not raw:
        return
    path = Path(raw).expanduser()
    if not path.is_dir():
        return
    if (path / "tianlian_voice_bridge" / "__init__.py").is_file():
        insert = str(path.resolve())
    elif path.name == "tianlian_voice_bridge" and (path / "__init__.py").is_file():
        insert = str(path.parent.resolve())
    else:
        insert = str(path.resolve())
    if insert not in sys.path:
        sys.path.insert(0, insert)


def load_bridge(robot_id: str, module_name: str | None = None) -> ModuleType:
    ensure_bridge_on_path()
    name = (module_name or os.environ.get("ROBOT_BRIDGE_MODULE") or DEFAULT_MODULES.get(robot_id) or "").strip()
    if not name:
        raise RuntimeError(
            f"本机没有为机械臂 {robot_id} 配置校验插件。请设置 ROBOT_BRIDGE_MODULE，或安装该臂的 voice_bridge。"
        )
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"本机没有机械臂校验插件 {name}。请设置 VOICE_HUB_BRIDGE_PATH，或把该臂的包装进 Python。"
        ) from exc
    if not hasattr(module, "validate_commands") or not hasattr(module, "execute_commands"):
        raise RuntimeError(f"{name} 必须提供 validate_commands 和 execute_commands")
    return module
