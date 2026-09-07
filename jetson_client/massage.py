"""Run VoiceOnline MassageController on the arm. Default is dry_run."""
from __future__ import annotations

import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

_TEACHING_BINARIES = frozenset({"drag_cartesian_pose_reader"})


def massage_enabled() -> bool:
    return os.environ.get("VOICE_ENABLE_MASSAGE", "0").strip().lower() in {"1", "true", "yes", "on"}


def voice_online_root() -> Path | None:
    raw = (os.environ.get("VOICE_ONLINE_ROOT") or "").strip()
    if raw:
        path = Path(raw).expanduser()
        if path.is_dir():
            return path
    home = Path.home() / "VoiceOnline"
    if home.is_dir():
        return home
    return None


def execute_commands(commands: list[dict[str, Any]]) -> str:
    if not commands:
        return "empty"
    if any(not isinstance(item, dict) or item.get("action") != "massage" for item in commands):
        return "rejected"
    if not massage_enabled():
        return "dry_run"
    controller = _controller()
    if controller is None:
        return "no_bridge"
    last: Any = None
    for item in commands:
        op = str(item.get("op") or "")
        if op in {"start_teaching", "start", "confirm_start"}:
            release_teaching_hold()
        last = _apply(controller, item)
        if isinstance(last, dict) and last.get("reply"):
            print(f"wake massage op={op} {last.get('reply')}", flush=True)
        if _massage_failed(last):
            msg = str(last.get("reply") or last.get("message") or "按摩未执行")
            return "error:" + msg[:80]
    return "ok"


def leftover_teaching_pids() -> list[int]:
    """PIDs of native teaching binaries that hold the arm SDK."""
    found: list[int] = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        try:
            exe = Path(f"/proc/{pid}/exe").resolve()
        except OSError:
            continue
        if exe.name in _TEACHING_BINARIES:
            found.append(pid)
    found.sort()
    return found


def release_teaching_hold() -> list[int]:
    """Free the arm SDK without wiping massage plan / confirmation state."""
    controller = _CONTROLLER
    if controller is not None:
        stop = getattr(controller, "_stop_teaching_process", None)
        if callable(stop):
            try:
                stop()
            except Exception as exc:
                print(f"⚠️ massage stop teaching: {type(exc).__name__}: {exc}", flush=True)
    return _kill_orphan_teaching()


def reclaim_teaching() -> list[int]:
    """Stop leftover teaching processes so a new arm client can connect."""
    return release_teaching_hold()


def close_runtime() -> None:
    """Called when the voice UI stops, so teaching cannot keep the arm locked."""
    global _CONTROLLER
    controller = _CONTROLLER
    if controller is not None:
        closer = getattr(controller, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception as exc:
                print(f"⚠️ massage close: {type(exc).__name__}: {exc}", flush=True)
    killed = _kill_orphan_teaching()
    _CONTROLLER = None
    return killed


def _kill_orphan_teaching() -> list[int]:
    killed: list[int] = []
    for pid in leftover_teaching_pids():
        if _stop_pid(pid):
            killed.append(pid)
            print(f"wake massage reclaim pid={pid} bin=drag_cartesian_pose_reader", flush=True)
    return killed


def teaching_capture() -> dict[str, Any]:
    if not massage_enabled():
        return {"ok": True, "status": "dry_run", "message": "演练模式，没有真正记点。"}
    controller = _controller()
    if controller is None:
        return {"ok": False, "error": "找不到 VoiceOnline 按摩程序。请设置 VOICE_ONLINE_ROOT。"}
    result = controller.capture_teaching_point()
    return {"ok": True, "status": "ok", "detail": result}


def teaching_finish() -> dict[str, Any]:
    if not massage_enabled():
        return {"ok": True, "status": "dry_run", "message": "演练模式，没有结束真示教。"}
    controller = _controller()
    if controller is None:
        return {"ok": False, "error": "找不到 VoiceOnline 按摩程序。请设置 VOICE_ONLINE_ROOT。"}
    result = controller.finish_teaching()
    return {"ok": True, "status": "ok", "detail": result}


_CONTROLLER = None


def _controller():
    global _CONTROLLER
    if _CONTROLLER is not None:
        return _CONTROLLER
    root = voice_online_root()
    if root is None:
        return None
    scripts = root / "scripts"
    if not scripts.is_dir():
        return None
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    os.environ["VOICE_ENABLE_MASSAGE"] = "1"
    os.environ.setdefault("VOICE_MASSAGE_DRY_RUN", "0")
    from voice_massage_controller import MassageController

    _CONTROLLER = MassageController(enabled=True)
    return _CONTROLLER


_FAIL_MARKS = (
    "不能开始",
    "还没有完整",
    "未启用",
    "不可用",
    "仍在运行",
    "没有等待确认",
    "启动失败",
    "参数已失效",
    "请先完成示教",
    "请先结束",
)


def _massage_failed(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("arm_busy"):
        return True
    reply = str(result.get("reply") or result.get("message") or "")
    return bool(result.get("started") is False and any(mark in reply for mark in _FAIL_MARKS))


def _stop_pid(pid: int) -> bool:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            return True
        except OSError:
            return False
        deadline = time.time() + (1.5 if sig == signal.SIGTERM else 1.0)
        while time.time() < deadline:
            if not Path(f"/proc/{pid}").exists():
                return True
            time.sleep(0.05)
    return not Path(f"/proc/{pid}").exists()


def _apply(controller, item: dict[str, Any]) -> Any:
    op = str(item.get("op") or "")
    if op == "start":
        return controller.handle(
            "start_massage",
            parts=list(item.get("parts") or []),
            force_level=item.get("force_level"),
        )
    if op:
        return controller.handle(op)
    return None
