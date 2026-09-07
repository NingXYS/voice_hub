"""Server-side hints that emit tianlian JSON. The arm still re-validates."""
from __future__ import annotations

import math
import re

_CARTESIAN_MAP = {"上": ("z", 1), "下": ("z", -1), "左": ("x", -1), "右": ("x", 1), "前": ("y", 1), "后": ("y", -1)}
_CHINESE_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000}
_NUMBER = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千点]+)"
_RE_CARTESIAN = re.compile(
    rf"(左臂|右臂)(?:向|往)?([上下左右前后])(?:抬|降|移动|移|走)?({_NUMBER})\s*(毫米|厘米|米)"
)
_RE_JOINT = re.compile(rf"(左臂|右臂)关节(\d)(反转)?(?:旋转)?({_NUMBER})\s*度")
_SCALE = {"毫米": 0.001, "厘米": 0.01, "米": 1.0}
_CLARIFY = "请说明左臂还是右臂，以及具体动作、方向、距离或关节角度。"


def parse_action(text: str):
    raw = re.sub(r"\s+", "", text or "")
    raw = raw.replace("左手", "左臂").replace("右手", "右臂")
    if not raw:
        return None
    massage = _parse_massage(text)
    if massage is not None:
        return massage
    if re.fullmatch(r"(左臂|右臂)向?[上下左右前后]", raw) or raw in {"左臂", "右臂", "动一下"}:
        return [{"reply": _CLARIFY}]
    positions = [m.start() for m in re.finditer(r"(左臂|右臂)", raw)]
    if not positions:
        return None
    segments = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(raw)
        seg = raw[start:end]
        if seg:
            segments.append(seg)
    if "".join(segments) != raw:
        return None
    commands = []
    for seg in segments:
        cmd = _one(seg)
        if cmd is None:
            return None
        commands.append(cmd)
    return commands or None


def _one(seg: str):
    m = _RE_JOINT.fullmatch(seg)
    if m:
        joint = int(m.group(2))
        if joint < 1 or joint > 6:
            return None
        deg = _number(m.group(4))
        if deg is None:
            return None
        value = round(deg * math.pi / 180.0, 6)
        if m.group(3):
            value = -value
        return {
            "action": "move_joint",
            "arm": "left" if m.group(1) == "左臂" else "right",
            "joint": joint,
            "value": value,
            "type": "delta",
        }
    m = _RE_CARTESIAN.fullmatch(seg)
    if not m:
        return None
    amount = _number(m.group(3))
    if amount is None:
        return None
    meters = round(float(amount) * _SCALE[m.group(4)], 6)
    axis, sign = _CARTESIAN_MAP[m.group(2)]
    cmd = {"action": "move_cartesian", "arm": "left" if m.group(1) == "左臂" else "right", "x": 0, "y": 0, "z": 0, "relative": True}
    cmd[axis] = meters * sign
    return cmd


_MASSAGE_MOD = None


def _parse_massage(text: str):
    global _MASSAGE_MOD
    if _MASSAGE_MOD is None:
        import importlib.util
        from pathlib import Path

        path = Path(__file__).with_name("massage.py")
        spec = importlib.util.spec_from_file_location("tianlian_dual_massage", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MASSAGE_MOD = module
    parser = getattr(_MASSAGE_MOD, "parse_action", None)
    if not callable(parser):
        return None
    return parser(text)


def _number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        pass
    integer_text, _, fraction_text = value.partition("点")
    integer = 0
    section = 0
    digit = 0
    for char in integer_text:
        if char in _CHINESE_DIGITS:
            digit = _CHINESE_DIGITS[char]
            continue
        unit = _CHINESE_UNITS.get(char)
        if unit is None:
            return None
        section += (digit or 1) * unit
        digit = 0
    integer += section + digit
    if not fraction_text:
        return float(integer)
    frac = "".join(str(_CHINESE_DIGITS[c]) for c in fraction_text if c in _CHINESE_DIGITS)
    if len(frac) != len(fraction_text):
        return None
    return float(f"{integer}.{frac}")
