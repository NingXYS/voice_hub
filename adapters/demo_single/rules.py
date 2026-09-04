"""Deliberately different JSON from tianlian_dual."""
from __future__ import annotations

import re

_RE = re.compile(r"(?:向)?([前后左右上下])(?:移动|移)?(\d+(?:\.\d+)?)(?:毫米|厘米|米)?")
_AXIS = {"前": "y", "后": "y", "左": "x", "右": "x", "上": "z", "下": "z"}
_SIGN = {"前": 1, "后": -1, "左": -1, "右": 1, "上": 1, "下": -1}


def parse_action(text: str):
    raw = re.sub(r"\s+", "", text or "")
    m = _RE.search(raw)
    if not m:
        return None
    direction, number = m.group(1), float(m.group(2))
    mm = number
    if "厘米" in raw:
        mm = number * 10
    elif "米" in raw and "毫米" not in raw and "厘米" not in raw:
        mm = number * 1000
    return [{"cmd": "jog", "axis": _AXIS[direction], "mm": mm * _SIGN[direction]}]
