"""Parse spoken massage flow. Does not move the arm."""
from __future__ import annotations

import re
from typing import Any, Optional

MASSAGE_BODY_PARTS = (
    "left-shoulder",
    "right-shoulder",
    "back",
    "left-waist",
    "right-waist",
)
MASSAGE_FORCE_NEWTONS = {1: 20, 2: 30, 3: 40, 4: 50, 5: 55}
MASSAGE_FORCE_LABELS = {1: "很轻", 2: "轻一点", 3: "适中", 4: "重一点", 5: "很重"}
_BODY_PART_NAMES = {
    "left-shoulder": "左肩",
    "right-shoulder": "右肩",
    "back": "背部",
    "left-waist": "左腰",
    "right-waist": "右腰",
    "all": "全部位置",
}
_BODY_PART_ALIASES = {
    "left-shoulder": ("左肩膀", "左肩"),
    "right-shoulder": ("右肩膀", "右肩"),
    "back": ("后背", "背部"),
    "left-waist": ("左腰部", "左腰"),
    "right-waist": ("右腰部", "右腰"),
}
_ALL_PART_ALIASES = ("全部位置", "所有位置", "全部部位", "所有部位", "全部", "所有")
_MASSAGE_START_EXPLICIT_RE = re.compile(
    r"^(?:请帮我|请|帮我|麻烦|我想|我要|给我)?(?:开始|启动|执行|进行)按摩"
)
_MASSAGE_START_SOFT_RE = re.compile(
    r"^(?:请帮我|请|帮我|麻烦|我想|我要|给我)?(?:来|做)?(?:一下)?按摩"
)
_FORCE_LEVEL_RE = re.compile(r"(?:力度)?([一二三四五12345])(?:级|档)(?:力度)?")
_START_FILLER_RE = re.compile(r"(?:和|以及|跟|还有|与|位置|部位|力度|一下|吧|谢谢|的|地|得)+")
_FORCE_COLLOQUIAL_ALIASES: tuple[tuple[str, int], ...] = (
    ("不轻不重", 3),
    ("轻一点力", 2),
    ("重一点力", 4),
    ("轻点力", 2),
    ("重点力", 4),
    ("轻一点", 2),
    ("轻一些", 2),
    ("重一点", 4),
    ("重一些", 4),
    ("轻轻的", 1),
    ("重重的", 5),
    ("非常轻", 1),
    ("非常重", 5),
    ("最轻", 1),
    ("最重", 5),
    ("很轻", 1),
    ("很重", 5),
    ("偏轻", 2),
    ("偏重", 4),
    ("轻点", 2),
    ("重点", 4),
    ("适中", 3),
    ("中等", 3),
    ("正常", 3),
    ("一般", 3),
)
_EXACT_INTENT = {
    "开始示教": "start_teaching",
    "启动示教": "start_teaching",
    "开始记录点位": "start_teaching",
    "开始记录位置": "start_teaching",
    "开始记录点": "start_teaching",
    "开始点位记录": "start_teaching",
    "开始按摩点位记录": "start_teaching",
    "记录按摩点位": "start_teaching",
    "开始记录按摩点位": "start_teaching",
    "开始记录按摩位置": "start_teaching",
    "记录完成": "finish_teaching",
    "结束示教": "finish_teaching",
    "示教完成": "finish_teaching",
    "记录结束": "finish_teaching",
    "点位记录完成": "finish_teaching",
    "按摩点位记录完成": "finish_teaching",
    "按摩点位完成": "finish_teaching",
    "点位记录结束": "finish_teaching",
    "点位记录好了": "finish_teaching",
    "点位保存完成": "finish_teaching",
    "结束记录点位": "finish_teaching",
    "停止记录点位": "finish_teaching",
    "结束记录按摩点位": "finish_teaching",
    "停止记录按摩点位": "finish_teaching",
    "开始按摩": "start_massage",
    "启动按摩": "start_massage",
    "执行按摩": "start_massage",
    "进行按摩": "start_massage",
    "开始按摩吧": "start_massage",
    "确认开始": "confirm_start",
    "确认按摩": "confirm_start",
    "确认启动": "confirm_start",
    "确定开始": "confirm_start",
    "可以开始按摩": "confirm_start",
    "取消启动": "cancel_start",
    "取消按摩启动": "cancel_start",
    "取消开始按摩": "cancel_start",
    "按摩状态": "status",
    "当前按摩状态": "status",
    "查看按摩状态": "status",
    "点位状态": "status",
    "重置按摩状态": "reset_state",
    "清除按摩异常": "reset_state",
    "清除异常状态": "reset_state",
    "恢复按摩就绪": "reset_state",
}
_SAY = {
    "start_teaching": "好的，开始记录点位。请按左肩、右肩、背部、左腰、右腰的顺序记录。",
    "finish_teaching": "好的，结束示教并构建方案。",
    "confirm_start": "好的，确认开始按摩。",
    "cancel_start": "好的，已取消启动。",
    "status": "好的，查询按摩状态。",
    "reset_state": "好的，已重置按摩状态。",
}
_MISSING = "请同时说明位置和力度，例如开始按摩左肩，轻一点。"
_MOVE_HINT = "上下左右请改说左臂或右臂，以及距离和单位，例如左臂向上10厘米。"


def _normalize(text: str) -> str:
    normalized = re.sub(r"[\s，。！？,.!?、；;：:]", "", text or "")
    for src, dst in (
        ("按摩擦", "按摩"),
        ("按麽", "按摩"),
        ("按魔", "按摩"),
        ("开使按摩", "开始按摩"),
        ("开始按磨", "开始按摩"),
        ("启动按磨", "启动按摩"),
        ("轻一点儿", "轻一点"),
        ("重一点儿", "重一点"),
    ):
        normalized = normalized.replace(src, dst)
    return normalized


def _extract_force_level(remainder: str) -> tuple[Optional[int], str, bool]:
    text = remainder or ""
    force_level: Optional[int] = None
    valid = True
    hits: list[tuple[int, int, int]] = []
    for phrase, level in _FORCE_COLLOQUIAL_ALIASES:
        start = 0
        while True:
            idx = text.find(phrase, start)
            if idx < 0:
                break
            hits.append((idx, idx + len(phrase), level))
            start = idx + len(phrase)
    if hits:
        hits.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        kept: list[tuple[int, int, int]] = []
        for start, end, level in hits:
            if any(start >= ks and end <= ke for ks, ke, _ in kept):
                continue
            if any(not (end <= ks or start >= ke) for ks, ke, _ in kept):
                valid = False
                break
            kept.append((start, end, level))
        if valid and kept:
            levels = {level for _, _, level in kept}
            if len(levels) != 1 or len(kept) != 1:
                valid = False
            else:
                force_level = kept[0][2]
                start, end, _ = kept[0]
                text = text[:start] + text[end:]
    force_matches = list(_FORCE_LEVEL_RE.finditer(text))
    if force_matches:
        if force_level is not None or len(force_matches) > 1:
            valid = False
        else:
            value = force_matches[0].group(1)
            force_level = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}.get(
                value, int(value) if value.isdigit() else None
            )
            start, end = force_matches[0].span()
            text = text[:start] + text[end:]
    return force_level, text, valid


def _extract_selection(payload: str) -> tuple[list[str], Optional[int], str, bool]:
    remainder = payload or ""
    parts: list[str] = []
    valid = True
    all_hits = 0
    for alias in _ALL_PART_ALIASES:
        count = remainder.count(alias)
        if count:
            all_hits += count
            remainder = remainder.replace(alias, "")
    for part in MASSAGE_BODY_PARTS:
        hit_count = 0
        for alias in _BODY_PART_ALIASES[part]:
            count = remainder.count(alias)
            if count:
                hit_count += count
                remainder = remainder.replace(alias, "")
        if hit_count:
            if hit_count > 1:
                valid = False
            parts.append(part)
    if all_hits:
        if all_hits > 1 or parts:
            valid = False
        else:
            parts = ["all"]
    force_level, remainder, force_valid = _extract_force_level(remainder)
    if not force_valid:
        valid = False
    remainder = _START_FILLER_RE.sub("", remainder)
    return parts, force_level, remainder, valid


def parse_massage_start_request(text: str) -> Optional[dict[str, Any]]:
    raw = text or ""
    if any(mark in raw for mark in ("?", "？", "吗", "能不能", "可不可以")):
        return None
    normalized = _normalize(raw)
    if not normalized:
        return None
    match = _MASSAGE_START_EXPLICIT_RE.match(normalized)
    explicit = match is not None
    if match is None:
        match = _MASSAGE_START_SOFT_RE.match(normalized)
        if match is None:
            return None
    payload = normalized[match.end() :]
    parts, force_level, remainder, valid = _extract_selection(payload)
    if not explicit and not parts and force_level is None:
        return None
    complete = bool(parts) and force_level in MASSAGE_FORCE_NEWTONS and valid and not remainder
    return {"parts": tuple(parts), "force_level": force_level, "complete": complete}


def parse_massage_intent(text: str) -> Optional[str]:
    normalized = _normalize(text)
    if not normalized:
        return None
    if any(mark in (text or "") for mark in ("?", "？", "吗", "能不能")):
        return None
    return _EXACT_INTENT.get(normalized)


def _cmd(op: str, say: str, **fields: Any) -> list[dict[str, Any]]:
    payload = {"action": "massage", "op": op, "say": say}
    payload.update(fields)
    return [payload]


def parse_action(text: str) -> list[dict[str, Any]] | None:
    """Return massage commands, a clarify reply, or None to fall through."""
    raw = text or ""
    if re.search(r"按摩机械臂|按摩臂", _normalize(raw)) and re.search(r"[上下左右]", raw):
        return [{"reply": _MOVE_HINT}]
    intent = parse_massage_intent(raw)
    if intent == "start_massage":
        return [{"reply": _MISSING}]
    if intent:
        return _cmd(intent, _SAY[intent])
    started = parse_massage_start_request(raw)
    if started is None:
        return None
    if not started.get("complete"):
        return [{"reply": _MISSING}]
    parts = list(started["parts"])
    level = int(started["force_level"])
    names = "、".join(_BODY_PART_NAMES.get(part, part) for part in parts)
    label = MASSAGE_FORCE_LABELS[level]
    say = f"好的，按摩{names}，力度{label}。确认现场安全后请说确认开始。"
    return _cmd(
        "start",
        say,
        parts=parts,
        force_level=level,
        force_n=MASSAGE_FORCE_NEWTONS[level],
    )
