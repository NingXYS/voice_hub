from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from server import config

# Provincial capitals and common prefecture-level cities (no 市 suffix).
CITIES = (
    "北京", "上海", "天津", "重庆",
    "石家庄", "太原", "呼和浩特", "沈阳", "大连", "长春", "吉林", "哈尔滨",
    "南京", "苏州", "无锡", "杭州", "宁波", "温州", "合肥", "福州", "厦门", "南昌",
    "济南", "青岛", "郑州", "武汉", "长沙", "广州", "深圳", "珠海", "佛山", "东莞",
    "南宁", "海口", "三亚", "成都", "绵阳", "贵阳", "昆明", "拉萨",
    "西安", "兰州", "西宁", "银川", "乌鲁木齐",
    "香港", "澳门", "台北",
)
_CITY_BY_LEN = tuple(sorted(CITIES, key=len, reverse=True))
_EXPLICIT = (
    re.compile(r"我住在(.{1,6}?)(?:市|县)?(?:[的了啊呀呢吧。！？,.!?]|$)"),
    re.compile(r"我住(.{1,6}?)(?:市|县)?(?:[的了啊呀呢吧。！？,.!?]|$)"),
    re.compile(r"我在(.{1,6}?)(?:市|县)?(?:[的了啊呀呢吧。！？,.!?]|$)"),
    re.compile(r"我是(.{1,6}?)人"),
    re.compile(r"我从(.{1,6}?)(?:市|县)?来"),
)
MAX_TURNS = 8


def city_in_text(text: str) -> str:
    needle = re.sub(r"\s+", "", text or "")
    if not needle:
        return ""
    for city in _CITY_BY_LEN:
        if city in needle:
            return city
    return ""


def extract_city(text: str) -> str:
    needle = re.sub(r"\s+", "", text or "")
    if not needle:
        return ""
    for pattern in _EXPLICIT:
        match = pattern.search(needle)
        if not match:
            continue
        hit = city_in_text(match.group(1))
        if hit:
            return hit
    return city_in_text(needle)


@dataclass
class ConversationMemory:
    robot_id: str
    city: str = ""
    name: str = ""
    turns: list[dict[str, str]] = field(default_factory=list)
    dialog_id: str = ""
    path: Path | None = None

    @classmethod
    def load(cls, robot_id: str, directory: Path | None = None) -> "ConversationMemory":
        robot_id = str(robot_id or "").strip() or "default"
        folder = directory or config.memory_dir()
        path = folder / f"{robot_id}.json"
        data: dict = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    data = loaded
            except (OSError, json.JSONDecodeError, UnicodeError):
                data = {}
        turns = data.get("turns") if isinstance(data.get("turns"), list) else []
        clean: list[dict[str, str]] = []
        for item in turns:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            content = str(item.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                clean.append({"role": role, "content": content})
        return cls(
            robot_id=robot_id,
            city=str(data.get("city") or "").strip(),
            name=str(data.get("name") or "").strip(),
            turns=clean[-MAX_TURNS * 2 :],
            dialog_id=str(data.get("dialog_id") or "").strip() if len(str(data.get("dialog_id") or "")) > 8 else "",
            path=path,
        )

    def observe_user(self, text: str) -> None:
        cleaned = " ".join((text or "").split()).strip()
        if not cleaned:
            return
        city = extract_city(cleaned)
        if city:
            self.city = city
        if self.turns and self.turns[-1].get("role") == "user" and self.turns[-1].get("content") == cleaned:
            return
        self.turns.append({"role": "user", "content": cleaned})
        self._trim()

    def observe_assistant(self, text: str) -> None:
        cleaned = " ".join((text or "").split()).strip()
        if cleaned:
            self.turns.append({"role": "assistant", "content": cleaned})
            self._trim()
        self.save()

    def messages(self) -> list[dict[str, str]]:
        return [dict(item) for item in self.turns]

    def prompt_block(self) -> str:
        parts: list[str] = []
        if self.city:
            parts.append(f"用户所在城市：{self.city}。问天气、明天、穿衣时默认这个城市，不要再问在哪。")
        else:
            parts.append("还不知道用户在哪个城市。问天气且用户没说城市时，先问城市，不要编造地点。")
        if self.name:
            parts.append(f"用户称呼：{self.name}。")
        return "".join(parts)

    def clear(self) -> None:
        self.city = ""
        self.name = ""
        self.turns = []
        self.dialog_id = ""
        self.save()

    def apply(self, city: str | None = None, name: str | None = None) -> None:
        if city is not None:
            self.city = str(city).strip()
        if name is not None:
            self.name = str(name).strip()
        self.save()

    def snapshot(self) -> dict:
        return {
            "city": self.city,
            "name": self.name,
            "turns": list(self.turns[-16:]),
            "path": str(self.path) if self.path else "",
        }

    def save(self) -> None:
        path = self.path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "robot_id": self.robot_id,
            "city": self.city,
            "name": self.name,
            "dialog_id": self.dialog_id,
            "turns": self.turns[-MAX_TURNS * 2 :],
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _trim(self) -> None:
        limit = MAX_TURNS * 2
        if len(self.turns) > limit:
            self.turns = self.turns[-limit:]
