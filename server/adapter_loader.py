from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Adapter:
    robot_id: str
    name: str
    schema_version: int
    ack_text: str
    knowledge: list[dict[str, str]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    parse_action: Any = None

    def match_knowledge(self, text: str) -> str | None:
        needle = "".join((text or "").split())
        if not needle:
            return None
        hits: list[tuple[int, str]] = []
        for item in self.knowledge:
            reply = item.get("reply") or ""
            for keyword in item.get("keywords") or []:
                if keyword and keyword in needle:
                    extra = len(needle) - len(keyword)
                    if extra <= 12:
                        hits.append((len(keyword), reply))
        if not hits:
            return None
        hits.sort(key=lambda pair: pair[0], reverse=True)
        return hits[0][1] or None


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"yaml root must be a mapping: {path}")
    return data


def load_adapter(robot_id: str, adapter_dir: Path) -> Adapter:
    folder = adapter_dir / robot_id
    spec_path = folder / "adapter.yaml"
    if not spec_path.is_file():
        raise FileNotFoundError(f"unknown robot_id: {robot_id}")
    spec = _load_yaml(spec_path)
    knowledge_name = spec.get("knowledge_file") or "knowledge.yaml"
    knowledge_path = folder / knowledge_name
    knowledge: list[dict[str, str]] = []
    if knowledge_path.is_file():
        loaded = _load_yaml(knowledge_path)
        items = loaded.get("items") if isinstance(loaded, dict) else loaded
        if isinstance(items, list):
            knowledge = [item for item in items if isinstance(item, dict)]

    parse_action = None
    rules_path = folder / "rules.py"
    if rules_path.is_file():
        import importlib.util

        spec_mod = importlib.util.spec_from_file_location(
            f"voice_hub_rules_{robot_id}", rules_path
        )
        if spec_mod and spec_mod.loader:
            module = importlib.util.module_from_spec(spec_mod)
            spec_mod.loader.exec_module(module)
            parse_action = getattr(module, "parse_action", None)

    return Adapter(
        robot_id=str(spec.get("id") or robot_id),
        name=str(spec.get("name") or robot_id),
        schema_version=int(spec.get("schema_version") or 1),
        ack_text=str(spec.get("ack_text") or "好的，正在执行"),
        knowledge=knowledge,
        tools=list(spec.get("tools") or []),
        raw=spec,
        parse_action=parse_action,
    )
