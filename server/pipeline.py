from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

from server.adapter_loader import Adapter
from server.memory import ConversationMemory, city_in_text, extract_city
from server.providers.llm_openai import OpenAICompatLLM, split_sentences, split_speakable
from server.providers.search_tavily import web_search

FILLERS = frozenset({"嗯", "啊", "哦", "呃", "额", "唔", "哈", "呵", "欸", "唉", "嗯嗯", "这个", "那个"})
CHAT_HINTS = (
    "天气", "新闻", "搜索", "查一下", "百科", "为什么", "怎么样",
    "多少钱", "是什么", "是谁", "讲个", "笑话", "聊聊", "热点", "最新",
)
SEARCH_HINTS = (
    "天气", "新闻", "搜索", "查一下", "百科", "热点", "最新", "多少钱",
    "中秋", "春节", "元宵", "端午", "国庆", "元旦", "除夕", "清明", "圣诞",
    "还有多少天", "几天后", "几号", "日期", "今年", "农历", "星期",
)
WEATHER_HINTS = ("天气", "气温", "下雨", "下雪", "温度", "穿什么", "出门")
STALL_PHRASES = ("正在搜索", "正在查询", "我去搜", "稍等一下", "请稍候", "正在为您查询")
CONTROL_HINTS = ("臂", "关节", "厘米", "毫米", "回零", "归零", "夹爪", "夹子")
UNCLEAR = "刚才没有听清，请再说一次。"
ASK_CITY = "你在哪个城市？说一次我记住，下次问天气就不用再说了。"
LLM_DOWN = "聊天服务还没配置。请在电脑上设置 CHAT_LLM_API_KEY 和 CHAT_LLM_MODEL（百炼 Qwen，例如 qwen-flash）。"


@dataclass
class TurnResult:
    route: str
    reply: str = ""
    commands: list[dict[str, Any]] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)
    search_q: str = ""
    takeover: bool = False


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def looks_like_chat(text: str) -> bool:
    n = _norm(text)
    if any(h in n for h in CHAT_HINTS):
        return True
    return bool(re.search(r"(?:吗|呢|什么|怎么)[啊呀吧呢]?$", n))


def looks_like_control(text: str) -> bool:
    n = _norm(text)
    return any(h in n for h in CONTROL_HINTS)


def looks_like_search(text: str) -> bool:
    n = _norm(text)
    return any(h in n for h in SEARCH_HINTS)


def looks_like_weather(text: str) -> bool:
    n = _norm(text)
    return any(h in n for h in WEATHER_HINTS)


def today_label(now: datetime | None = None) -> str:
    now = now or datetime.now()
    week = "一二三四五六日"[now.weekday()]
    return f"{now.year}年{now.month}月{now.day}日星期{week}"


def date_label(when: datetime) -> str:
    return f"{when.year}年{when.month}月{when.day}日"


_FACT_MARKS = (
    "天气", "气温", "温度", "℃", "°C", "阴", "晴", "雨", "雪", "多云", "雾", "风",
    "度", "高温", "低温", "穿",
)
_JUNK_MARKS = ("景点", "美食", "特产", "排行榜", "点击查看", "页面还提供", "旅游攻略", "十大")


def _search_line_body(line: str) -> str:
    text = (line or "").strip()
    if "：" in text:
        _title, _, rest = text.partition("：")
        if rest.strip():
            return rest.strip()
    return text


def spoken_search_reply(notes: str, *, city: str = "") -> str:
    text = (notes or "").strip()
    if not text:
        return "网上没有查到有用的信息。"
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    head = lines[0]
    if any(mark in head for mark in ("密钥", "超时", "连不上", "次数已用完", "还没配置", "暂时不可用")):
        return head[:80]
    bodies = [_search_line_body(line) for line in lines]
    bodies = [body for body in bodies if body]
    facts = [body for body in bodies if any(mark in body for mark in _FACT_MARKS)]
    blob = "".join(facts or bodies)
    sentences = [part.strip() for part in re.split(r"(?<=[。！？!?])", blob) if part.strip()]
    chosen: list[str] = []
    for sentence in sentences:
        if any(mark in sentence for mark in _JUNK_MARKS):
            continue
        chosen.append(sentence)
        if len("".join(chosen)) >= 180 or len(chosen) >= 3:
            break
    piece = "".join(chosen).strip() or blob
    piece = piece[:220]
    if city and city not in piece:
        piece = f"{city}，{piece}"
    if piece[-1:] not in "。！？!?":
        piece += "。"
    return piece


def search_query(text: str, now: datetime | None = None, city: str = "") -> str:
    now = now or datetime.now()
    cleaned = " ".join((text or "").split())
    parts: list[str] = []
    needle = _norm(cleaned)
    if "明天" in needle:
        stamp = date_label(now + timedelta(days=1))
        if stamp not in cleaned:
            parts.append(stamp)
    elif any(mark in needle for mark in WEATHER_HINTS):
        stamp = date_label(now)
        if stamp not in cleaned:
            parts.append(stamp)
    elif str(now.year) not in cleaned:
        parts.append(f"{now.year}年")
    if city and city not in cleaned:
        parts.append(city)
    parts.append(cleaned)
    return " ".join(part for part in parts if part).strip()


class Pipeline:
    def __init__(self, llm: OpenAICompatLLM | None = None, search_fn=None):
        self.llm = llm or OpenAICompatLLM()
        self.search_fn = search_fn or web_search

    def decide(
        self,
        text: str,
        adapter: Adapter,
        memory: ConversationMemory | None = None,
    ) -> TurnResult:
        cleaned = (text or "").strip()
        if memory is not None:
            memory.observe_user(cleaned)
        if not cleaned or _norm(cleaned) in FILLERS:
            return TurnResult(route="ambient", reply=UNCLEAR, sentences=split_speakable(UNCLEAR), takeover=True)

        stated = extract_city(cleaned)
        if stated and re.fullmatch(rf"(?:我在|我住在|我住|我是){re.escape(stated)}(?:市|人)?", _norm(cleaned)):
            reply = f"好，记住你在{stated}。"
            return TurnResult(
                route="memory",
                reply=reply,
                sentences=split_speakable(reply),
                takeover=True,
            )

        knowledge = adapter.match_knowledge(cleaned)
        if knowledge:
            sentences = split_speakable(knowledge)
            return TurnResult(route="knowledge", reply=knowledge, sentences=sentences, takeover=True)

        if callable(adapter.parse_action):
            parsed = adapter.parse_action(cleaned)
            if isinstance(parsed, list) and parsed and all(isinstance(x, dict) for x in parsed):
                if any("reply" in item and "action" not in item and len(item) == 1 for item in parsed):
                    reply = str(parsed[0].get("reply") or "") or UNCLEAR
                    return TurnResult(
                        route="clarify",
                        reply=reply,
                        sentences=split_speakable(reply),
                        takeover=True,
                    )
                commands = [item for item in parsed if "reply" not in item]
                if commands:
                    ack = adapter.ack_text
                    return TurnResult(
                        route="local_action",
                        reply=ack,
                        commands=commands,
                        sentences=[ack],
                        takeover=True,
                    )

        city = (memory.city if memory is not None else "") or city_in_text(cleaned)
        want_search = looks_like_search(cleaned) and not looks_like_control(cleaned)
        if want_search and looks_like_weather(cleaned) and not city:
            return TurnResult(
                route="ask_city",
                reply=ASK_CITY,
                sentences=split_speakable(ASK_CITY),
                takeover=True,
            )
        if want_search:
            query = search_query(cleaned, city=city)
            return TurnResult(route="search", search_q=query, takeover=False)
        return TurnResult(route="chat", takeover=False)

    def handle_text(
        self,
        text: str,
        adapter: Adapter,
        *,
        on_sentence: Callable[[str], None] | None = None,
        memory: ConversationMemory | None = None,
    ) -> TurnResult:
        decision = self.decide(text, adapter, memory)
        if decision.takeover:
            # Local actions must send robot_command before ack audio.
            if decision.route != "local_action" and on_sentence:
                for part in decision.sentences:
                    on_sentence(part)
            if memory is not None:
                memory.observe_assistant(decision.reply)
            return decision

        cleaned = (text or "").strip()
        today = today_label()
        extra = (
            f"今天是{today}。这是服务器上的真实日期，禁止用模型记忆里的2023或2024年。"
            f"当前机械臂是{adapter.name}。规格问答已由知识库处理。"
            f"不要输出其它型号的控制 JSON。"
        )
        if memory is not None:
            extra += memory.prompt_block()
        enable_search = looks_like_chat(cleaned) and not looks_like_control(cleaned) and decision.route != "search"

        if looks_like_control(cleaned) and not looks_like_chat(cleaned):
            extra += "用户像在下控制指令但本地规则没有命中。请用一句话要求补充手臂、方向、距离或关节角度，不要编造动作 JSON。"

        def emit(sentence: str) -> None:
            if not on_sentence or not sentence:
                return
            if any(p in sentence for p in STALL_PHRASES):
                return
            on_sentence(sentence)

        if decision.route == "search":
            emit("我查一下")
            query = decision.search_q or search_query(cleaned, city=memory.city if memory else "")
            import time as _time

            t0 = _time.monotonic()
            notes = self.search_fn(query)
            print(f"search_q={query!r} search_ms={int((_time.monotonic() - t0) * 1000)}", flush=True)
            city = memory.city if memory is not None else ""
            answer = spoken_search_reply(notes, city=city)
            emit(answer)
            if memory is not None:
                memory.observe_assistant(answer)
            sentences = [s for s in ("我查一下", answer) if s]
            return TurnResult(route="search", reply=answer, sentences=sentences, search_q=query)

        if not self.llm.configured:
            result = self._reply("chat", LLM_DOWN, on_sentence)
            if memory is not None:
                memory.observe_assistant(result.reply)
            return result

        history = memory.messages()[:-1] if memory is not None else None
        text_out = self.llm.chat(
            cleaned,
            on_sentence=emit,
            extra_system=extra,
            enable_search=enable_search,
            search_fn=self.search_fn if enable_search else None,
            history=history,
        )
        sentences = split_sentences(text_out) or ([text_out] if text_out else [])
        if memory is not None:
            memory.observe_assistant(text_out)
        return TurnResult(route=decision.route if decision.route == "search" else "chat", reply=text_out, sentences=sentences, search_q=decision.search_q)

    @staticmethod
    def _reply(route: str, text: str, on_sentence) -> TurnResult:
        sentences = split_speakable(text)
        if on_sentence:
            for part in sentences:
                on_sentence(part)
        return TurnResult(route=route, reply=text, sentences=sentences)
