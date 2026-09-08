from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable

from server import config
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
CONTROL_HINTS = ("臂", "关节", "厘米", "毫米", "回零", "归零", "夹爪", "夹子", "按摩", "示教", "点位", "左肩", "右肩")
UNCLEAR = "刚才没有听清，请再说一次。"
ASK_CITY = "你在哪个城市？说一次我记住，下次问天气就不用再说了。"
LLM_DOWN = "聊天服务还没配置。请在电脑上设置 CHAT_LLM_API_KEY 和 CHAT_LLM_MODEL（百炼 Qwen，例如 qwen-flash）。"
CHAT_RETRY = "刚才聊天没接上，请再说一次。"
_LLM_SETUP_MARKS = ("CHAT_LLM_API_KEY", "聊天密钥无效", "百炼控制台", "还没有填写模型名", "聊天服务还没配置")


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
    "天气", "气温", "温度", "℃", "°C", "阴", "晴", "雨", "雪", "多云", "雾",
    "高温", "低温",
)
_WEATHER_MARKS = ("气温", "℃", "°C", "阴", "晴", "雨", "雪", "多云", "雾", "高温", "低温")
_JUNK_MARKS = (
    "景点", "美食", "特产", "排行榜", "点击查看", "页面还提供", "旅游攻略", "十大",
    "喜神", "福神", "财神", "阳贵", "阴贵", "黄历", "宜：", "忌：", "宜疗", "忌安葬",
    "安葬", "动土", "针灸", "入仓", "求职", "疗病", "结婚", "当日为星期",
    "紫外线", "空气质量", "建议穿", "单层", "套装", "T恤", "湿度",
    "温馨提示", "出行时间", "全国多地", "生活指数", "还包含", "根据天气变化",
    "天气预报及", "点击进入", "详情请", "以上信息", "本文整理",
    "关键词", "头像", "官方账号", "hPa", "m/s",
)
_DATE_FULL = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")
_DATE_MD = re.compile(r"(\d{1,2})月(\d{1,2})日")
_HI_TEMP = re.compile(
    r"最高(?:气温)?(?:一般)?为?\s*(\d{1,2})(?:\.\d+)?(?:\s*[~～\-至到]\s*(\d{1,2})(?:\.\d+)?)?"
)
_LO_TEMP = re.compile(
    r"最低(?:气温)?(?:一般)?为?\s*(\d{1,2})(?:\.\d+)?(?:\s*[~～\-至到]\s*(\d{1,2})(?:\.\d+)?)?"
)
_RANGE_TEMP = re.compile(
    r"气温\s*(\d{1,2})(?:\.\d+)?\s*[~～\-至到]\s*(\d{1,2})(?:\.\d+)?\s*(?:℃|°C|度)"
)
_PLAIN_RANGE = re.compile(
    r"(\d{1,2})(?:\.\d+)?(?:\s*(?:℃|°C|度))?\s*[~～\-至到]\s*(\d{1,2})(?:\.\d+)?\s*(?:℃|°C|度)"
)
_TWO_DAY = re.compile(r"\d{1,2}日\d{1,2}日两天")
_WEATHER_CONDS = (
    "小雨转晴", "阴转晴", "晴转阴", "晴转多云", "多云转晴", "多云转阴", "阴转多云", "阴转小雨",
    "雨夹雪", "雷阵雨", "阵雨", "暴雨", "大雨", "中雨", "小雨",
    "大雪", "中雪", "小雪", "多云", "阴", "晴", "雨", "雪", "雾",
)
_COND_SKIP = (
    "预计", "未来一周", "未来三天", "关键词", "雨具", "备好雨",
    "降雨来袭", "下周一", "即将",
)
_NO_RAIN_MARKS = ("无雨", "没有雨", "不会下雨", "降雨量0", "降水0")
_OTHER_PLACES = ("白山", "白城", "松原", "四平", "通化", "辽源", "延边", "哈尔滨", "沈阳")
_HOLIDAY_ASK = ("多少天", "几天", "几号", "哪天", "什么时候", "何时", "放假")
_HOLIDAY_LABEL = {
    "元旦": "元旦",
    "除夕": "除夕",
    "春节": "春节",
    "元宵": "元宵节",
    "清明": "清明节",
    "端午": "端午节",
    "中秋": "中秋节",
    "国庆": "国庆节",
    "圣诞": "圣诞节",
}
# Public-facing solar dates. Prefer this over web snippets for countdowns.
_HOLIDAYS = {
    2025: {
        "元旦": (1, 1), "除夕": (1, 28), "春节": (1, 29), "元宵": (2, 12),
        "清明": (4, 4), "端午": (5, 31), "中秋": (10, 6), "国庆": (10, 1), "圣诞": (12, 25),
    },
    2026: {
        "元旦": (1, 1), "除夕": (2, 16), "春节": (2, 17), "元宵": (3, 3),
        "清明": (4, 5), "端午": (6, 19), "中秋": (9, 25), "国庆": (10, 1), "圣诞": (12, 25),
    },
    2027: {
        "元旦": (1, 1), "除夕": (2, 5), "春节": (2, 6), "元宵": (2, 20),
        "清明": (4, 5), "端午": (6, 9), "中秋": (9, 15), "国庆": (10, 1), "圣诞": (12, 25),
    },
}


def looks_like_llm_setup(text: str) -> bool:
    blob = text or ""
    return any(mark in blob for mark in _LLM_SETUP_MARKS)


def _chat_fallback_reply() -> str:
    return CHAT_RETRY if config.dialog_configured() else LLM_DOWN


def _search_line_body(line: str) -> str:
    text = (line or "").strip()
    if "：" in text:
        _title, _, rest = text.partition("：")
        if rest.strip():
            return rest.strip()
    return text


def _allowed_dates(query: str, now: datetime | None = None) -> set[tuple[int, int]]:
    now = now or datetime.now()
    allowed: set[tuple[int, int]] = set()
    blob = query or ""
    for match in _DATE_FULL.finditer(blob):
        allowed.add((int(match.group(2)), int(match.group(3))))
    for match in _DATE_MD.finditer(blob):
        allowed.add((int(match.group(1)), int(match.group(2))))
    if "明天" in blob:
        nxt = now + timedelta(days=1)
        allowed.add((nxt.month, nxt.day))
    elif "今天" in blob or looks_like_weather(blob):
        allowed.add((now.month, now.day))
    return allowed


def _weather_target(query: str, now: datetime) -> datetime:
    if "明天" in (query or ""):
        return now + timedelta(days=1)
    return now


def _weather_condition(text: str) -> str:
    blob = text or ""
    if any(mark in blob for mark in _NO_RAIN_MARKS):
        for word in ("阴转晴", "多云转晴", "晴转多云", "多云", "阴", "晴"):
            if word in blob:
                return word
        return ""
    for word in _WEATHER_CONDS:
        if word not in blob:
            continue
        if word == "雨" and not any(mark in blob for mark in ("有雨", "下雨", "阴雨")):
            continue
        return word
    return ""


def _is_precip(cond: str) -> bool:
    return any(mark in (cond or "") for mark in ("雨", "雪"))


def _pick_condition(primary: str, extras: list[str]) -> str:
    ordered = [item for item in ([primary] if primary else []) + extras if item]
    dry = [item for item in ordered if not _is_precip(item)]
    pool = dry or ordered
    if not pool:
        return ""
    for word in _WEATHER_CONDS:
        if word in pool:
            return word
    return pool[0]


def _window_dates(window: str) -> list[tuple[int, int]]:
    dates = [(int(m.group(1)), int(m.group(2))) for m in _DATE_MD.finditer(window)]
    dates.extend((int(m.group(2)), int(m.group(3))) for m in _DATE_FULL.finditer(window))
    return dates


def _window_matches_day(window: str, target: datetime, want_tomorrow: bool) -> bool:
    dates = _window_dates(window)
    if dates:
        return (target.month, target.day) in dates
    if _TWO_DAY.search(window):
        days = {int(n) for n in re.findall(r"(?<!\d)(\d{1,2})日", window)}
        return target.day in days
    if re.search(rf"(?<!\d){target.day}日", window):
        return True
    if want_tomorrow:
        return "明天" in window
    return "今天" in window


def _read_hi_lo(blob: str) -> tuple[int | None, int | None]:
    hi = lo = None
    hit = _HI_TEMP.search(blob)
    if hit:
        a, b = hit.group(1), hit.group(2)
        hi = max(int(a), int(b or a))
    low = _LO_TEMP.search(blob)
    if low:
        a, b = low.group(1), low.group(2)
        lo = min(int(a), int(b or a))
    if hi is None or lo is None:
        rng = _RANGE_TEMP.search(blob) or _PLAIN_RANGE.search(blob)
        if rng:
            a, b = int(rng.group(1)), int(rng.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
    if hi is not None and lo is not None and hi < lo:
        hi, lo = lo, hi
    return hi, lo


def _compose_weather(notes: str, *, city: str, query: str, now: datetime) -> str | None:
    blob = re.sub(r"\s+", "", notes or "")
    if not blob:
        return None
    want_tomorrow = "明天" in (query or "")
    target = _weather_target(query, now)
    day_word = "明天" if want_tomorrow else "今天"
    extras: list[str] = []
    for window in re.split(r"[。！？!?；;\n]", blob):
        if not window or any(mark in window for mark in _JUNK_MARKS) or any(mark in window for mark in _COND_SKIP):
            continue
        if "今天" in window and "明天" in window:
            continue
        if not _window_matches_day(window, target, want_tomorrow):
            continue
        cond = _weather_condition(window)
        if cond:
            extras.append(cond)
    cands: list[tuple[int, bool, bool, int, int, str]] = []
    for window in re.split(r"[。！？!?；;\n]", blob):
        if not window or any(mark in window for mark in ("hPa", "m/s", "关键词")):
            continue
        if city and any(place in window for place in _OTHER_PLACES) and city not in window:
            continue
        two_day = bool(_TWO_DAY.search(window))
        if two_day and not want_tomorrow:
            continue
        days_hit = {int(n) for n in re.findall(r"(?<!\d)(\d{1,2})日", window)}
        dated_ok = (target.month, target.day) in _window_dates(window)
        if (not dated_ok) and len(days_hit) > 1 and not two_day:
            continue
        if not _window_matches_day(window, target, want_tomorrow):
            continue
        hi, lo = _read_hi_lo(window)
        if hi is None or lo is None:
            continue
        cond = _weather_condition(window)
        precip = _is_precip(cond)
        score = 0
        if dated_ok or f"{target.year}年{target.month}月{target.day}日" in window:
            score += 4
        if day_word in window:
            score += 5
        if city and city in window:
            score += 3
        if any(mark in window for mark in ("实况", "当前", "现在", "目前")):
            score += 3
        if two_day:
            score -= 4
        if cond:
            score += 1
        if cond and not precip:
            score += 3
        if precip:
            score -= 3
        cands.append((score, dated_ok, not precip, hi, lo, cond))
    if not cands:
        return None
    if any(item[5] and not _is_precip(item[5]) for item in cands):
        cands = [item for item in cands if not _is_precip(item[5])]
    cands.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _, _, _, hi, lo, cond = cands[0]
    cond = _pick_condition(cond, extras)
    when = "明天" if want_tomorrow else "今天"
    lead = f"{city or ''}{when}{cond}"
    if cond:
        return f"{lead}，最高{hi}度，最低{lo}度。"
    return f"{lead}最高{hi}度，最低{lo}度。"


def _sentence_date_ok(sentence: str, allowed: set[tuple[int, int]]) -> bool:
    if not allowed:
        return True
    found: list[tuple[int, int]] = []
    for match in _DATE_FULL.finditer(sentence):
        found.append((int(match.group(2)), int(match.group(3))))
    if not found:
        for match in _DATE_MD.finditer(sentence):
            found.append((int(match.group(1)), int(match.group(2))))
    if not found:
        return True
    return any(item in allowed for item in found)


def spoken_search_reply(
    notes: str,
    *,
    city: str = "",
    query: str = "",
    now: datetime | None = None,
) -> str:
    text = (notes or "").strip()
    if not text:
        return "网上没有查到有用的信息。"
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    head = lines[0]
    if any(mark in head for mark in ("密钥", "超时", "连不上", "次数已用完", "还没配置", "暂时不可用")):
        return head[:80]
    bodies = [_search_line_body(line) for line in lines]
    bodies = [body for body in bodies if body]
    weather_query = looks_like_weather(query) or any(mark in "".join(bodies) for mark in _WEATHER_MARKS)
    if weather_query:
        composed = _compose_weather(text, city=city, query=query, now=now or datetime.now())
        if composed:
            return composed
    marks = _WEATHER_MARKS if weather_query else _FACT_MARKS
    facts = [body for body in bodies if any(mark in body for mark in marks)]
    blob = "".join(facts or (bodies if not weather_query else []))
    sentences = [part.strip() for part in re.split(r"(?<=[。！？!?;；])", blob) if part.strip()]
    allowed = _allowed_dates(query, now=now)
    want_tomorrow = "明天" in (query or "")
    chosen: list[str] = []

    def consider(sentence: str) -> bool:
        if any(mark in sentence for mark in _JUNK_MARKS):
            return False
        if not _sentence_date_ok(sentence, allowed):
            return False
        if weather_query and not any(mark in sentence for mark in _WEATHER_MARKS):
            return False
        if want_tomorrow and "今天" in sentence and "明天" not in sentence:
            return False
        return True

    ranked = sentences
    if want_tomorrow:
        ranked = [s for s in sentences if "明天" in s] + [s for s in sentences if "明天" not in s]
    for sentence in ranked:
        if not consider(sentence):
            continue
        chosen.append(sentence)
        limit = 90 if weather_query else 160
        if len("".join(chosen)) >= limit or len(chosen) >= (2 if weather_query else 3):
            break
    piece = "".join(chosen).strip()
    if not piece:
        if weather_query:
            return "网上没有查到这次要的天气，请稍后再问。"
        return "网上没有查到有用的信息。"
    limit = 90 if weather_query else 160
    if len(piece) > limit:
        cut = piece[:limit]
        for sep in "。！？!?；;":
            at = cut.rfind(sep)
            if at >= 8:
                cut = cut[: at + 1]
                break
        piece = cut
    if city and city not in piece and any(mark in piece for mark in marks):
        piece = f"{city}，{piece}"
    if piece[-1:] not in "。！？!?":
        piece += "。"
    return piece


def holiday_countdown_reply(text: str, now: datetime | None = None) -> str | None:
    now = now or datetime.now()
    needle = _norm(text)
    if not any(mark in needle for mark in _HOLIDAY_ASK):
        return None
    if "放假" in needle and any(mark in needle for mark in ("安排", "调休", "放几天")):
        return None
    hit = next((name for name in _HOLIDAY_LABEL if name in needle), "")
    if not hit:
        return None
    year = now.year
    names = _HOLIDAYS.get(year) or {}
    if hit not in names:
        year += 1
        names = _HOLIDAYS.get(year) or {}
    if hit not in names:
        return None
    month, day = names[hit]
    target = datetime(year, month, day)
    if target.date() < now.date():
        names = _HOLIDAYS.get(year + 1) or {}
        if hit not in names:
            return None
        month, day = names[hit]
        target = datetime(year + 1, month, day)
    days = (target.date() - now.date()).days
    label = _HOLIDAY_LABEL[hit]
    today = f"{now.month}月{now.day}日"
    when = f"{target.month}月{target.day}日"
    if days <= 0:
        return f"今天是{today}，就是{label}。"
    if days == 1:
        return f"今天是{today}，{label}是明天{when}。"
    return f"今天是{today}，{label}是{when}，还有{days}天。"


def search_query(text: str, now: datetime | None = None, city: str = "") -> str:
    now = now or datetime.now()
    cleaned = " ".join((text or "").split())
    parts: list[str] = []
    needle = _norm(cleaned)
    weather = any(mark in needle for mark in WEATHER_HINTS)
    if weather:
        day = now + timedelta(days=1) if "明天" in needle else now
        if city and city not in cleaned:
            parts.append(city)
        md = f"{day.month}月{day.day}日"
        if md not in cleaned:
            parts.append(md)
        parts.append("天气预报")
        parts.append("最高气温")
        parts.append("阴晴")
        parts.append("实况")
        parts.append(cleaned)
        return " ".join(part for part in parts if part).strip()
    if "明天" in needle:
        stamp = date_label(now + timedelta(days=1))
        if stamp not in cleaned:
            parts.append(stamp)
    elif str(now.year) not in cleaned:
        parts.append(f"{now.year}年")
    if city and city not in cleaned and any(mark in needle for mark in WEATHER_HINTS):
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
                    ack = str(commands[0].get("say") or adapter.ack_text)
                    cleaned_commands = [
                        {key: value for key, value in item.items() if key != "say"} for item in commands
                    ]
                    return TurnResult(
                        route="local_action",
                        reply=ack,
                        commands=cleaned_commands,
                        sentences=split_speakable(ack) if "massage" in {
                            str(item.get("action") or "") for item in cleaned_commands
                        } else [ack],
                        takeover=True,
                    )

        calendar = holiday_countdown_reply(cleaned)
        if calendar:
            return TurnResult(
                route="calendar",
                reply=calendar,
                sentences=split_speakable(calendar),
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
            if looks_like_llm_setup(sentence) and config.dialog_configured():
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
            answer = spoken_search_reply(notes, city=city, query=query)
            emit(answer)
            if memory is not None:
                memory.observe_assistant(answer)
            sentences = [s for s in ("我查一下", answer) if s]
            return TurnResult(route="search", reply=answer, sentences=sentences, search_q=query)

        if not self.llm.configured:
            result = self._reply("chat", _chat_fallback_reply(), on_sentence)
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
        if looks_like_llm_setup(text_out) and config.dialog_configured():
            text_out = CHAT_RETRY
            if on_sentence:
                on_sentence(text_out)
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
