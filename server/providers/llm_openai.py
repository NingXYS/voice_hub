from __future__ import annotations

import json
import re
from typing import Any, Callable

import requests

from server import config

CHAT_SYSTEM = """你是机械臂旁的语音助手。可以闲聊和（在需要实时信息时）根据系统提供的网上资料回答。
不要声称自己会抓取、视觉识别。不要输出 markdown。用短句中文回答，方便播放。
日期、节日、倒计时必须用系统给出的「今天是…」，不要用训练数据里的2023或2024年。
机械臂动作由本机规则处理；闲聊时不要编造该臂的 JSON 控制指令。"""

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "搜索公开网页。问天气、新闻、时事、机构时使用。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

_FIRST = ("。", "！", "？", "!", "?", "；", ";", "\n", "，", ",")
_LATER = ("。", "！", "？", "!", "?", "；", ";", "\n")


class SentenceAssembler:
    def __init__(self):
        self.buffer = ""
        self.emitted = 0

    def push(self, delta: str) -> list[str]:
        if not delta:
            return []
        self.buffer += delta
        return self._flush(False)

    def finish(self) -> list[str]:
        return self._flush(True)

    def _flush(self, final: bool) -> list[str]:
        punct = _FIRST if self.emitted == 0 else _LATER
        out: list[str] = []
        while self.buffer:
            hits = [self.buffer.find(p) for p in punct if p in self.buffer]
            if not hits:
                break
            cut = min(hits) + 1
            piece = self.buffer[:cut].strip()
            self.buffer = self.buffer[cut:]
            if piece:
                out.append(piece)
                self.emitted += 1
        if final:
            tail = self.buffer.strip()
            self.buffer = ""
            if tail:
                out.append(tail)
                self.emitted += 1
        return out


def split_sentences(text: str) -> list[str]:
    if not text or not str(text).strip():
        return []
    parts = re.split(r"(?<=[。！？!?；;\n])", str(text).strip())
    return [p.strip() for p in parts if p.strip()]


def split_speakable(text: str) -> list[str]:
    """Split a finished reply the same way streaming does (first comma, then periods)."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    assembler = SentenceAssembler()
    out = assembler.push(cleaned)
    out.extend(assembler.finish())
    return out or [cleaned]


class OpenAICompatLLM:
    """Aliyun Bailian (DashScope) or any OpenAI-compatible /chat/completions endpoint."""

    def __init__(self, session=None):
        self.session = session or requests.Session()
        self.last_error = ""

    @property
    def configured(self) -> bool:
        return bool(config.llm_api_key() and config.llm_model())

    def _url(self) -> str:
        base = config.llm_base_url()
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _payload(self, messages: list[dict[str, Any]], *, tools: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": config.llm_model(),
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": config.llm_max_tokens(),
            "stream": True,
            "thinking": {"type": "disabled"},
            "enable_thinking": False,
            # DashScope Qwen3 compatible-mode also reads this nested flag.
            "extra_body": {"enable_thinking": False},
        }
        if tools:
            payload["tools"] = [WEB_SEARCH_TOOL]
            payload["tool_choice"] = "auto"
        return payload

    def chat(
        self,
        user_text: str,
        *,
        on_sentence: Callable[[str], None] | None = None,
        extra_system: str = "",
        enable_search: bool = False,
        search_fn: Callable[[str], str] | None = None,
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        if not config.llm_api_key():
            self.last_error = "CHAT_LLM_API_KEY is not configured"
            return "聊天服务还没配置。请在电脑上设置 CHAT_LLM_API_KEY（百炼，sk- 开头）。"
        if not config.llm_model():
            self.last_error = "CHAT_LLM_MODEL is not configured"
            return "还没有填写模型名。请设置 CHAT_LLM_MODEL，例如 qwen-flash。"
        system = CHAT_SYSTEM
        if extra_system:
            system = system + "\n" + extra_system
        messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for item in history or []:
            role = str(item.get("role") or "")
            content = str(item.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != user_text:
            messages.append({"role": "user", "content": user_text})
        payload = self._payload(messages, tools=enable_search)
        try:
            text, tool_calls = self._stream(payload, on_sentence if not enable_search else None)
            if tool_calls and search_fn is not None:
                if on_sentence:
                    on_sentence("我查一下")
                messages.append({"role": "assistant", "content": text, "tool_calls": tool_calls})
                for call in tool_calls:
                    fn = call.get("function") or {}
                    query = _tool_query(fn.get("arguments") or "")
                    result = search_fn(query)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id") or "search",
                            "name": fn.get("name") or "web_search",
                            "content": result,
                        }
                    )
                follow = self._payload(messages, tools=False)
                text, _ = self._stream(follow, on_sentence)
            elif text and on_sentence and enable_search and not tool_calls:
                for part in split_sentences(text) or [text]:
                    on_sentence(part)
        except (requests.RequestException, ValueError, TypeError) as exc:
            self.last_error = str(exc)
            return "我这边网络有点慢，请再说一次。"
        return (text or "").strip() or "我这边网络有点慢，请再说一次。"

    def _stream(self, payload: dict[str, Any], on_sentence) -> tuple[str, list[dict[str, Any]]]:
        assembler = SentenceAssembler()
        collected: list[str] = []
        tools: dict[int, dict[str, str]] = {}
        mode: str | None = None
        response = self.session.post(
            self._url(),
            headers={
                "Authorization": f"Bearer {config.llm_api_key()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(5.0, config.llm_timeout()),
            stream=True,
        )
        try:
            if response.status_code != 200:
                self.last_error = f"chat HTTP {response.status_code}"
                print(f"⚠️ LLM {self.last_error}", flush=True)
                return _llm_http_message(response.status_code), []
            for raw in response.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line == "[DONE]":
                    if line == "[DONE]":
                        break
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                tcs = delta.get("tool_calls") or []
                content = delta.get("content") or ""
                # GLM/Qwen thinking traces must not be spoken.
                _ = delta.get("reasoning_content")
                if mode is None:
                    mode = "tools" if tcs else ("content" if content else None)
                if mode == "tools":
                    for item in tcs:
                        if not isinstance(item, dict):
                            continue
                        idx = int(item.get("index") or 0)
                        slot = tools.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if item.get("id"):
                            slot["id"] = str(item["id"])
                        fn = item.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = str(fn["name"])
                        if fn.get("arguments"):
                            slot["arguments"] += str(fn["arguments"])
                    continue
                if content:
                    collected.append(content)
                    if on_sentence:
                        for part in assembler.push(content):
                            on_sentence(part)
            if on_sentence and mode != "tools":
                for part in assembler.finish():
                    on_sentence(part)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
        tool_list = []
        for idx in sorted(tools):
            slot = tools[idx]
            if slot.get("name"):
                tool_list.append(
                    {
                        "id": slot.get("id") or f"call_{idx}",
                        "type": "function",
                        "function": {"name": slot["name"], "arguments": slot.get("arguments") or "{}"},
                    }
                )
        return "".join(collected).strip(), tool_list


def _llm_http_message(status: int) -> str:
    if status in (401, 403):
        return "聊天密钥无效。请到百炼控制台复制新的 CHAT_LLM_API_KEY（sk- 开头）。"
    if status == 429:
        return "聊天请求太多，请稍后再试。"
    if status >= 500:
        return "聊天服务暂时不可用，请稍后再试。"
    return "聊天服务返回异常，请稍后再试。"


def _tool_query(arguments: str) -> str:
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return str(arguments or "").strip()
    if isinstance(parsed, dict):
        return str(parsed.get("query") or parsed.get("q") or "").strip()
    return str(arguments or "").strip()
