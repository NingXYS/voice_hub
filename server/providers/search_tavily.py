from __future__ import annotations

import json

import requests

from server import config

METASO_SEARCH_URL = "https://metaso.cn/api/v1/search"


class SearchError(Exception):
    """Speakable search failure. `str(exc)` is safe to play over TTS."""


def web_search(query: str) -> str:
    cleaned = " ".join(str(query or "").split())[:200]
    if not cleaned:
        return "搜索词为空。"
    api_key = config.search_api_key()
    if not api_key:
        provider = config.search_provider()
        if provider == "tavily":
            return "联网搜索还没配置，请设置 TAVILY_API_KEY。"
        return "联网搜索还没配置，请设置 METASO_API_KEY（秘塔，mk- 开头）。"
    try:
        if config.search_provider() == "tavily":
            payload = _tavily(cleaned, api_key)
        else:
            payload = _metaso_with_retry(cleaned, api_key)
    except SearchError as exc:
        print(f"⚠️ search: {exc}", flush=True)
        return str(exc)
    except requests.exceptions.SSLError as exc:
        print(f"⚠️ search TLS: {type(exc).__name__}", flush=True)
        return "电脑连不上秘塔搜索，HTTPS 被掐断。请把 metaso.cn 设为直连，或给 Python 配系统代理。"
    except requests.exceptions.Timeout:
        print("⚠️ search timeout", flush=True)
        return "秘塔搜索超时，请稍后再问。"
    except requests.exceptions.ConnectionError as exc:
        print(f"⚠️ search connection: {type(exc).__name__}", flush=True)
        return "电脑连不上秘塔搜索。请检查网络或代理。"
    except (requests.RequestException, ValueError, json.JSONDecodeError, TypeError) as exc:
        print(f"⚠️ search failed: {type(exc).__name__}", flush=True)
        return "搜索服务暂时不可用，请稍后再问。"
    return _format(payload) or "没有找到可用的公开资料。"


def _metaso_with_retry(query: str, api_key: str) -> dict:
    last_timeout: Exception | None = None
    for timeout in (5.0, 8.0):
        try:
            return _metaso(query, api_key, timeout=timeout)
        except requests.exceptions.Timeout as exc:
            last_timeout = exc
            print(f"⚠️ search timeout {timeout:.0f}s, retry" if timeout < 8 else "⚠️ search timeout", flush=True)
    raise last_timeout or requests.exceptions.Timeout("search timeout")


def _metaso(query: str, api_key: str, timeout: float = 8.0) -> dict:
    response = requests.post(
        METASO_SEARCH_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json={
            "q": query,
            "scope": "webpage",
            "includeSummary": True,
            "size": 5,
        },
        timeout=timeout,
    )
    payload = _json_object(response)
    spoken = _metaso_error(response.status_code, payload)
    if spoken:
        raise SearchError(spoken)
    return payload


def _tavily(query: str, api_key: str) -> dict:
    response = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": 5,
            "search_depth": "basic",
            "include_answer": True,
        },
        timeout=12,
    )
    payload = _json_object(response)
    if response.status_code == 401:
        raise SearchError("Tavily 密钥无效，请检查 TAVILY_API_KEY。")
    if response.status_code >= 400:
        raise SearchError("搜索服务暂时不可用，请稍后再问。")
    return payload


def _json_object(response: requests.Response) -> dict:
    try:
        payload = response.json()
    except json.JSONDecodeError:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _metaso_error(status: int, payload: dict) -> str:
    code = payload.get("errCode")
    if code is None:
        code = payload.get("code")
    try:
        code_i = int(code) if code is not None and str(code) != "" else 0
    except (TypeError, ValueError):
        code_i = 0
    if status == 401 or code_i == 401:
        return "秘塔搜索密钥无效或已过期。请到 metaso.cn 重新复制 METASO_API_KEY，必须是 mk- 开头。"
    if status in {402, 429} or code_i == 4002:
        return "秘塔搜索次数已用完，请到控制台充值或等免费额度刷新。"
    if code_i == 4009:
        return "这个问题秘塔不能搜，请换个问法。"
    if status == 403 or code_i == 403:
        return "秘塔账号被限制，请登录控制台查看。"
    if status == 400 or code_i == 400:
        return "秘塔搜索参数不对，请稍后再问。"
    if status >= 400 or (code_i not in {0, 200} and code is not None):
        return "搜索服务暂时不可用，请稍后再问。"
    return ""


def _format(payload: dict) -> str:
    lines: list[str] = []
    for key in ("answer", "summary"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(value.strip())
            break
    rows = payload.get("results") or payload.get("webpages") or payload.get("data") or []
    if isinstance(rows, dict):
        rows = rows.get("list") or rows.get("results") or []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or row.get("name") or "").strip()
        snippet = str(
            row.get("snippet") or row.get("summary") or row.get("content") or row.get("desc") or ""
        ).strip()
        if title or snippet:
            lines.append(f"{title}：{snippet}"[:400])
        if len(lines) >= 5:
            break
    return "\n".join(lines)
