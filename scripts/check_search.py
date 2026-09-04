#!/usr/bin/env python3
"""Check METASO_API_KEY shape and connectivity. Never prints the key."""
from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    for path in (Path("/etc/voice_hub.env"), ROOT / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def _dns(host: str) -> str:
    try:
        return socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)[0][4][0]
    except OSError as exc:
        return f"failed:{exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check 秘塔 / Tavily search config")
    parser.add_argument("--live", action="store_true", help="POST a real search (uses quota)")
    parser.add_argument("--query", default="北京天气")
    args = parser.parse_args(argv)
    _load_env()

    from server import config
    from server.providers.search_tavily import METASO_SEARCH_URL, web_search

    key = config.search_api_key()
    provider = config.search_provider()
    print(f"provider={provider}")
    print(f"key_set={bool(key)}")
    print(f"key_len={len(key)}")
    print(f"key_prefix_ok={key.startswith('mk-') if provider != 'tavily' else key.startswith('tvly-')}")
    print(f"key_has_space={any(ch.isspace() for ch in key)}")

    host = "api.tavily.com" if provider == "tavily" else "metaso.cn"
    ip = _dns(host)
    print(f"dns {host} -> {ip}")
    if isinstance(ip, str) and ip.startswith("198.18."):
        print("warn: Clash fake-ip. If HTTPS fails, bypass this domain or set HTTPS_PROXY.")

    import requests

    try:
        probe = requests.get(METASO_SEARCH_URL if provider != "tavily" else "https://api.tavily.com/", timeout=8)
        print(f"tls_probe HTTP {probe.status_code} (405 on metaso GET is normal)")
    except requests.exceptions.SSLError as exc:
        print(f"tls_probe SSLError {type(exc).__name__}")
        print("fail: HTTPS to search API is blocked. Key is probably fine; fix proxy/fake-ip.")
        return 2
    except requests.RequestException as exc:
        print(f"tls_probe {type(exc).__name__}")
        return 2

    if not args.live:
        print("shape/tls only. Pass --live to POST a real query (uses 秘塔 credits).")
        return 0 if key else 1

    text = web_search(args.query)
    preview = " ".join(text.split())[:160]
    print(f"live_preview={preview}")
    bad = ("密钥无效", "还没配置", "连不上", "超时", "暂时不可用", "次数已用完")
    return 0 if text and not any(token in text for token in bad) else 1


if __name__ == "__main__":
    raise SystemExit(main())
