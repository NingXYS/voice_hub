#!/usr/bin/env python3
"""Check Doubao streaming ASR config. Never prints the key."""
from __future__ import annotations

import argparse
import asyncio
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
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _dns(host: str) -> str:
    try:
        return socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)[0][4][0]
    except OSError as exc:
        return f"failed:{exc}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Doubao / Volcengine streaming ASR")
    parser.add_argument("--live", action="store_true", help="Open a real ASR session (uses quota)")
    args = parser.parse_args(argv)
    _load_env()

    from server import config

    key = config.doubao_asr_api_key()
    host = "openspeech.bytedance.com"
    ip = _dns(host)
    print(f"ASR_PROVIDER={config.asr_provider()}")
    print(f"resource={config.doubao_asr_resource_id()}")
    print(f"url={config.doubao_asr_url()}")
    print(f"key={'set' if key else 'missing'} len={len(key or '')}")
    print(f"dns {host} -> {ip}")
    fake_ip = str(ip).startswith("198.18.")
    if fake_ip:
        print("Clash 假 IP：若 --live 失败，把 openspeech.bytedance.com 设为直连")
        if not args.live:
            return 2
    if not key and not (config.doubao_appid() and config.doubao_access_token()):
        print("还没配置 DOUBAO_ASR_API_KEY（可与豆包 TTS 共用）")
        return 2
    if not args.live:
        print("配置看起来可用。加 --live 才会真正连火山（扣额度）。")
        return 0

    from server.providers.asr_doubao import DoubaoASR

    pcm = b"\x00\x00" * 16000  # 1s silence, checks auth/session only
    asr = DoubaoASR()

    async def probe() -> str:
        return await asr.transcribe_async(pcm)

    try:
        text = asyncio.run(probe())
    except Exception as exc:
        print(f"live_fail: {exc}")
        return 1
    print(f"live_ok text_len={len(text)} preview={text[:40]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
