#!/usr/bin/env python3
"""Check Doubao realtime-dialog config without printing secrets."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_env() -> None:
    path = ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        import os

        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Open a real dialog session (uses quota)")
    parser.add_argument("--tts", action="store_true", help="Also send ChatTTSText and wait for audio")
    args = parser.parse_args()
    _load_env()
    from server import config

    print(f"dialog_provider={config.dialog_provider()}")
    print(f"configured={config.dialog_configured()}")
    print(f"url={config.dialog_url()}")
    print(f"resource={config.dialog_resource_id()}")
    print(f"appid={'set' if config.dialog_appid() else 'missing'}")
    print(f"token={'set' if config.dialog_access_token() else 'missing'}")
    print(f"api_key={'set' if config.dialog_api_key() else 'missing'}")
    print(f"speaker={config.dialog_speaker()}")
    if not config.dialog_configured():
        print("missing key: set DIALOG_API_KEY, or DIALOG_APPID+DIALOG_ACCESS_TOKEN, or reuse DOUBAO_TTS_API_KEY")
        return 1
    if not args.live and not args.tts:
        print("ok (config only). pass --live to handshake, --tts to probe ChatTTSText")
        return 0
    import asyncio

    from server.providers.realtime_doubao import DoubaoRealtimeSession

    async def probe() -> int:
        session = DoubaoRealtimeSession()
        try:
            await session.start(system_role="你是测试助手。", bot_name="测试")
            print("live_ok")
            if not args.tts:
                return 0
            await session.prepare_external_speech()
            await session.chat_tts_text("我查一下。")
            got_audio = False
            kinds = []
            deadline = __import__("time").monotonic() + 4.0
            while __import__("time").monotonic() < deadline:
                event = await session.wait_event(0.5)
                kind = event.kind
                if kind == "timeout":
                    continue
                kinds.append(f"{kind}:{event.event}:{event.tts_type}")
                if kind == "audio" and event.pcm:
                    got_audio = True
                    break
                if kind in {"ended", "closed", "error"}:
                    break
            print(f"chat_tts_audio={'yes' if got_audio else 'no'} seen={kinds[:12]}")
            return 0 if got_audio else 2
        except Exception as exc:
            print(str(exc))
            return 1
        finally:
            await session.close()

    return asyncio.run(probe())


if __name__ == "__main__":
    raise SystemExit(main())
