import asyncio
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path("/home/ning/model/lyz/src/tianlian_voice_bridge")))

import websockets

from jetson_client.session import HubSession
from jetson_client.voice_client import run_text
from server.app import VoiceHub
from server.pipeline import Pipeline
from server.providers.asr_local import NullASR
from server.providers.tts_dummy import DummyTTS
from tests.test_adapters import FakeLLM


class ClientTests(unittest.TestCase):
    def test_text_turn_dry_runs_tianlian_command(self):
        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=DummyTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                return await run_text(
                    f"ws://127.0.0.1:{port}",
                    "tianlian_dual",
                    1,
                    "左臂向上10厘米",
                    play=False,
                )

        events = asyncio.run(scenario())
        types = [e.get("type") for e in events]
        self.assertIn("hello_ok", types)
        self.assertIn("robot_command", types)
        self.assertIn("bridge", types)
        bridge = next(e for e in events if e.get("type") == "bridge")
        self.assertEqual(bridge["status"], "dry_run")

    def test_listen_pcm_uses_stream_final_text(self):
        class LiveSession:
            def __init__(self, on_partial):
                self.on_partial = on_partial

            async def feed(self, pcm):
                if self.on_partial:
                    await self.on_partial("左臂向上")

            async def finish(self):
                return "左臂向上10厘米"

            async def close(self):
                return None

        class LiveASR:
            name = "live"

            def transcribe(self, pcm, sample_rate=16000):
                return "should_not_use"

            async def open_stream(self, sample_rate=16000, on_partial=None):
                return LiveSession(on_partial)

        async def scenario():
            hub = VoiceHub(asr=LiveASR(), tts=DummyTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                session = HubSession()
                try:
                    hello = await session.connect(f"ws://127.0.0.1:{port}", "tianlian_dual", 1)
                    self.assertEqual(hello.get("type"), "hello_ok")
                    result = await session.listen_pcm(b"\x00\x00" * 320)
                    return result.events
                finally:
                    await session.close()

        events = asyncio.run(scenario())
        types = [e.get("type") for e in events]
        stt = [e.get("text") for e in events if e.get("type") == "stt"]
        self.assertIn("左臂向上10厘米", stt)
        self.assertIn("robot_command", types)
        self.assertIn("turn_end", types)

    def test_listen_stream_chunks(self):
        class LiveSession:
            def __init__(self, on_partial):
                self.on_partial = on_partial
                self.n = 0

            async def feed(self, pcm):
                self.n += len(pcm)
                if self.on_partial:
                    await self.on_partial("你好")

            async def finish(self):
                return "你好"

            async def close(self):
                return None

        class LiveASR:
            name = "live"

            async def open_stream(self, sample_rate=16000, on_partial=None):
                return LiveSession(on_partial)

        async def chunks():
            yield b"\x00\x00" * 160
            yield b"\x00\x00" * 160

        async def scenario():
            hub = VoiceHub(asr=LiveASR(), tts=DummyTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                session = HubSession()
                try:
                    await session.connect(f"ws://127.0.0.1:{port}", "tianlian_dual", 1)
                    result = await session.listen_stream(chunks())
                    return [e.get("type") for e in result.events]
                finally:
                    await session.close()

        types = asyncio.run(scenario())
        self.assertIn("stt", types)
        self.assertIn("turn_end", types)

    def test_connect_reuses_open_socket(self):
        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=DummyTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                session = HubSession()
                try:
                    hello1 = await session.connect(f"ws://127.0.0.1:{port}", "tianlian_dual", 1)
                    ws1 = session._ws
                    hello2 = await session.connect(f"ws://127.0.0.1:{port}", "tianlian_dual", 1)
                    return hello1.get("type"), hello2.get("type"), session._ws is ws1
                finally:
                    await session.close()

        kind1, kind2, same = asyncio.run(scenario())
        self.assertEqual(kind1, "hello_ok")
        self.assertEqual(kind2, "hello_ok")
        self.assertTrue(same)
