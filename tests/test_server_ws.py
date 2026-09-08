import asyncio
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path("/home/ning/model/lyz/src/tianlian_voice_bridge")))

import websockets

from server.app import VoiceHub
from server.pipeline import Pipeline
from server.providers.asr_local import NullASR
from server.providers.tts_dummy import DummyTTS
from tests.test_adapters import FakeLLM


async def _turn(robot_id: str, schema_version: int, text: str) -> list[dict]:
    hub = VoiceHub(asr=NullASR(), tts=DummyTTS(), pipeline=Pipeline(llm=FakeLLM()))
    async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        uri = f"ws://127.0.0.1:{port}"
        async with websockets.connect(uri) as ws:
            await ws.send(json.dumps({
                "type": "hello",
                "robot_id": robot_id,
                "schema_version": schema_version,
            }))
            events = [json.loads(await ws.recv())]
            if events[0].get("type") != "hello_ok":
                return events
            await ws.send(json.dumps({"type": "text", "text": text}))
            try:
                while True:
                    message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    if isinstance(message, bytes):
                        events.append({"type": "audio", "n": len(message)})
                    else:
                        events.append(json.loads(message))
            except asyncio.TimeoutError:
                return events
    return events


class WebsocketServerTests(unittest.TestCase):
    def test_unknown_robot_is_rejected(self):
        events = asyncio.run(_turn("no_such_arm", 1, "你好"))
        self.assertEqual(events[0]["type"], "hello_reject")

    def test_schema_mismatch_is_rejected(self):
        events = asyncio.run(_turn("tianlian_dual", 99, "你好"))
        self.assertEqual(events[0]["type"], "hello_reject")
        self.assertIn("schema_version", events[0]["reason"])

    def test_knowledge_turn_has_no_robot_command(self):
        events = asyncio.run(_turn("tianlian_dual", 1, "你有几个关节"))
        types = [e.get("type") for e in events]
        self.assertIn("hello_ok", types)
        self.assertIn("stt", types)
        self.assertIn("tts_text", types)
        self.assertIn("turn_end", types)
        self.assertNotIn("robot_command", types)

    def test_action_turn_suggests_commands(self):
        events = asyncio.run(_turn("tianlian_dual", 1, "左臂向上10厘米"))
        command_event = next(e for e in events if e.get("type") == "robot_command")
        self.assertEqual(command_event["commands"][0]["action"], "move_cartesian")
        types = [e.get("type") for e in events]
        self.assertLess(types.index("robot_command"), types.index("tts_text"))
        tts_event = next(e for e in events if e.get("type") == "tts_text")
        self.assertEqual(tts_event["text"], "好的，正在执行")

    def test_two_robots_on_one_server(self):
        tianlian = asyncio.run(_turn("tianlian_dual", 1, "左臂向上10厘米"))
        demo = asyncio.run(_turn("demo_single", 1, "向前50毫米"))
        t_cmd = next(e for e in tianlian if e.get("type") == "robot_command")["commands"][0]
        d_cmd = next(e for e in demo if e.get("type") == "robot_command")["commands"][0]
        self.assertIn("action", t_cmd)
        self.assertIn("cmd", d_cmd)
        self.assertNotEqual(set(t_cmd), set(d_cmd))

    def test_first_sentence_streams_before_second(self):
        import time

        class SlowLLM:
            configured = True

            def chat(self, user_text, on_sentence=None, extra_system="", enable_search=False, search_fn=None, history=None):
                if on_sentence:
                    on_sentence("第一句。")
                    time.sleep(0.2)
                    on_sentence("第二句。")
                return "第一句。第二句。"

        class StampTTS:
            name = "stamp"

            def __init__(self):
                self.times: list[float] = []
                self.texts: list[str] = []

            def synthesize(self, text: str) -> bytes:
                self.times.append(time.monotonic())
                self.texts.append(text)
                return b"\x00\x00"

        tts = StampTTS()

        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=tts, pipeline=Pipeline(llm=SlowLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "text", "text": "你好"}))
                    try:
                        while True:
                            await asyncio.wait_for(ws.recv(), timeout=1.5)
                    except asyncio.TimeoutError:
                        return

        asyncio.run(scenario())
        self.assertEqual(tts.texts, ["第一句。", "第二句。"])
        self.assertGreater(tts.times[1] - tts.times[0], 0.12)

    def test_streamed_comma_reply_is_not_spoken_twice(self):
        class CommaLLM:
            configured = True

            def chat(self, user_text, on_sentence=None, extra_system="", enable_search=False, search_fn=None, history=None):
                if on_sentence:
                    on_sentence("你好，")
                    on_sentence("我是助手。")
                return "你好，我是助手。"

        class StampTTS:
            name = "stamp"

            def __init__(self):
                self.texts: list[str] = []

            def synthesize(self, text: str) -> bytes:
                self.texts.append(text)
                return b"\x00\x00"

        tts = StampTTS()

        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=tts, pipeline=Pipeline(llm=CommaLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "text", "text": "你好"}))
                    try:
                        while True:
                            await asyncio.wait_for(ws.recv(), timeout=1.5)
                    except asyncio.TimeoutError:
                        return

        asyncio.run(scenario())
        self.assertEqual(tts.texts, ["你好，", "我是助手。"])

    def test_pcm_chunk_streams_before_rest(self):
        import threading

        class ChunkTTS:
            name = "chunk"

            def __init__(self):
                self.gate = threading.Event()
                self.second = False

            def synthesize_iter(self, text):
                yield b"\x00\x01"
                self.gate.wait(timeout=2)
                self.second = True
                yield b"\x00\x02"

        tts = ChunkTTS()

        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=tts, pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "text", "text": "你有几个关节"}))
                    first = None
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        if isinstance(msg, bytes):
                            first = msg
                            break
                    self.assertEqual(first, b"\x00\x01")
                    self.assertFalse(tts.second)
                    caption = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    self.assertFalse(isinstance(caption, bytes))
                    self.assertEqual(json.loads(caption).get("type"), "tts_text")
                    with self.assertRaises(asyncio.TimeoutError):
                        await asyncio.wait_for(ws.recv(), timeout=0.2)
                    tts.gate.set()
                    second = await asyncio.wait_for(ws.recv(), timeout=2)
                    self.assertEqual(second, b"\x00\x02")
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        if isinstance(msg, bytes):
                            continue
                        if json.loads(msg).get("type") == "turn_end":
                            break

        asyncio.run(scenario())
        self.assertTrue(tts.second)

    def test_abort_stops_tts(self):
        import threading

        class GatedTTS:
            name = "gated"

            def __init__(self):
                self.gate = threading.Event()
                self.second = False

            def synthesize_iter(self, text):
                yield b"\x01\x00"
                self.gate.wait(timeout=2)
                self.second = True
                yield b"\x02\x00"

        tts = GatedTTS()

        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=tts, pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "text", "text": "你有几个关节"}))
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        if isinstance(msg, bytes):
                            self.assertEqual(msg, b"\x01\x00")
                            break
                    await ws.send(json.dumps({"type": "abort"}))
                    ended = False
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        if isinstance(msg, bytes):
                            self.fail("should not send remaining TTS after abort")
                        payload = json.loads(msg)
                        if payload.get("type") == "turn_end":
                            ended = True
                            break
                    self.assertTrue(ended)
            tts.gate.set()

        asyncio.run(scenario())

    def test_listen_during_tts_does_not_barge_in(self):
        import threading

        class GatedTTS:
            name = "gated"

            def __init__(self):
                self.gate = threading.Event()
                self.second = False

            def synthesize_iter(self, text):
                yield b"\x01\x00"
                self.gate.wait(timeout=2)
                self.second = True
                yield b"\x02\x00"

        tts = GatedTTS()

        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=tts, pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "text", "text": "你有几个关节"}))
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        if isinstance(msg, bytes):
                            self.assertEqual(msg, b"\x01\x00")
                            break
                    await ws.send(json.dumps({"type": "listen", "state": "start"}))
                    await ws.send(b"\x00\x00" * 160)
                    await ws.send(json.dumps({"type": "listen", "state": "stop"}))
                    tts.gate.set()
                    got_second = False
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        if isinstance(msg, bytes) and msg == b"\x02\x00":
                            got_second = True
                        elif not isinstance(msg, bytes) and json.loads(msg).get("type") == "turn_end":
                            break
                    self.assertTrue(got_second)
                    self.assertTrue(tts.second)

        asyncio.run(scenario())

    def test_orphan_listen_stop_unblocks_client(self):
        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=DummyTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "listen", "state": "stop"}))
                    msg = await asyncio.wait_for(ws.recv(), timeout=2)
                    payload = json.loads(msg)
                    self.assertEqual(payload.get("type"), "turn_end")

        asyncio.run(scenario())

    def test_memory_get_set_clear(self):
        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=DummyTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "memory_set", "city": "长春"}))
                    got = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
                    self.assertEqual(got.get("type"), "memory")
                    self.assertEqual(got.get("city"), "长春")
                    await ws.send(json.dumps({"type": "memory_get"}))
                    got = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
                    self.assertEqual(got.get("city"), "长春")
                    await ws.send(json.dumps({"type": "memory_clear"}))
                    got = json.loads(await asyncio.wait_for(ws.recv(), timeout=2))
                    self.assertEqual(got.get("city"), "")

        asyncio.run(scenario())

    def test_echo_asr_is_ignored(self):
        from server.app import _is_echo

        self.assertTrue(_is_echo("你说吧", ["请继续说。你说吧。"]))
        self.assertTrue(_is_echo("结果怎么样", ["你验光结果怎么样？"]))
        self.assertFalse(_is_echo("你好", ["好呀，我们接着聊。"]))
        self.assertFalse(_is_echo("长春明天天气怎么样", ["你好，我是助手。"]))

    def test_odd_pcm_is_padded_even(self):
        class OddTTS:
            name = "odd"

            def synthesize_iter(self, text):
                yield b"\x01"
                yield b"\x02\x03"

        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=OddTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "text", "text": "你好"}))
                    frames = []
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        if isinstance(msg, bytes):
                            frames.append(msg)
                            continue
                        if json.loads(msg).get("type") == "turn_end":
                            break
                    return frames

        frames = asyncio.run(scenario())
        self.assertTrue(frames)
        for frame in frames:
            self.assertEqual(len(frame) % 2, 0)
        self.assertEqual(b"".join(frames), b"\x01\x02\x03\x00")

    def test_listen_stream_sends_partial_then_final(self):
        class LiveSession:
            def __init__(self, on_partial):
                self.on_partial = on_partial

            async def feed(self, pcm):
                if self.on_partial:
                    await self.on_partial("你有几个")

            async def finish(self):
                return "你有几个关节"

            async def close(self):
                return None

        class LiveASR:
            name = "live"

            def transcribe(self, pcm, sample_rate=16000):
                return "should_not_use"

            async def transcribe_async(self, pcm, sample_rate=16000, on_partial=None):
                return "should_not_use"

            async def open_stream(self, sample_rate=16000, on_partial=None):
                return LiveSession(on_partial)

        async def scenario():
            hub = VoiceHub(asr=LiveASR(), tts=DummyTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "listen", "state": "start"}))
                    await ws.send(b"\x00\x00" * 160)
                    await ws.send(json.dumps({"type": "listen", "state": "stop"}))
                    events = []
                    try:
                        while True:
                            msg = await asyncio.wait_for(ws.recv(), timeout=2)
                            if isinstance(msg, bytes):
                                events.append({"type": "audio"})
                            else:
                                events.append(json.loads(msg))
                            if events[-1].get("type") == "turn_end":
                                break
                    except asyncio.TimeoutError:
                        pass
                    return events

        events = asyncio.run(scenario())
        stt = [e.get("text") for e in events if e.get("type") == "stt"]
        self.assertIn("你有几个", stt)
        self.assertIn("你有几个关节", stt)
        self.assertNotIn("should_not_use", stt)
        types = [e.get("type") for e in events]
        self.assertIn("tts_text", types)
        self.assertIn("turn_end", types)

    def test_listen_falls_back_if_stream_returns_empty(self):
        class EmptySession:
            async def feed(self, pcm):
                return None

            async def finish(self):
                return ""

            async def close(self):
                return None

        captured: list[int] = []

        class FallbackASR:
            name = "fallback"

            async def open_stream(self, sample_rate=16000, on_partial=None):
                return EmptySession()

            async def transcribe_async(self, pcm, sample_rate=16000, on_partial=None):
                captured.append(len(pcm))
                return "你有几个关节"

        asr = FallbackASR()

        async def scenario():
            hub = VoiceHub(asr=asr, tts=DummyTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "listen", "state": "start"}))
                    await ws.send(b"\x00\x00" * 160)
                    await ws.send(json.dumps({"type": "listen", "state": "stop"}))
                    events = []
                    try:
                        while True:
                            msg = await asyncio.wait_for(ws.recv(), timeout=2)
                            if isinstance(msg, bytes):
                                continue
                            events.append(json.loads(msg))
                            if events[-1].get("type") == "turn_end":
                                break
                    except asyncio.TimeoutError:
                        pass
                    return events

        events = asyncio.run(scenario())
        stt = [e.get("text") for e in events if e.get("type") == "stt"]
        self.assertIn("你有几个关节", stt)
        self.assertTrue(captured)
        self.assertGreater(captured[0], 0)

    def test_dialog_chat_and_knowledge_takeover(self):
        from server.providers.realtime_doubao import RealtimeEvent

        calls: list[str] = []

        class FakeDialog:
            def __init__(self, on_asr=None):
                self.on_asr = on_asr
                self.alive = True
                self.muted = False
                self._asr_text = ""
                self.dialog_id = "dlg-1"
                self._queue: asyncio.Queue = asyncio.Queue()

            async def start(self, **kwargs):
                calls.append("start:" + str(kwargs.get("city") or ""))

            async def feed(self, pcm):
                return None

            async def end_utterance(self):
                return None

            async def wait_asr_final(self, timeout=8.0):
                return ""

            async def chat_text(self, text):
                calls.append("chat:" + text)
                await self._queue.put(RealtimeEvent(kind="text", text="端到端在。"))
                await self._queue.put(RealtimeEvent(kind="audio", pcm=b"\x00\x00" * 8))
                await self._queue.put(RealtimeEvent(kind="ended"))

            async def chat_rag(self, notes):
                calls.append("rag")

            async def chat_tts_text(self, text, start=True, end=True):
                calls.append("tts:" + str(text))
                await self._queue.put(RealtimeEvent(kind="text", text=str(text)))
                await self._queue.put(RealtimeEvent(kind="audio", pcm=b"\x00\x00" * 8))
                await self._queue.put(RealtimeEvent(kind="ended"))

            async def wait_event(self, timeout=2.0):
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    return RealtimeEvent(kind="timeout")
                if item is None:
                    return RealtimeEvent(kind="closed")
                return item

            def mute(self):
                self.muted = True
                calls.append("mute")

            def drain(self):
                return None

            async def events(self):
                while True:
                    item = await self._queue.get()
                    yield item
                    if item is not None and item.kind == "ended":
                        return

            async def finish(self):
                return None

            async def close(self):
                return None

        async def scenario():
            hub = VoiceHub(
                asr=NullASR(),
                tts=DummyTTS(),
                pipeline=Pipeline(llm=FakeLLM()),
                realtime_factory=FakeDialog,
            )
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "text", "text": "你好"}))
                    chat_events = []
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        if isinstance(msg, bytes):
                            chat_events.append({"type": "audio"})
                            continue
                        payload = json.loads(msg)
                        chat_events.append(payload)
                        if payload.get("type") == "turn_end":
                            break
                    await ws.send(json.dumps({"type": "text", "text": "你有几个关节"}))
                    kn_events = []
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        if isinstance(msg, bytes):
                            kn_events.append({"type": "audio"})
                            continue
                        payload = json.loads(msg)
                        kn_events.append(payload)
                        if payload.get("type") == "turn_end":
                            break
                    return chat_events, kn_events

        chat_events, kn_events = asyncio.run(scenario())
        self.assertTrue(any(c.startswith("chat:") for c in calls))
        self.assertIn("mute", calls)
        self.assertTrue(any(e.get("type") == "tts_text" and "端到端" in (e.get("text") or "") for e in chat_events))
        self.assertTrue(any(e.get("type") == "audio" for e in chat_events))
        self.assertTrue(any(e.get("type") == "tts_text" and "关节" in (e.get("text") or "") for e in kn_events))
        self.assertNotIn("robot_command", [e.get("type") for e in kn_events])

    def test_dialog_chat_keeps_audio_after_chat_end(self):
        from server.providers.realtime_doubao import RealtimeEvent

        class FakeDialog:
            def __init__(self, on_asr=None):
                self.on_asr = on_asr
                self.alive = True
                self.muted = False
                self._asr_text = ""
                self.dialog_id = "dlg-chat-end"
                self._queue: asyncio.Queue = asyncio.Queue()

            async def start(self, **kwargs):
                return None

            async def chat_text(self, text):
                await self._queue.put(RealtimeEvent(kind="text", text="长春中午可以吃锅包肉。"))
                await self._queue.put(RealtimeEvent(kind="audio", pcm=b"\x01\x00" * 8))
                await self._queue.put(RealtimeEvent(kind="chat_end"))
                await self._queue.put(RealtimeEvent(kind="audio", pcm=b"\x02\x00" * 16))
                await self._queue.put(RealtimeEvent(kind="ended"))

            async def chat_tts_text(self, text, start=True, end=True):
                raise AssertionError("chat audio after ChatEnded must not fall back to ChatTTSText")

            async def wait_event(self, timeout=2.0):
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    return RealtimeEvent(kind="timeout")
                if item is None:
                    return RealtimeEvent(kind="closed")
                return item

            async def chat_rag(self, notes):
                return None

            def mute(self):
                self.muted = True

            def drain(self):
                return None

            async def finish(self):
                return None

            async def close(self):
                return None

        async def scenario():
            hub = VoiceHub(
                asr=NullASR(),
                tts=DummyTTS(),
                pipeline=Pipeline(llm=FakeLLM()),
                realtime_factory=FakeDialog,
            )
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "text", "text": "中午吃什么"}))
                    audio_bytes = 0
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=3)
                        if isinstance(msg, bytes):
                            audio_bytes += len(msg)
                            continue
                        if json.loads(msg).get("type") == "turn_end":
                            break
                    return audio_bytes

        self.assertEqual(asyncio.run(scenario()), 48)

    def test_dialog_search_speaks_result_with_same_voice(self):
        from server.providers.realtime_doubao import RealtimeEvent

        calls: list[str] = []

        class FakeDialog:
            def __init__(self, on_asr=None):
                self.on_asr = on_asr
                self.alive = True
                self.muted = False
                self._asr_text = ""
                self.dialog_id = "dlg-search-1"
                self.city = "长春"
                self._queue: asyncio.Queue = asyncio.Queue()

            async def start(self, **kwargs):
                return None

            async def chat_text(self, text):
                calls.append("chat")

            async def chat_tts_text(self, text, start=True, end=True):
                calls.append("tts:" + str(text))
                await self._queue.put(RealtimeEvent(kind="text", text=str(text)))
                await self._queue.put(RealtimeEvent(kind="audio", pcm=b"\x00\x00" * 8))
                await self._queue.put(RealtimeEvent(kind="ended"))

            async def chat_rag(self, notes):
                calls.append("rag")
                await self._queue.put(RealtimeEvent(kind="text", text="长春今天多云，气温二十二度。"))
                await self._queue.put(RealtimeEvent(kind="audio", pcm=b"\x00\x00" * 8))
                await self._queue.put(RealtimeEvent(kind="ended"))

            async def wait_event(self, timeout=2.0):
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    return RealtimeEvent(kind="timeout")
                if item is None:
                    return RealtimeEvent(kind="closed")
                return item

            def mute(self):
                self.muted = True

            def drain(self):
                return None

            async def finish(self):
                return None

            async def close(self):
                return None

        class ForbiddenTTS:
            name = "forbid"

            def synthesize(self, text: str) -> bytes:
                raise AssertionError("search must not use HTTP Doubao TTS")

            def synthesize_iter(self, text: str):
                raise AssertionError("search must not use HTTP Doubao TTS")
                yield b""

        async def scenario():
            hub = VoiceHub(
                asr=NullASR(),
                tts=ForbiddenTTS(),
                pipeline=Pipeline(llm=FakeLLM(), search_fn=lambda q: "天气：长春今天多云，气温二十二度。"),
                realtime_factory=FakeDialog,
            )
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "text", "text": "长春天气怎么样"}))
                    texts = []
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=3)
                        if isinstance(msg, bytes):
                            continue
                        payload = json.loads(msg)
                        if payload.get("type") == "tts_text":
                            texts.append(payload.get("text") or "")
                        if payload.get("type") == "turn_end":
                            break
                    return texts

        texts = asyncio.run(scenario())
        blob = "".join(texts)
        self.assertIn("我查一下", blob)
        self.assertIn("多云", blob)
        self.assertNotIn("rag", calls)
        self.assertTrue(any(c.startswith("tts:") for c in calls))

    def test_dialog_text_without_audio_falls_back_to_tts(self):
        from server.providers.realtime_doubao import RealtimeEvent

        class TextOnlyDialog:
            def __init__(self, on_asr=None):
                self.on_asr = on_asr
                self.alive = True
                self.muted = False
                self._asr_text = ""
                self.dialog_id = "real-dialog-id-0001"
                self.city = ""
                self._queue: asyncio.Queue = asyncio.Queue()

            async def start(self, **kwargs):
                return None

            async def chat_text(self, text):
                await self._queue.put(RealtimeEvent(kind="text", text="只有文字没有音频。"))
                await self._queue.put(RealtimeEvent(kind="ended"))

            async def chat_tts_text(self, text, start=True, end=True):
                await self._queue.put(RealtimeEvent(kind="text", text=str(text)))
                await self._queue.put(RealtimeEvent(kind="audio", pcm=b"\x00\x10" * 8))
                await self._queue.put(RealtimeEvent(kind="ended"))

            async def wait_event(self, timeout=2.0):
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    return RealtimeEvent(kind="timeout")
                if item is None:
                    return RealtimeEvent(kind="closed")
                return item

            async def chat_rag(self, notes):
                return None

            def mute(self):
                self.muted = True

            def drain(self):
                return None

            async def events(self):
                while True:
                    item = await self._queue.get()
                    yield item
                    if item is not None and item.kind == "ended":
                        return

            async def finish(self):
                return None

            async def close(self):
                return None

        class BeepTTS:
            name = "beep"

            def synthesize(self, text: str) -> bytes:
                return b"\x00\x10" * 32

        async def scenario():
            hub = VoiceHub(
                asr=NullASR(),
                tts=BeepTTS(),
                pipeline=Pipeline(llm=FakeLLM()),
                realtime_factory=TextOnlyDialog,
            )
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "text", "text": "你好"}))
                    types = []
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        if isinstance(msg, bytes):
                            types.append("audio")
                            continue
                        payload = json.loads(msg)
                        types.append(payload.get("type"))
                        if payload.get("type") == "turn_end":
                            break
                    return types

        types = asyncio.run(scenario())
        self.assertIn("tts_text", types)
        self.assertIn("audio", types)

    def test_dialog_listen_native_chat_skips_chat_text(self):
        from server.providers.realtime_doubao import RealtimeEvent

        calls: list[str] = []

        class FakeDialog:
            def __init__(self, on_asr=None):
                self.on_asr = on_asr
                self.alive = True
                self.muted = False
                self.dialog_id = "dlg-native-1"
                self._queue: asyncio.Queue = asyncio.Queue()

            async def start(self, **kwargs):
                calls.append("start")

            async def feed(self, pcm):
                return None

            def begin_listen(self):
                calls.append("begin")

            async def end_utterance(self):
                calls.append("end")
                await self._queue.put(RealtimeEvent(kind="text", text="端到端在。"))
                await self._queue.put(RealtimeEvent(kind="audio", pcm=b"\x11\x00" * 8))
                await self._queue.put(RealtimeEvent(kind="ended"))

            async def wait_asr_final(self, timeout=8.0):
                return "你好呀"

            async def chat_text(self, text):
                calls.append("chat:" + text)
                raise AssertionError("voice chat must use native dialog audio")

            async def chat_tts_text(self, text, start=True, end=True):
                calls.append("tts:" + str(text))
                raise AssertionError("voice chat must not take over TTS")

            async def wait_event(self, timeout=2.0):
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    return RealtimeEvent(kind="timeout")
                return item if item is not None else RealtimeEvent(kind="closed")

            def mute(self):
                self.muted = True
                calls.append("mute")
                while True:
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            def drain(self):
                self.mute()

            async def finish(self):
                return None

            async def close(self):
                return None

        async def scenario():
            hub = VoiceHub(
                asr=NullASR(),
                tts=DummyTTS(),
                pipeline=Pipeline(llm=FakeLLM()),
                realtime_factory=FakeDialog,
            )
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "listen", "state": "start"}))
                    await ws.send(b"\x00\x00" * 160)
                    await ws.send(json.dumps({"type": "listen", "state": "stop"}))
                    events = []
                    audio = b""
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        if isinstance(msg, bytes):
                            audio += msg
                            events.append({"type": "audio"})
                            continue
                        payload = json.loads(msg)
                        events.append(payload)
                        if payload.get("type") == "turn_end":
                            break
                    return events, audio

        events, audio = asyncio.run(scenario())
        self.assertIn("end", calls)
        self.assertFalse(any(c.startswith("chat:") for c in calls))
        self.assertNotIn("mute", calls)
        self.assertEqual(audio, b"\x11\x00" * 8)
        self.assertTrue(any(e.get("type") == "stt" and "你好" in (e.get("text") or "") for e in events))
        self.assertTrue(any(e.get("type") == "tts_text" for e in events))

    def test_dialog_listen_knowledge_mutes_native(self):
        from server.providers.realtime_doubao import RealtimeEvent

        calls: list[str] = []

        class FakeDialog:
            def __init__(self, on_asr=None):
                self.on_asr = on_asr
                self.alive = True
                self.muted = False
                self.dialog_id = "dlg-native-kn"
                self._queue: asyncio.Queue = asyncio.Queue()

            async def start(self, **kwargs):
                return None

            async def feed(self, pcm):
                return None

            def begin_listen(self):
                return None

            async def end_utterance(self):
                await self._queue.put(RealtimeEvent(kind="audio", pcm=b"\x99\x00" * 8))
                await self._queue.put(RealtimeEvent(kind="ended"))

            async def wait_asr_final(self, timeout=8.0):
                return "你有几个关节"

            async def chat_text(self, text):
                calls.append("chat")
                raise AssertionError("knowledge must not use chat_text")

            async def chat_tts_text(self, text, start=True, end=True):
                calls.append("tts:" + str(text))
                await self._queue.put(RealtimeEvent(kind="text", text=str(text)))
                await self._queue.put(RealtimeEvent(kind="audio", pcm=b"\x22\x00" * 8))
                await self._queue.put(RealtimeEvent(kind="ended"))

            async def wait_event(self, timeout=2.0):
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    return RealtimeEvent(kind="timeout")
                return item if item is not None else RealtimeEvent(kind="closed")

            def mute(self):
                self.muted = True
                calls.append("mute")
                while True:
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            def drain(self):
                self.mute()

            async def finish(self):
                return None

            async def close(self):
                return None

        async def scenario():
            hub = VoiceHub(
                asr=NullASR(),
                tts=DummyTTS(),
                pipeline=Pipeline(llm=FakeLLM()),
                realtime_factory=FakeDialog,
            )
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "listen", "state": "start"}))
                    await ws.send(b"\x00\x00" * 160)
                    await ws.send(json.dumps({"type": "listen", "state": "stop"}))
                    events = []
                    audio = b""
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        if isinstance(msg, bytes):
                            audio += msg
                            continue
                        payload = json.loads(msg)
                        events.append(payload)
                        if payload.get("type") == "turn_end":
                            break
                    return events, audio

        events, audio = asyncio.run(scenario())
        self.assertIn("mute", calls)
        self.assertTrue(any(c.startswith("tts:") for c in calls))
        self.assertNotIn("chat", calls)
        self.assertNotIn("robot_command", [e.get("type") for e in events])
        self.assertTrue(any("关节" in (e.get("text") or "") for e in events if e.get("type") == "tts_text"))
        self.assertEqual(audio, b"\x22\x00" * 8)

    def test_dialog_listen_empty_asr_does_not_leak_native(self):
        from server.providers.realtime_doubao import RealtimeEvent

        calls: list[str] = []

        class FakeDialog:
            def __init__(self, on_asr=None):
                self.on_asr = on_asr
                self.alive = True
                self.muted = False
                self.dialog_id = "dlg-empty"
                self._queue: asyncio.Queue = asyncio.Queue()

            async def start(self, **kwargs):
                return None

            async def feed(self, pcm):
                return None

            def begin_listen(self):
                return None

            async def end_utterance(self):
                await self._queue.put(RealtimeEvent(kind="audio", pcm=b"\x77\x00" * 8))
                await self._queue.put(RealtimeEvent(kind="ended"))

            async def wait_asr_final(self, timeout=8.0):
                return ""

            async def chat_text(self, text):
                calls.append("chat")
                return None

            async def wait_event(self, timeout=2.0):
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    return RealtimeEvent(kind="timeout")
                return item if item is not None else RealtimeEvent(kind="closed")

            def mute(self):
                self.muted = True
                calls.append("mute")
                while True:
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            def drain(self):
                self.mute()

            async def finish(self):
                return None

            async def close(self):
                return None

        async def scenario():
            hub = VoiceHub(
                asr=NullASR(),
                tts=DummyTTS(),
                pipeline=Pipeline(llm=FakeLLM()),
                realtime_factory=FakeDialog,
            )
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "listen", "state": "start"}))
                    await ws.send(b"\x00\x00" * 160)
                    await ws.send(json.dumps({"type": "listen", "state": "stop"}))
                    audio = 0
                    ended = False
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        if isinstance(msg, bytes):
                            audio += len(msg)
                            continue
                        if json.loads(msg).get("type") == "turn_end":
                            ended = True
                            break
                    return audio, ended

        audio, ended = asyncio.run(scenario())
        self.assertIn("mute", calls)
        self.assertNotIn("chat", calls)
        self.assertEqual(audio, 0)
        self.assertTrue(ended)

    def test_dialog_listen_control_still_commands_first(self):
        from server.providers.realtime_doubao import RealtimeEvent

        class FakeDialog:
            def __init__(self, on_asr=None):
                self.on_asr = on_asr
                self.alive = True
                self.muted = False
                self.dialog_id = "dlg-ctrl"
                self._queue: asyncio.Queue = asyncio.Queue()

            async def start(self, **kwargs):
                return None

            async def feed(self, pcm):
                return None

            def begin_listen(self):
                return None

            async def end_utterance(self):
                await self._queue.put(RealtimeEvent(kind="audio", pcm=b"\x01\x00" * 4))
                await self._queue.put(RealtimeEvent(kind="ended"))

            async def wait_asr_final(self, timeout=8.0):
                return "左臂向上10厘米"

            async def chat_text(self, text):
                raise AssertionError("control must not chat_text")

            async def chat_tts_text(self, text, start=True, end=True):
                await self._queue.put(RealtimeEvent(kind="text", text=str(text)))
                await self._queue.put(RealtimeEvent(kind="audio", pcm=b"\x02\x00" * 4))
                await self._queue.put(RealtimeEvent(kind="ended"))

            async def wait_event(self, timeout=2.0):
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    return RealtimeEvent(kind="timeout")
                return item if item is not None else RealtimeEvent(kind="closed")

            def mute(self):
                self.muted = True
                while True:
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            def drain(self):
                self.mute()

            async def finish(self):
                return None

            async def close(self):
                return None

        async def scenario():
            hub = VoiceHub(
                asr=NullASR(),
                tts=DummyTTS(),
                pipeline=Pipeline(llm=FakeLLM()),
                realtime_factory=FakeDialog,
            )
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "listen", "state": "start"}))
                    await ws.send(b"\x00\x00" * 160)
                    await ws.send(json.dumps({"type": "listen", "state": "stop"}))
                    types = []
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        if isinstance(msg, bytes):
                            types.append("audio")
                            continue
                        payload = json.loads(msg)
                        types.append(payload.get("type"))
                        if payload.get("type") == "robot_command":
                            self.assertEqual(payload["commands"][0]["action"], "move_cartesian")
                        if payload.get("type") == "turn_end":
                            break
                    return types

        types = asyncio.run(scenario())
        self.assertIn("robot_command", types)
        self.assertLess(types.index("robot_command"), types.index("tts_text"))
        self.assertIn("audio", types)

    def test_dialog_listen_search_mutes_native(self):
        from server.providers.realtime_doubao import RealtimeEvent

        calls: list[str] = []

        class FakeDialog:
            def __init__(self, on_asr=None):
                self.on_asr = on_asr
                self.alive = True
                self.muted = False
                self.dialog_id = "dlg-search-listen"
                self.city = "长春"
                self._queue: asyncio.Queue = asyncio.Queue()

            async def start(self, **kwargs):
                return None

            async def feed(self, pcm):
                return None

            def begin_listen(self):
                return None

            async def end_utterance(self):
                await self._queue.put(RealtimeEvent(kind="audio", pcm=b"\x33\x00" * 8))
                await self._queue.put(RealtimeEvent(kind="ended"))

            async def wait_asr_final(self, timeout=8.0):
                return "长春天气怎么样"

            async def chat_text(self, text):
                calls.append("chat")
                raise AssertionError("search must not chat_text")

            async def chat_tts_text(self, text, start=True, end=True):
                calls.append("tts:" + str(text))
                await self._queue.put(RealtimeEvent(kind="text", text=str(text)))
                await self._queue.put(RealtimeEvent(kind="audio", pcm=b"\x44\x00" * 8))
                await self._queue.put(RealtimeEvent(kind="ended"))

            async def wait_event(self, timeout=2.0):
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    return RealtimeEvent(kind="timeout")
                return item if item is not None else RealtimeEvent(kind="closed")

            def mute(self):
                self.muted = True
                calls.append("mute")
                while True:
                    try:
                        self._queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            def drain(self):
                self.mute()

            async def finish(self):
                return None

            async def close(self):
                return None

        async def scenario():
            hub = VoiceHub(
                asr=NullASR(),
                tts=DummyTTS(),
                pipeline=Pipeline(llm=FakeLLM(), search_fn=lambda q: "天气：长春今天多云，气温二十二度。"),
                realtime_factory=FakeDialog,
            )
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with websockets.connect(f"ws://127.0.0.1:{port}") as ws:
                    await ws.send(json.dumps({
                        "type": "hello",
                        "robot_id": "tianlian_dual",
                        "schema_version": 1,
                    }))
                    await ws.recv()
                    await ws.send(json.dumps({"type": "listen", "state": "start"}))
                    await ws.send(b"\x00\x00" * 160)
                    await ws.send(json.dumps({"type": "listen", "state": "stop"}))
                    texts = []
                    audio = b""
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=3)
                        if isinstance(msg, bytes):
                            audio += msg
                            continue
                        payload = json.loads(msg)
                        if payload.get("type") == "tts_text":
                            texts.append(payload.get("text") or "")
                        if payload.get("type") == "turn_end":
                            break
                    return texts, audio

        texts, audio = asyncio.run(scenario())
        blob = "".join(texts)
        self.assertIn("mute", calls)
        self.assertNotIn("chat", calls)
        self.assertIn("我查一下", blob)
        self.assertIn("多云", blob)
        self.assertNotIn(b"\x33\x00", audio)
        self.assertIn(b"\x44\x00", audio)


if __name__ == "__main__":
    unittest.main()
