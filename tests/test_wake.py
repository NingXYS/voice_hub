import asyncio
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jetson_client.wake_kws import (
    KEYWORD_TOKENS,
    KwsCooldown,
    keyword_line,
    pcm_rms,
    pcm_s16le_to_float,
    verify_keyword_tokens,
    wake_word,
)
from jetson_client.wake_loop import begin_command_stream, collect_command_from_queue

KWS_DIR = Path("/home/ning/model/lyz/src/voice_control_online/models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20")
POS_WAV = Path("/home/ning/model/lyz/src/voice_control_online/datasets/wake/positive/close_10cm/xiaoxi_1.wav")
NEG_DIR = Path("/home/ning/model/lyz/src/voice_control_online/datasets/wake/negative")


class WakeTokenTests(unittest.TestCase):
    def test_wake_word_is_xiaoxi(self):
        self.assertEqual(wake_word(), "小溪小溪")

    def test_keyword_line_uses_ppinyin_tokens(self):
        line = keyword_line("小溪小溪")
        self.assertTrue(line.startswith(" ".join(KEYWORD_TOKENS)))
        self.assertIn("@小溪小溪", line)

    def test_official_tokens_file_has_xiaoxi_pieces(self):
        if not (KWS_DIR / "tokens.txt").is_file():
            self.skipTest("KWS tokens.txt not on this machine")
        tokens = verify_keyword_tokens(KWS_DIR / "tokens.txt", "小溪小溪")
        self.assertEqual(tokens, KEYWORD_TOKENS)


class WakeHelperTests(unittest.TestCase):
    def test_cooldown_drops_then_allows(self):
        cool = KwsCooldown(0.01)
        self.assertTrue(cool.allow(2))
        cool.arm()
        self.assertFalse(cool.allow(2))
        self.assertTrue(cool.allow(16000))

    def test_pcm_rms_silence_is_low(self):
        silent = b"\x00\x00" * 160
        self.assertLess(pcm_rms(silent), 0.01)
        self.assertEqual(len(pcm_s16le_to_float(silent)), 160)

    def test_collect_command_times_out_without_speech(self):
        async def scenario():
            queue: asyncio.Queue = asyncio.Queue()
            task = asyncio.create_task(
                collect_command_from_queue(queue, start_timeout=0.3, max_duration=1.0, skip_sec=0.0)
            )
            await queue.put(("pcm", b"\x00\x00" * 160))
            return await task

        pcm = asyncio.run(scenario())
        self.assertEqual(pcm, b"")

    def test_begin_stream_starts_on_first_speech(self):
        async def scenario():
            queue: asyncio.Queue = asyncio.Queue()
            loud = b"\x00\x40" * 400
            stream = None

            async def wait_stream():
                return await begin_command_stream(
                    queue,
                    start_timeout=1.0,
                    max_duration=2.0,
                    skip_sec=0.0,
                    silence_sec=0.8,
                )

            task = asyncio.create_task(wait_stream())
            await queue.put(("pcm", loud))
            stream = await asyncio.wait_for(task, timeout=1.0)
            self.assertIsNotNone(stream)
            first = await asyncio.wait_for(stream.__anext__(), timeout=0.5)
            self.assertEqual(first, loud)
            await queue.put(("eof", None))
            rest = [chunk async for chunk in stream]
            self.assertEqual(rest, [])

        asyncio.run(scenario())

    def test_listen_stop_during_prime_keeps_pending(self):
        async def scenario():
            queue: asyncio.Queue = asyncio.Queue()
            quiet = b"\x80\x01" * 400
            self.assertLess(pcm_rms(quiet), 0.012)
            await queue.put(("pcm", quiet))
            await queue.put(("json", {"type": "listen", "state": "stop"}))
            stream = await begin_command_stream(
                queue, start_timeout=1.0, max_duration=2.0, skip_sec=0.0, silence_sec=0.8
            )
            self.assertIsNotNone(stream)
            chunks = [chunk async for chunk in stream]
            self.assertEqual(b"".join(chunks), quiet)

        asyncio.run(scenario())

    def test_eof_is_requeued_after_stream(self):
        async def scenario():
            queue: asyncio.Queue = asyncio.Queue()
            loud = b"\x00\x40" * 400
            await queue.put(("pcm", loud))
            await queue.put(("eof", None))
            stream = await begin_command_stream(
                queue, start_timeout=1.0, max_duration=2.0, skip_sec=0.0
            )
            self.assertIsNotNone(stream)
            chunks = [chunk async for chunk in stream]
            self.assertTrue(b"".join(chunks))
            kind, _ = await asyncio.wait_for(queue.get(), timeout=0.2)
            self.assertEqual(kind, "eof")

        asyncio.run(scenario())

    def test_skip_keeps_tail_of_chunk(self):
        async def scenario():
            queue: asyncio.Queue = asyncio.Queue()
            loud = b"\x00\x40" * 2000
            await queue.put(("pcm", loud))
            await queue.put(("eof", None))
            stream = await begin_command_stream(
                queue, start_timeout=1.0, max_duration=2.0, skip_sec=0.1
            )
            self.assertIsNotNone(stream)
            joined = b"".join([chunk async for chunk in stream])
            self.assertEqual(joined, loud[3200:])

        asyncio.run(scenario())

    def test_abort_during_prime_with_pending_streams(self):
        async def scenario():
            queue: asyncio.Queue = asyncio.Queue()
            loud = b"\x00\x40" * 400
            await queue.put(("pcm", loud))
            await queue.put(("json", {"type": "abort"}))
            stream = await begin_command_stream(
                queue, start_timeout=1.0, max_duration=2.0, skip_sec=0.0
            )
            self.assertIsNotNone(stream)
            joined = b"".join([chunk async for chunk in stream])
            self.assertEqual(joined, loud)

        asyncio.run(scenario())


class WakeSpotterLiveTests(unittest.TestCase):
    def test_positive_wav_hits_xiaoxi(self):
        if not (KWS_DIR / "encoder.onnx").is_file() or not POS_WAV.is_file():
            self.skipTest("KWS model or positive wav missing")
        try:
            from jetson_client.wake_kws import KeywordSpotter
        except RuntimeError:
            self.skipTest("sherpa-onnx not installed")
        import wave

        detector = KeywordSpotter(str(KWS_DIR), "小溪小溪")
        with wave.open(str(POS_WAV), "rb") as handle:
            pcm = handle.readframes(handle.getnframes())
            rate = handle.getframerate()
        if rate != 16000:
            self.skipTest(f"wav sample rate {rate}")
        hit = detector.accept_pcm(pcm)
        self.assertTrue(hit, "positive 小西/小溪 clip should wake")

    def test_conversation_wav_does_not_wake(self):
        neg = NEG_DIR / "conversation" / "chat_1.wav"
        if not (KWS_DIR / "encoder.onnx").is_file() or not neg.is_file():
            self.skipTest("KWS model or negative wav missing")
        try:
            from jetson_client.wake_kws import KeywordSpotter
        except RuntimeError:
            self.skipTest("sherpa-onnx not installed")
        import wave

        detector = KeywordSpotter(str(KWS_DIR), "小溪小溪")
        with wave.open(str(neg), "rb") as handle:
            pcm = handle.readframes(handle.getnframes())
            rate = handle.getframerate()
        if rate != 16000:
            self.skipTest(f"wav sample rate {rate}")
        self.assertFalse(detector.accept_pcm(pcm), "conversation clip should not wake")

    def test_silence_does_not_wake(self):
        if not (KWS_DIR / "encoder.onnx").is_file():
            self.skipTest("KWS model missing")
        try:
            from jetson_client.wake_kws import KeywordSpotter
        except RuntimeError:
            self.skipTest("sherpa-onnx not installed")
        detector = KeywordSpotter(str(KWS_DIR), "小溪小溪")
        self.assertFalse(detector.accept_pcm(b"\x00\x00" * 16000))


class DeviceWakeWsTests(unittest.TestCase):
    def test_pcm_then_hit_then_listen(self):
        import tempfile

        from starlette.testclient import TestClient

        from device_ui.app import create_app
        from jetson_client.session import SayResult
        from tests.test_device_ui import _isolate_env, _restore_env

        class Detector:
            def __init__(self):
                self.n = 0

            def accept_pcm(self, pcm):
                self.n += 1
                return self.n >= 2

            def reset(self):
                self.n = 0

        class FakeSession:
            connected = True
            hello = {"type": "hello_ok"}
            url = "ws://x"
            robot_id = "tianlian_dual"
            schema_version = 1
            bridge_error = ""
            heard = b""
            stream_calls = 0

            async def listen_stream(self, chunks, idle_after_speech=8.0, on_audio=None, on_event=None):
                self.stream_calls += 1
                parts = []
                async for pcm in chunks:
                    parts.append(pcm)
                self.heard = b"".join(parts)
                if on_event:
                    on_event({"type": "stt", "text": "你有几个关节"})
                    on_event({"type": "tts_text", "text": "十二个关节。"})
                    on_event({"type": "turn_end"})
                if on_audio:
                    on_audio(b"\x00\x10" * 8)
                return SayResult(events=[{"type": "turn_end"}], pcm=b"\x00\x10" * 8)

            async def abort_playback(self):
                return None

            async def close(self):
                return None

        saved = _isolate_env()
        tmp = tempfile.NamedTemporaryFile("w", delete=False)
        tmp.close()
        os.environ["VOICE_HUB_CLIENT_ENV"] = tmp.name
        os.environ.pop("VOICE_KWS_DISABLE", None)
        try:
            session = FakeSession()
            detector = Detector()
            with TestClient(create_app(session, wake_detector=detector)) as client:
                with client.websocket_connect("/ws/voice") as ws:
                    first = ws.receive_json()
                    self.assertEqual(first.get("type"), "wake")
                    self.assertEqual(first.get("status"), "waiting")
                    ws.send_bytes(b"\x00\x00" * 160)
                    ws.send_bytes(b"\x10\x00" * 160)
                    hit = ws.receive_json()
                    self.assertEqual(hit.get("type"), "wake")
                    self.assertEqual(hit.get("status"), "hit")
                    self.assertEqual(hit.get("word"), "小溪小溪")
                    loud = b"\x00\x40" * 4000
                    ws.send_bytes(loud)
                    ws.send_bytes(loud)
                    ws.send_json({"type": "listen", "state": "stop"})
                    kinds = []
                    statuses = []
                    for _ in range(12):
                        message = ws.receive()
                        if message.get("text"):
                            payload = json.loads(message["text"])
                            kinds.append(payload.get("type"))
                            if payload.get("type") == "wake":
                                statuses.append(payload.get("status"))
                        elif message.get("bytes"):
                            kinds.append("audio")
                        if statuses and statuses[-1] == "waiting" and "stt" in kinds:
                            break
                    self.assertIn("listen", statuses)
                    self.assertIn("stt", kinds)
                    self.assertIn("tts_text", kinds)
                    self.assertEqual(session.stream_calls, 1)
                    self.assertTrue(session.heard)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
            _restore_env(saved)


if __name__ == "__main__":
    unittest.main()
