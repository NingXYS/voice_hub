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
    _sherpa_search_dirs,
    boost_for_kws,
    keyword_line,
    pcm_rms,
    pcm_s16le_to_float,
    verify_keyword_tokens,
    wake_word,
)
from jetson_client.tone import render_prompt_tone_pcm, write_prompt_tone_wav
from jetson_client.wake_loop import (
    ENDPOINT_SILENCE_SEC,
    begin_command_stream,
    collect_command_from_queue,
    listen_with_barge,
    watch_wake_barge,
)

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
        self.assertIn("#0.25", line)

    def test_official_tokens_file_has_xiaoxi_pieces(self):
        if not (KWS_DIR / "tokens.txt").is_file():
            self.skipTest("KWS tokens.txt not on this machine")
        tokens = verify_keyword_tokens(KWS_DIR / "tokens.txt", "小溪小溪")
        self.assertEqual(tokens, KEYWORD_TOKENS)


class WakeHelperTests(unittest.TestCase):
    def test_sherpa_search_includes_user_local(self):
        dirs = [str(path) for path in _sherpa_search_dirs()]
        self.assertTrue(any("/.local/lib/" in item and item.endswith("site-packages") for item in dirs))

    def test_sherpa_search_skips_other_python(self):
        from jetson_client.wake_kws import _path_matches_this_python, _python_tag
        from pathlib import Path

        mine = Path.home() / ".local" / "lib" / _python_tag() / "site-packages"
        other = Path.home() / ".local" / "lib" / "python3.10" / "site-packages"
        self.assertTrue(_path_matches_this_python(mine))
        if _python_tag() != "python3.10":
            self.assertFalse(_path_matches_this_python(other))
        for path in _sherpa_search_dirs():
            self.assertTrue(_path_matches_this_python(path), str(path))

    def test_endpoint_silence_is_under_600ms(self):
        self.assertGreaterEqual(ENDPOINT_SILENCE_SEC, 0.4)
        self.assertLessEqual(ENDPOINT_SILENCE_SEC, 0.6)

    def test_watch_wake_barge_aborts_after_start(self):
        class Detector:
            def __init__(self):
                self.n = 0

            def accept_pcm(self, pcm):
                self.n += 1
                return self.n >= 1

        async def scenario():
            incoming: asyncio.Queue = asyncio.Queue()
            started = asyncio.Event()
            done = asyncio.Event()
            aborted = asyncio.Event()

            async def abort_fn():
                aborted.set()

            task = asyncio.create_task(
                watch_wake_barge(
                    incoming,
                    Detector(),
                    None,
                    abort_fn,
                    started=started,
                    done=done,
                )
            )
            await incoming.put(("pcm", b"\x00\x40" * 160))
            await asyncio.sleep(0.05)
            self.assertFalse(aborted.is_set())
            started.set()
            await asyncio.sleep(0.05)
            self.assertFalse(aborted.is_set())
            await incoming.put(("pcm", b"\x00\x40" * 160))
            hit = await asyncio.wait_for(task, timeout=1.0)
            self.assertTrue(hit)
            self.assertTrue(aborted.is_set())

        asyncio.run(scenario())

    def test_watch_wake_barge_drops_pcm_queued_before_tts(self):
        class Detector:
            def __init__(self):
                self.n = 0

            def accept_pcm(self, pcm):
                self.n += 1
                return True

        async def scenario():
            incoming: asyncio.Queue = asyncio.Queue()
            started = asyncio.Event()
            done = asyncio.Event()
            aborted = asyncio.Event()

            async def abort_fn():
                aborted.set()

            detector = Detector()
            task = asyncio.create_task(
                watch_wake_barge(
                    incoming,
                    detector,
                    None,
                    abort_fn,
                    started=started,
                    done=done,
                )
            )
            await incoming.put(("pcm", b"\x7f\x7f" * 160))
            started.set()
            await asyncio.sleep(0.08)
            self.assertFalse(aborted.is_set())
            done.set()
            hit = await asyncio.wait_for(task, timeout=1.0)
            self.assertFalse(hit)
            self.assertEqual(detector.n, 0)

        asyncio.run(scenario())

    def test_listen_with_barge_calls_abort_during_tts(self):
        from jetson_client.session import SayResult

        class Detector:
            def accept_pcm(self, pcm):
                return True

            def reset(self):
                return None

        class FakeSession:
            def __init__(self):
                self.abort_calls = 0
                self._hold = asyncio.Event()

            async def listen_stream(self, chunks, idle_after_speech=8.0, on_audio=None, on_event=None, on_utterance_end=None):
                async for _pcm in chunks:
                    pass
                if on_utterance_end:
                    on_utterance_end()
                if on_event:
                    on_event({"type": "tts_text", "text": "正在说"})
                await self._hold.wait()
                if on_event:
                    on_event({"type": "turn_end"})
                return SayResult(events=[{"type": "turn_end"}], pcm=b"")

            async def abort_playback(self):
                self.abort_calls += 1
                self._hold.set()

        async def scenario():
            incoming: asyncio.Queue = asyncio.Queue()
            session = FakeSession()

            async def chunks():
                yield b"\x00\x10" * 160

            async def later_pcm():
                await asyncio.sleep(0.05)
                await incoming.put(("pcm", b"\x00\x40" * 160))

            feeder = asyncio.create_task(later_pcm())
            result, barged = await listen_with_barge(
                session, chunks(), incoming, Detector(), None
            )
            await feeder
            return session.abort_calls, barged, result is not None

        abort_calls, barged, ok = asyncio.run(scenario())
        self.assertEqual(abort_calls, 1)
        self.assertTrue(barged)
        self.assertTrue(ok)

    def test_cooldown_drops_then_allows(self):
        cool = KwsCooldown(0.01)
        self.assertTrue(cool.allow(2))
        cool.arm()
        self.assertFalse(cool.allow(2))
        self.assertTrue(cool.allow(16000))

    def test_cooldown_arm_adds_tts_duration(self):
        cool = KwsCooldown(0.0)
        cool.arm(extra_sec=0.01)
        self.assertFalse(cool.allow(2))
        self.assertTrue(cool.allow(16000))

    def test_boost_raises_quiet_speech_not_silence(self):
        import numpy as np

        silent, rms_s = boost_for_kws(np.zeros(1600, dtype=np.float32))
        self.assertLess(rms_s, 0.001)
        self.assertLess(float(np.max(np.abs(silent))), 1e-6)
        quiet = np.full(1600, 0.01, dtype=np.float32)
        boosted, rms_q = boost_for_kws(quiet)
        self.assertGreater(float(np.sqrt(np.mean(np.square(boosted)))), rms_q)
        loud = np.full(1600, 0.2, dtype=np.float32)
        same, rms_l = boost_for_kws(loud)
        self.assertAlmostEqual(float(np.max(np.abs(same))), 0.2, places=4)
        self.assertGreater(rms_l, 0.1)

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
        pcm = pcm + b"\x00\x00" * 5600
        hit = False
        for offset in range(0, len(pcm), 2730):
            if detector.accept_pcm(pcm[offset : offset + 2730]):
                hit = True
                break
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

    def test_positive_set_hit_rate_with_live_mic_tail(self):
        pos_dir = POS_WAV.parent.parent
        if not (KWS_DIR / "encoder.onnx").is_file() or not pos_dir.is_dir():
            self.skipTest("KWS model or positive set missing")
        try:
            from jetson_client.wake_kws import KeywordSpotter
        except RuntimeError:
            self.skipTest("sherpa-onnx not installed")
        import wave

        files = sorted(pos_dir.rglob("*.wav"))
        if len(files) < 5:
            self.skipTest("not enough positive clips")
        hits = 0
        for path in files:
            with wave.open(str(path), "rb") as handle:
                if handle.getframerate() != 16000 or handle.getnchannels() != 1:
                    continue
                pcm = handle.readframes(handle.getnframes())
            pcm = pcm + b"\x00\x00" * 5600
            detector = KeywordSpotter(str(KWS_DIR), "小溪小溪")
            for offset in range(0, len(pcm), 2730):
                if detector.accept_pcm(pcm[offset : offset + 2730]):
                    hits += 1
                    break
        self.assertGreaterEqual(hits, 9, f"wake hit rate too low: {hits}/{len(files)}")

    def test_conversation_set_has_no_false_wake(self):
        conv = NEG_DIR / "conversation"
        if not (KWS_DIR / "encoder.onnx").is_file() or not conv.is_dir():
            self.skipTest("KWS model or negative set missing")
        try:
            from jetson_client.wake_kws import KeywordSpotter
        except RuntimeError:
            self.skipTest("sherpa-onnx not installed")
        import wave

        fa = 0
        files = sorted(conv.glob("*.wav"))
        for path in files:
            with wave.open(str(path), "rb") as handle:
                if handle.getframerate() != 16000:
                    continue
                pcm = handle.readframes(handle.getnframes())
            pcm = pcm + b"\x00\x00" * 5600
            detector = KeywordSpotter(str(KWS_DIR), "小溪小溪")
            for offset in range(0, len(pcm), 2730):
                if detector.accept_pcm(pcm[offset : offset + 2730]):
                    fa += 1
                    break
        self.assertEqual(fa, 0, f"false wakes on conversation clips: {fa}/{len(files)}")


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

            async def listen_stream(self, chunks, idle_after_speech=8.0, on_audio=None, on_event=None, **kwargs):
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

    def test_wake_during_tts_aborts_and_starts_next(self):
        import tempfile

        from starlette.testclient import TestClient

        from device_ui.app import create_app
        from jetson_client.session import SayResult
        from tests.test_device_ui import _isolate_env, _restore_env

        class Detector:
            def __init__(self):
                self.hits = 0

            def accept_pcm(self, pcm):
                if pcm and pcm[:2] == b"\x7f\x7f":
                    self.hits += 1
                    return True
                return False

            def reset(self):
                return None

        class FakeSession:
            connected = True
            hello = {"type": "hello_ok"}
            url = "ws://x"
            robot_id = "tianlian_dual"
            schema_version = 1
            bridge_error = ""
            stream_calls = 0
            abort_calls = 0

            def __init__(self):
                self._holds: list[asyncio.Event] = []

            async def listen_stream(self, chunks, idle_after_speech=8.0, on_audio=None, on_event=None, **kwargs):
                self.stream_calls += 1
                async for _pcm in chunks:
                    pass
                end = kwargs.get("on_utterance_end")
                if callable(end):
                    end()
                hold = asyncio.Event()
                self._holds.append(hold)
                if on_event:
                    on_event({"type": "tts_text", "text": "正在回答"})
                if on_audio:
                    on_audio(b"\x00\x10" * 8)
                if self.stream_calls == 1:
                    await hold.wait()
                if on_event:
                    on_event({"type": "turn_end"})
                return SayResult(events=[{"type": "turn_end"}], pcm=b"\x00\x10" * 8)

            async def abort_playback(self):
                self.abort_calls += 1
                if self._holds:
                    self._holds[-1].set()

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
                    self.assertEqual(first.get("status"), "waiting")
                    ws.send_bytes(b"\x7f\x7f" * 160)
                    hit = ws.receive_json()
                    self.assertEqual(hit.get("status"), "hit")
                    loud = b"\x00\x40" * 4000
                    ws.send_bytes(loud)
                    ws.send_bytes(loud)
                    ws.send_json({"type": "listen", "state": "stop"})
                    saw_tts = False
                    for _ in range(12):
                        message = ws.receive()
                        if message.get("text"):
                            payload = json.loads(message["text"])
                            if payload.get("type") == "tts_text":
                                saw_tts = True
                                break
                            if payload.get("type") == "wake" and payload.get("status") == "listen":
                                continue
                    self.assertTrue(saw_tts)
                    ws.send_bytes(b"\x7f\x7f" * 160)
                    ws.send_bytes(loud)
                    ws.send_bytes(loud)
                    ws.send_json({"type": "listen", "state": "stop"})
                    statuses = []
                    for _ in range(24):
                        message = ws.receive()
                        if not message.get("text"):
                            continue
                        payload = json.loads(message["text"])
                        if payload.get("type") == "wake":
                            statuses.append(payload.get("status"))
                        if statuses and statuses[-1] == "waiting":
                            break
                    self.assertGreaterEqual(session.abort_calls, 1)
                    self.assertGreaterEqual(session.stream_calls, 2)
                    self.assertIn("hit", statuses)
        finally:
            Path(tmp.name).unlink(missing_ok=True)
            _restore_env(saved)


class ToneTests(unittest.TestCase):
    def test_prompt_tone_matches_original_ding(self):
        pcm = render_prompt_tone_pcm()
        self.assertEqual(len(pcm), int(0.12 * 16000) * 2)
        self.assertGreater(max(pcm), 0)
        wav_path = ROOT / "device_ui" / "static" / "wake_ding.wav"
        self.assertTrue(wav_path.is_file())
        self.assertGreater(wav_path.stat().st_size, 44)

    def test_write_prompt_tone_wav_header(self):
        import tempfile
        import wave

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            path = write_prompt_tone_wav(tmp.name)
            with wave.open(str(path), "rb") as wav:
                self.assertEqual(wav.getnchannels(), 1)
                self.assertEqual(wav.getsampwidth(), 2)
                self.assertEqual(wav.getframerate(), 16000)
                self.assertEqual(wav.getnframes(), int(0.12 * 16000))
        finally:
            Path(tmp.name).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
