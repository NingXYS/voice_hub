import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.providers.realtime_doubao import (
    EVENT_ASR_ENDED,
    EVENT_CHAT_ENDED,
    EVENT_CHAT_RESPONSE,
    EVENT_CHAT_TTS_TEXT,
    EVENT_CONNECTION_STARTED,
    EVENT_REPLY_DONE,
    EVENT_START_CONNECTION,
    EVENT_START_SESSION,
    EVENT_TASK_REQUEST,
    EVENT_TTS_RESPONSE,
    MSG_ACK,
    encode_event,
    format_external_rag,
    parse_frame,
    pcm16_resample,
    to_event,
)


class RealtimeProtocolTests(unittest.TestCase):
    def test_start_connection_has_no_session_id(self):
        frame = encode_event(EVENT_START_CONNECTION, {})
        parsed = parse_frame(frame)
        self.assertEqual(parsed["event"], EVENT_START_CONNECTION)
        self.assertEqual(parsed["session_id"], "")
        self.assertEqual(parsed["payload"], {})

    def test_start_session_roundtrip(self):
        payload = {"dialog": {"bot_name": "助手", "location": {"city": "长春"}}}
        frame = encode_event(EVENT_START_SESSION, payload, session_id="abc")
        parsed = parse_frame(frame)
        self.assertEqual(parsed["event"], EVENT_START_SESSION)
        self.assertEqual(parsed["session_id"], "abc")
        self.assertEqual(parsed["payload"]["dialog"]["location"]["city"], "长春")

    def test_audio_task_is_gzip_pcm(self):
        pcm = b"\x01\x00" * 10
        frame = encode_event(EVENT_TASK_REQUEST, pcm, session_id="s1", audio=True)
        parsed = parse_frame(frame)
        self.assertEqual(parsed["event"], EVENT_TASK_REQUEST)
        self.assertEqual(parsed["payload"], pcm)

    def test_asr_end_and_tts_events(self):
        asr = encode_event(EVENT_ASR_ENDED, {"content": "你好"}, session_id="s")
        event = to_event(parse_frame(asr))
        self.assertEqual(event.kind, "asr_end")
        self.assertEqual(event.text, "你好")
        tts = encode_event(EVENT_TTS_RESPONSE, b"\x00\x01" * 4, session_id="s", audio=True)
        audio = to_event(parse_frame(tts))
        self.assertEqual(audio.kind, "audio")
        self.assertEqual(audio.pcm, b"\x00\x01" * 4)
        chat = encode_event(EVENT_CHAT_RESPONSE, {"content": "在的"}, session_id="s")
        text = to_event(parse_frame(chat))
        self.assertEqual(text.kind, "text")
        self.assertEqual(text.text, "在的")
        chat_end = to_event({"event": EVENT_CHAT_ENDED, "payload": {}})
        self.assertEqual(chat_end.kind, "chat_end")
        tts_done = to_event({"event": EVENT_REPLY_DONE, "payload": {}})
        self.assertEqual(tts_done.kind, "ended")

    def test_connection_started_without_session(self):
        frame = encode_event(EVENT_CONNECTION_STARTED, {})
        parsed = parse_frame(frame)
        self.assertEqual(parsed["event"], EVENT_CONNECTION_STARTED)

    def test_tts_json_base64_payload(self):
        import base64

        pcm = b"\x11\x22" * 4
        event = to_event(
            {"event": EVENT_TTS_RESPONSE, "payload": {"audio": base64.b64encode(pcm).decode()}}
        )
        self.assertEqual(event.kind, "audio")
        self.assertEqual(event.pcm, pcm)

    def test_begin_listen_clears_old_asr(self):
        import asyncio

        from server.providers.realtime_doubao import DoubaoRealtimeSession

        session = DoubaoRealtimeSession()
        session._asr_text = "旧的一句"
        session._asr_done.set()
        session.begin_listen()
        self.assertEqual(session._asr_text, "")
        self.assertFalse(session._asr_done.is_set())

        async def keep_final():
            session._asr_text = "你好"
            session._asr_done.set()
            await session.end_utterance()
            return await session.wait_asr_final(0.2)

        self.assertEqual(asyncio.run(keep_final()), "你好")

    def test_drain_keeps_close_sentinel(self):
        from server.providers.realtime_doubao import DoubaoRealtimeSession, RealtimeEvent

        session = DoubaoRealtimeSession()
        session._queue.put_nowait(RealtimeEvent(kind="text", text="x"))
        session._queue.put_nowait(None)
        session.drain()
        self.assertIsNone(session._queue.get_nowait())

    def test_resample_24k_to_16k_length(self):
        # 24 samples at 24kHz = 1ms; 16 samples at 16kHz
        pcm = b"\x00\x10" * 24
        out = pcm16_resample(pcm, 24000, 16000)
        self.assertEqual(len(out), 32)

    def test_ack_binary_is_audio(self):
        pcm = b"\x11\x22" * 8
        event = to_event({"message_type": MSG_ACK, "event": 0, "payload": pcm})
        self.assertEqual(event.kind, "audio")
        self.assertEqual(event.pcm, pcm)

    def test_gzip_failure_keeps_payload(self):
        import struct

        from server.providers.realtime_doubao import FLAG_EVENT, HEADER_SIZE, MSG_FULL_SERVER, PROTOCOL_VERSION, SERIAL_NONE

        pcm = b"\x33\x44" * 6
        frame = bytearray([(PROTOCOL_VERSION << 4) | HEADER_SIZE, (MSG_FULL_SERVER << 4) | FLAG_EVENT, (SERIAL_NONE << 4) | 0x01, 0x00])
        frame.extend(struct.pack(">I", EVENT_TTS_RESPONSE))
        sid = b"s"
        frame.extend(struct.pack(">I", len(sid)))
        frame.extend(sid)
        frame.extend(struct.pack(">I", len(pcm)))
        frame.extend(pcm)
        parsed = parse_frame(bytes(frame))
        self.assertEqual(parsed["event"], EVENT_TTS_RESPONSE)
        audio = to_event(parsed)
        self.assertEqual(audio.kind, "audio")
        self.assertEqual(audio.pcm, pcm)

    def test_hold_drops_default_audio_until_chat_tts(self):
        from server.providers.realtime_doubao import DoubaoRealtimeSession, RealtimeEvent

        session = DoubaoRealtimeSession()
        session._hold_default_audio = True
        auto = RealtimeEvent(kind="audio", pcm=b"\x01\x00", tts_type="")
        self.assertFalse(session._accept_dialog_output(auto))
        self.assertTrue(session._hold_default_audio)
        caption = RealtimeEvent(kind="text", text="我查一下。", tts_type="chat_tts_text")
        self.assertTrue(session._accept_dialog_output(caption))
        self.assertFalse(session._hold_default_audio)
        follow = RealtimeEvent(kind="audio", pcm=b"\x02\x00", tts_type="")
        self.assertTrue(session._accept_dialog_output(follow))
        session.mute()
        self.assertFalse(session._accept_dialog_output(follow))
        self.assertTrue(session._accept_dialog_output(RealtimeEvent(kind="ended")))

    def test_chat_tts_text_sends_start_then_end(self):
        import asyncio

        from server.providers.realtime_doubao import DoubaoRealtimeSession

        class FakeWS:
            def __init__(self):
                self.sent = []

            async def send(self, data):
                self.sent.append(data)

        session = DoubaoRealtimeSession()
        session._ws = FakeWS()
        asyncio.run(session.chat_tts_text("我查一下。"))
        self.assertEqual(len(session._ws.sent), 2)
        first = parse_frame(session._ws.sent[0])
        last = parse_frame(session._ws.sent[1])
        self.assertEqual(first["event"], EVENT_CHAT_TTS_TEXT)
        self.assertEqual(first["payload"]["start"], True)
        self.assertEqual(first["payload"]["end"], False)
        self.assertEqual(first["payload"]["content"], "我查一下。")
        self.assertEqual(last["payload"]["start"], False)
        self.assertEqual(last["payload"]["end"], True)
        self.assertEqual(last["payload"]["content"], "")

    def test_prepare_external_speech_times_out_quickly(self):
        import asyncio
        import time

        from server.providers.realtime_doubao import DoubaoRealtimeSession

        class FakeWS:
            async def send(self, data):
                return None

        session = DoubaoRealtimeSession()
        session._ws = FakeWS()

        async def run():
            started = time.monotonic()
            await session.prepare_external_speech()
            return time.monotonic() - started

        elapsed = asyncio.run(run())
        self.assertLess(elapsed, 0.45)

    def test_format_external_rag_json_list(self):
        blob = format_external_rag("天气：长春今天多云，气温二十二度。\n其它：忽略")
        rows = __import__("json").loads(blob)
        self.assertIsInstance(rows, list)
        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("二十二", rows[0]["content"])


if __name__ == "__main__":
    unittest.main()
