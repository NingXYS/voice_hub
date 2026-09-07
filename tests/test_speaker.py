import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jetson_client.speaker_gate import allows_host, run_gated_listen, verify_pcm
from jetson_client.session import SayResult
from tests.test_device_ui import _isolate_env, _restore_env


class _Decision:
    def __init__(self, accepted, score=0.91, name="操作员", reason="matched"):
        self.accepted = accepted
        self.score = score
        self.speaker_id = "ab" * 16
        self.display_name = name
        self.reason = reason
        self.duration_ms = 8
        self.threshold = 0.6
        self.audio_ms = 1200
        self.rms = 0.04
        self.score_mode = "max_template"

    def event_payload(self, *, mode):
        return {
            "mode": mode,
            "accepted": self.accepted,
            "score": round(self.score, 4),
            "speaker_id": self.speaker_id,
            "display_name": self.display_name,
            "reason": self.reason,
            "duration_ms": self.duration_ms,
        }


class SpeakerGateTests(unittest.TestCase):
    def setUp(self):
        self._saved = _isolate_env()
        self.tmp = tempfile.NamedTemporaryFile("w", delete=False)
        self.tmp.close()
        os.environ["VOICE_HUB_CLIENT_ENV"] = self.tmp.name

    def tearDown(self):
        Path(self.tmp.name).unlink(missing_ok=True)
        _restore_env(self._saved)

    def test_off_skips_verifier(self):
        os.environ["VOICE_SPEAKER_VERIFY_MODE"] = "off"
        verifier = mock.Mock()
        with mock.patch("jetson_client.speaker_gate._verifier", return_value=verifier):
            with mock.patch("jetson_client.speaker_gate.list_profiles", return_value=[{"speaker_id": "x"}]):
                decision = verify_pcm(b"\x00\x10" * 1600)
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["reason"], "off")
        verifier.verify.assert_not_called()
        self.assertTrue(allows_host(decision))

    def test_shadow_forwards_mismatch(self):
        os.environ["VOICE_SPEAKER_VERIFY_MODE"] = "shadow"
        verifier = mock.Mock()
        verifier.verify.return_value = _Decision(False, score=0.12, reason="speaker_mismatch")
        with mock.patch("jetson_client.speaker_gate._verifier", return_value=verifier):
            with mock.patch("jetson_client.speaker_gate.list_profiles", return_value=[{"speaker_id": "x"}]):
                decision = verify_pcm(b"\x00\x10" * 1600)
        self.assertTrue(decision["accepted"])
        self.assertFalse(decision["matched"])
        self.assertEqual(decision["reason"], "speaker_mismatch")
        self.assertFalse(decision["enforce_would_accept"])
        self.assertEqual(verifier.verify.call_count, 1)
        self.assertTrue(allows_host(decision))

    def test_enforce_blocks_mismatch(self):
        os.environ["VOICE_SPEAKER_VERIFY_MODE"] = "enforce"
        verifier = mock.Mock()
        verifier.verify.return_value = _Decision(False, score=0.12, reason="speaker_mismatch")
        with mock.patch("jetson_client.speaker_gate._verifier", return_value=verifier):
            with mock.patch("jetson_client.speaker_gate.list_profiles", return_value=[{"speaker_id": "x"}]):
                decision = verify_pcm(b"\x00\x10" * 1600)
        self.assertFalse(decision["accepted"])
        self.assertFalse(decision["matched"])
        self.assertFalse(allows_host(decision))

    def test_enforce_no_profiles_blocks(self):
        os.environ["VOICE_SPEAKER_VERIFY_MODE"] = "enforce"
        with mock.patch("jetson_client.speaker_gate.list_profiles", return_value=[]):
            decision = verify_pcm(b"\x00\x10" * 1600)
        self.assertFalse(decision["accepted"])
        self.assertEqual(decision["reason"], "no_profiles")
        self.assertFalse(allows_host(decision))

    def test_shadow_no_profiles_still_forwards(self):
        os.environ["VOICE_SPEAKER_VERIFY_MODE"] = "shadow"
        with mock.patch("jetson_client.speaker_gate.list_profiles", return_value=[]):
            decision = verify_pcm(b"\x00\x10" * 1600)
        self.assertTrue(decision["accepted"])
        self.assertEqual(decision["reason"], "no_profiles")
        self.assertTrue(allows_host(decision))


class GatedListenTests(unittest.TestCase):
    def setUp(self):
        self._saved = _isolate_env()

    def tearDown(self):
        _restore_env(self._saved)

    def test_off_streams_live(self):
        os.environ["VOICE_SPEAKER_VERIFY_MODE"] = "off"

        class Session:
            def __init__(self):
                self.calls = []

            async def listen_stream(self, chunks, **kwargs):
                parts = []
                async for pcm in chunks:
                    parts.append(pcm)
                self.calls.append(("stream", b"".join(parts)))
                return SayResult(events=[{"type": "turn_end"}], pcm=b"")

            async def listen_pcm(self, pcm, **kwargs):
                self.calls.append(("pcm", pcm))
                return SayResult(events=[{"type": "turn_end"}], pcm=b"")

        async def source():
            yield b"abc"
            yield b"def"

        session = Session()
        result, decision = asyncio.run(run_gated_listen(session, source()))
        self.assertIsNotNone(result)
        self.assertEqual(decision["reason"], "off")
        self.assertEqual(session.calls, [("stream", b"abcdef")])

    def test_shadow_streams_then_scores(self):
        os.environ["VOICE_SPEAKER_VERIFY_MODE"] = "shadow"

        class Session:
            def __init__(self):
                self.calls = []

            async def listen_stream(self, chunks, **kwargs):
                parts = []
                async for pcm in chunks:
                    parts.append(pcm)
                self.calls.append(("stream", b"".join(parts)))
                return SayResult(events=[{"type": "turn_end"}], pcm=b"")

        async def source():
            yield b"abc"

        session = Session()
        events = []
        with mock.patch(
            "jetson_client.speaker_gate.verify_pcm",
            return_value={
                "mode": "shadow",
                "accepted": True,
                "matched": False,
                "reason": "speaker_mismatch",
                "score": 0.1,
                "display_name": None,
            },
        ) as verify:
            result, decision = asyncio.run(
                run_gated_listen(session, source(), on_event=events.append)
            )
        self.assertIsNotNone(result)
        self.assertEqual(session.calls, [("stream", b"abc")])
        verify.assert_called_once_with(b"abc")
        self.assertTrue(decision["accepted"])
        self.assertEqual(events[0]["type"], "speaker")

    def test_enforce_reject_skips_host(self):
        os.environ["VOICE_SPEAKER_VERIFY_MODE"] = "enforce"

        class Session:
            def __init__(self):
                self.calls = []

            async def listen_stream(self, chunks, **kwargs):
                self.calls.append("stream")
                return SayResult(events=[], pcm=b"")

            async def listen_pcm(self, pcm, **kwargs):
                self.calls.append(("pcm", pcm))
                return SayResult(events=[], pcm=b"")

        async def source():
            yield b"nope"

        session = Session()
        with mock.patch(
            "jetson_client.speaker_gate.verify_pcm",
            return_value={
                "mode": "enforce",
                "accepted": False,
                "matched": False,
                "reason": "speaker_mismatch",
                "score": 0.1,
                "display_name": None,
            },
        ):
            result, decision = asyncio.run(run_gated_listen(session, source()))
        self.assertIsNone(result)
        self.assertFalse(decision["accepted"])
        self.assertEqual(session.calls, [])

    def test_enforce_match_replays_pcm(self):
        os.environ["VOICE_SPEAKER_VERIFY_MODE"] = "enforce"

        class Session:
            def __init__(self):
                self.calls = []

            async def listen_stream(self, chunks, **kwargs):
                self.calls.append("stream")
                return SayResult(events=[], pcm=b"")

            async def listen_pcm(self, pcm, **kwargs):
                self.calls.append(("pcm", pcm))
                return SayResult(events=[{"type": "turn_end"}], pcm=b"")

        async def source():
            yield b"ok"

        session = Session()
        with mock.patch(
            "jetson_client.speaker_gate.verify_pcm",
            return_value={
                "mode": "enforce",
                "accepted": True,
                "matched": True,
                "reason": "matched",
                "score": 0.9,
                "display_name": "操作员",
            },
        ):
            result, decision = asyncio.run(run_gated_listen(session, source()))
        self.assertIsNotNone(result)
        self.assertEqual(session.calls, [("pcm", b"ok")])
        self.assertTrue(decision["accepted"])


class SpeakerWsEnforceTests(unittest.TestCase):
    def setUp(self):
        self._saved = _isolate_env()
        self.tmp = tempfile.NamedTemporaryFile("w", delete=False)
        self.tmp.write("VOICE_SPEAKER_VERIFY_MODE=enforce\n")
        self.tmp.close()
        os.environ["VOICE_HUB_CLIENT_ENV"] = self.tmp.name

    def tearDown(self):
        Path(self.tmp.name).unlink(missing_ok=True)
        _restore_env(self._saved)

    def test_wake_enforce_reject_does_not_call_host(self):
        import json

        from starlette.testclient import TestClient
        from device_ui.app import create_app

        class Detector:
            def accept_pcm(self, pcm):
                return True

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
            pcm_calls = 0

            async def listen_stream(self, chunks, idle_after_speech=8.0, on_audio=None, on_event=None):
                self.stream_calls += 1
                async for _pcm in chunks:
                    pass
                return SayResult(events=[{"type": "turn_end"}], pcm=b"")

            async def listen_pcm(self, pcm, idle_after_speech=8.0, on_audio=None, on_event=None):
                self.pcm_calls += 1
                return SayResult(events=[{"type": "turn_end"}], pcm=b"")

            async def abort_playback(self):
                return None

            async def close(self):
                return None

        session = FakeSession()
        with mock.patch(
            "jetson_client.speaker_gate.verify_pcm",
            return_value={
                "mode": "enforce",
                "accepted": False,
                "matched": False,
                "reason": "speaker_mismatch",
                "score": 0.11,
                "display_name": None,
            },
        ):
            with TestClient(create_app(session, wake_detector=Detector())) as client:
                with client.websocket_connect("/ws/voice") as ws:
                    first = ws.receive_json()
                    self.assertEqual(first.get("status"), "waiting")
                    ws.send_bytes(b"\x10\x00" * 160)
                    hit = ws.receive_json()
                    self.assertEqual(hit.get("status"), "hit")
                    loud = b"\x00\x40" * 4000
                    ws.send_bytes(loud)
                    ws.send_bytes(loud)
                    ws.send_json({"type": "listen", "state": "stop"})
                    statuses = []
                    for _ in range(12):
                        message = ws.receive()
                        if not message.get("text"):
                            continue
                        payload = json.loads(message["text"])
                        if payload.get("type") == "wake":
                            statuses.append(payload.get("status"))
                        if payload.get("status") == "rejected":
                            break
                    self.assertIn("rejected", statuses)
        self.assertEqual(session.stream_calls, 0)
        self.assertEqual(session.pcm_calls, 0)


if __name__ == "__main__":
    unittest.main()
