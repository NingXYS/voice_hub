import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path("/home/ning/model/lyz/src/tianlian_voice_bridge")))

import httpx
import websockets

from device_ui.app import create_app
from device_ui.client_settings import SettingsError, public_view, save, settings_path
from device_ui.wavutil import pcm_s16le_to_wav
from jetson_client.voice_client import run_text
from server.app import VoiceHub
from server.pipeline import Pipeline
from server.providers.asr_local import NullASR
from server.providers.tts_dummy import DummyTTS
from tests.test_adapters import FakeLLM


class _PcmTTS:
    name = "pcm"

    def synthesize(self, text: str) -> bytes:
        return b"\x00\x10" * 160


_CLIENT_ENV_KEYS = ("VOICE_HUB_URL", "ROBOT_ID", "ROBOT_SCHEMA_VERSION", "VOICE_HUB_CLIENT_ENV")


def _isolate_env():
    saved = {key: os.environ.get(key) for key in _CLIENT_ENV_KEYS}
    for key in ("VOICE_HUB_URL", "ROBOT_ID", "ROBOT_SCHEMA_VERSION"):
        os.environ.pop(key, None)
    return saved


def _restore_env(saved):
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self._saved = _isolate_env()
        self.tmp = tempfile.NamedTemporaryFile("w", delete=False)
        self.tmp.close()
        os.environ["VOICE_HUB_CLIENT_ENV"] = self.tmp.name

    def tearDown(self):
        Path(self.tmp.name).unlink(missing_ok=True)
        _restore_env(self._saved)

    def test_path_uses_override(self):
        self.assertEqual(settings_path(), Path(self.tmp.name))

    def test_save_and_mask_no_cloud_keys(self):
        view = save({"url": "ws://10.101.67.111:8765", "robot_id": "demo_single", "schema_version": 1})
        self.assertEqual(view["url"], "ws://10.101.67.111:8765")
        self.assertEqual(view["robot_id"], "demo_single")
        text = Path(self.tmp.name).read_text(encoding="utf-8")
        self.assertNotIn("CHAT_LLM", text)
        self.assertNotIn("METASO", text)
        self.assertNotIn("DOUBAO", text)
        self.assertEqual(Path(self.tmp.name).stat().st_mode & 0o777, 0o600)

    def test_rejects_cloud_keys(self):
        with self.assertRaises(SettingsError):
            save({"CHAT_LLM_API_KEY": "sk-test"})
        with self.assertRaises(SettingsError):
            save({"METASO_API_KEY": "mk-test"})

    def test_loopback_warning(self):
        save({"url": "ws://127.0.0.1:8765", "robot_id": "tianlian_dual"})
        self.assertIn("127.0.0.1", public_view()["warning"])

    def test_fake_ip_warning(self):
        save({"url": "ws://198.18.0.8:8765", "robot_id": "tianlian_dual"})
        self.assertIn("假 IP", public_view()["warning"])

    def test_empty_url_asks_to_fill(self):
        self.assertIn("请填写", public_view()["warning"])
        self.assertEqual(public_view()["url"], "")


class WavTests(unittest.TestCase):
    def test_header(self):
        pcm = b"\x00\x00" * 16
        wav = pcm_s16le_to_wav(pcm)
        self.assertTrue(wav.startswith(b"RIFF"))
        self.assertIn(b"WAVE", wav[:16])
        self.assertEqual(wav[40:44], (len(pcm)).to_bytes(4, "little"))
        self.assertEqual(wav[44:], pcm)

    def test_empty(self):
        self.assertEqual(pcm_s16le_to_wav(b""), b"")


class DeviceApiTests(unittest.TestCase):
    def setUp(self):
        self._saved = _isolate_env()
        self.tmp = tempfile.NamedTemporaryFile("w", delete=False)
        self.tmp.close()
        os.environ["VOICE_HUB_CLIENT_ENV"] = self.tmp.name

    def tearDown(self):
        Path(self.tmp.name).unlink(missing_ok=True)
        _restore_env(self._saved)

    def test_settings_http_forbids_keys(self):
        async def scenario():
            transport = httpx.ASGITransport(app=create_app())
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                res = await client.put("/api/settings", json={"METASO_API_KEY": "mk-nope"})
                self.assertEqual(res.status_code, 400)
                self.assertIn("密钥", res.json()["error"])

        asyncio.run(scenario())

    def test_connect_and_say(self):
        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=_PcmTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                save({"url": f"ws://127.0.0.1:{port}", "robot_id": "tianlian_dual", "schema_version": 1})
                transport = httpx.ASGITransport(app=create_app())
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    connected = await client.post("/api/connect")
                    body = connected.json()
                    self.assertTrue(body.get("ok"), body)
                    said = await client.post("/api/say", json={"text": "你有几个关节"})
                    payload = said.json()
                    self.assertTrue(payload.get("ok"), payload)
                    types = [e.get("type") for e in payload.get("events") or []]
                    self.assertIn("tts_text", types)
                    self.assertIn("turn_end", types)
                    self.assertTrue(payload.get("has_audio"))
                    self.assertIn("wav_base64", payload)
                    health = await client.get("/api/health")
                    self.assertTrue(health.json().get("ok"))
                    await client.post("/api/disconnect")

        asyncio.run(scenario())

    def test_say_stream_audio_before_turn_end(self):
        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=_PcmTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                save({"url": f"ws://127.0.0.1:{port}", "robot_id": "tianlian_dual", "schema_version": 1})
                transport = httpx.ASGITransport(app=create_app())
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    connected = await client.post("/api/connect")
                    self.assertTrue(connected.json().get("ok"), connected.json())
                    saw_audio = False
                    saw_end = False
                    audio_before_end = False
                    async with client.stream(
                        "POST", "/api/say/stream", json={"text": "你有几个关节"}
                    ) as res:
                        self.assertEqual(res.status_code, 200)
                        async for line in res.aiter_lines():
                            if line.startswith("event: audio"):
                                saw_audio = True
                                if not saw_end:
                                    audio_before_end = True
                            if line.startswith("event: turn_end"):
                                saw_end = True
                    self.assertTrue(saw_audio)
                    self.assertTrue(saw_end)
                    self.assertTrue(audio_before_end)
                    await client.post("/api/disconnect")

        asyncio.run(scenario())

    def test_bad_robot_is_speakable(self):
        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=DummyTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                save({"url": f"ws://127.0.0.1:{port}", "robot_id": "no_such_arm", "schema_version": 1})
                transport = httpx.ASGITransport(app=create_app())
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    body = (await client.post("/api/connect")).json()
                    self.assertFalse(body.get("ok"))
                    self.assertTrue(body.get("error"))

    def test_say_without_connect_is_speakable(self):
        async def scenario():
            transport = httpx.ASGITransport(app=create_app())
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                body = (await client.post("/api/say", json={"text": "你好"})).json()
                self.assertFalse(body.get("ok"))
                self.assertIn("连", body.get("error") or "")

        asyncio.run(scenario())

    def test_stream_empty_text_is_speakable(self):
        async def scenario():
            transport = httpx.ASGITransport(app=create_app())
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                body = (await client.post("/api/say/stream", json={"text": "  "})).json()
                self.assertEqual(body.get("ok"), False)
                self.assertIn("请输入", body.get("error") or "")

        asyncio.run(scenario())

    def test_abort_without_connect_is_speakable(self):
        async def scenario():
            transport = httpx.ASGITransport(app=create_app())
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                body = (await client.post("/api/abort")).json()
                self.assertFalse(body.get("ok"))
                self.assertIn("连", body.get("error") or "")

        asyncio.run(scenario())

    def test_cli_still_dry_runs(self):
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

    def test_voice_ws_requires_connect(self):
        from starlette.testclient import TestClient

        with TestClient(create_app()) as client:
            with client.websocket_connect("/ws/voice") as ws:
                msg = ws.receive_json()
                self.assertEqual(msg.get("type"), "error")
                self.assertIn("连", msg.get("message") or "")

    def test_voice_ws_listen_turn(self):
        from starlette.testclient import TestClient
        from jetson_client.session import SayResult

        class FakeSession:
            connected = True
            hello = {"type": "hello_ok"}
            url = "ws://x"
            robot_id = "tianlian_dual"
            schema_version = 1
            bridge_error = ""

            async def listen_stream(self, chunks, idle_after_speech=8.0, on_audio=None, on_event=None):
                async for _pcm in chunks:
                    pass
                if on_event:
                    on_event({"type": "stt", "text": "你好"})
                    on_event({"type": "tts_text", "text": "你好。"})
                if on_audio:
                    on_audio(b"\x00\x10" * 8)
                if on_event:
                    on_event({"type": "turn_end"})
                return SayResult(events=[{"type": "turn_end"}], pcm=b"\x00\x10" * 8)

            async def abort_playback(self):
                return None

            async def close(self):
                return None

        with TestClient(create_app(FakeSession())) as client:
            with client.websocket_connect("/ws/voice") as ws:
                ws.send_json({"type": "listen", "state": "start"})
                ws.send_bytes(b"\x00\x00" * 160)
                ws.send_json({"type": "listen", "state": "stop"})
                kinds = []
                for _ in range(8):
                    message = ws.receive()
                    if message.get("text"):
                        kinds.append(json.loads(message["text"]).get("type"))
                    elif message.get("bytes"):
                        kinds.append("audio")
                    if "turn_end" in kinds:
                        break
                self.assertIn("stt", kinds)
                self.assertIn("tts_text", kinds)
                self.assertIn("audio", kinds)
                self.assertIn("turn_end", kinds)


if __name__ == "__main__":
    unittest.main()
