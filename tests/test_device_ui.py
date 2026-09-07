import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


_CLIENT_ENV_KEYS = (
    "VOICE_HUB_URL",
    "ROBOT_ID",
    "ROBOT_SCHEMA_VERSION",
    "VOICE_HUB_CLIENT_ENV",
    "VOICE_ENABLE_ROBOT_COMMAND",
    "VOICE_ENABLE_MASSAGE",
    "VOICE_SPEAKER_VERIFY_MODE",
    "VOICE_HUB_BRIDGE_PATH",
    "ROBOT_BRIDGE_MODULE",
    "VOICE_KWS_DISABLE",
)
_BRIDGE = "/home/ning/model/lyz/src/tianlian_voice_bridge"


def _isolate_env():
    saved = {key: os.environ.get(key) for key in _CLIENT_ENV_KEYS}
    for key in ("VOICE_HUB_URL", "ROBOT_ID", "ROBOT_SCHEMA_VERSION", "ROBOT_BRIDGE_MODULE"):
        os.environ.pop(key, None)
    os.environ["VOICE_ENABLE_ROBOT_COMMAND"] = "0"
    os.environ.pop("VOICE_SPEAKER_VERIFY_MODE", None)
    os.environ["VOICE_HUB_BRIDGE_PATH"] = _BRIDGE
    os.environ["VOICE_KWS_DISABLE"] = "1"
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
        warning = public_view()["warning"]
        self.assertIn("127.0.0.1", warning)
        self.assertIn("本机联调", warning)

    def test_fake_ip_warning(self):
        save({"url": "ws://198.18.0.8:8765", "robot_id": "tianlian_dual"})
        self.assertIn("假 IP", public_view()["warning"])

    def test_empty_url_asks_to_fill(self):
        self.assertIn("请填写", public_view()["warning"])
        self.assertEqual(public_view()["url"], "")

    def test_save_keeps_kws_and_bridge_paths(self):
        Path(self.tmp.name).write_text(
            "\n".join(
                [
                    "VOICE_HUB_URL=ws://10.101.67.111:8765",
                    "ROBOT_ID=tianlian_dual",
                    "ROBOT_SCHEMA_VERSION=1",
                    "APLAY_DEVICE=default",
                    "VOICE_KWS_PYTHONPATH=/home/li/VoiceOnline/.venv_online/lib/python3.10/site-packages",
                    "VOICE_KWS_MODEL_DIR=/home/li/VoiceOnline/models/kws",
                    "WAKE_WORD=小溪小溪",
                    "VOICE_ENABLE_ROBOT_COMMAND=1",
                    "VOICE_ENABLE_MASSAGE=0",
                    "VOICE_SPEAKER_VERIFY_MODE=shadow",
                    "CHAT_LLM_API_KEY=should-not-keep",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        save({"url": "ws://10.101.67.111:8765", "robot_id": "tianlian_dual", "schema_version": 1})
        text = Path(self.tmp.name).read_text(encoding="utf-8")
        self.assertIn("VOICE_KWS_PYTHONPATH=/home/li/VoiceOnline/.venv_online/lib/python3.10/site-packages", text)
        self.assertIn("VOICE_KWS_MODEL_DIR=/home/li/VoiceOnline/models/kws", text)
        self.assertIn("WAKE_WORD=小溪小溪", text)
        self.assertIn("VOICE_ENABLE_ROBOT_COMMAND=1", text)
        self.assertIn("VOICE_ENABLE_MASSAGE=0", text)
        self.assertIn("VOICE_SPEAKER_VERIFY_MODE=shadow", text)
        self.assertNotIn("CHAT_LLM_API_KEY", text)


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
                    self.assertNotIn("robot_command", types)
                    self.assertNotIn("bridge", types)
                    self.assertTrue(payload.get("has_audio"))
                    self.assertIn("wav_base64", payload)
                    status = (await client.get("/api/status")).json()
                    self.assertTrue(status.get("bridge_loaded"), status)
                    self.assertFalse(status.get("robot_publish_enabled"), status)
                    self.assertIn(status.get("last_command"), (None, []))
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

    def test_connect_and_say_control_dry_run(self):
        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=DummyTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                save({"url": f"ws://127.0.0.1:{port}", "robot_id": "tianlian_dual", "schema_version": 1})
                transport = httpx.ASGITransport(app=create_app())
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    connected = await client.post("/api/connect")
                    body = connected.json()
                    self.assertTrue(body.get("ok"), body)
                    self.assertTrue(body.get("bridge_loaded"), body)
                    self.assertFalse(body.get("bridge_error"), body)
                    said = await client.post("/api/say", json={"text": "左臂向上10厘米"})
                    payload = said.json()
                    self.assertTrue(payload.get("ok"), payload)
                    events = payload.get("events") or []
                    types = [e.get("type") for e in events]
                    self.assertIn("robot_command", types)
                    self.assertIn("bridge", types)
                    self.assertLess(types.index("robot_command"), types.index("bridge"))
                    command = next(e for e in events if e.get("type") == "robot_command")
                    move = (command.get("commands") or [None])[0]
                    self.assertEqual(move["action"], "move_cartesian")
                    self.assertEqual(move["arm"], "left")
                    self.assertEqual(move["z"], 0.1)
                    bridge = next(e for e in events if e.get("type") == "bridge")
                    self.assertEqual(bridge["status"], "dry_run")
                    status = (await client.get("/api/status")).json()
                    self.assertEqual(status.get("last_bridge_status"), "dry_run")
                    self.assertEqual(status.get("last_command")[0]["z"], 0.1)
                    self.assertFalse(status.get("robot_publish_enabled"))
                    await client.post("/api/disconnect")

        asyncio.run(scenario())

    def test_create_app_pins_dry_run_even_if_env_was_on(self):
        os.environ["VOICE_ENABLE_ROBOT_COMMAND"] = "1"
        os.environ["VOICE_ENABLE_MASSAGE"] = "1"
        os.environ["VOICE_SPEAKER_VERIFY_MODE"] = "enforce"
        tmp = tempfile.NamedTemporaryFile("w", delete=False)
        tmp.close()
        os.environ["VOICE_HUB_CLIENT_ENV"] = tmp.name

        async def scenario():
            transport = httpx.ASGITransport(app=create_app())
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                status = (await client.get("/api/status")).json()
                self.assertFalse(status.get("robot_publish_enabled"))
                self.assertFalse(status.get("massage_enabled"))
                self.assertEqual(status.get("speaker_mode"), "off")

        try:
            asyncio.run(scenario())
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def test_create_app_honors_client_env_live_flags(self):
        tmp = tempfile.NamedTemporaryFile("w", delete=False)
        tmp.write("VOICE_ENABLE_ROBOT_COMMAND=1\nVOICE_ENABLE_MASSAGE=1\nVOICE_SPEAKER_VERIFY_MODE=shadow\n")
        tmp.close()
        os.environ["VOICE_HUB_CLIENT_ENV"] = tmp.name
        os.environ["VOICE_ENABLE_ROBOT_COMMAND"] = "0"
        os.environ["VOICE_SPEAKER_VERIFY_MODE"] = "enforce"

        async def scenario():
            transport = httpx.ASGITransport(app=create_app())
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                status = (await client.get("/api/status")).json()
                self.assertTrue(status.get("robot_publish_enabled"))
                self.assertTrue(status.get("massage_enabled"))
                self.assertEqual(status.get("speaker_mode"), "shadow")

        try:
            asyncio.run(scenario())
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def test_static_and_html_are_not_cached(self):
        async def scenario():
            transport = httpx.ASGITransport(app=create_app())
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                home = await client.get("/")
                self.assertEqual(home.status_code, 200)
                self.assertIn("no-store", (home.headers.get("cache-control") or "").lower())
                self.assertIn("armCard", home.text)
                js = await client.get("/static/dash.js")
                self.assertEqual(js.status_code, 200)
                self.assertIn("no-store", (js.headers.get("cache-control") or "").lower())
                self.assertIn("小溪小溪", js.text)
                home_text = home.text
                self.assertIn("dash.js?v=dash-5", home_text)
                self.assertIn("msgChain", (await client.get("/static/dash.js")).text)

        asyncio.run(scenario())

    def test_say_stream_control_dry_run_before_tts(self):
        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=_PcmTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                save({"url": f"ws://127.0.0.1:{port}", "robot_id": "tianlian_dual", "schema_version": 1})
                transport = httpx.ASGITransport(app=create_app())
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    self.assertTrue((await client.post("/api/connect")).json().get("ok"))
                    order = []
                    async with client.stream(
                        "POST", "/api/say/stream", json={"text": "左臂向上10厘米"}
                    ) as res:
                        self.assertEqual(res.status_code, 200)
                        async for line in res.aiter_lines():
                            if line.startswith("event: "):
                                order.append(line.split(" ", 1)[1].strip())
                    self.assertIn("robot_command", order)
                    self.assertIn("bridge", order)
                    self.assertIn("tts_text", order)
                    self.assertLess(order.index("robot_command"), order.index("bridge"))
                    self.assertLess(order.index("bridge"), order.index("tts_text"))
                    status = (await client.get("/api/status")).json()
                    self.assertEqual(status.get("last_bridge_status"), "dry_run")
                    await client.post("/api/disconnect")

        asyncio.run(scenario())

    def test_underspecified_control_has_no_robot_command(self):
        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=DummyTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                save({"url": f"ws://127.0.0.1:{port}", "robot_id": "tianlian_dual", "schema_version": 1})
                transport = httpx.ASGITransport(app=create_app())
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    self.assertTrue((await client.post("/api/connect")).json().get("ok"))
                    payload = (await client.post("/api/say", json={"text": "左臂向上"})).json()
                    types = [e.get("type") for e in payload.get("events") or []]
                    self.assertNotIn("robot_command", types)
                    self.assertNotIn("bridge", types)
                    tts = " ".join(e.get("text") or "" for e in payload.get("events") or [] if e.get("type") == "tts_text")
                    self.assertIn("请说明", tts)
                    await client.post("/api/disconnect")

        asyncio.run(scenario())

    def test_missing_bridge_is_speakable_and_does_not_publish(self):
        os.environ["ROBOT_BRIDGE_MODULE"] = "no_such_voice_bridge_mod"

        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=DummyTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                save({"url": f"ws://127.0.0.1:{port}", "robot_id": "tianlian_dual", "schema_version": 1})
                transport = httpx.ASGITransport(app=create_app())
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    body = (await client.post("/api/connect")).json()
                    self.assertTrue(body.get("ok"), body)
                    self.assertFalse(body.get("bridge_loaded"))
                    self.assertIn("插件", body.get("bridge_error") or "")
                    payload = (await client.post("/api/say", json={"text": "左臂向上10厘米"})).json()
                    types = [e.get("type") for e in payload.get("events") or []]
                    self.assertIn("robot_command", types)
                    bridge = next(e for e in payload.get("events") or [] if e.get("type") == "bridge")
                    self.assertEqual(bridge["status"], "no_bridge")
                    await client.post("/api/disconnect")

        asyncio.run(scenario())

    def test_bridge_exception_does_not_drop_the_turn(self):
        class Boom:
            def validate_commands(self, commands):
                return commands

            def execute_commands(self, commands):
                raise RuntimeError("boom")

        async def scenario():
            hub = VoiceHub(asr=NullASR(), tts=DummyTTS(), pipeline=Pipeline(llm=FakeLLM()))
            async with websockets.serve(hub.handle, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                save({"url": f"ws://127.0.0.1:{port}", "robot_id": "tianlian_dual", "schema_version": 1})
                with mock.patch("jetson_client.session.load_bridge", return_value=Boom()):
                    transport = httpx.ASGITransport(app=create_app())
                    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                        self.assertTrue((await client.post("/api/connect")).json().get("ok"))
                        payload = (await client.post("/api/say", json={"text": "左臂向上10厘米"})).json()
                        self.assertTrue(payload.get("ok"), payload)
                        types = [e.get("type") for e in payload.get("events") or []]
                        self.assertIn("tts_text", types)
                        self.assertIn("turn_end", types)
                        bridge = next(e for e in payload.get("events") or [] if e.get("type") == "bridge")
                        self.assertEqual(bridge["status"], "error")
                        await client.post("/api/disconnect")

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

    def test_speakers_status_default_off(self):
        async def scenario():
            transport = httpx.ASGITransport(app=create_app())
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                body = (await client.get("/api/speakers")).json()
                self.assertEqual(body.get("mode"), "off")
                self.assertIn("profiles", body)
                self.assertFalse(body.get("voice_busy"))
                bad = await client.put("/api/speakers", json={"mode": "always"})
                self.assertEqual(bad.status_code, 400)
                ok = (await client.put("/api/speakers", json={"mode": "shadow"})).json()
                self.assertTrue(ok.get("ok"))
                self.assertEqual(ok.get("mode"), "shadow")
                text = Path(self.tmp.name).read_text(encoding="utf-8")
                self.assertIn("VOICE_SPEAKER_VERIFY_MODE=shadow", text)
                save({"url": "ws://10.101.67.111:8765", "robot_id": "tianlian_dual"})
                text = Path(self.tmp.name).read_text(encoding="utf-8")
                self.assertIn("VOICE_SPEAKER_VERIFY_MODE=shadow", text)

        asyncio.run(scenario())

    def test_enroll_rejected_while_voice_ws_live(self):
        from starlette.testclient import TestClient

        class FakeSession:
            connected = True
            hello = {"type": "hello_ok"}
            url = "ws://x"
            robot_id = "tianlian_dual"
            schema_version = 1
            bridge_error = ""

            async def abort_playback(self):
                return None

            async def close(self):
                return None

        with TestClient(create_app(FakeSession())) as client:
            with client.websocket_connect("/ws/voice") as _ws:
                res = client.post("/api/speakers/enrollments", json={"display_name": "测试员"})
                self.assertEqual(res.status_code, 409)
                self.assertIn("停止语音", res.json().get("error") or "")


class DeviceLaunchScriptTests(unittest.TestCase):
    def test_run_device_ui_pins_dry_run_and_skips_host_env(self):
        text = (ROOT / "scripts" / "run_device_ui.sh").read_text(encoding="utf-8")
        self.assertNotIn('source "$ROOT/.env"', text)
        self.assertNotIn("voice_hub.env", text)
        self.assertIn("VOICE_HUB_BRIDGE_PATH", text)
        pin = text.find("export VOICE_ENABLE_ROBOT_COMMAND=0")
        source = text.rfind("source")
        self.assertGreater(pin, -1)
        self.assertGreater(source, -1)
        self.assertLess(pin, source)
        self.assertLess(text.rfind("export VOICE_ENABLE_ROBOT_COMMAND=0"), source)

    def test_run_device_ui_strips_host_keys_at_runtime(self):
        script = (ROOT / "scripts" / "run_device_ui.sh").read_text(encoding="utf-8")
        probe = script.replace(
            'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"',
            f'ROOT="{ROOT}"',
        ).replace(
            'exec "$PYTHON_BIN" -m device_ui',
            "exec python3 -c \"import os; print(os.environ.get('VOICE_ENABLE_ROBOT_COMMAND','')); print(os.environ.get('DOUBAO_TTS_API_KEY','UNSET')); print(os.environ.get('CHAT_LLM_API_KEY','UNSET')); print(os.environ.get('VOICE_HUB_BRIDGE_PATH',''))\"",
        )
        env = os.environ.copy()
        env["VOICE_ENABLE_ROBOT_COMMAND"] = "1"
        env["DOUBAO_TTS_API_KEY"] = "secret"
        env["CHAT_LLM_API_KEY"] = "sk-test"
        env["VOICE_HUB_CLIENT_ENV"] = str(Path(tempfile.gettempdir()) / "voice_hub_missing_client.env")
        result = subprocess.run(
            ["bash", "-c", probe],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
        self.assertGreaterEqual(len(lines), 4, result.stdout)
        self.assertEqual(lines[0], "0")
        self.assertEqual(lines[1], "UNSET")
        self.assertEqual(lines[2], "UNSET")
        self.assertTrue(lines[3].endswith("tianlian_voice_bridge"), lines[3])

    def test_run_client_pins_dry_run_and_skips_host_env(self):
        text = (ROOT / "scripts" / "run_client.sh").read_text(encoding="utf-8")
        self.assertNotIn('source "$ROOT/.env"', text)
        self.assertNotIn("voice_hub.env", text)
        pin = text.find("export VOICE_ENABLE_ROBOT_COMMAND=0")
        source = text.rfind("source")
        self.assertGreater(pin, -1)
        self.assertGreater(source, -1)
        self.assertLess(pin, source)
        self.assertLess(text.rfind("export VOICE_ENABLE_ROBOT_COMMAND=0"), source)

    def test_run_client_strips_host_keys_at_runtime(self):
        script = (ROOT / "scripts" / "run_client.sh").read_text(encoding="utf-8")
        probe = script.replace(
            'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"',
            f'ROOT="{ROOT}"',
        ).replace(
            'exec "$PYTHON_BIN" "$ROOT/jetson_client/voice_client.py" "$@"',
            "exec python3 -c \"import os; print(os.environ.get('VOICE_ENABLE_ROBOT_COMMAND','')); print(os.environ.get('DOUBAO_TTS_API_KEY','UNSET')); print(os.environ.get('CHAT_LLM_API_KEY','UNSET'))\"",
        )
        env = os.environ.copy()
        env["VOICE_ENABLE_ROBOT_COMMAND"] = "1"
        env["DOUBAO_TTS_API_KEY"] = "secret"
        env["CHAT_LLM_API_KEY"] = "sk-test"
        env["VOICE_HUB_CLIENT_ENV"] = str(Path(tempfile.gettempdir()) / "voice_hub_missing_client.env")
        result = subprocess.run(
            ["bash", "-c", probe],
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
        self.assertGreaterEqual(len(lines), 3, result.stdout)
        self.assertEqual(lines[0], "0")
        self.assertEqual(lines[1], "UNSET")
        self.assertEqual(lines[2], "UNSET")

    def test_run_device_ui_honors_client_env_live_flags(self):
        script = (ROOT / "scripts" / "run_device_ui.sh").read_text(encoding="utf-8")
        probe = script.replace(
            'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"',
            f'ROOT="{ROOT}"',
        ).replace(
            'exec "$PYTHON_BIN" -m device_ui',
            "exec python3 -c \"import os; print(os.environ.get('VOICE_ENABLE_ROBOT_COMMAND','')); print(os.environ.get('VOICE_ENABLE_MASSAGE',''))\"",
        )
        tmp = tempfile.NamedTemporaryFile("w", delete=False)
        tmp.write("VOICE_ENABLE_ROBOT_COMMAND=1\nVOICE_ENABLE_MASSAGE=1\n")
        tmp.close()
        env = os.environ.copy()
        env["VOICE_ENABLE_ROBOT_COMMAND"] = "0"
        env["VOICE_HUB_CLIENT_ENV"] = tmp.name
        try:
            result = subprocess.run(
                ["bash", "-c", probe],
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
            self.assertEqual(lines[:2], ["1", "1"])
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def test_sync_device_client_excludes_host_secrets(self):
        text = (ROOT / "scripts" / "sync_device_client.sh").read_text(encoding="utf-8")
        self.assertIn("li@10.101.67.221", text)
        self.assertIn("--exclude '.env'", text)
        self.assertIn("jetson_client/", text)
        self.assertNotIn("server/", text)
        self.assertNotIn("adapters/", text)


if __name__ == "__main__":
    unittest.main()
