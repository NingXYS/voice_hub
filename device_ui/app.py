"""Local-only HTTP UI for connecting a Linux device to a voice_hub host."""
from __future__ import annotations

import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket, WebSocketDisconnect

from device_ui import client_settings
from device_ui.wavutil import pcm_s16le_to_wav
from jetson_client.session import HubConnectError, HubSession
from jetson_client.wake_kws import KwsCooldown, describe_kws, wake_word as kws_wake_word
from jetson_client.wake_loop import begin_command_stream, listen_with_barge, requeue_eof

_WAKE_UNSET = object()

_NO_STORE = {"Cache-Control": "no-store"}


class _NoStoreStatic(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response

STATIC = Path(__file__).resolve().parent / "static"


class DeviceState:
    def __init__(self, session: HubSession | None = None, wake_detector=None):
        self.session = session or HubSession()
        self.wake_detector = wake_detector
        self.wake_load_error = ""
        self._wake_loaded = wake_detector is not None
        self.voice_ws = 0

    def ensure_wake(self):
        if self._wake_loaded:
            return self.wake_detector
        if os.environ.get("VOICE_KWS_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}:
            self._wake_loaded = True
            self.wake_load_error = "唤醒未启用"
            return None
        self._wake_loaded = True
        try:
            from jetson_client.wake_kws import KeywordSpotter

            self.wake_detector = KeywordSpotter()
            print(
                f"wake kws_ready=true word={kws_wake_word()} model={getattr(self.wake_detector, 'model_path', '')}",
                flush=True,
            )
        except Exception as exc:
            self.wake_detector = None
            self.wake_load_error = str(exc)
            print(f"wake kws_ready=false error={exc}", flush=True)
        return self.wake_detector


def create_app(session: HubSession | None = None, wake_detector=_WAKE_UNSET) -> Starlette:
    _pin_device_safety()
    injected = None if wake_detector is _WAKE_UNSET else wake_detector
    state = DeviceState(session, wake_detector=injected if injected is not _WAKE_UNSET else None)
    if wake_detector is not _WAKE_UNSET:
        state._wake_loaded = True
        if injected is None:
            state.wake_load_error = "唤醒未启用"

    async def homepage(_: Request):
        return FileResponse(STATIC / "index.html", headers=_NO_STORE)

    async def api_health(_: Request):
        return JSONResponse({"ok": True, "name": "voice-hub-device"})

    async def api_settings(request: Request):
        if request.method == "GET":
            return JSONResponse(client_settings.public_view())
        try:
            body = await request.json()
            view = client_settings.save(body if isinstance(body, dict) else {})
        except client_settings.SettingsError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, **view})

    async def api_status(_: Request):
        session = state.session
        hello = session.hello if session.connected else None
        view = client_settings.public_view()
        return JSONResponse(
            {
                "connected": session.connected,
                "url": session.url or view["url"],
                "robot_id": session.robot_id or view["robot_id"],
                "schema_version": session.schema_version or view["schema_version"],
                "hello": hello,
                "bridge_error": session.bridge_error if session.connected else "",
                "bridge_loaded": bool(getattr(session, "bridge_loaded", False)) if session.connected else False,
                "last_command": getattr(session, "last_command", None) if session.connected else None,
                "last_bridge_status": getattr(session, "last_bridge_status", "") if session.connected else "",
                "robot_publish_enabled": _robot_publish_enabled(),
                "massage_enabled": _massage_enabled(),
                "speaker_mode": _speaker_mode(),
                "voice_busy": state.voice_ws > 0,
                **_wake_view(state),
            }
        )

    def _connected_payload(hello: dict[str, Any]) -> dict[str, Any]:
        session = state.session
        return {
            "ok": True,
            "connected": True,
            "hello": hello,
            "bridge_error": session.bridge_error,
            "bridge_loaded": bool(getattr(session, "bridge_loaded", False)),
            "last_command": getattr(session, "last_command", None),
            "last_bridge_status": getattr(session, "last_bridge_status", "") or "",
            "robot_publish_enabled": _robot_publish_enabled(),
            "massage_enabled": _massage_enabled(),
        }

    async def api_connect(_: Request):
        view = client_settings.public_view()
        url = str(view["url"]).strip()
        robot_id = str(view["robot_id"]).strip()
        schema_version = int(view["schema_version"])
        session = state.session
        if state.voice_ws > 0:
            same = (
                session.connected
                and session.url == url
                and session.robot_id == robot_id
                and int(session.schema_version or 0) == schema_version
            )
            if not same:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "语音正在进行，请先停止再更换主机。",
                        "connected": session.connected,
                    },
                    status_code=409,
                )
            hello = session.hello if isinstance(session.hello, dict) else {"type": "hello_ok"}
            return JSONResponse(_connected_payload(hello))
        try:
            hello = await session.connect(url, robot_id, schema_version)
        except HubConnectError as exc:
            return JSONResponse({"ok": False, "error": str(exc), "connected": False})
        if hello.get("type") != "hello_ok":
            reason = str(hello.get("reason") or hello.get("message") or "主机拒绝连接")
            return JSONResponse({"ok": False, "error": reason, "hello": hello, "connected": False})
        return JSONResponse(_connected_payload(hello))

    async def api_disconnect(_: Request):
        from jetson_client.massage import close_runtime

        close_runtime()
        await state.session.close()
        return JSONResponse({"ok": True, "connected": False})

    async def api_say(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        text = str((body or {}).get("text") or "").strip()
        if not text:
            return JSONResponse({"ok": False, "error": "请输入要说的话。"}, status_code=400)
        if len(text) > 400:
            return JSONResponse({"ok": False, "error": "这句话太长了，请再短一些。"}, status_code=400)
        try:
            result = await state.session.say(text)
        except HubConnectError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        wav = pcm_s16le_to_wav(result.pcm)
        payload: dict[str, Any] = {
            "ok": True,
            "events": result.events,
            "has_audio": bool(wav),
        }
        if wav:
            payload["wav_base64"] = base64.b64encode(wav).decode("ascii")
        return JSONResponse(payload)

    async def api_say_stream(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        text = str((body or {}).get("text") or "").strip()
        if not text:
            return JSONResponse({"ok": False, "error": "请输入要说的话。"}, status_code=400)
        if len(text) > 400:
            return JSONResponse({"ok": False, "error": "这句话太长了，请再短一些。"}, status_code=400)
        if not state.session.connected:
            return JSONResponse({"ok": False, "error": "还没有连上主机。请先点连接。"}, status_code=400)

        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        def on_event(ev: dict[str, Any]) -> None:
            if ev.get("type") == "audio":
                return
            queue.put_nowait({"event": str(ev.get("type") or "message"), "data": ev})

        def on_audio(pcm: bytes) -> None:
            queue.put_nowait(
                {
                    "event": "audio",
                    "data": {
                        "b64": base64.b64encode(pcm).decode("ascii"),
                        "bytes": len(pcm),
                    },
                }
            )

        async def run() -> None:
            try:
                await state.session.say(text, on_audio=on_audio, on_event=on_event)
            except HubConnectError as exc:
                queue.put_nowait({"event": "error", "data": {"type": "error", "message": str(exc)}})
            except Exception as exc:
                print(f"⚠️ device_ui stream: {type(exc).__name__}: {exc}", flush=True)
                queue.put_nowait(
                    {"event": "error", "data": {"type": "error", "message": "设备端内部错误，请重试。"}}
                )
            finally:
                queue.put_nowait(None)

        async def gen():
            task = asyncio.create_task(run())
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    payload = json.dumps(item["data"], ensure_ascii=False)
                    yield f"event: {item['event']}\ndata: {payload}\n\n"
            except asyncio.CancelledError:
                try:
                    await state.session.abort_playback()
                except Exception:
                    pass
                raise
            finally:
                await task

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def ws_voice(websocket: WebSocket):
        await websocket.accept()
        session = state.session
        if not session.connected:
            await websocket.send_json({"type": "error", "message": "还没有连上主机。请先点连接。"})
            await websocket.close()
            return
        state.voice_ws += 1
        incoming: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        outgoing: asyncio.Queue[tuple[str, Any] | None] = asyncio.Queue()

        async def reader() -> None:
            try:
                while True:
                    message = await websocket.receive()
                    if message.get("type") == "websocket.disconnect":
                        break
                    data = message.get("bytes")
                    if data is not None:
                        await incoming.put(("pcm", data))
                        continue
                    text = message.get("text")
                    if text:
                        try:
                            payload = json.loads(text)
                        except json.JSONDecodeError:
                            continue
                        await incoming.put(("json", payload))
            except WebSocketDisconnect:
                pass
            except Exception as exc:
                print(f"⚠️ device_ui ws reader: {type(exc).__name__}: {exc}", flush=True)
            finally:
                await incoming.put(("eof", None))

        async def writer() -> None:
            try:
                while True:
                    item = await outgoing.get()
                    if item is None:
                        break
                    kind, data = item
                    if kind == "json":
                        await websocket.send_json(data)
                    else:
                        await websocket.send_bytes(data)
            except (WebSocketDisconnect, RuntimeError):
                pass

        async def turns() -> None:
            detector = state.ensure_wake()
            cooldown = KwsCooldown()
            outgoing.put_nowait(
                (
                    "json",
                    {
                        "type": "wake",
                        "status": "waiting" if detector else "error",
                        "word": kws_wake_word(),
                        "message": "" if detector else (state.wake_load_error or "唤醒未就绪"),
                    },
                )
            )

            def on_event(ev: dict[str, Any]) -> None:
                if ev.get("type") == "audio":
                    return
                outgoing.put_nowait(("json", ev))

            tts_bytes = 0

            def on_audio(pcm: bytes) -> None:
                nonlocal tts_bytes
                if pcm:
                    tts_bytes += len(pcm)
                    outgoing.put_nowait(("bin", pcm))

            def emit_rejected(decision: dict[str, Any]) -> None:
                outgoing.put_nowait(
                    (
                        "json",
                        {
                            "type": "wake",
                            "status": "rejected",
                            "word": kws_wake_word(),
                            "message": "未通过声纹，请本人再说",
                            "reason": decision.get("reason"),
                            "score": decision.get("score"),
                            "display_name": decision.get("display_name"),
                            "mode": decision.get("mode"),
                        },
                    )
                )

            async def host_listen(chunks):
                nonlocal tts_bytes
                tts_bytes = 0
                return await listen_with_barge(
                    session,
                    chunks,
                    incoming,
                    detector,
                    cooldown,
                    on_event=on_event,
                    on_audio=on_audio,
                )

            async def command_loop(*, skip_sec: float = 0.25) -> bool:
                did_turn = False
                while True:
                    stream = await begin_command_stream(incoming, skip_sec=skip_sec)
                    if stream is None:
                        print("wake=idle reason=no_speech", flush=True)
                        outgoing.put_nowait(
                            ("json", {"type": "wake", "status": "idle", "reason": "no_speech"})
                        )
                        return did_turn
                    streamed = 0
                    print("wake=stream", flush=True)
                    outgoing.put_nowait(
                        ("json", {"type": "wake", "status": "listen", "word": kws_wake_word()})
                    )

                    async def live_chunks():
                        nonlocal streamed
                        async for part in stream:
                            streamed += len(part)
                            yield part

                    (result, decision), barged = await host_listen(live_chunks())
                    if barged:
                        print("wake=barge word=" + kws_wake_word(), flush=True)
                        outgoing.put_nowait(
                            ("json", {"type": "wake", "status": "hit", "word": kws_wake_word()})
                        )
                        if detector is not None:
                            detector.reset()
                        skip_sec = 0.25
                        continue
                    if result is None:
                        print(
                            f"wake=rejected bytes={streamed} reason={decision.get('reason')}",
                            flush=True,
                        )
                        emit_rejected(decision)
                        return did_turn
                    did_turn = True
                    print(f"wake=listen bytes={streamed} streamed=1", flush=True)
                    return True

            async def run_listen_from_queue():
                async def chunks():
                    while True:
                        item_kind, item_data = await incoming.get()
                        if item_kind == "eof":
                            requeue_eof(incoming, item_kind)
                            return
                        if item_kind == "pcm" and item_data:
                            yield item_data
                        elif (
                            item_kind == "json"
                            and isinstance(item_data, dict)
                            and item_data.get("type") in {"abort", "listen"}
                            and (item_data.get("type") == "abort" or item_data.get("state") == "stop")
                        ):
                            return

                (result, decision), barged = await host_listen(chunks())
                if barged:
                    print("wake=barge word=" + kws_wake_word(), flush=True)
                    outgoing.put_nowait(
                        ("json", {"type": "wake", "status": "hit", "word": kws_wake_word()})
                    )
                    if detector is not None:
                        detector.reset()
                    await command_loop(skip_sec=0.25)
                    return
                if result is None:
                    emit_rejected(decision)

            while True:
                kind, data = await incoming.get()
                if kind == "eof":
                    return
                if kind == "json" and isinstance(data, dict) and data.get("type") == "abort":
                    try:
                        await session.abort_playback()
                    except Exception:
                        pass
                    continue
                if kind == "json" and isinstance(data, dict) and data.get("type") == "listen" and data.get("state") == "start":
                    try:
                        await run_listen_from_queue()
                    except HubConnectError as exc:
                        outgoing.put_nowait(("json", {"type": "error", "message": str(exc)}))
                    except Exception as exc:
                        print(f"⚠️ device_ui ws turn: {type(exc).__name__}: {exc}", flush=True)
                        outgoing.put_nowait(("json", {"type": "error", "message": "设备端内部错误，请重试。"}))
                    continue
                if kind != "pcm" or not data or detector is None:
                    continue
                if not cooldown.allow(len(data)):
                    continue
                try:
                    hit = detector.accept_pcm(data)
                except Exception as exc:
                    print(f"⚠️ wake kws: {type(exc).__name__}: {exc}", flush=True)
                    continue
                if not hit:
                    continue
                print(f"wake=hit word={kws_wake_word()}", flush=True)
                outgoing.put_nowait(("json", {"type": "wake", "status": "hit", "word": kws_wake_word()}))
                did_turn = False
                try:
                    did_turn = await command_loop()
                except HubConnectError as exc:
                    outgoing.put_nowait(("json", {"type": "error", "message": str(exc)}))
                except Exception as exc:
                    print(f"⚠️ device_ui wake turn: {type(exc).__name__}: {exc}", flush=True)
                    outgoing.put_nowait(("json", {"type": "error", "message": "设备端内部错误，请重试。"}))
                finally:
                    if detector is not None:
                        detector.reset()
                    if did_turn:
                        cooldown.arm(extra_sec=tts_bytes / 32000.0)
                    outgoing.put_nowait(("json", {"type": "wake", "status": "waiting", "word": kws_wake_word()}))

        reader_task = asyncio.create_task(reader())
        writer_task = asyncio.create_task(writer())
        turns_task = asyncio.create_task(turns())
        try:
            await turns_task
        except Exception as exc:
            print(f"⚠️ device_ui voice loop: {type(exc).__name__}: {exc}", flush=True)
            try:
                await websocket.send_json({"type": "error", "message": "语音通道出错，请重新启动语音。"})
            except Exception:
                pass
        finally:
            from jetson_client.massage import close_runtime

            close_runtime()
            state.voice_ws = max(0, state.voice_ws - 1)
            reader_task.cancel()
            outgoing.put_nowait(None)
            for task in (reader_task, writer_task):
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

    async def api_abort(_: Request):
        from jetson_client.massage import close_runtime

        close_runtime()
        try:
            await state.session.abort_playback()
        except HubConnectError:
            pass
        except Exception as exc:
            print(f"⚠️ device_ui abort: {type(exc).__name__}: {exc}", flush=True)
        return JSONResponse({"ok": True})

    async def api_memory(request: Request):
        if not state.session.connected:
            return JSONResponse({"ok": False, "error": "还没有连上主机。请先点连接。"}, status_code=400)
        try:
            if request.method == "GET":
                data = await state.session.memory_get()
            elif request.method == "DELETE":
                data = await state.session.memory_clear()
            else:
                body = await request.json()
                data = await state.session.memory_set(
                    city=None if not isinstance(body, dict) else body.get("city"),
                    name=None if not isinstance(body, dict) else body.get("name"),
                )
        except HubConnectError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        except Exception:
            return JSONResponse({"ok": False, "error": "读取记忆失败，请重试。"}, status_code=500)
        if data.get("type") == "error":
            return JSONResponse({"ok": False, "error": data.get("message") or "记忆操作失败"}, status_code=400)
        return JSONResponse({"ok": True, **{k: data.get(k) for k in ("city", "name", "turns", "path")}})

    async def api_massage_capture(_: Request):
        from jetson_client.massage import teaching_capture

        result = teaching_capture()
        code = 200 if result.get("ok") else 400
        return JSONResponse(result, status_code=code)

    async def api_massage_finish(_: Request):
        from jetson_client.massage import teaching_finish

        result = teaching_finish()
        code = 200 if result.get("ok") else 400
        return JSONResponse(result, status_code=code)

    def _speaker_payload() -> dict[str, Any]:
        from jetson_client import speaker_gate

        data = speaker_gate.status()
        data["voice_busy"] = state.voice_ws > 0
        return data

    def _require_voice_stopped() -> JSONResponse | None:
        if state.voice_ws > 0:
            return JSONResponse(
                {"ok": False, "error": "请先停止语音，再用机器人麦克风登记或试听。"},
                status_code=409,
            )
        return None

    async def api_speakers(request: Request):
        from jetson_client import speaker_gate

        if request.method == "GET":
            return JSONResponse(_speaker_payload())
        try:
            body = await request.json()
        except Exception:
            body = {}
        mode = str((body or {}).get("mode") or "").strip().lower()
        try:
            client_settings.set_speaker_mode(mode)
        except client_settings.SettingsError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        speaker_gate.reset_verifier()
        payload = _speaker_payload()
        payload["ok"] = True
        return JSONResponse(payload)

    async def api_speaker_enroll_start(request: Request):
        from jetson_client import speaker_gate

        blocked = _require_voice_stopped()
        if blocked is not None:
            return blocked
        try:
            body = await request.json()
        except Exception:
            body = {}
        name = str((body or {}).get("display_name") or (body or {}).get("name") or "").strip()
        try:
            enrollment = speaker_gate.start_enrollment(name)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, **enrollment})

    async def api_speaker_enroll_sample(request: Request):
        from jetson_client import speaker_gate

        blocked = _require_voice_stopped()
        if blocked is not None:
            return blocked
        session_id = request.path_params.get("session_id") or ""
        try:
            pcm, _audio_ms = await asyncio.to_thread(speaker_gate.capture_locked)
            result = speaker_gate.add_enrollment_pcm(session_id, pcm)
        except KeyError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse({"ok": True, **result})

    async def api_speaker_enroll_commit(request: Request):
        from jetson_client import speaker_gate

        blocked = _require_voice_stopped()
        if blocked is not None:
            return blocked
        session_id = request.path_params.get("session_id") or ""
        try:
            result = speaker_gate.commit_enrollment(session_id)
        except KeyError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        return JSONResponse(result)

    async def api_speaker_enroll_cancel(request: Request):
        from jetson_client import speaker_gate

        session_id = request.path_params.get("session_id") or ""
        try:
            speaker_gate.cancel_enrollment(session_id)
        except KeyError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        return JSONResponse({"ok": True, "enrollment": None})

    async def api_speaker_delete(request: Request):
        from jetson_client import speaker_gate

        blocked = _require_voice_stopped()
        if blocked is not None:
            return blocked
        speaker_id = request.path_params.get("speaker_id") or ""
        try:
            ok = speaker_gate.delete_profile(speaker_id)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        if not ok:
            return JSONResponse({"ok": False, "error": "没有这个声纹档案。"}, status_code=404)
        return JSONResponse({"ok": True, "speaker_id": speaker_id})

    async def api_speaker_probe(_: Request):
        from jetson_client import speaker_gate

        blocked = _require_voice_stopped()
        if blocked is not None:
            return blocked
        try:
            pcm, audio_ms = await asyncio.to_thread(speaker_gate.capture_locked)
            decision = speaker_gate.score_pcm(pcm)
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
        decision["ok"] = True
        decision["audio_ms"] = audio_ms
        return JSONResponse(decision)

    @asynccontextmanager
    async def lifespan(_: Starlette):
        yield
        from jetson_client.massage import close_runtime

        close_runtime()
        await state.session.close()

    app = Starlette(
        routes=[
            Route("/", homepage),
            Route("/api/health", api_health, methods=["GET"]),
            Route("/api/settings", api_settings, methods=["GET", "PUT"]),
            Route("/api/status", api_status, methods=["GET"]),
            Route("/api/connect", api_connect, methods=["POST"]),
            Route("/api/disconnect", api_disconnect, methods=["POST"]),
            Route("/api/say", api_say, methods=["POST"]),
            Route("/api/say/stream", api_say_stream, methods=["POST"]),
            Route("/api/abort", api_abort, methods=["POST"]),
            Route("/api/memory", api_memory, methods=["GET", "PUT", "DELETE"]),
            Route("/api/massage/teaching/capture", api_massage_capture, methods=["POST"]),
            Route("/api/massage/teaching/finish", api_massage_finish, methods=["POST"]),
            Route("/api/speakers", api_speakers, methods=["GET", "PUT"]),
            Route("/api/speakers/enrollments", api_speaker_enroll_start, methods=["POST"]),
            Route(
                "/api/speakers/enrollments/{session_id}/samples",
                api_speaker_enroll_sample,
                methods=["POST"],
            ),
            Route(
                "/api/speakers/enrollments/{session_id}/commit",
                api_speaker_enroll_commit,
                methods=["POST"],
            ),
            Route(
                "/api/speakers/enrollments/{session_id}",
                api_speaker_enroll_cancel,
                methods=["DELETE"],
            ),
            Route("/api/speakers/probes", api_speaker_probe, methods=["POST"]),
            Route("/api/speakers/{speaker_id}", api_speaker_delete, methods=["DELETE"]),
            WebSocketRoute("/ws/voice", ws_voice),
            Mount("/static", _NoStoreStatic(directory=str(STATIC)), name="static"),
        ],
        lifespan=lifespan,
    )
    app.state.device = state

    async def unhandled(request: Request, exc: Exception):
        print(f"⚠️ device_ui: {type(exc).__name__}: {exc}", flush=True)
        if str(request.url.path).startswith("/api/"):
            return JSONResponse({"ok": False, "error": "设备端内部错误，请重试。"}, status_code=500)
        raise exc

    app.add_exception_handler(Exception, unhandled)
    return app


def main() -> None:
    import webbrowser

    import uvicorn

    _pin_device_safety()

    host = os.environ.get("DEVICE_UI_HOST", "127.0.0.1").strip() or "127.0.0.1"
    try:
        port = int(os.environ.get("DEVICE_UI_PORT", "8780") or "8780")
    except ValueError:
        port = 8780
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print("DEVICE_UI_HOST 只能是 127.0.0.1（设备端页面含连接控制，不要绑局域网）", flush=True)
        host = "127.0.0.1"

    if _port_open(host, port):
        url = f"http://127.0.0.1:{port}/"
        print(f"设备端已在运行，打开 {url}", flush=True)
        if os.environ.get("DEVICE_UI_OPEN_BROWSER", "1").strip().lower() not in {"0", "false", "no"}:
            webbrowser.open(url)
        return

    url = f"http://127.0.0.1:{port}/"
    print(f"Voice Hub 设备端 {url}", flush=True)
    if os.environ.get("DEVICE_UI_OPEN_BROWSER", "1").strip().lower() not in {"0", "false", "no"}:
        webbrowser.open(url)
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


def _wake_view(state: DeviceState) -> dict[str, Any]:
    info = describe_kws()
    if state._wake_loaded:
        info["kws_ready"] = bool(state.wake_detector)
        if state.wake_load_error:
            info["kws_error"] = state.wake_load_error
    return {
        "wake_word": info.get("wake_word") or kws_wake_word(),
        "kws_model": info.get("kws_model") or "",
        "kws_ready": bool(info.get("kws_ready")),
        "kws_error": str(info.get("kws_error") or ""),
    }


def _pin_device_safety() -> None:
    """Cloud keys stay off. Live/demo switches come from client.env, default 0."""
    client_settings.apply_device_flags()


def _flag_on(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _robot_publish_enabled() -> bool:
    return _flag_on("VOICE_ENABLE_ROBOT_COMMAND")


def _massage_enabled() -> bool:
    return _flag_on("VOICE_ENABLE_MASSAGE")


def _speaker_mode() -> str:
    from jetson_client.speaker_gate import verify_mode

    return verify_mode()


def _port_open(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        try:
            return sock.connect_ex((host, port)) == 0
        except OSError:
            return False
