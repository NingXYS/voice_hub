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

STATIC = Path(__file__).resolve().parent / "static"


class DeviceState:
    def __init__(self, session: HubSession | None = None):
        self.session = session or HubSession()


def create_app(session: HubSession | None = None) -> Starlette:
    state = DeviceState(session)

    async def homepage(_: Request):
        return FileResponse(STATIC / "index.html")

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
            }
        )

    async def api_connect(_: Request):
        view = client_settings.public_view()
        try:
            hello = await state.session.connect(
                str(view["url"]),
                str(view["robot_id"]),
                int(view["schema_version"]),
            )
        except HubConnectError as exc:
            return JSONResponse({"ok": False, "error": str(exc), "connected": False})
        if hello.get("type") != "hello_ok":
            reason = str(hello.get("reason") or hello.get("message") or "主机拒绝连接")
            return JSONResponse({"ok": False, "error": reason, "hello": hello, "connected": False})
        return JSONResponse(
            {
                "ok": True,
                "connected": True,
                "hello": hello,
                "bridge_error": state.session.bridge_error,
            }
        )

    async def api_disconnect(_: Request):
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
            while True:
                kind, data = await incoming.get()
                if kind == "eof":
                    return
                if not (kind == "json" and isinstance(data, dict) and data.get("type") == "listen" and data.get("state") == "start"):
                    if kind == "json" and isinstance(data, dict) and data.get("type") == "abort":
                        try:
                            await session.abort_playback()
                        except Exception:
                            pass
                    continue

                async def chunks():
                    while True:
                        item_kind, item_data = await incoming.get()
                        if item_kind == "eof":
                            return
                        if item_kind == "pcm" and item_data:
                            yield item_data
                        elif (
                            item_kind == "json"
                            and isinstance(item_data, dict)
                            and item_data.get("type") == "listen"
                            and item_data.get("state") == "stop"
                        ):
                            return

                def on_event(ev: dict[str, Any]) -> None:
                    if ev.get("type") == "audio":
                        return
                    outgoing.put_nowait(("json", ev))

                def on_audio(pcm: bytes) -> None:
                    if pcm:
                        outgoing.put_nowait(("bin", pcm))

                try:
                    await session.listen_stream(chunks(), on_event=on_event, on_audio=on_audio)
                except HubConnectError as exc:
                    outgoing.put_nowait(("json", {"type": "error", "message": str(exc)}))
                except Exception as exc:
                    print(f"⚠️ device_ui ws turn: {type(exc).__name__}: {exc}", flush=True)
                    outgoing.put_nowait(("json", {"type": "error", "message": "设备端内部错误，请重试。"}))

        reader_task = asyncio.create_task(reader())
        writer_task = asyncio.create_task(writer())
        turns_task = asyncio.create_task(turns())
        try:
            await turns_task
        finally:
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
        try:
            await state.session.abort_playback()
        except HubConnectError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
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

    @asynccontextmanager
    async def lifespan(_: Starlette):
        yield
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
            WebSocketRoute("/ws/voice", ws_voice),
            Mount("/static", StaticFiles(directory=str(STATIC)), name="static"),
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


def _port_open(host: str, port: int) -> bool:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        try:
            return sock.connect_ex((host, port)) == 0
        except OSError:
            return False
