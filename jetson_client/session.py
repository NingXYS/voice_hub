"""Reusable WebSocket session to a voice_hub host. No cloud API keys here."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import websockets
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidURI

from jetson_client.plugin import load_bridge

OnEvent = Callable[[dict[str, Any]], None]
OnAudio = Callable[[bytes], None]


class HubConnectError(Exception):
    """Speakable failure connecting to the host."""


@dataclass
class SayResult:
    events: list[dict[str, Any]] = field(default_factory=list)
    pcm: bytes = b""


def speakable_connect_error(exc: BaseException, url: str) -> str:
    raw = f"{type(exc).__name__}: {exc}"
    lowered = raw.lower()
    if "198.18." in (url or ""):
        return "地址是 Clash 假 IP（198.18.x），请改成主机的局域网 IP，例如 ws://10.101.67.111:8765。"
    if "timed out" in lowered or "timeout" in lowered:
        return "连接主机超时。请确认同一局域网，并且主机已经启动语音服务。"
    if "refused" in lowered or "connect call failed" in lowered or "name or service not known" in lowered:
        if "127.0.0.1" in url or "localhost" in url:
            return "连不上本机语音服务。请先在主机上运行 bash scripts/run_server.sh，并确认 8765 已打开。"
        return "连不上主机。请确认主机已启动、地址和端口正确，并且设备与主机在同一局域网。"
    if isinstance(exc, InvalidURI):
        return "主机地址格式不对。请用 ws://<局域网IP>:8765。"
    return f"连不上主机：{exc}"


class HubSession:
    def __init__(self, on_event: OnEvent | None = None):
        self.on_event = on_event
        self.url = ""
        self.robot_id = ""
        self.schema_version = 1
        self.hello: dict[str, Any] | None = None
        self.bridge_error = ""
        self._ws: Any = None
        self._bridge: Any = None
        self._lock = asyncio.Lock()
        self._open = False
        self._watch_task: asyncio.Task | None = None

    @property
    def connected(self) -> bool:
        return self._open and self._ws is not None

    async def connect(self, url: str, robot_id: str, schema_version: int) -> dict[str, Any]:
        url = (url or "").strip()
        robot_id = (robot_id or "").strip()
        if not url:
            raise HubConnectError("还没有填写主机地址。请填 ws://<局域网IP>:8765。")
        if not robot_id:
            raise HubConnectError("还没有选择机械臂（robot_id）。")
        async with self._lock:
            return await self._connect_locked(url, robot_id, schema_version)

    async def _connect_locked(self, url: str, robot_id: str, schema_version: int) -> dict[str, Any]:
        await self._close_unlocked()
        try:
            ws = await websockets.connect(
                url,
                max_size=8 * 1024 * 1024,
                open_timeout=8,
                ping_interval=20,
                ping_timeout=20,
            )
        except Exception as exc:
            raise HubConnectError(speakable_connect_error(exc, url)) from exc
        self._ws = ws
        self._open = True
        self.url = url
        self.robot_id = robot_id
        self.schema_version = int(schema_version)
        try:
            await ws.send(
                json.dumps(
                    {
                        "type": "hello",
                        "robot_id": robot_id,
                        "schema_version": int(schema_version),
                        "audio_params": {"format": "pcm_s16le", "sample_rate": 16000, "channels": 1},
                    },
                    ensure_ascii=False,
                )
            )
            raw = await asyncio.wait_for(ws.recv(), timeout=8)
            hello = _as_json(raw)
        except Exception as exc:
            await self._close_unlocked()
            if isinstance(exc, HubConnectError):
                raise
            raise HubConnectError(speakable_connect_error(exc, url)) from exc
        if not isinstance(hello, dict):
            await self._close_unlocked()
            raise HubConnectError("主机握手返回无法识别。")
        self.hello = hello
        self._emit(hello)
        if hello.get("type") != "hello_ok":
            await self._close_unlocked()
            return hello
        try:
            self._bridge = load_bridge(robot_id)
            self.bridge_error = ""
        except Exception as exc:
            self._bridge = None
            self.bridge_error = str(exc)
        self._watch_task = asyncio.create_task(self._watch_closed(ws))
        return hello

    async def say(
        self,
        text: str,
        *,
        idle_after_speech: float = 20.0,
        on_audio: OnAudio | None = None,
        on_event: OnEvent | None = None,
    ) -> SayResult:
        async with self._lock:
            return await self._say_locked(
                text,
                idle_after_speech=idle_after_speech,
                on_audio=on_audio,
                on_event=on_event,
            )

    async def _say_locked(
        self,
        text: str,
        *,
        idle_after_speech: float,
        on_audio: OnAudio | None,
        on_event: OnEvent | None = None,
    ) -> SayResult:
        if not self.connected:
            raise HubConnectError("还没有连上主机。请先点连接。")
        cleaned = (text or "").strip()
        if not cleaned:
            raise HubConnectError("请输入要说的话。")
        if len(cleaned) > 400:
            raise HubConnectError("这句话太长了，请再短一些。")
        ws = self._ws
        await ws.send(json.dumps({"type": "text", "text": cleaned}, ensure_ascii=False))
        return await self._recv_turn_locked(idle_after_speech=idle_after_speech, on_audio=on_audio, on_event=on_event)

    async def listen_pcm(
        self,
        pcm: bytes,
        *,
        idle_after_speech: float = 8.0,
        on_audio: OnAudio | None = None,
        on_event: OnEvent | None = None,
    ) -> SayResult:
        if not pcm:
            raise HubConnectError("没有收到录音。")

        async def chunks():
            offset = 0
            while offset < len(pcm):
                yield pcm[offset : offset + 6400]
                offset += 6400

        return await self.listen_stream(
            chunks(),
            idle_after_speech=idle_after_speech,
            on_audio=on_audio,
            on_event=on_event,
        )

    async def listen_stream(
        self,
        chunks,
        *,
        idle_after_speech: float = 20.0,
        on_audio: OnAudio | None = None,
        on_event: OnEvent | None = None,
    ) -> SayResult:
        async with self._lock:
            if not self.connected:
                raise HubConnectError("还没有连上主机。请先点连接。")
            ws = self._ws
            await ws.send(json.dumps({"type": "listen", "state": "start"}))

            async def pump() -> None:
                try:
                    async for chunk in _as_aiter(chunks):
                        if chunk:
                            await ws.send(chunk)
                finally:
                    try:
                        await ws.send(json.dumps({"type": "listen", "state": "stop"}))
                    except Exception:
                        pass

            pump_task = asyncio.create_task(pump())
            try:
                return await self._recv_turn_locked(
                    idle_after_speech=idle_after_speech,
                    on_audio=on_audio,
                    on_event=on_event,
                )
            finally:
                if not pump_task.done():
                    pump_task.cancel()
                try:
                    await pump_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass

    async def _recv_turn_locked(
        self,
        *,
        idle_after_speech: float,
        on_audio: OnAudio | None,
        on_event: OnEvent | None = None,
    ) -> SayResult:
        ws = self._ws
        result = SayResult()
        pcm_parts: list[bytes] = []
        got_spoken = False
        while True:
            timeout = 45.0 if not got_spoken else max(8.0, idle_after_speech)
            try:
                message = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                break
            except ConnectionClosed as exc:
                self._open = False
                self._ws = None
                raise HubConnectError("和主机的连接断开了，请重新连接。") from exc
            if isinstance(message, bytes):
                pcm_parts.append(message)
                event = {"type": "audio", "bytes": len(message)}
                result.events.append(event)
                self._emit(event)
                if on_event:
                    on_event(event)
                got_spoken = True
                if on_audio:
                    on_audio(message)
                continue
            payload = _as_json(message)
            if not isinstance(payload, dict):
                continue
            result.events.append(payload)
            self._emit(payload)
            if on_event:
                on_event(payload)
            kind = payload.get("type")
            if kind == "robot_command" and self._bridge is not None:
                status = self._bridge.execute_commands(payload.get("commands") or [])
                bridge_event = {"type": "bridge", "status": status}
                result.events.append(bridge_event)
                self._emit(bridge_event)
            if kind in {"error", "hello_reject", "turn_end"}:
                break
            if kind == "tts_text":
                got_spoken = True
        result.pcm = b"".join(pcm_parts)
        return result

    async def memory_get(self) -> dict[str, Any]:
        return await self._memory_rpc({"type": "memory_get"})

    async def memory_set(self, *, city: str | None = None, name: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "memory_set"}
        if city is not None:
            payload["city"] = city
        if name is not None:
            payload["name"] = name
        return await self._memory_rpc(payload)

    async def memory_clear(self) -> dict[str, Any]:
        return await self._memory_rpc({"type": "memory_clear"})

    async def _memory_rpc(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            if not self.connected:
                raise HubConnectError("还没有连上主机。请先点连接。")
            ws = self._ws
            await ws.send(json.dumps(payload, ensure_ascii=False))
            deadline = asyncio.get_running_loop().time() + 8
            while True:
                timeout = max(0.2, deadline - asyncio.get_running_loop().time())
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=timeout)
                except asyncio.TimeoutError as exc:
                    raise HubConnectError("读取记忆超时。请先结束连续对话再试。") from exc
                if isinstance(message, bytes):
                    continue
                data = _as_json(message)
                if not isinstance(data, dict):
                    continue
                if data.get("type") in {"memory", "error", "hello_reject"}:
                    return data

    async def abort_playback(self) -> None:
        """Stop the host's current TTS. Safe to call while say() is receiving."""
        ws = self._ws
        if not self._open or ws is None:
            raise HubConnectError("还没有连上主机。请先点连接。")
        await ws.send(json.dumps({"type": "abort"}, ensure_ascii=False))

    async def close(self) -> None:
        async with self._lock:
            await self._close_unlocked()

    async def _watch_closed(self, ws: Any) -> None:
        try:
            await ws.wait_closed()
        except Exception:
            pass
        if self._ws is ws:
            self._open = False
            self._ws = None

    async def _close_unlocked(self) -> None:
        task = self._watch_task
        self._watch_task = None
        if task is not None:
            task.cancel()
        ws = self._ws
        self._ws = None
        self._open = False
        self.hello = None
        self._bridge = None
        self.bridge_error = ""
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:
            pass

    def _emit(self, payload: dict[str, Any]) -> None:
        if self.on_event:
            self.on_event(payload)


async def _as_aiter(chunks):
    if hasattr(chunks, "__aiter__"):
        async for item in chunks:
            yield item
        return
    for item in chunks:
        yield item


def _as_json(raw: Any) -> Any:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    return None
