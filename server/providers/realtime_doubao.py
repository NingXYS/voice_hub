from __future__ import annotations

"""Doubao end-to-end realtime dialogue (docs 6561/2549778, v3 binary events).

wss://openspeech.bytedance.com/api/v3/realtime/dialogue
Resource-Id: volc.speech.dialog
"""

import asyncio
import base64
import gzip
import json
import struct
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Awaitable, Callable

OnAsr = Callable[[str], Any]

import websockets

from server import config

PROTOCOL_VERSION = 0x01
HEADER_SIZE = 0x01
MSG_FULL_CLIENT = 0x01
MSG_AUDIO_ONLY = 0x02
MSG_FULL_SERVER = 0x09
MSG_ACK = 0x0B
MSG_ERROR = 0x0F
FLAG_EVENT = 0x04
SERIAL_NONE = 0x00
SERIAL_JSON = 0x01
COMP_GZIP = 0x01

EVENT_START_CONNECTION = 1
EVENT_FINISH_CONNECTION = 2
EVENT_CONNECTION_STARTED = 50
EVENT_CONNECTION_FAILED = 51
EVENT_CONNECTION_FINISHED = 52
EVENT_START_SESSION = 100
EVENT_FINISH_SESSION = 102
EVENT_SESSION_STARTED = 150
EVENT_SESSION_FINISHED = 152
EVENT_SESSION_FAILED = 153
EVENT_USAGE = 154
EVENT_TASK_REQUEST = 200
EVENT_SAY_HELLO = 300
EVENT_TTS_SENTENCE_START = 350
EVENT_TTS_SENTENCE_END = 351
EVENT_TTS_RESPONSE = 352
EVENT_END_ASR = 400
EVENT_ASR_INFO = 450
EVENT_ASR_RESPONSE = 451
EVENT_ASR_ENDED = 459
EVENT_CHAT_TTS_TEXT = 500
EVENT_CHAT_TEXT_QUERY = 501
EVENT_CHAT_RAG_TEXT = 502
EVENT_CHAT_RESPONSE = 550
EVENT_CHAT_ENDED = 559
EVENT_REPLY_DONE = 359
EVENT_DIALOG_COMMON_ERROR = 599
CHAT_TTS_TYPES = frozenset({"chat_tts_text", "external_rag"})

CONNECTION_EVENTS = frozenset(
    {
        EVENT_START_CONNECTION,
        EVENT_FINISH_CONNECTION,
        EVENT_CONNECTION_STARTED,
        EVENT_CONNECTION_FAILED,
        EVENT_CONNECTION_FINISHED,
    }
)
TURN_END_EVENTS = frozenset({EVENT_REPLY_DONE, EVENT_SESSION_FINISHED})


@dataclass
class RealtimeEvent:
    kind: str
    event: int = 0
    text: str = ""
    pcm: bytes = b""
    dialog_id: str = ""
    error: str = ""
    payload: Any = None
    tts_type: str = ""


def _header(msg_type: int, flags: int, serial: int, compress: int) -> bytes:
    return bytes(
        [
            (PROTOCOL_VERSION << 4) | HEADER_SIZE,
            (msg_type << 4) | flags,
            (serial << 4) | compress,
            0x00,
        ]
    )


def encode_event(
    event: int,
    payload: dict[str, Any] | bytes | None = None,
    *,
    session_id: str | None = None,
    audio: bool = False,
) -> bytes:
    if audio:
        raw_payload = payload if isinstance(payload, (bytes, bytearray)) else b""
        packed = gzip.compress(bytes(raw_payload))
        serial = SERIAL_NONE
        msg_type = MSG_AUDIO_ONLY
    else:
        body = payload if isinstance(payload, dict) else {}
        packed = gzip.compress(json.dumps(body, ensure_ascii=False).encode("utf-8"))
        serial = SERIAL_JSON
        msg_type = MSG_FULL_CLIENT
    frame = bytearray(_header(msg_type, FLAG_EVENT, serial, COMP_GZIP))
    frame.extend(struct.pack(">I", int(event)))
    if session_id is not None:
        sid = session_id.encode("utf-8")
        frame.extend(struct.pack(">I", len(sid)))
        frame.extend(sid)
    frame.extend(struct.pack(">I", len(packed)))
    frame.extend(packed)
    return bytes(frame)


def parse_frame(data: bytes) -> dict[str, Any]:
    if not data or len(data) < 4:
        return {"error": True, "message": "empty frame"}
    header_size = data[0] & 0x0F
    message_type = data[1] >> 4
    flags = data[1] & 0x0F
    serial = data[2] >> 4
    compress = data[2] & 0x0F
    offset = header_size * 4
    result: dict[str, Any] = {"message_type": message_type, "event": 0, "session_id": ""}
    if message_type == MSG_ERROR:
        code = 0
        if offset + 4 <= len(data):
            code = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4
        size = 0
        if offset + 4 <= len(data):
            size = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4
        blob = data[offset : offset + size] if size else data[offset:]
        if compress == COMP_GZIP and blob:
            try:
                blob = gzip.decompress(blob)
            except Exception:
                pass
        text = blob.decode("utf-8", "replace") if isinstance(blob, (bytes, bytearray)) else str(blob)
        return {"error": True, "code": code, "message": text, "event": 0}
    if flags & FLAG_EVENT and offset + 4 <= len(data):
        result["event"] = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
    event = int(result["event"] or 0)
    if event not in CONNECTION_EVENTS and offset + 4 <= len(data):
        sid_len = struct.unpack(">I", data[offset : offset + 4])[0]
        if 0 <= sid_len <= 256 and offset + 4 + sid_len <= len(data):
            offset += 4
            result["session_id"] = data[offset : offset + sid_len].decode("utf-8", "replace")
            offset += sid_len
    if offset + 4 <= len(data):
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        blob = data[offset : offset + size]
        if compress == COMP_GZIP and blob:
            try:
                blob = gzip.decompress(blob)
            except Exception:
                pass
        if serial == SERIAL_JSON and blob:
            try:
                result["payload"] = json.loads(blob.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                result["payload"] = blob
        else:
            result["payload"] = blob
    return result


def pcm16_resample(pcm: bytes, src_rate: int, dst_rate: int = 16000) -> bytes:
    if not pcm or src_rate == dst_rate or src_rate <= 0:
        return pcm if len(pcm) % 2 == 0 else pcm[:-1]
    if len(pcm) % 2:
        pcm = pcm[:-1]
    if not pcm:
        return b""
    import array

    samples = array.array("h")
    samples.frombytes(pcm)
    ratio = src_rate / float(dst_rate)
    count = max(1, int(len(samples) / ratio))
    out = array.array("h")
    last = len(samples) - 1
    for index in range(count):
        src = index * ratio
        left = int(src)
        frac = src - left
        a = samples[left] if left <= last else samples[last]
        b = samples[left + 1] if left + 1 <= last else a
        out.append(int(a + (b - a) * frac))
    return out.tobytes()


def _payload_pcm(payload: Any) -> bytes:
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    if not isinstance(payload, dict):
        return b""
    for key in ("audio", "data", "pcm", "speech"):
        value = payload.get(key)
        if isinstance(value, (bytes, bytearray)) and value:
            return bytes(value)
        if isinstance(value, str) and value.strip():
            try:
                return base64.b64decode(value)
            except Exception:
                continue
    return b""


def _payload_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload.strip()
    if not isinstance(payload, dict):
        return ""
    for key in ("content", "text", "question"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    results = payload.get("results") or payload.get("result")
    if isinstance(results, dict):
        return _payload_text(results)
    if isinstance(results, list):
        parts = [_payload_text(item) for item in results]
        joined = "".join(part for part in parts if part)
        if joined:
            return joined
    extra = payload.get("extra")
    if isinstance(extra, dict):
        return _payload_text(extra)
    return ""


def _dialog_id(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("dialog_id") or payload.get("dialogId") or "").strip()
    return ""


def _tts_type(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("tts_type") or "").strip()
    return ""


def format_external_rag(notes: str) -> str:
    """ChatRAGText wants a JSON list of {title, content}, max ~4k chars."""
    items: list[dict[str, str]] = []
    for index, raw in enumerate((notes or "").split("\n")):
        line = raw.strip()
        if not line:
            continue
        if "：" in line:
            title, _, content = line.partition("：")
            title = title.strip() or f"资料{index + 1}"
            content = content.strip()
        else:
            title, content = f"资料{index + 1}", line
        if not content:
            continue
        items.append({"title": title[:80], "content": content[:1500]})
        if len(items) >= 5:
            break
    if not items:
        cleaned = (notes or "").strip()
        if cleaned:
            items = [{"title": "检索", "content": cleaned[:3500]}]
    return json.dumps(items, ensure_ascii=False)


def to_event(parsed: dict[str, Any]) -> RealtimeEvent:
    if parsed.get("error"):
        return RealtimeEvent(
            kind="error",
            error=str(parsed.get("message") or "端到端语音失败"),
            payload=parsed,
        )
    event = int(parsed.get("event") or 0)
    payload = parsed.get("payload")
    text = _payload_text(payload)
    tts_type = _tts_type(payload)
    message_type = int(parsed.get("message_type") or 0)
    if event in {EVENT_ASR_INFO, EVENT_ASR_RESPONSE}:
        return RealtimeEvent(kind="asr", event=event, text=text, payload=payload)
    if event == EVENT_ASR_ENDED:
        return RealtimeEvent(kind="asr_end", event=event, text=text, payload=payload)
    if event in {EVENT_CHAT_RESPONSE, EVENT_TTS_SENTENCE_START}:
        return RealtimeEvent(kind="text", event=event, text=text, payload=payload, tts_type=tts_type)
    ack_pcm = isinstance(payload, (bytes, bytearray)) and bytes(payload)
    if event == EVENT_TTS_RESPONSE or (message_type == MSG_ACK and ack_pcm):
        pcm = _payload_pcm(payload) if event == EVENT_TTS_RESPONSE else bytes(payload)
        if event == EVENT_TTS_RESPONSE and not pcm:
            pcm = bytes(payload) if isinstance(payload, (bytes, bytearray)) else b""
        return RealtimeEvent(kind="audio", event=event, pcm=pcm, payload=payload, tts_type=tts_type)
    if event == EVENT_CHAT_ENDED:
        return RealtimeEvent(kind="chat_end", event=event, text=text, payload=payload)
    if event in TURN_END_EVENTS:
        return RealtimeEvent(kind="ended", event=event, text=text, payload=payload)
    if event == EVENT_SESSION_STARTED:
        return RealtimeEvent(
            kind="session",
            event=event,
            dialog_id=_dialog_id(payload),
            payload=payload,
        )
    if event in {EVENT_SESSION_FAILED, EVENT_DIALOG_COMMON_ERROR}:
        return RealtimeEvent(kind="error", event=event, error=text or "端到端会话失败", payload=payload)
    if event == EVENT_CONNECTION_FAILED:
        return RealtimeEvent(kind="error", event=event, error=text or "端到端连接失败", payload=payload)
    return RealtimeEvent(kind="other", event=event, text=text, payload=payload, tts_type=tts_type)


class DoubaoRealtimeSession:
    name = "doubao_dialog"

    def __init__(self, on_asr: OnAsr | None = None):
        self.session_id = uuid.uuid4().hex
        self.dialog_id = ""
        self.city = ""
        self.sample_rate = 16000
        self.muted = False
        self.on_asr = on_asr
        self._ws: Any = None
        self._queue: asyncio.Queue[RealtimeEvent | None] = asyncio.Queue()
        self._recv_task: asyncio.Task | None = None
        self._asr_text = ""
        self._asr_done = asyncio.Event()
        self._ready = asyncio.Event()
        self._session_ready = asyncio.Event()
        self._error = ""
        self._odd = b""
        self._lock = asyncio.Lock()
        self._hold_default_audio = False

    @property
    def alive(self) -> bool:
        return self._ws is not None and not self._error

    async def start(
        self,
        *,
        city: str = "",
        system_role: str = "",
        speaking_style: str = "",
        dialog_id: str = "",
        bot_name: str = "",
    ) -> None:
        self.city = city or ""
        if not config.dialog_configured():
            raise RuntimeError("端到端实时语音还没配置。请设置 DIALOG_APPID 和 DIALOG_ACCESS_TOKEN，或 DIALOG_API_KEY。")
        headers = _auth_headers()
        url = config.dialog_url()
        try:
            self._ws = await _connect(url, headers)
        except Exception as exc:
            raise RuntimeError(_connect_error(exc)) from exc
        await self._ws.send(encode_event(EVENT_START_CONNECTION, {}))
        self._recv_task = asyncio.create_task(self._recv_loop())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=8.0)
        except asyncio.TimeoutError as exc:
            await self.close()
            raise RuntimeError("端到端语音服务没有应答，请稍后重试。") from exc
        if self._error:
            msg = self._error
            await self.close()
            raise RuntimeError(msg)
        payload = _start_session_payload(
            city=city,
            system_role=system_role,
            speaking_style=speaking_style,
            dialog_id=dialog_id,
            bot_name=bot_name,
        )
        await self._ws.send(encode_event(EVENT_START_SESSION, payload, session_id=self.session_id))
        try:
            await asyncio.wait_for(self._session_ready.wait(), timeout=8.0)
        except asyncio.TimeoutError:
            print("⚠️ dialog session start had no ack; continuing", flush=True)
        if self._error:
            msg = self._error
            await self.close()
            raise RuntimeError(msg)

    def begin_listen(self) -> None:
        self._asr_text = ""
        self._asr_done.clear()
        self.muted = False
        self._hold_default_audio = False

    async def feed(self, pcm: bytes) -> None:
        if not pcm or self._ws is None:
            return
        async with self._lock:
            data = self._odd + pcm
            if len(data) % 2:
                self._odd = data[-1:]
                data = data[:-1]
            else:
                self._odd = b""
            if data:
                await self._ws.send(
                    encode_event(EVENT_TASK_REQUEST, data, session_id=self.session_id, audio=True)
                )

    async def end_utterance(self) -> None:
        leftover = b""
        async with self._lock:
            leftover = self._odd
            self._odd = b""
            if leftover:
                leftover = leftover + b"\x00"
                if self._ws is not None:
                    await self._ws.send(
                        encode_event(EVENT_TASK_REQUEST, leftover, session_id=self.session_id, audio=True)
                    )
        if self._ws is not None and not self._asr_done.is_set():
            await self._ws.send(encode_event(EVENT_END_ASR, {}, session_id=self.session_id))

    async def wait_asr_final(self, timeout: float = 5.0) -> str:
        try:
            await asyncio.wait_for(self._asr_done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return self._asr_text.strip()
        if self._error and not self._asr_text:
            raise RuntimeError(self._error)
        return self._asr_text.strip()

    async def chat_text(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned or self._ws is None:
            return
        self.muted = False
        await self._ws.send(
            encode_event(EVENT_CHAT_TEXT_QUERY, {"content": cleaned}, session_id=self.session_id)
        )

    async def chat_tts_text(self, text: str, *, start: bool = True, end: bool = True) -> None:
        """Custom TTS. Docs require two packets: start=true/end=false then start=false/end=true."""
        if self._ws is None:
            return
        cleaned = (text or "").strip()
        self.muted = False
        self._hold_default_audio = True
        if start and end:
            if cleaned:
                await self._send_chat_tts(start=True, end=False, content=cleaned)
                await asyncio.sleep(0)
            await self._send_chat_tts(start=False, end=True, content="")
            return
        if not cleaned and not end:
            return
        await self._send_chat_tts(start=start, end=end, content=cleaned)

    async def _send_chat_tts(self, *, start: bool, end: bool, content: str) -> None:
        if self._ws is None:
            return
        await self._ws.send(
            encode_event(
                EVENT_CHAT_TTS_TEXT,
                {"start": bool(start), "end": bool(end), "content": content},
                session_id=self.session_id,
            )
        )

    async def say_hello(self, text: str) -> None:
        cleaned = (text or "").strip()
        if not cleaned or self._ws is None:
            return
        self.muted = False
        self._hold_default_audio = True
        await self._ws.send(encode_event(EVENT_SAY_HELLO, {"content": cleaned}, session_id=self.session_id))

    async def prepare_external_speech(self) -> None:
        """ChatTTSText must follow ASREnded. Typed turns send EndASR and wait briefly."""
        if self._ws is None or self._asr_done.is_set():
            return
        await self._ws.send(encode_event(EVENT_END_ASR, {}, session_id=self.session_id))
        try:
            await asyncio.wait_for(self._asr_done.wait(), timeout=0.2)
        except asyncio.TimeoutError:
            self._asr_done.set()

    async def wait_event(self, timeout: float = 2.0) -> RealtimeEvent:
        try:
            item = await asyncio.wait_for(self._queue.get(), timeout=max(0.05, timeout))
        except asyncio.TimeoutError:
            return RealtimeEvent(kind="timeout")
        if item is None:
            return RealtimeEvent(kind="closed")
        return item

    async def chat_rag(self, notes: str) -> None:
        cleaned = (notes or "").strip()
        if not cleaned or self._ws is None:
            return
        self.muted = False
        self._hold_default_audio = True
        payload = cleaned if cleaned[:1] in "[{" else format_external_rag(cleaned)
        await self._ws.send(
            encode_event(EVENT_CHAT_RAG_TEXT, {"external_rag": payload}, session_id=self.session_id)
        )

    def mute(self) -> None:
        self.muted = True
        self.drain()

    def drain(self) -> None:
        sentinel = False
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                sentinel = True
        if sentinel:
            self._queue.put_nowait(None)

    async def events(self) -> AsyncIterator[RealtimeEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item

    async def finish(self) -> None:
        if self._ws is None:
            return
        try:
            await self._ws.send(encode_event(EVENT_FINISH_SESSION, {}, session_id=self.session_id))
            await self._ws.send(encode_event(EVENT_FINISH_CONNECTION, {}))
        except Exception:
            pass

    async def close(self) -> None:
        task = self._recv_task
        self._recv_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        ws = self._ws
        self._ws = None
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:
            pass

    async def _recv_loop(self) -> None:
        ws = self._ws
        if ws is None:
            return
        try:
            async for message in ws:
                raw = message if isinstance(message, (bytes, bytearray)) else None
                if raw is None:
                    continue
                parsed = parse_frame(bytes(raw))
                event = to_event(parsed)
                if not self._ready.is_set() and parsed.get("event") in {
                    EVENT_CONNECTION_STARTED,
                    EVENT_CONNECTION_FAILED,
                    EVENT_SESSION_STARTED,
                    EVENT_SESSION_FAILED,
                }:
                    if parsed.get("event") in {EVENT_CONNECTION_FAILED, EVENT_SESSION_FAILED} or event.kind == "error":
                        self._error = event.error or "端到端连接失败"
                    self._ready.set()
                if event.kind == "session" and event.dialog_id:
                    self.dialog_id = event.dialog_id
                if event.kind in {"asr", "asr_end"} and event.text:
                    self._asr_text = event.text
                    await _emit_asr(self.on_asr, event.text)
                if event.kind == "session":
                    self._session_ready.set()
                if event.kind == "asr_end":
                    self._asr_done.set()
                if event.kind == "error":
                    self._error = event.error or self._error
                    self._asr_done.set()
                    self._ready.set()
                    self._session_ready.set()
                if event.kind == "text" and event.tts_type in CHAT_TTS_TYPES:
                    self._hold_default_audio = False
                if self._hold_default_audio and event.kind in {"audio", "text"}:
                    if event.tts_type and event.tts_type not in CHAT_TTS_TYPES:
                        continue
                    if event.kind == "audio":
                        self._hold_default_audio = False
                if self.muted and event.kind in {"audio", "text"}:
                    continue
                if event.kind == "audio" and event.pcm:
                    rate = config.dialog_output_rate()
                    if rate != 16000:
                        event.pcm = pcm16_resample(event.pcm, rate, 16000)
                await self._queue.put(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._error = _connect_error(exc)
            await self._queue.put(RealtimeEvent(kind="error", error=self._error))
        finally:
            self._ready.set()
            self._asr_done.set()
            try:
                self._queue.put_nowait(None)
            except Exception:
                pass


def _start_session_payload(
    *,
    city: str,
    system_role: str,
    speaking_style: str,
    dialog_id: str,
    bot_name: str,
) -> dict[str, Any]:
    dialog: dict[str, Any] = {
        "bot_name": bot_name or "语音助手",
        "system_role": system_role or "你是机械臂旁的语音助手，用短句中文回答。",
        "speaking_style": speaking_style or "说话简洁，语速适中。",
        "extra": {
            "strict_audit": False,
            "recv_timeout": 120,
            "input_mod": "audio",
        },
    }
    if city:
        dialog["location"] = {"city": city, "country": "中国", "country_code": "CN"}
    if dialog_id:
        dialog["dialog_id"] = dialog_id
    speaker = config.dialog_speaker()
    return {
        "asr": {"extra": {"end_smooth_window_ms": 900}},
        "tts": {
            "speaker": speaker,
            "audio_config": {
                "channel": 1,
                "format": "pcm_s16le",
                "sample_rate": config.dialog_output_rate(),
            },
            "extra": {},
        },
        "dialog": dialog,
    }


def _auth_headers() -> dict[str, str]:
    headers = {
        "X-Api-Resource-Id": config.dialog_resource_id(),
        "X-Api-App-Key": config.dialog_app_key(),
        "X-Api-Connect-Id": str(uuid.uuid4()),
        "X-Api-Request-Id": str(uuid.uuid4()),
    }
    api_key = config.dialog_api_key()
    appid = config.dialog_appid()
    token = config.dialog_access_token()
    if appid and token:
        headers["X-Api-App-ID"] = appid
        headers["X-Api-Access-Key"] = token
    elif api_key:
        headers["X-Api-Key"] = api_key
    return headers


async def _connect(url: str, headers: dict[str, str]):
    kwargs = {"max_size": 8 * 1024 * 1024, "open_timeout": 8, "ping_interval": None}
    try:
        return await websockets.connect(url, additional_headers=headers, **kwargs)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, **kwargs)


async def _emit_asr(callback: OnAsr | None, text: str) -> None:
    if not callback or not text:
        return
    result = callback(text)
    if isinstance(result, Awaitable):
        await result


def _connect_error(exc: BaseException) -> str:
    raw = f"{type(exc).__name__}: {exc}"
    lowered = raw.lower()
    if "198.18." in raw:
        return "端到端语音连不上火山（Clash 假 IP）。请把 openspeech.bytedance.com 设为直连。"
    if "ssl" in lowered or "eof" in lowered or "certificate" in lowered:
        return "端到端语音 HTTPS 被掐断。请把 openspeech.bytedance.com 设为直连。"
    if "403" in raw:
        return "端到端语音返回 403。请确认火山控制台已开通「豆包端到端实时语音」，并把 openspeech.bytedance.com 设为直连。"
    if "401" in raw or "auth" in lowered:
        return "端到端语音密钥无效。请检查 DOUBAO_TTS_API_KEY，或填写 DIALOG_APPID 和 DIALOG_ACCESS_TOKEN。"
    return "端到端语音暂时不可用，已改用原来的对话链路。"
