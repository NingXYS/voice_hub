from __future__ import annotations

"""Volcengine / Doubao streaming ASR (SAUC v3 WebSocket).

Docs: https://docs.volcengine.com/docs/6561/2630027
      https://www.volcengine.com/docs/6561/1354869
"""

import asyncio
import gzip
import json
import struct
import uuid
from typing import Any, Awaitable, Callable, Optional

import websockets

from server import config

OnPartial = Callable[[str], Any]

PROTOCOL_VERSION = 0x01
HEADER_SIZE = 0x01
MSG_FULL_CLIENT = 0x01
MSG_AUDIO_ONLY = 0x02
MSG_FULL_SERVER = 0x09
MSG_ERROR = 0x0F
FLAG_NONE = 0x00
FLAG_POS_SEQ = 0x01
FLAG_LAST = 0x02
FLAG_NEG_SEQ = 0x03
SERIAL_NONE = 0x00
SERIAL_JSON = 0x01
COMP_NONE = 0x00
COMP_GZIP = 0x01

CHUNK_BYTES = 6400  # 200ms of 16 kHz S16LE mono


def _header(msg_type: int, flags: int, serial: int, compress: int) -> bytes:
    return bytes(
        [
            (PROTOCOL_VERSION << 4) | HEADER_SIZE,
            (msg_type << 4) | flags,
            (serial << 4) | compress,
            0x00,
        ]
    )


def encode_full_client_request(payload: dict[str, Any]) -> bytes:
    raw = gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    return _header(MSG_FULL_CLIENT, FLAG_NONE, SERIAL_JSON, COMP_GZIP) + struct.pack(">I", len(raw)) + raw


def encode_full_server_response(payload: dict[str, Any], *, last: bool = False, seq: int = 1) -> bytes:
    raw = gzip.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    flags = FLAG_NEG_SEQ if last else FLAG_POS_SEQ
    return (
        _header(MSG_FULL_SERVER, flags, SERIAL_JSON, COMP_GZIP)
        + struct.pack(">i", -seq if last else seq)
        + struct.pack(">I", len(raw))
        + raw
    )


def encode_audio(seq: int, pcm: bytes, *, last: bool) -> bytes:
    flags = FLAG_NEG_SEQ if last else FLAG_POS_SEQ
    seq_out = -abs(seq) if last else seq
    return (
        _header(MSG_AUDIO_ONLY, flags, SERIAL_NONE, COMP_NONE)
        + struct.pack(">i", seq_out)
        + struct.pack(">I", len(pcm))
        + pcm
    )


def parse_response(data: bytes) -> dict[str, Any]:
    if not data or len(data) < 4:
        return {}
    header_size = data[0] & 0x0F
    message_type = data[1] >> 4
    flags = data[1] & 0x0F
    serial = data[2] >> 4
    compress = data[2] & 0x0F
    offset = header_size * 4
    sequence = None
    if flags in (FLAG_POS_SEQ, FLAG_NEG_SEQ) and offset + 4 <= len(data):
        sequence = struct.unpack(">i", data[offset : offset + 4])[0]
        offset += 4
    if message_type == MSG_ERROR:
        code = 0
        if offset + 4 <= len(data):
            code = struct.unpack(">I", data[offset : offset + 4])[0]
            offset += 4
        msg = data[offset:]
        if compress == COMP_GZIP and msg:
            try:
                msg = gzip.decompress(msg)
            except Exception:
                pass
        text = msg.decode("utf-8", "replace") if isinstance(msg, (bytes, bytearray)) else str(msg)
        extra = _json_from_text(text)
        if extra:
            code = extra.get("code") or extra.get("error_code") or code
            text = str(extra.get("error") or extra.get("message") or extra.get("msg") or text)
        return {"error": True, "code": code, "message": text, "sequence": sequence}
    payload: dict[str, Any] = {}
    if offset + 4 <= len(data):
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        offset += 4
        blob = data[offset : offset + size]
        if compress == COMP_GZIP and blob:
            try:
                blob = gzip.decompress(blob)
            except Exception:
                blob = b""
        if serial == SERIAL_JSON and blob:
            try:
                parsed = json.loads(blob.decode("utf-8"))
                if isinstance(parsed, dict):
                    payload = parsed
            except json.JSONDecodeError:
                payload = {}
    if isinstance(payload.get("payload_msg"), dict):
        payload = payload["payload_msg"]
    text = _result_text(payload)
    is_last = bool(payload.get("is_last_package"))
    err = False
    code = payload.get("code")
    if code not in (None, 0, "0", 20000000, "20000000"):
        err = True
    return {
        "error": err,
        "code": code,
        "message": str(payload.get("message") or payload.get("msg") or ""),
        "payload": payload,
        "text": text,
        "is_last": is_last,
        "sequence": sequence,
    }


def _result_text(payload: dict[str, Any]) -> str:
    result = payload.get("result")
    if isinstance(result, dict):
        text = str(result.get("text") or "").strip()
        if text:
            return text
        utterances = result.get("utterances")
        if isinstance(utterances, list):
            definite = [
                str(item.get("text") or "").strip()
                for item in utterances
                if isinstance(item, dict) and item.get("definite")
            ]
            if definite:
                return "".join(definite)
            tails = [str(item.get("text") or "").strip() for item in utterances if isinstance(item, dict)]
            tails = [part for part in tails if part]
            if tails:
                return tails[-1]
    if isinstance(result, list):
        parts = [str(item.get("text") or "").strip() for item in result if isinstance(item, dict)]
        joined = "".join(parts).strip()
        if joined:
            return joined
    return str(payload.get("text") or "").strip()


def _config_payload(sample_rate: int) -> dict[str, Any]:
    return {
        "user": {"uid": "voice_hub"},
        "audio": {
            "format": "pcm",
            "codec": "raw",
            "rate": int(sample_rate or 16000),
            "bits": 16,
            "channel": 1,
            "language": "zh-CN",
        },
        "request": {
            "model_name": "bigmodel",
            "enable_punc": True,
            "enable_itn": True,
            "enable_ddc": False,
            "result_type": "full",
            "show_utterances": False,
        },
    }


class DoubaoASR:
    name = "doubao"

    def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> str:
        if not pcm:
            return ""
        self._require_key()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.transcribe_async(pcm, sample_rate))
        raise RuntimeError("DoubaoASR.transcribe must run in a worker thread")

    async def transcribe_async(
        self,
        pcm: bytes,
        sample_rate: int = 16000,
        on_partial: Optional[OnPartial] = None,
    ) -> str:
        session = await self.open_stream(sample_rate=sample_rate, on_partial=on_partial)
        try:
            await session.feed(pcm)
            return await session.finish()
        finally:
            await session.close()

    async def open_stream(
        self,
        sample_rate: int = 16000,
        on_partial: Optional[OnPartial] = None,
    ) -> "DoubaoAsrSession":
        self._require_key()
        session = DoubaoAsrSession(sample_rate=sample_rate, on_partial=on_partial)
        await session.start()
        return session

    def _require_key(self) -> None:
        if config.doubao_asr_api_key() or (config.doubao_appid() and config.doubao_access_token()):
            return
        raise RuntimeError("语音识别还没配置。请在电脑上设置 DOUBAO_ASR_API_KEY（可与豆包 TTS 共用同一把 Key）。")


class DoubaoAsrSession:
    def __init__(self, sample_rate: int = 16000, on_partial: Optional[OnPartial] = None):
        self.sample_rate = int(sample_rate or 16000)
        self.on_partial = on_partial
        self._ws: Any = None
        self._seq = 2  # full client request is implicit sequence 1
        self._text = ""
        self._odd = b""
        self._ready = asyncio.Event()
        self._done = asyncio.Event()
        self._error = ""
        self._recv_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        headers = _auth_headers()
        url = config.doubao_asr_url()
        try:
            self._ws = await _connect(url, headers)
        except Exception as exc:
            raise RuntimeError(_connect_error(exc)) from exc
        await self._ws.send(encode_full_client_request(_config_payload(self.sample_rate)))
        self._recv_task = asyncio.create_task(self._recv_loop())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=8.0)
        except asyncio.TimeoutError as exc:
            await self.close()
            raise RuntimeError("语音识别服务没有应答，请稍后重试。") from exc
        if self._error and not self._text:
            msg = self._error
            await self.close()
            raise RuntimeError(msg)

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
            offset = 0
            while offset < len(data):
                chunk = data[offset : offset + CHUNK_BYTES]
                offset += len(chunk)
                await self._ws.send(encode_audio(self._seq, chunk, last=False))
                self._seq += 1

    async def finish(self, timeout: float = 12.0) -> str:
        if self._ws is None:
            return self._text
        async with self._lock:
            leftover = self._odd
            self._odd = b""
            if leftover:
                leftover = leftover + b"\x00"
            try:
                await self._ws.send(encode_audio(self._seq, leftover, last=True))
            except Exception:
                pass
        try:
            await asyncio.wait_for(self._done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        if self._error and not self._text:
            raise RuntimeError(self._error)
        return self._text.strip()

    async def close(self) -> None:
        self._done.set()
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
        if ws is None:
            return
        try:
            await ws.close()
        except Exception:
            pass

    async def _recv_loop(self) -> None:
        ws = self._ws
        if ws is None:
            self._done.set()
            return
        try:
            async for message in ws:
                raw = message if isinstance(message, (bytes, bytearray)) else None
                if raw is None:
                    continue
                parsed = parse_response(bytes(raw))
                if not self._ready.is_set():
                    self._ready.set()
                if parsed.get("error"):
                    speakable = _speakable_asr_error(parsed)
                    if speakable:
                        self._error = speakable
                    if not self._text:
                        self._done.set()
                        return
                    continue
                text = str(parsed.get("text") or "").strip()
                if text:
                    self._text = text
                    await _emit(self.on_partial, text)
                if parsed.get("is_last"):
                    self._done.set()
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._error = _connect_error(exc)
        finally:
            self._done.set()


async def _emit(callback: Optional[OnPartial], text: str) -> None:
    if not callback or not text:
        return
    result = callback(text)
    if isinstance(result, Awaitable):
        await result


def _auth_headers() -> dict[str, str]:
    headers = {
        "X-Api-Resource-Id": config.doubao_asr_resource_id(),
        "X-Api-Request-Id": str(uuid.uuid4()),
        "X-Api-Connect-Id": str(uuid.uuid4()),
    }
    api_key = config.doubao_asr_api_key()
    if api_key:
        headers["X-Api-Key"] = api_key
    else:
        headers["X-Api-App-Key"] = config.doubao_appid()
        headers["X-Api-Access-Key"] = config.doubao_access_token()
    return headers


async def _connect(url: str, headers: dict[str, str]):
    kwargs = {"max_size": 8 * 1024 * 1024, "open_timeout": 8, "ping_interval": None}
    try:
        return await websockets.connect(url, additional_headers=headers, **kwargs)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, **kwargs)


def _connect_error(exc: BaseException) -> str:
    raw = f"{type(exc).__name__}: {exc}"
    lowered = raw.lower()
    if "198.18." in raw:
        return "语音识别连不上火山（Clash 假 IP）。请把 openspeech.bytedance.com 设为直连。"
    if "ssl" in lowered or "eof" in lowered or "certificate" in lowered:
        return "语音识别 HTTPS 被掐断。请把 openspeech.bytedance.com 设为直连，或设置 HTTPS_PROXY。"
    if "401" in raw or "403" in raw or "auth" in lowered:
        return "语音识别密钥无效。请检查 DOUBAO_ASR_API_KEY 或与 TTS 共用的豆包 Key。"
    return "语音识别暂时不可用，请再说一次。"


def _json_from_text(text: str) -> dict[str, Any]:
    start = (text or "").find("{")
    if start < 0:
        return {}
    try:
        parsed = json.loads(text[start:])
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _speakable_asr_error(parsed: dict[str, Any]) -> str:
    code = str(parsed.get("code") or "")
    msg = str(parsed.get("message") or "")
    blob = f"{code} {msg}".lower()
    if "sequence" in blob:
        return "语音识别协议错误，请再说一次。"
    if "empty" in blob or code in {"45000002"}:
        return ""
    if code in {"401", "403", "45000001", "45000003"} or "auth" in blob or "unauthor" in blob:
        return "语音识别密钥无效。请检查 DOUBAO_ASR_API_KEY 或与 TTS 共用的豆包 Key。"
    if "resource" in blob:
        return "语音识别资源 ID 不对。请把 DOUBAO_ASR_RESOURCE_ID 改成 volc.seedasr.sauc.duration 或 volc.bigasr.sauc.duration。"
    if code and code not in {"0", "None"}:
        return "语音识别失败，请再说一次。"
    return "语音识别失败，请再说一次。"
