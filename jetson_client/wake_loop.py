"""Wait for 小溪小溪, then one host turn, then wait again."""
from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncIterator
from typing import Any, Callable

from jetson_client.session import HubConnectError, HubSession
from jetson_client.speaker_gate import run_gated_listen
from jetson_client.tone import play_prompt_tone
from jetson_client.wake_kws import KwsCooldown, KeywordSpotter, describe_kws, pcm_rms, wake_word

OnEvent = Callable[[dict[str, Any]], None]
ENDPOINT_SILENCE_SEC = 0.55


def _is_listen_start(kind: str, data: Any) -> bool:
    return kind == "json" and isinstance(data, dict) and data.get("type") == "listen" and data.get("state") == "start"


def _is_stream_end(kind: str, data: Any) -> bool:
    if kind == "eof":
        return True
    if kind != "json" or not isinstance(data, dict):
        return False
    if data.get("type") == "abort":
        return True
    return data.get("type") == "listen" and data.get("state") == "stop"


def requeue_eof(incoming: asyncio.Queue, kind: str) -> None:
    """Keep a single eof readable by the outer wait loop after a command stream."""
    if kind != "eof":
        return
    try:
        incoming.put_nowait(("eof", None))
    except Exception:
        pass


def _trim_skip(data: bytes, skipped: int, skip_bytes: int) -> tuple[bytes, int]:
    if skipped >= skip_bytes or not data:
        return data, skipped
    take = min(len(data), skip_bytes - skipped)
    return data[take:], skipped + take


def _pending_stream(pending: list[bytes]) -> AsyncIterator[bytes]:
    async def live() -> AsyncIterator[bytes]:
        for part in pending:
            if part:
                yield part

    return live()


async def begin_command_stream(
    incoming: asyncio.Queue,
    *,
    start_timeout: float = 5.0,
    max_duration: float = 12.0,
    silence_sec: float = ENDPOINT_SILENCE_SEC,
    speech_rms: float = 0.012,
    skip_sec: float = 0.25,
    preroll_sec: float = 0.3,
) -> AsyncIterator[bytes] | None:
    """Wait for first speech, then yield PCM live until silence/stop. None = idle."""
    started = time.monotonic()
    skip_bytes = max(0, int(skip_sec * 16000 * 2))
    preroll_bytes = max(0, int(preroll_sec * 16000 * 2))
    skipped = 0
    pending: list[bytes] = []
    pending_n = 0
    voiced = False

    while time.monotonic() - started < max_duration:
        try:
            kind, data = await asyncio.wait_for(incoming.get(), timeout=0.2)
        except asyncio.TimeoutError:
            if time.monotonic() - started >= start_timeout:
                return None
            continue
        if _is_stream_end(kind, data):
            requeue_eof(incoming, kind)
            return _pending_stream(pending) if pending else None
        if _is_listen_start(kind, data):
            continue
        if kind != "pcm" or not data:
            continue
        data, skipped = _trim_skip(data, skipped, skip_bytes)
        if not data:
            continue
        pending.append(data)
        pending_n += len(data)
        while preroll_bytes and pending_n > preroll_bytes and len(pending) > 1:
            dropped = pending.pop(0)
            pending_n -= len(dropped)
        if pcm_rms(data) >= speech_rms:
            voiced = True
            break
        if time.monotonic() - started >= start_timeout:
            return None
    if not voiced:
        return None

    async def live() -> AsyncIterator[bytes]:
        for part in pending:
            yield part
        silence = 0.0
        while time.monotonic() - started < max_duration:
            try:
                kind, data = await asyncio.wait_for(incoming.get(), timeout=0.2)
            except asyncio.TimeoutError:
                if silence >= silence_sec:
                    return
                continue
            if _is_stream_end(kind, data):
                requeue_eof(incoming, kind)
                return
            if _is_listen_start(kind, data) or kind != "pcm" or not data:
                continue
            yield data
            energy = pcm_rms(data)
            dt = (len(data) / 2) / 16000.0
            if energy >= speech_rms:
                silence = 0.0
            else:
                silence += dt
            if silence >= silence_sec:
                return

    return live()


async def collect_command_from_queue(
    incoming: asyncio.Queue,
    *,
    start_timeout: float = 5.0,
    max_duration: float = 12.0,
    silence_sec: float = ENDPOINT_SILENCE_SEC,
    speech_rms: float = 0.012,
    skip_sec: float = 0.25,
) -> bytes:
    """Read until silence, listen-stop, or timeout. Empty = no speech."""
    stream = await begin_command_stream(
        incoming,
        start_timeout=start_timeout,
        max_duration=max_duration,
        silence_sec=silence_sec,
        speech_rms=speech_rms,
        skip_sec=skip_sec,
    )
    if stream is None:
        return b""
    parts: list[bytes] = []
    async for chunk in stream:
        parts.append(chunk)
    return b"".join(parts)


def _drop_queued_pcm(incoming: asyncio.Queue) -> None:
    """Throw away mic PCM captured before TTS actually started."""
    kept: list[tuple[str, Any]] = []
    while True:
        try:
            item = incoming.get_nowait()
        except asyncio.QueueEmpty:
            break
        kind, _data = item
        if kind == "pcm":
            continue
        kept.append(item)
    for item in kept:
        try:
            incoming.put_nowait(item)
        except Exception:
            pass


async def watch_wake_barge(
    incoming: asyncio.Queue,
    detector: Any,
    cooldown: KwsCooldown | None,
    abort_fn: Callable[..., Any],
    *,
    started: asyncio.Event,
    done: asyncio.Event,
) -> bool:
    """After TTS starts, listen for 小溪小溪 and abort playback. True = hit."""
    await started.wait()
    if done.is_set() or detector is None:
        return False
    _drop_queued_pcm(incoming)
    while not done.is_set():
        try:
            kind, data = await asyncio.wait_for(incoming.get(), timeout=0.15)
        except asyncio.TimeoutError:
            continue
        if kind == "eof":
            requeue_eof(incoming, kind)
            return False
        if kind != "pcm" or not data:
            continue
        if cooldown is not None and not cooldown.allow(len(data)):
            continue
        accept = getattr(detector, "accept_pcm", None)
        if not callable(accept):
            continue
        try:
            hit = accept(data)
        except Exception as exc:
            print(f"⚠️ wake kws: {type(exc).__name__}: {exc}", flush=True)
            continue
        if not hit:
            continue
        try:
            await abort_fn()
        except Exception:
            pass
        return True
    return False


async def listen_with_barge(session, chunks, incoming, detector, cooldown, **listen_kwargs):
    """Host listen + optional wake-word abort while TTS is playing."""
    started = asyncio.Event()
    done = asyncio.Event()
    user_end = listen_kwargs.pop("on_utterance_end", None)
    user_event = listen_kwargs.pop("on_event", None)
    user_audio = listen_kwargs.pop("on_audio", None)

    def mark() -> None:
        if callable(user_end):
            user_end()

    def on_event(ev: dict[str, Any]) -> None:
        if isinstance(ev, dict) and ev.get("type") in {"tts_text", "audio"}:
            started.set()
        if callable(user_event):
            user_event(ev)

    def on_audio(pcm: bytes) -> None:
        if pcm:
            started.set()
        if callable(user_audio):
            user_audio(pcm)

    listen_kwargs["on_utterance_end"] = mark
    listen_kwargs["on_event"] = on_event
    listen_kwargs["on_audio"] = on_audio

    async def do_listen():
        try:
            return await run_gated_listen(session, chunks, **listen_kwargs)
        finally:
            done.set()
            started.set()

    task = asyncio.create_task(do_listen())
    try:
        barged = await watch_wake_barge(
            incoming,
            detector,
            cooldown,
            session.abort_playback,
            started=started,
            done=done,
        )
        return await task, barged
    except BaseException:
        done.set()
        started.set()
        if not task.done():
            try:
                await session.abort_playback()
            except Exception:
                pass
        await task
        raise


async def pcm_source(device: str, chunk_bytes: int = 3200):
    cmd = [
        "arecord",
        "-q",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-c",
        "1",
        "-r",
        "16000",
        "-t",
        "raw",
        "-",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise HubConnectError("本机没有 arecord。请安装 alsa-utils。") from exc
    print(f"wake audio_device={device}", flush=True)
    try:
        while True:
            assert proc.stdout is not None
            chunk = await proc.stdout.read(chunk_bytes)
            if not chunk:
                err = b""
                if proc.stderr:
                    try:
                        err = await asyncio.wait_for(proc.stderr.read(), timeout=0.2)
                    except Exception:
                        err = b""
                detail = err.decode("utf-8", "replace").strip()
                raise HubConnectError("麦克风读不到数据。" + (f" arecord：{detail}" if detail else ""))
            yield chunk
    finally:
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()


async def run_wake(
    url: str,
    robot_id: str,
    schema_version: int,
    play: bool,
    on_event: OnEvent | None = None,
    detector: Any | None = None,
) -> int:
    info = describe_kws()
    print(
        f"wake word={info['wake_word']} kws_model={info['kws_model']} "
        f"kws_ready={info['kws_ready']} kws_error={info['kws_error'] or '-'}",
        flush=True,
    )
    if detector is None:
        if not info["kws_ready"]:
            print(f"error: {info['kws_error']}", flush=True)
            return 2
        detector = KeywordSpotter()
    device = os.environ.get("ARECORD_DEVICE", "default") or "default"
    session = HubSession(on_event=on_event)
    backoff = 1.0
    try:
        while True:
            try:
                if not session.connected:
                    hello = await session.connect(url, robot_id, schema_version)
                    if hello.get("type") != "hello_ok":
                        print(f"host=rejected reason={hello.get('reason')}", flush=True)
                        await asyncio.sleep(backoff)
                        backoff = min(8.0, backoff * 2)
                        continue
                    print(f"host=connected robot={hello.get('robot_id')}", flush=True)
                    backoff = 1.0
                print(f"wake=waiting word={wake_word()}", flush=True)
                incoming: asyncio.Queue = asyncio.Queue()
                stop_mic = asyncio.Event()
                mic_error: list[BaseException] = []

                async def mic_to_queue() -> None:
                    try:
                        async for chunk in pcm_source(device):
                            if stop_mic.is_set():
                                return
                            await incoming.put(("pcm", chunk))
                    except Exception as exc:
                        mic_error.append(exc)
                    finally:
                        try:
                            incoming.put_nowait(("eof", None))
                        except Exception:
                            pass

                mic_task = asyncio.create_task(mic_to_queue())
                cooldown = KwsCooldown()
                did_turn = False
                tts_bytes = 0
                try:
                    while True:
                        kind, data = await incoming.get()
                        if kind == "eof":
                            if mic_error:
                                raise mic_error[0]
                            raise HubConnectError("麦克风读不到数据。")
                        if not session.connected:
                            raise HubConnectError("和主机的连接断开了，请重新连接。")
                        if kind != "pcm" or not data:
                            continue
                        if not cooldown.allow(len(data)):
                            continue
                        if detector.accept_pcm(data):
                            print(f"wake=hit word={wake_word()} device={device}", flush=True)
                            played = await asyncio.to_thread(play_prompt_tone)
                            print(f"wake=prompt played={played}", flush=True)
                            break
                    skip_sec = 0.25
                    while True:
                        stream = await begin_command_stream(incoming, skip_sec=skip_sec)
                        if stream is None:
                            print("wake=idle reason=no_speech", flush=True)
                            break
                        did_turn = True
                        streamed = 0

                        async def chunks():
                            nonlocal streamed
                            async for part in stream:
                                streamed += len(part)
                                if streamed == len(part):
                                    print(f"wake=stream bytes={streamed}", flush=True)
                                yield part

                        sink = None
                        if play:
                            from jetson_client.voice_client import _AplayStream

                            sink = _AplayStream()
                        tts_bytes = 0

                        def on_audio(pcm: bytes, _sink=sink) -> None:
                            nonlocal tts_bytes
                            if pcm:
                                tts_bytes += len(pcm)
                            if _sink is not None:
                                _sink.write(pcm)

                        try:
                            (result, decision), barged = await listen_with_barge(
                                session,
                                chunks(),
                                incoming,
                                detector,
                                cooldown,
                                idle_after_speech=12.0 if play else 20.0,
                                on_audio=on_audio,
                                on_event=on_event,
                            )
                            if barged:
                                print(f"wake=barge word={wake_word()}", flush=True)
                                if on_event:
                                    on_event(
                                        {"type": "wake", "status": "hit", "word": wake_word()}
                                    )
                                detector.reset()
                                skip_sec = 0.25
                                continue
                            if result is None:
                                print(
                                    f"wake=rejected bytes={streamed} "
                                    f"reason={decision.get('reason')}",
                                    flush=True,
                                )
                                if on_event:
                                    on_event(
                                        {
                                            "type": "wake",
                                            "status": "rejected",
                                            "word": wake_word(),
                                            "message": "未通过声纹，请本人再说",
                                            "reason": decision.get("reason"),
                                            "score": decision.get("score"),
                                            "display_name": decision.get("display_name"),
                                            "mode": decision.get("mode"),
                                        }
                                    )
                            else:
                                print(
                                    f"wake=listen bytes={streamed} streamed=1 "
                                    f"stt={_event_text(result.events, 'stt')!r} "
                                    f"tts={_event_text(result.events, 'tts_text')!r} "
                                    f"pcm={len(result.pcm)}",
                                    flush=True,
                                )
                        finally:
                            if sink is not None:
                                sink.close()
                        break
                finally:
                    stop_mic.set()
                    mic_task.cancel()
                    try:
                        await mic_task
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass
                detector.reset()
                if did_turn:
                    cooldown.arm(extra_sec=tts_bytes / 32000.0)
                continue
            except HubConnectError as exc:
                detail = str(exc)
                if "麦克风" in detail or "arecord" in detail:
                    print(f"wake=error error={detail}", flush=True)
                else:
                    print(f"host=disconnected error={detail}", flush=True)
                    try:
                        await session.close()
                    except Exception:
                        pass
                await asyncio.sleep(backoff)
                backoff = min(8.0, backoff * 2)
    finally:
        await session.close()
    return 0


def _event_text(events: list[dict[str, Any]], kind: str) -> str:
    return " ".join(str(e.get("text") or "") for e in events if e.get("type") == kind).strip()
