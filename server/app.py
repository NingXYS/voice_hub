from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import websockets

from server import config, protocol
from server.adapter_loader import load_adapter
from server.memory import ConversationMemory
from server.pipeline import Pipeline, spoken_search_reply, today_label
from server.providers import build_asr, build_llm, build_tts
from server.providers.realtime_doubao import DoubaoRealtimeSession, RealtimeEvent

_TTS_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vh-tts")
_JOB_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="vh-job")


class VoiceHub:
    def __init__(
        self,
        asr=None,
        tts=None,
        llm=None,
        pipeline: Pipeline | None = None,
        realtime_factory=None,
    ):
        self.asr = asr
        self.tts = tts
        self.llm = llm
        self.pipeline = pipeline
        self.realtime_factory = realtime_factory

    def ensure(self) -> None:
        if self.asr is None:
            try:
                self.asr = build_asr()
            except Exception as exc:
                print(f"⚠️ ASR init failed ({exc}); text-only mode")
                from server.providers.asr_local import NullASR

                self.asr = NullASR()
        if self.tts is None:
            self.tts = build_tts()
        if self.pipeline is None:
            self.pipeline = Pipeline(llm=self.llm or build_llm())
        print(
            f"voice_hub providers: asr={getattr(self.asr, 'name', '?')} "
            f"tts={getattr(self.tts, 'name', '?')} "
            f"llm={config.llm_model() or 'unconfigured'} thinking=off "
            f"search={config.search_provider()} "
            f"search_key={'set' if config.search_api_key() else 'missing'} "
            f"dialog={'on' if config.dialog_configured() else 'off'} "
            f"dialog_auth={'appid+token' if (config.dialog_appid() and config.dialog_access_token()) else ('api_key' if config.dialog_api_key() else 'missing')} "
            f"dialog_speaker={config.dialog_speaker()}",
            flush=True,
        )

    async def handle(self, websocket):
        adapter = None
        pcm = bytearray()
        listening = False
        abort = asyncio.Event()
        turn_task: asyncio.Task | None = None
        send_lock = asyncio.Lock()
        live_asr = None
        last_partial = ""
        memory: ConversationMemory | None = None
        dialog: DoubaoRealtimeSession | None = None
        speaking = False
        user_listening = False
        recent_tts: list[str] = []

        async def send_json(payload: dict[str, Any]) -> None:
            if payload.get("type") == "tts_text":
                piece = str(payload.get("text") or "").strip()
                if piece:
                    recent_tts.append(piece)
                    del recent_tts[:-12]
            async with send_lock:
                await websocket.send(json.dumps(payload, ensure_ascii=False))

        async def send_bin(data: bytes) -> None:
            async with send_lock:
                await websocket.send(data)

        async def run_turn(spoken_text: str, token: asyncio.Event) -> None:
            nonlocal speaking
            cleaned = (spoken_text or "").strip()
            if not cleaned or _is_echo(cleaned, recent_tts):
                if cleaned:
                    print(f"⚠️ ignore echo/stt={cleaned[:24]!r}", flush=True)
                try:
                    await send_json(protocol.turn_end())
                except Exception:
                    pass
                return
            speaking = True
            try:
                async def refresh_dialog():
                    nonlocal dialog
                    await close_dialog()
                    dialog = await ensure_dialog()
                    return dialog

                await self._turn(
                    adapter,
                    spoken_text,
                    token,
                    send_json,
                    send_bin,
                    memory=memory,
                    dialog=dialog,
                    refresh_dialog=refresh_dialog,
                )
            except Exception as exc:
                print(f"⚠️ turn failed: {type(exc).__name__}: {exc}", flush=True)
                try:
                    await send_json(protocol.error("处理失败，请再说一次。"))
                    await send_json(protocol.turn_end())
                except Exception:
                    pass
            finally:
                speaking = False

        async def run_asr_turn(captured: bytes, token: asyncio.Event) -> None:
            text = ""
            if self.asr is not None:
                try:
                    transcribe_async = getattr(self.asr, "transcribe_async", None)
                    if callable(transcribe_async):
                        text = await transcribe_async(captured)
                    else:
                        text = await asyncio.get_running_loop().run_in_executor(
                            _JOB_POOL, self.asr.transcribe, captured
                        )
                except Exception as exc:
                    msg = str(exc)
                    print(f"⚠️ ASR failed: {type(exc).__name__}: {msg}", flush=True)
                    if any(mark in msg for mark in ("请", "密钥", "连不上", "识别")):
                        await send_json(protocol.error(msg))
            await run_turn(text or "", token)

        async def close_live_asr() -> None:
            nonlocal live_asr
            session = live_asr
            live_asr = None
            if session is None:
                return
            closer = getattr(session, "close", None)
            if callable(closer):
                try:
                    await closer()
                except Exception:
                    pass

        async def close_dialog() -> None:
            nonlocal dialog
            session = dialog
            dialog = None
            if session is None:
                return
            try:
                await session.finish()
            except Exception:
                pass
            try:
                await session.close()
            except Exception:
                pass

        async def ensure_dialog() -> DoubaoRealtimeSession | None:
            nonlocal dialog
            wanted_city = memory.city if memory is not None else ""
            if dialog is not None and dialog.alive:
                if not wanted_city or getattr(dialog, "city", "") == wanted_city:
                    return dialog
                await close_dialog()
            elif dialog is not None:
                await close_dialog()
            factory = self.realtime_factory
            if factory is None and config.dialog_configured():
                factory = DoubaoRealtimeSession
            if factory is None:
                return None

            async def on_partial(text: str) -> None:
                nonlocal last_partial
                if speaking:
                    return
                cleaned = (text or "").strip()
                if cleaned and cleaned != last_partial:
                    last_partial = cleaned
                    await send_json(protocol.stt(cleaned))

            try:
                session = factory(on_asr=on_partial) if _accepts_on_asr(factory) else factory()
                if session is None:
                    return None
                if getattr(session, "on_asr", None) is None:
                    session.on_asr = on_partial
                await session.start(
                    city=memory.city if memory is not None else "",
                    system_role=_dialog_system(adapter, memory),
                    dialog_id=memory.dialog_id if memory is not None else "",
                    bot_name=adapter.name if adapter is not None else "",
                )
                if memory is not None and len(str(getattr(session, "dialog_id", "") or "")) > 8:
                    memory.dialog_id = session.dialog_id
                    memory.save()
                dialog = session
                return session
            except Exception as exc:
                print(f"⚠️ dialog start failed ({exc}); cascade fallback", flush=True)
                dialog = None
                return None

        async def stop_turn() -> None:
            nonlocal turn_task
            abort.set()
            task = turn_task
            turn_task = None
            if task is not None:
                try:
                    await task
                except Exception:
                    pass

        try:
            raw = await websocket.recv()
            hello = _as_json(raw)
            if not isinstance(hello, dict) or hello.get("type") != "hello":
                await send_json(protocol.hello_reject("first message must be hello"))
                return
            robot_id = str(hello.get("robot_id") or "").strip()
            schema_version = int(hello.get("schema_version") or 1)
            try:
                adapter = load_adapter(robot_id, config.adapter_dir())
            except FileNotFoundError:
                await send_json(protocol.hello_reject(f"unknown robot_id: {robot_id}"))
                return
            if adapter.schema_version != schema_version:
                await send_json(
                    protocol.hello_reject(
                        f"schema_version mismatch: server={adapter.schema_version} client={schema_version}"
                    ),
                )
                return
            memory = ConversationMemory.load(adapter.robot_id)
            await send_json(
                protocol.hello_ok(adapter.robot_id, adapter.schema_version, adapter.name),
            )

            async for message in websocket:
                if isinstance(message, bytes):
                    if listening and user_listening and not speaking:
                        pcm.extend(message)
                        target = dialog if dialog is not None and dialog.alive else live_asr
                        if target is not None:
                            try:
                                await target.feed(message)
                            except Exception as exc:
                                print(f"⚠️ audio feed failed: {type(exc).__name__}: {exc}", flush=True)
                    continue
                payload = _as_json(message)
                if not isinstance(payload, dict):
                    await send_json(protocol.error("invalid json"))
                    continue
                kind = payload.get("type")
                if kind == "listen":
                    state = payload.get("state")
                    if state == "start":
                        if speaking:
                            # Do not barge in; listen stop will still get turn_end.
                            continue
                        pcm.clear()
                        last_partial = ""
                        listening = True
                        user_listening = True
                        await close_live_asr()
                        opened = await ensure_dialog()
                        beginner = getattr(opened, "begin_listen", None) if opened is not None else None
                        if callable(beginner):
                            beginner()
                        if opened is None:
                            opener = getattr(self.asr, "open_stream", None)
                            if callable(opener):
                                try:
                                    async def on_partial(text: str) -> None:
                                        nonlocal last_partial
                                        if speaking:
                                            return
                                        cleaned = (text or "").strip()
                                        if cleaned and cleaned != last_partial:
                                            last_partial = cleaned
                                            await send_json(protocol.stt(cleaned))

                                    live_asr = await opener(on_partial=on_partial)
                                except Exception as exc:
                                    print(f"⚠️ ASR stream open failed ({exc}); buffering until stop", flush=True)
                                    live_asr = None
                    elif state == "stop":
                        if not user_listening:
                            if not speaking:
                                try:
                                    await send_json(protocol.turn_end())
                                except Exception:
                                    pass
                            continue
                        listening = False
                        user_listening = False
                        captured = bytes(pcm)
                        pcm.clear()
                        await stop_turn()
                        abort = asyncio.Event()
                        if dialog is not None and dialog.alive:
                            async def finish_dialog() -> None:
                                text = ""
                                try:
                                    await dialog.end_utterance()
                                    text = await dialog.wait_asr_final()
                                except Exception as exc:
                                    msg = str(exc)
                                    print(f"⚠️ dialog ASR failed: {type(exc).__name__}: {msg}", flush=True)
                                    if any(mark in msg for mark in ("请", "密钥", "连不上", "识别", "端到端")):
                                        await send_json(protocol.error(msg))
                                if not (text or "").strip() and captured and self.asr is not None:
                                    await run_asr_turn(captured, abort)
                                    return
                                await run_turn(text or "", abort)

                            turn_task = asyncio.create_task(finish_dialog())
                        else:
                            session = live_asr
                            live_asr = None
                            if session is not None:
                                async def finish_live() -> None:
                                    text = ""
                                    try:
                                        text = await session.finish()
                                    except Exception as exc:
                                        msg = str(exc)
                                        print(f"⚠️ ASR failed: {type(exc).__name__}: {msg}", flush=True)
                                        if any(mark in msg for mark in ("请", "密钥", "连不上", "识别", "资源")):
                                            await send_json(protocol.error(msg))
                                    finally:
                                        closer = getattr(session, "close", None)
                                        if callable(closer):
                                            try:
                                                await closer()
                                            except Exception:
                                                pass
                                    if not (text or "").strip() and captured:
                                        await run_asr_turn(captured, abort)
                                        return
                                    await run_turn(text or "", abort)

                                turn_task = asyncio.create_task(finish_live())
                            else:
                                turn_task = asyncio.create_task(run_asr_turn(captured, abort))
                elif kind == "text":
                    listening = False
                    user_listening = False
                    await close_live_asr()
                    await stop_turn()
                    abort = asyncio.Event()
                    if config.dialog_configured() or self.realtime_factory is not None:
                        await ensure_dialog()
                    turn_task = asyncio.create_task(
                        run_turn(str(payload.get("text") or ""), abort)
                    )
                elif kind == "memory_get":
                    snap = memory.snapshot() if memory is not None else {"city": "", "name": "", "turns": [], "path": ""}
                    await send_json(protocol.memory_state(**snap))
                elif kind == "memory_set":
                    if memory is not None:
                        if "city" in payload:
                            memory.apply(city=str(payload.get("city") or ""))
                        if "name" in payload:
                            memory.apply(name=str(payload.get("name") or ""))
                        snap = memory.snapshot()
                        await send_json(protocol.memory_state(**snap))
                    else:
                        await send_json(protocol.error("还没有机械臂会话，无法改记忆。"))
                elif kind == "memory_clear":
                    if memory is not None:
                        memory.clear()
                        await send_json(protocol.memory_state(**memory.snapshot()))
                    else:
                        await send_json(protocol.error("还没有机械臂会话，无法改记忆。"))
                elif kind == "abort":
                    pcm.clear()
                    listening = False
                    user_listening = False
                    speaking = False
                    await close_live_asr()
                    if dialog is not None:
                        dialog.mute()
                    await stop_turn()
                else:
                    await send_json(protocol.error(f"unknown type: {kind}"))
            await stop_turn()
            await close_live_asr()
            await close_dialog()
        except websockets.ConnectionClosed:
            abort.set()
            try:
                await close_live_asr()
            except Exception:
                pass
            try:
                await close_dialog()
            except Exception:
                pass
            if turn_task is not None:
                try:
                    await turn_task
                except Exception:
                    pass
            return

    async def _turn(
        self,
        adapter,
        text: str,
        abort: asyncio.Event,
        send_json: Callable,
        send_bin: Callable,
        memory: ConversationMemory | None = None,
        dialog: DoubaoRealtimeSession | None = None,
        refresh_dialog=None,
    ) -> None:
        started = time.monotonic()
        first_audio_logged = False
        text = (text or "").strip()
        if text:
            await send_json(protocol.stt(text))

        async def on_first_audio(sentence: str) -> None:
            nonlocal first_audio_logged
            if first_audio_logged:
                return
            first_audio_logged = True
            ms = int((time.monotonic() - started) * 1000)
            print(f"tts_first_audio_ms={ms} text={sentence[:24]!r}", flush=True)

        async def speak(sentence: str) -> None:
            await self._speak(sentence, abort, send_json, send_bin, on_first=on_first_audio)

        use_dialog = dialog is not None and getattr(dialog, "alive", False)
        if use_dialog:
            decision = self.pipeline.decide(text, adapter, memory)
            try:
                await self._dialog_turn(
                    decision,
                    text,
                    abort,
                    send_json,
                    send_bin,
                    speak,
                    on_first_audio,
                    memory,
                    dialog,
                    started,
                )
                return
            except Exception as exc:
                print(f"⚠️ dialog turn failed ({type(exc).__name__}: {exc}); retry", flush=True)
                if callable(refresh_dialog) and not abort.is_set():
                    try:
                        dialog = await refresh_dialog()
                    except Exception as refresh_exc:
                        print(f"⚠️ dialog refresh failed: {type(refresh_exc).__name__}: {refresh_exc}", flush=True)
                        dialog = None
                    if dialog is not None and getattr(dialog, "alive", False):
                        try:
                            await self._dialog_turn(
                                decision,
                                text,
                                abort,
                                send_json,
                                send_bin,
                                speak,
                                on_first_audio,
                                memory,
                                dialog,
                                started,
                            )
                            return
                        except Exception as exc2:
                            print(f"⚠️ dialog retry failed ({type(exc2).__name__}: {exc2}); cascade", flush=True)

        await self._cascade_turn(adapter, text, abort, send_json, send_bin, speak, memory)
        await send_json(protocol.turn_end())

    async def _dialog_turn(
        self,
        decision,
        text: str,
        abort: asyncio.Event,
        send_json,
        send_bin,
        speak,
        on_first_audio,
        memory,
        dialog: DoubaoRealtimeSession,
        started: float,
    ) -> None:
        if decision.takeover or decision.route == "search":
            dialog.mute()
        else:
            dialog.muted = False
        collected: list[str] = []

        async def say_dialog(content: str, *, until: str = "ended", allow_http: bool = True) -> tuple[str, bool]:
            cleaned = (content or "").strip()
            if not cleaned:
                return "", False
            prep = getattr(dialog, "prepare_external_speech", None)
            if callable(prep):
                await prep()
            await dialog.chat_tts_text(cleaned)
            spoken, got_audio = await self._pump_dialog(
                dialog,
                abort,
                send_json,
                send_bin,
                on_first_audio,
                until=until,
                timeout=20.0,
                first_audio=5.0,
                caption=cleaned,
            )
            if got_audio or abort.is_set():
                return spoken or cleaned, got_audio
            hello = getattr(dialog, "say_hello", None)
            if callable(hello):
                await hello(cleaned)
                spoken, got_audio = await self._pump_dialog(
                    dialog,
                    abort,
                    send_json,
                    send_bin,
                    on_first_audio,
                    until=until,
                    timeout=3.0,
                    first_audio=1.5,
                    caption=cleaned,
                )
                if got_audio or abort.is_set():
                    return spoken or cleaned, got_audio
            if not allow_http:
                print("⚠️ dialog ChatTTSText had no audio; skip HTTP filler", flush=True)
                return cleaned, False
            print("⚠️ dialog ChatTTSText had no audio; using Doubao TTS for this sentence", flush=True)
            await speak(cleaned)
            return cleaned, True

        if decision.takeover:
            if decision.commands:
                await send_json(protocol.robot_command(decision.commands))
            spoken, _ = await say_dialog("".join(decision.sentences) or decision.reply)
            if memory is not None:
                memory.observe_assistant(decision.reply or spoken)
            await send_json(protocol.turn_end())
            return

        if decision.route == "search":
            loop = asyncio.get_running_loop()
            query = decision.search_q or text
            search_task = loop.run_in_executor(_JOB_POOL, self.pipeline.search_fn, query)
            filler, _ = await say_dialog("我查一下。", until="ended", allow_http=True)
            if filler:
                collected.append(filler)
            notes = await search_task
            print(
                f"search_q={query!r} search_ms={int((time.monotonic() - started) * 1000)} "
                f"notes={len(notes or '')}",
                flush=True,
            )
            city = memory.city if memory is not None else ""
            answer = spoken_search_reply(notes, city=city, query=query)
            spoken, _ = await say_dialog(answer, until="ended", allow_http=True)
            if spoken:
                collected.append(spoken)
            if memory is not None and (spoken or answer):
                memory.observe_assistant(spoken or answer)
            await send_json(protocol.turn_end())
            return

        drain = getattr(dialog, "drain", None)
        if callable(drain):
            drain()
        await dialog.chat_text(text)
        spoken, got_audio = await self._pump_dialog(
            dialog,
            abort,
            send_json,
            send_bin,
            on_first_audio,
            until="ended",
            timeout=30.0,
        )
        if spoken:
            collected.append(spoken)
        reply = "".join(collected)
        if not got_audio and not abort.is_set():
            if reply:
                spoken, _ = await say_dialog(reply, until="ended")
                if spoken:
                    collected.append(spoken)
                reply = "".join(collected)
            else:
                raise RuntimeError("dialog produced no audio")
        if memory is not None and reply:
            memory.observe_assistant(reply)
        await send_json(protocol.turn_end())

    async def _pump_dialog(
        self,
        dialog,
        abort: asyncio.Event,
        send_json,
        send_bin,
        on_first_audio,
        *,
        until: str = "ended",
        timeout: float = 8.0,
        first_audio: float = 0.0,
        caption: str = "",
    ) -> tuple[str, bool]:
        collected: list[str] = []
        got_audio = False
        caption_sent = False
        stale_ended = 0
        seen: list[str] = []
        started_at = time.monotonic()
        deadline = started_at + timeout
        first_deadline = started_at + first_audio if first_audio > 0 else deadline
        waiter = getattr(dialog, "wait_event", None)

        async def flush_caption() -> None:
            nonlocal caption_sent
            if caption_sent:
                return
            caption_sent = True
            blob = "".join(collected).strip() or (caption or "").strip()
            if blob:
                await send_json(protocol.tts_text(blob))

        async def nxt(wait: float):
            if callable(waiter):
                return await waiter(wait)
            queue = getattr(dialog, "_queue", None)
            if queue is None:
                return RealtimeEvent(kind="closed")
            try:
                item = await asyncio.wait_for(queue.get(), timeout=wait)
            except asyncio.TimeoutError:
                return RealtimeEvent(kind="timeout")
            if item is None:
                return RealtimeEvent(kind="closed")
            return item

        while time.monotonic() < deadline and not abort.is_set():
            event = await nxt(min(2.0, max(0.05, deadline - time.monotonic())))
            kind = getattr(event, "kind", "")
            if kind and kind != "timeout":
                ev = getattr(event, "event", 0) or 0
                tts_type = getattr(event, "tts_type", "") or ""
                seen.append(f"{kind}:{ev}:{tts_type}" if tts_type else f"{kind}:{ev}")
            if kind == "timeout":
                if got_audio and until == "audio":
                    break
                if not got_audio and time.monotonic() >= first_deadline:
                    break
                continue
            if kind == "closed":
                break
            if kind == "error" and getattr(event, "error", ""):
                print(f"⚠️ dialog event error: {event.error}", flush=True)
                break
            if kind == "chat_end":
                deadline = max(deadline, time.monotonic() + 8.0)
                continue
            if kind == "text" and event.text:
                collected.append(event.text)
                deadline = max(deadline, time.monotonic() + 4.0)
                if got_audio:
                    await send_json(protocol.tts_text(event.text))
            elif kind == "audio" and event.pcm:
                pcm = event.pcm
                if len(pcm) % 2:
                    pcm = pcm + b"\x00"
                if not got_audio:
                    await flush_caption()
                got_audio = True
                deadline = max(deadline, time.monotonic() + 4.0)
                await send_bin(pcm)
                await on_first_audio(collected[-1] if collected else caption)
                if until == "audio":
                    break
            elif kind == "ended":
                if got_audio:
                    drain_until = time.monotonic() + 0.5
                    while time.monotonic() < drain_until and not abort.is_set():
                        extra = await nxt(max(0.05, drain_until - time.monotonic()))
                        extra_kind = getattr(extra, "kind", "")
                        if extra_kind == "audio" and extra.pcm:
                            pcm = extra.pcm + (b"\x00" if len(extra.pcm) % 2 else b"")
                            await send_bin(pcm)
                            continue
                        if extra_kind in {"timeout", "closed", "ended"}:
                            break
                    break
                if collected:
                    break
                stale_ended += 1
                if stale_ended >= 3:
                    break
        if not got_audio and seen:
            print(f"⚠️ dialog pump no audio until={until} seen={seen[:16]}", flush=True)
        return "".join(collected), got_audio

    async def _cascade_turn(
        self,
        adapter,
        text: str,
        abort: asyncio.Event,
        send_json,
        send_bin,
        speak,
        memory: ConversationMemory | None,
    ) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        spoken: list[str] = []

        def on_sentence(sentence: str) -> None:
            if sentence and not abort.is_set():
                _offer(loop, queue, sentence)

        holder: dict[str, Any] = {}

        def work() -> None:
            try:
                holder["result"] = self.pipeline.handle_text(
                    text, adapter, on_sentence=on_sentence, memory=memory
                )
            except Exception as exc:
                holder["error"] = exc
            finally:
                _offer(loop, queue, None)

        worker = loop.run_in_executor(_JOB_POOL, work)
        while not abort.is_set():
            item = await _queue_get(queue, abort)
            if item is None:
                break
            spoken.append(item)
            await speak(item)
        if abort.is_set():
            await worker
            return
        await worker

        if holder.get("error") is not None:
            exc = holder["error"]
            print(f"⚠️ pipeline failed: {type(exc).__name__}: {exc}", flush=True)
            await send_json(protocol.error("处理失败，请再说一次。"))
            await speak("处理失败，请再说一次。")
            return

        result = holder.get("result")
        if result is None:
            return
        if result.commands:
            await send_json(protocol.robot_command(result.commands))
        if spoken:
            joined = "".join(spoken)
            reply = result.reply or ""
            tail = reply[len(joined):].strip() if reply.startswith(joined) else ""
            if tail and tail not in spoken:
                spoken.append(tail)
                await speak(tail)
        else:
            parts = result.sentences or ([result.reply] if result.reply else [])
            for sentence in parts:
                if abort.is_set():
                    break
                if not sentence or sentence in spoken:
                    continue
                spoken.append(sentence)
                await speak(sentence)

    async def _speak(
        self,
        sentence: str,
        abort: asyncio.Event,
        send_json: Callable,
        send_bin: Callable,
        on_first=None,
    ) -> bool:
        if not sentence or abort.is_set():
            return False
        if self.tts is None:
            await send_json(protocol.tts_text(sentence))
            return False
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()

        def work() -> None:
            odd = b""
            try:
                for chunk in _iter_pcm(self.tts, sentence):
                    if abort.is_set():
                        break
                    data = odd + (chunk or b"")
                    if len(data) % 2:
                        odd = data[-1:]
                        data = data[:-1]
                    else:
                        odd = b""
                    if data:
                        _offer(loop, queue, data)
                if odd and not abort.is_set():
                    _offer(loop, queue, odd + b"\x00")
            except Exception as exc:
                _offer(loop, queue, exc)
            finally:
                _offer(loop, queue, None)

        worker = loop.run_in_executor(_TTS_POOL, work)
        sent = False
        try:
            while not abort.is_set():
                item = await _queue_get(queue, abort)
                if item is None:
                    break
                if isinstance(item, BaseException):
                    print(f"⚠️ TTS failed: {type(item).__name__}: {item}", flush=True)
                    break
                await send_bin(item)
                if not sent:
                    await send_json(protocol.tts_text(sentence))
                    if on_first is not None:
                        await on_first(sentence)
                sent = True
        finally:
            if not abort.is_set():
                await worker
        if not sent and not abort.is_set():
            await send_json(protocol.tts_text(sentence))
        return sent


async def _send_json(websocket, payload: dict[str, Any]) -> None:
    await websocket.send(json.dumps(payload, ensure_ascii=False))


async def _queue_get(queue: asyncio.Queue, abort: asyncio.Event):
    if abort.is_set():
        return None
    getter = asyncio.create_task(queue.get())
    waiter = asyncio.create_task(abort.wait())
    done, pending = await asyncio.wait({getter, waiter}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    if abort.is_set() and getter not in done:
        return None
    if getter in done:
        return getter.result()
    return None


def _offer(loop: asyncio.AbstractEventLoop, queue: asyncio.Queue, item) -> None:
    try:
        loop.call_soon_threadsafe(queue.put_nowait, item)
    except RuntimeError:
        pass


def _iter_pcm(tts, text: str):
    if tts is None or not text:
        return
    iterator = getattr(tts, "synthesize_iter", None)
    if callable(iterator):
        yield from iterator(text)
        return
    pcm = tts.synthesize(text)
    if pcm:
        yield pcm


def _norm_echo(text: str) -> str:
    return "".join(ch for ch in (text or "") if ch.isalnum())


def _is_echo(heard: str, recent_tts: list[str]) -> bool:
    needle = _norm_echo(heard)
    if len(needle) < 2:
        return True
    spoken = _norm_echo("".join(recent_tts[-8:]))
    if not spoken:
        return False
    if len(needle) >= 4 and needle in spoken:
        return True
    if spoken.endswith(needle) and len(spoken) >= len(needle) + 4:
        return True
    return False


def _dialog_system(adapter, memory: ConversationMemory | None) -> str:
    name = getattr(adapter, "name", "机械臂") if adapter is not None else "机械臂"
    parts = [
        f"你是{name}旁的语音助手。用短句中文回答。不要编造机械臂控制 JSON。",
        f"今天是{today_label()}。日期必须用这句话，不要用训练数据里的2023或2024年。",
    ]
    if memory is not None:
        parts.append(memory.prompt_block())
    return "".join(parts)


def _accepts_on_asr(factory) -> bool:
    try:
        import inspect

        return "on_asr" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False


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


async def serve(hub: VoiceHub | None = None):
    hub = hub or VoiceHub()
    hub.ensure()
    async with websockets.serve(hub.handle, config.host(), config.port(), max_size=8 * 1024 * 1024):
        print(f"voice_hub listening ws://{config.host()}:{config.port()}{protocol.PROTOCOL_PATH}", flush=True)
        await asyncio.Future()


def main():
    asyncio.run(serve())
