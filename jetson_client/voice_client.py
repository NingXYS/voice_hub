#!/usr/bin/env python3
"""Thin Jetson client: mic/speaker + WebSocket. No robot JSON schema in this file."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jetson_client.plugin import ensure_bridge_on_path  # noqa: E402
from jetson_client.session import HubConnectError, HubSession  # noqa: E402

ensure_bridge_on_path()


def _read_pcm(path: str) -> bytes:
    target = Path(path)
    if not target.is_file():
        raise SystemExit(
            f"找不到录音文件：{path}\n"
            "请换成电脑上真实存在的 16kHz 单声道 wav，例如：\n"
            "  bash scripts/run_client.sh --pcm tests/fixtures/asr_demo.wav --play"
        )
    data = target.read_bytes()
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        pos = 12
        while pos + 8 <= len(data):
            kind = data[pos : pos + 4]
            size = int.from_bytes(data[pos + 4 : pos + 8], "little")
            pos += 8
            if kind == b"data":
                return data[pos : pos + size]
            pos += size
        raise SystemExit("WAV 里没有 data 块")
    return data


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


async def _mic_chunks(seconds: float, device: str):
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
        raise HubConnectError("本机没有 arecord。请安装 alsa-utils，或用 --pcm 指定 wav 文件。") from exc
    if seconds and seconds > 0:
        print(f"请对着麦克风说话（最多 {seconds:.0f} 秒）…", flush=True)
        stopper = asyncio.create_task(asyncio.sleep(seconds))
    elif sys.stdin.isatty():
        print("请对着麦克风说话，说完按回车。", flush=True)
        stopper = asyncio.create_task(asyncio.to_thread(input, ""))
    else:
        print("没有终端输入，改为录 5 秒。", flush=True)
        stopper = asyncio.create_task(asyncio.sleep(5))
    got = 0
    try:
        while not stopper.done():
            assert proc.stdout is not None
            try:
                chunk = await asyncio.wait_for(proc.stdout.read(6400), timeout=0.25)
            except asyncio.TimeoutError:
                continue
            if not chunk:
                break
            got += len(chunk)
            yield chunk
    finally:
        if not stopper.done():
            stopper.cancel()
            try:
                await stopper
            except asyncio.CancelledError:
                pass
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.5)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
        err = b""
        if proc.stderr:
            try:
                err = await asyncio.wait_for(proc.stderr.read(), timeout=0.3)
            except Exception:
                err = b""
        if got == 0:
            detail = err.decode("utf-8", "replace").strip()
            raise HubConnectError(
                "没有从麦克风录到声音。请检查是否静音、是否选对设备（ARECORD_DEVICE），"
                + (f" arecord：{detail}" if detail else "")
            )


async def run_mic(
    url: str,
    robot_id: str,
    schema_version: int,
    play: bool,
    seconds: float,
    loop: bool = False,
) -> list[dict]:
    session = HubSession(on_event=_print_event)
    events: list[dict] = []
    device = _env("ARECORD_DEVICE", "default") or "default"
    try:
        hello = await session.connect(url, robot_id, schema_version)
        events.append(hello)
        if hello.get("type") != "hello_ok":
            return events
        turn = 0
        while True:
            turn += 1
            if loop:
                print(f"--- 第 {turn} 轮，说完按回车（Ctrl+C 结束连续对话）---", flush=True)
            sink = _AplayStream() if play else None
            try:
                result = await session.listen_stream(
                    _mic_chunks(seconds, device),
                    idle_after_speech=12.0 if play else 20.0,
                    on_audio=sink.write if sink else None,
                )
            finally:
                if sink is not None:
                    sink.close()
            events.extend(result.events)
            if not loop:
                break
    except HubConnectError as exc:
        payload = {"type": "error", "message": str(exc)}
        events.append(payload)
        _print_event(payload)
    finally:
        await session.close()
    return events


async def run_pcm(url: str, robot_id: str, schema_version: int, pcm: bytes, play: bool) -> list[dict]:
    session = HubSession(on_event=_print_event)
    events: list[dict] = []
    sink = _AplayStream() if play else None
    try:
        hello = await session.connect(url, robot_id, schema_version)
        events.append(hello)
        if hello.get("type") != "hello_ok":
            return events
        result = await session.listen_pcm(
            pcm,
            idle_after_speech=12.0 if play else 20.0,
            on_audio=sink.write if sink else None,
        )
        events.extend(result.events)
    except HubConnectError as exc:
        payload = {"type": "error", "message": str(exc)}
        events.append(payload)
        _print_event(payload)
    finally:
        if sink is not None:
            sink.close()
        await session.close()
    return events


async def run_text(url: str, robot_id: str, schema_version: int, text: str, play: bool) -> list[dict]:
    session = HubSession(on_event=_print_event)
    events: list[dict] = []
    sink = _AplayStream() if play else None
    try:
        hello = await session.connect(url, robot_id, schema_version)
        events.append(hello)
        if hello.get("type") != "hello_ok":
            return events
        result = await session.say(
            text,
            idle_after_speech=12.0 if play else 20.0,
            on_audio=sink.write if sink else None,
        )
        events.extend(result.events)
    except HubConnectError as exc:
        payload = {"type": "error", "message": str(exc)}
        events.append(payload)
        _print_event(payload)
    finally:
        if sink is not None:
            sink.close()
        await session.close()
    return events


def _print_event(payload: dict) -> None:
    kind = payload.get("type")
    if kind == "stt":
        print(f"stt: {payload.get('text')}", flush=True)
    elif kind == "tts_text":
        print(f"tts: {payload.get('text')}", flush=True)
    elif kind == "error":
        print(f"error: {payload.get('message')}", flush=True)
    elif kind == "hello_ok":
        print(
            f"hello_ok robot={payload.get('robot_id')} schema={payload.get('schema_version')}",
            flush=True,
        )
    elif kind == "hello_reject":
        print(f"hello_reject: {payload.get('reason')}", flush=True)
    elif kind == "robot_command":
        print(f"robot_command: {len(payload.get('commands') or [])} item(s)", flush=True)
    elif kind == "audio":
        print(f"audio: {payload.get('bytes')} bytes", flush=True)
    elif kind == "bridge":
        print(f"arm bridge: {payload.get('status')}", flush=True)


class _AplayStream:
    """Keep one aplay process so chunks play without a gap per chunk."""

    def __init__(self, rate: int = 16000):
        self.rate = rate
        self.proc: subprocess.Popen | None = None
        self.failed = False

    def write(self, pcm: bytes) -> None:
        if self.failed or not pcm:
            return
        if self.proc is None or self.proc.poll() is not None:
            device = _env("APLAY_DEVICE", "default") or "default"
            try:
                self.proc = subprocess.Popen(
                    [
                        "aplay",
                        "-q",
                        "-D",
                        device,
                        "-f",
                        "S16_LE",
                        "-c",
                        "1",
                        "-r",
                        str(self.rate),
                        "-t",
                        "raw",
                        "-",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                self.failed = True
                return
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(pcm)
            self.proc.stdin.flush()
        except BrokenPipeError:
            self.proc = None

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=3)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="voice_hub Jetson client")
    parser.add_argument("--url", default=_env("VOICE_HUB_URL", "ws://127.0.0.1:8765"))
    parser.add_argument("--robot-id", default=_env("ROBOT_ID", "tianlian_dual"))
    parser.add_argument("--schema-version", type=int, default=int(_env("ROBOT_SCHEMA_VERSION", "1") or "1"))
    parser.add_argument("--text", default="")
    parser.add_argument("--pcm", default="", help="16 kHz S16LE mono raw PCM (or .wav) to send as listen")
    parser.add_argument("--mic", action="store_true", help="use the local microphone (arecord)")
    parser.add_argument("--loop", action="store_true", help="with --mic, keep talking after each reply")
    parser.add_argument("--seconds", type=float, default=0, help="with --mic, record this many seconds instead of waiting for Enter")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--wake", action="store_true", help="wait for 小溪小溪, then one host turn, repeat")
    args = parser.parse_args(argv)
    if args.wake:
        from jetson_client.wake_loop import run_wake

        try:
            return asyncio.run(
                run_wake(args.url, args.robot_id, args.schema_version, args.play, on_event=_print_event)
            )
        except KeyboardInterrupt:
            print("interrupted", flush=True)
            return 130
    if not args.text and not args.pcm:
        args.mic = True
    try:
        if args.pcm:
            events = asyncio.run(
                run_pcm(args.url, args.robot_id, args.schema_version, _read_pcm(args.pcm), args.play)
            )
        elif args.text:
            events = asyncio.run(run_text(args.url, args.robot_id, args.schema_version, args.text, args.play))
        else:
            events = asyncio.run(
                run_mic(
                    args.url,
                    args.robot_id,
                    args.schema_version,
                    args.play,
                    args.seconds,
                    loop=args.loop,
                )
            )
    except KeyboardInterrupt:
        print("interrupted", flush=True)
        return 130
    print(json.dumps(events, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
