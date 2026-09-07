"""Device-side speaker gate. Reuses VoiceOnline verifier and on-disk profiles."""
from __future__ import annotations

import os
import secrets
import subprocess
import sys
import threading
import time
from typing import Any

from jetson_client.massage import voice_online_root
from jetson_client.wake_kws import pcm_rms, pcm_s16le_to_float

REQUIRED_SAMPLES = 3
ENROLL_PROMPTS = (
    "小溪小溪，请帮我介绍一下机器人",
    "现在开始按摩",
    "请停止按摩",
)

_VERIFIER = None
_VERIFIER_ERROR = ""
_EXTRACTOR = None
_EXTRACTOR_ERROR = ""
_ENROLL: dict[str, Any] | None = None
_CAPTURE_LOCK = threading.Lock()


def verify_mode() -> str:
    raw = (os.environ.get("VOICE_SPEAKER_VERIFY_MODE") or "off").strip().lower()
    if raw not in {"off", "shadow", "enforce"}:
        return "off"
    return raw


def profile_dir():
    from pathlib import Path

    raw = (os.environ.get("VOICE_SPEAKER_PROFILE_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser()
    root = voice_online_root()
    if root is not None:
        return root / "runtime_data" / "speaker_profiles"
    return Path.home() / "VoiceOnline" / "runtime_data" / "speaker_profiles"


def model_path():
    from pathlib import Path

    raw = (os.environ.get("VOICE_SPEAKER_MODEL_PATH") or "").strip()
    if raw:
        return Path(raw).expanduser()
    root = voice_online_root()
    base = (root or Path.home() / "VoiceOnline") / "models" / "speaker"
    return base / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"


def threshold() -> float:
    raw = (os.environ.get("VOICE_SPEAKER_THRESHOLD") or "0.60").strip()
    try:
        value = float(raw)
    except ValueError:
        return 0.6
    if value < 0 or value > 1:
        return 0.6
    return value


def min_speech_ms() -> int:
    raw = (os.environ.get("VOICE_SPEAKER_MIN_SPEECH_MS") or "1000").strip()
    try:
        value = int(raw)
    except ValueError:
        return 1000
    return max(0, value)


def _scripts_dir():
    root = voice_online_root()
    if root is None:
        return None
    scripts = root / "scripts"
    return scripts if scripts.is_dir() else None


def _ensure_scripts_path():
    scripts = _scripts_dir()
    if scripts is None:
        return None
    inserted = str(scripts)
    if inserted not in sys.path:
        sys.path.insert(0, inserted)
    return scripts


def status() -> dict[str, Any]:
    profiles = []
    error = ""
    try:
        profiles = list_profiles()
    except Exception as exc:
        error = str(exc)
    path = model_path()
    return {
        "mode": verify_mode(),
        "threshold": threshold(),
        "model": str(path),
        "profile_dir": str(profile_dir()),
        "profiles": profiles,
        "error": error,
        "ready": not error and path.is_file(),
        "enrollment": enrollment_public(),
        "required_samples": REQUIRED_SAMPLES,
    }


def list_profiles() -> list[dict[str, Any]]:
    _ensure_scripts_path()
    from voice_speaker_verifier import SpeakerProfileStore

    store = SpeakerProfileStore(profile_dir())
    return [item.public_payload() for item in store.list_profiles()]


def delete_profile(speaker_id: str) -> bool:
    _ensure_scripts_path()
    from voice_speaker_verifier import SpeakerProfileStore

    store = SpeakerProfileStore(profile_dir())
    ok = bool(store.delete_profile(speaker_id))
    reset_verifier()
    return ok


def save_embeddings(name: str, embeddings: list) -> dict[str, Any]:
    _ensure_scripts_path()
    from voice_speaker_verifier import SpeakerProfileStore

    store = SpeakerProfileStore(profile_dir())
    profile = store.save_profile(name, embeddings)
    reset_verifier()
    return profile.public_payload()


def reset_verifier() -> None:
    global _VERIFIER, _VERIFIER_ERROR
    _VERIFIER = None
    _VERIFIER_ERROR = ""


def reset_extractor() -> None:
    global _EXTRACTOR, _EXTRACTOR_ERROR
    _EXTRACTOR = None
    _EXTRACTOR_ERROR = ""


def _extractor():
    global _EXTRACTOR, _EXTRACTOR_ERROR
    if _EXTRACTOR is not None:
        return _EXTRACTOR
    if _EXTRACTOR_ERROR:
        raise RuntimeError(_EXTRACTOR_ERROR)
    try:
        from jetson_client.wake_kws import _import_sherpa

        _import_sherpa()
    except Exception:
        pass
    _ensure_scripts_path()
    from voice_speaker_verifier import SherpaSpeakerEmbeddingExtractor

    path = model_path()
    if not path.is_file():
        _EXTRACTOR_ERROR = f"找不到声纹模型：{path}"
        raise RuntimeError(_EXTRACTOR_ERROR)
    try:
        threads = int(os.environ.get("VOICE_SPEAKER_NUM_THREADS") or "2")
    except ValueError:
        threads = 2
    _EXTRACTOR = SherpaSpeakerEmbeddingExtractor(path, num_threads=max(1, threads))
    return _EXTRACTOR


def _verifier():
    global _VERIFIER, _VERIFIER_ERROR
    if _VERIFIER is not None:
        return _VERIFIER
    if _VERIFIER_ERROR:
        raise RuntimeError(_VERIFIER_ERROR)
    _ensure_scripts_path()
    from voice_speaker_verifier import SpeakerProfileStore, SpeakerVerifier

    store = SpeakerProfileStore(profile_dir())
    profiles = store.list_profiles()
    if not profiles:
        raise RuntimeError("还没有登记声纹。")
    extractor = _extractor()
    _VERIFIER = SpeakerVerifier(
        store,
        extractor,
        threshold=threshold(),
        min_speech_ms=min_speech_ms(),
    )
    return _VERIFIER


def extract_embedding(pcm: bytes):
    audio = pcm_s16le_to_float(pcm)
    return _extractor().extract(audio, 16000)


def verify_pcm(pcm: bytes) -> dict[str, Any]:
    mode = verify_mode()
    payload: dict[str, Any] = {
        "mode": mode,
        "accepted": True,
        "score": 0.0,
        "speaker_id": None,
        "display_name": None,
        "reason": "off",
        "duration_ms": 0,
        "matched": False,
    }
    if mode == "off":
        return payload
    if not pcm:
        payload.update({"accepted": mode != "enforce", "reason": "empty_audio", "matched": False})
        return payload
    try:
        profiles = list_profiles()
    except Exception as exc:
        payload.update(
            {
                "accepted": mode != "enforce",
                "reason": "verifier_error",
                "message": str(exc),
                "matched": False,
            }
        )
        return payload
    if not profiles:
        payload.update({"accepted": mode != "enforce", "reason": "no_profiles", "matched": False})
        return payload
    try:
        decision = _verifier().verify(pcm_s16le_to_float(pcm), 16000)
        payload = decision.event_payload(mode=mode)
        payload["matched"] = bool(decision.accepted)
        if mode == "shadow":
            payload["accepted"] = True
            payload["enforce_would_accept"] = bool(decision.accepted)
    except Exception as exc:
        payload.update(
            {
                "accepted": mode != "enforce",
                "reason": "verifier_error",
                "message": str(exc),
                "matched": False,
            }
        )
    return payload


def score_pcm(pcm: bytes) -> dict[str, Any]:
    """Always score against profiles. Used by the dashboard probe, ignores mode."""
    if not pcm:
        return {
            "mode": "probe",
            "accepted": False,
            "score": 0.0,
            "speaker_id": None,
            "display_name": None,
            "reason": "empty_audio",
            "duration_ms": 0,
            "matched": False,
        }
    decision = _verifier().verify(pcm_s16le_to_float(pcm), 16000)
    payload = decision.event_payload(mode="probe")
    payload["matched"] = bool(decision.accepted)
    return payload


def allows_host(decision: dict[str, Any]) -> bool:
    if verify_mode() != "enforce":
        return True
    return bool(decision.get("accepted"))


def log_decision(decision: dict[str, Any]) -> None:
    print(
        "wake speaker "
        f"mode={decision.get('mode')} accepted={decision.get('accepted')} "
        f"matched={decision.get('matched')} speaker={decision.get('display_name') or '-'} "
        f"score={decision.get('score')} reason={decision.get('reason')}",
        flush=True,
    )


def _emit_decision(decision: dict[str, Any], on_event) -> None:
    log_decision(decision)
    if on_event:
        on_event({"type": "speaker", **decision})


async def run_gated_listen(session, stream, **kwargs):
    """Send command PCM to the host, gated by verify mode.

    Returns (SayResult | None, decision). None means enforce rejected and
    nothing was sent to the host.
    """
    mode = verify_mode()
    on_event = kwargs.get("on_event")
    if mode == "off":
        result = await session.listen_stream(stream, **kwargs)
        return result, {"mode": "off", "accepted": True, "reason": "off", "matched": False}

    if mode == "shadow":
        parts: list[bytes] = []

        async def live():
            async for part in stream:
                if part:
                    parts.append(part)
                    yield part

        result = await session.listen_stream(live(), **kwargs)
        decision = verify_pcm(b"".join(parts))
        _emit_decision(decision, on_event)
        return result, decision

    parts = []
    async for part in stream:
        if part:
            parts.append(part)
    pcm = b"".join(parts)
    decision = verify_pcm(pcm)
    _emit_decision(decision, on_event)
    if not allows_host(decision):
        return None, decision
    result = await session.listen_pcm(pcm, **kwargs)
    return result, decision


def enrollment_public() -> dict[str, Any] | None:
    if _ENROLL is None:
        return None
    count = len(_ENROLL["embeddings"])
    return {
        "session_id": _ENROLL["session_id"],
        "display_name": _ENROLL["display_name"],
        "sample_count": count,
        "required_samples": REQUIRED_SAMPLES,
        "prompt": ENROLL_PROMPTS[min(count, len(ENROLL_PROMPTS) - 1)],
        "complete": count >= REQUIRED_SAMPLES,
    }


def start_enrollment(display_name: str) -> dict[str, Any]:
    global _ENROLL
    if _ENROLL is not None:
        raise RuntimeError("已有声纹登记正在进行，请先完成或取消。")
    _ensure_scripts_path()
    from voice_speaker_verifier import SpeakerProfileStore

    name = SpeakerProfileStore(profile_dir()).validate_name(display_name)
    if not model_path().is_file():
        raise RuntimeError(f"找不到声纹模型：{model_path()}")
    _extractor()
    _ENROLL = {
        "session_id": secrets.token_hex(16),
        "display_name": name,
        "embeddings": [],
    }
    public = enrollment_public()
    assert public is not None
    return public


def add_enrollment_pcm(session_id: str, pcm: bytes) -> dict[str, Any]:
    if _ENROLL is None or _ENROLL["session_id"] != session_id:
        raise KeyError("声纹登记会话不存在")
    if len(_ENROLL["embeddings"]) >= REQUIRED_SAMPLES:
        raise RuntimeError("样本已经足够，请保存声纹。")
    audio_ms = int(len(pcm) / 32) if pcm else 0
    public = enrollment_public()
    assert public is not None
    try:
        embedding = extract_embedding(pcm)
    except Exception as exc:
        public["sample"] = {"accepted": False, "reason": str(exc), "audio_ms": audio_ms}
        return public
    _ENROLL["embeddings"].append(embedding)
    public = enrollment_public()
    assert public is not None
    public["sample"] = {"accepted": True, "audio_ms": audio_ms}
    return public


def commit_enrollment(session_id: str) -> dict[str, Any]:
    global _ENROLL
    if _ENROLL is None or _ENROLL["session_id"] != session_id:
        raise KeyError("声纹登记会话不存在")
    if len(_ENROLL["embeddings"]) < REQUIRED_SAMPLES:
        raise RuntimeError(f"需要 {REQUIRED_SAMPLES} 条有效样本")
    profile = save_embeddings(_ENROLL["display_name"], list(_ENROLL["embeddings"]))
    _ENROLL = None
    return {"ok": True, "profile": profile}


def cancel_enrollment(session_id: str | None = None) -> bool:
    global _ENROLL
    if _ENROLL is None:
        return False
    if session_id and _ENROLL["session_id"] != session_id:
        raise KeyError("声纹登记会话不存在")
    _ENROLL = None
    return True


def capture_robot_pcm(
    *,
    device: str | None = None,
    start_timeout: float = 8.0,
    max_duration: float = 8.0,
    min_duration: float = 1.0,
    silence_sec: float = 0.85,
    speech_rms: float = 0.012,
    skip_sec: float = 0.15,
) -> tuple[bytes, int]:
    """Record one utterance from the robot microphone (arecord)."""
    chosen = (device or os.environ.get("ARECORD_DEVICE") or "default").strip() or "default"
    cmd = [
        "arecord",
        "-q",
        "-D",
        chosen,
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
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError("本机没有 arecord。请安装 alsa-utils。") from exc
    chunk = 3200
    skip_bytes = max(0, int(skip_sec * 16000 * 2))
    preroll = max(0, int(0.3 * 16000 * 2))
    skipped = 0
    pending: list[bytes] = []
    pending_n = 0
    voiced = False
    silence = 0.0
    started = time.monotonic()
    try:
        while time.monotonic() - started < max_duration:
            assert proc.stdout is not None
            data = proc.stdout.read(chunk)
            if not data:
                err = b""
                if proc.stderr:
                    try:
                        err = proc.stderr.read() or b""
                    except Exception:
                        err = b""
                detail = err.decode("utf-8", "replace").strip()
                raise RuntimeError("麦克风读不到数据。" + (f" arecord：{detail}" if detail else ""))
            if skipped < skip_bytes:
                take = min(len(data), skip_bytes - skipped)
                skipped += take
                data = data[take:]
                if not data:
                    continue
            if not voiced:
                pending.append(data)
                pending_n += len(data)
                while preroll and pending_n > preroll and len(pending) > 1:
                    dropped = pending.pop(0)
                    pending_n -= len(dropped)
                if pcm_rms(data) >= speech_rms:
                    voiced = True
                    silence = 0.0
                elif time.monotonic() - started >= start_timeout:
                    break
                continue
            pending.append(data)
            pending_n += len(data)
            energy = pcm_rms(data)
            dt = (len(data) / 2) / 16000.0
            if energy >= speech_rms:
                silence = 0.0
            else:
                silence += dt
            if silence >= silence_sec:
                break
        pcm = b"".join(pending) if voiced else b""
        audio_ms = int(len(pcm) / 32) if pcm else 0
        if audio_ms < int(min_duration * 1000):
            raise RuntimeError("没有听到足够长的语音，请靠近麦克风再说。")
        return pcm, audio_ms
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.5)
            except Exception:
                proc.kill()
                proc.wait()


def capture_locked(**kwargs) -> tuple[bytes, int]:
    with _CAPTURE_LOCK:
        return capture_robot_pcm(**kwargs)
