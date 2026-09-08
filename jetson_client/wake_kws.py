"""Offline keyword spotting for 小溪小溪. Uses an existing sherpa-onnx KWS model."""
from __future__ import annotations

import os
import sys
from pathlib import Path

WAKE_WORD = "小溪小溪"
KEYWORD_TOKENS = ["x", "iǎo", "x", "ī", "x", "iǎo", "x", "ī"]
DEFAULT_MODEL_DIR = (
    "/home/ning/model/lyz/src/voice_control_online/models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
)
KEYWORDS_FILE = Path(__file__).resolve().parent / "config" / "keywords_xiaoxi.txt"


def model_dir() -> str:
    raw = (os.environ.get("VOICE_KWS_MODEL_DIR") or "").strip()
    if raw:
        return raw
    home_model = Path.home() / "VoiceOnline" / "models" / "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
    if (home_model / "encoder.onnx").is_file():
        return str(home_model)
    return DEFAULT_MODEL_DIR


def _python_tag() -> str:
    return f"python{sys.version_info.major}.{sys.version_info.minor}"


def _path_matches_this_python(path: Path) -> bool:
    blob = str(path)
    found = [piece for piece in blob.split("/") if piece.startswith("python3.")]
    if not found:
        return True
    return _python_tag() in found


def _sherpa_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    extra = (os.environ.get("VOICE_KWS_PYTHONPATH") or "").strip()
    if extra:
        dirs.append(Path(extra))
    home_sp = Path.home() / "VoiceOnline" / ".venv_online" / "lib"
    dirs.extend(sorted(home_sp.glob(f"{_python_tag()}/site-packages")))
    try:
        import site

        user_site = site.getusersitepackages()
        if user_site:
            dirs.append(Path(user_site))
    except Exception:
        pass
    dirs.append(Path.home() / ".local" / "lib" / _python_tag() / "site-packages")
    seen: set[str] = set()
    unique: list[Path] = []
    for path in dirs:
        if not _path_matches_this_python(path):
            continue
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _import_sherpa():
    try:
        import sherpa_onnx

        return sherpa_onnx
    except ImportError:
        pass
    for path in _sherpa_search_dirs():
        if not (path / "sherpa_onnx").is_dir():
            continue
        inserted = str(path)
        if inserted not in sys.path:
            sys.path.insert(0, inserted)
        try:
            import sherpa_onnx

            return sherpa_onnx
        except ImportError:
            continue
    raise ImportError("本机没有 sherpa-onnx。唤醒需要在设备 venv 安装 sherpa-onnx==1.13.3。")


def wake_word() -> str:
    return (os.environ.get("WAKE_WORD") or WAKE_WORD).replace(" ", "")


def verify_keyword_tokens(tokens_path: str | Path, phrase: str = WAKE_WORD) -> list[str]:
    """Check the official KWS tokens.txt can represent 小溪小溪 (ppinyin)."""
    path = Path(tokens_path)
    if not path.is_file():
        raise RuntimeError(f"找不到 KWS 词表：{path}")
    table = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        piece = line.strip().split()
        if piece:
            table.add(piece[0])
    missing = [tok for tok in KEYWORD_TOKENS if tok not in table]
    if missing:
        raise RuntimeError(f"词表不支持「{phrase}」，缺 token：{missing}")
    return list(KEYWORD_TOKENS)


def keywords_score() -> float:
    raw = os.environ.get("WAKE_KWS_SCORE", "2.0")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 2.0
    return max(0.0, min(10.0, value))


def keywords_threshold() -> float:
    raw = os.environ.get("WAKE_KWS_THRESHOLD", "0.25")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.25
    return max(0.0, min(1.0, value))


def keyword_line(phrase: str = WAKE_WORD) -> str:
    return (
        " ".join(KEYWORD_TOKENS)
        + f" :{keywords_score():g} #{keywords_threshold():g} @{phrase.replace(' ', '')}"
    )


def pcm_s16le_to_float(pcm: bytes):
    import numpy as np

    if not pcm:
        return np.zeros(0, dtype=np.float32)
    if len(pcm) % 2:
        pcm = pcm[:-1]
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    samples /= 32768.0
    return samples


def pcm_rms(pcm: bytes) -> float:
    samples = pcm_s16le_to_float(pcm)
    if samples.size == 0:
        return 0.0
    import numpy as np

    return float(np.sqrt(np.mean(np.square(samples))))


def boost_for_kws(samples, *, target: float = 0.05, max_gain: float = 6.0, floor: float = 0.004):
    """Raise quiet speech toward `target` RMS. Leave silence and loud speech alone."""
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0:
        return audio, 0.0
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < floor or rms >= target:
        return audio, rms
    gain = min(float(max_gain), float(target) / rms)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32), rms


class KwsCooldown:
    """Drop captured PCM after TTS so speaker echo cannot re-trigger wake."""

    def __init__(self, sec: float | None = None):
        raw = os.environ.get("WAKE_COOLDOWN_SEC", "1.2") if sec is None else sec
        try:
            self.sec = float(raw)
        except (TypeError, ValueError):
            self.sec = 1.2
        if self.sec < 0:
            self.sec = 0.0
        self.left = 0

    def arm(self, extra_sec: float = 0.0) -> None:
        try:
            extra = float(extra_sec)
        except (TypeError, ValueError):
            extra = 0.0
        if extra < 0:
            extra = 0.0
        self.left = int((self.sec + extra) * 16000 * 2)

    def allow(self, nbytes: int) -> bool:
        if self.left <= 0:
            return True
        self.left -= max(0, int(nbytes))
        return self.left <= 0


class KeywordSpotter:
    def __init__(self, model_path: str | None = None, word: str | None = None, num_threads: int = 2):
        try:
            sherpa_onnx = _import_sherpa()
        except ImportError as exc:
            raise RuntimeError("本机没有 sherpa-onnx。唤醒需要在设备 venv 安装 sherpa-onnx==1.13.3。") from exc
        base = Path(model_path or model_dir())
        files = {
            "tokens": base / "tokens.txt",
            "encoder": base / "encoder.onnx",
            "decoder": base / "decoder.onnx",
            "joiner": base / "joiner.onnx",
        }
        missing = [name for name, path in files.items() if not path.is_file()]
        if missing:
            raise RuntimeError("找不到唤醒模型文件：" + ", ".join(missing) + f"（目录 {base}）")
        verify_keyword_tokens(files["tokens"], word or wake_word())
        keywords_path = KEYWORDS_FILE
        self.word = (word or wake_word()).replace(" ", "")
        expected = keyword_line(self.word)
        if not keywords_path.is_file() or keywords_path.read_text(encoding="utf-8").strip() != expected:
            keywords_path.parent.mkdir(parents=True, exist_ok=True)
            keywords_path.write_text(expected + "\n", encoding="utf-8")
        self.model_path = str(base)
        score = keywords_score()
        threshold = keywords_threshold()
        self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(files["tokens"]),
            encoder=str(files["encoder"]),
            decoder=str(files["decoder"]),
            joiner=str(files["joiner"]),
            keywords_file=str(keywords_path),
            num_threads=int(num_threads),
            max_active_paths=4,
            num_trailing_blanks=2,
            keywords_score=score,
            keywords_threshold=threshold,
            sample_rate=16000,
            feature_dim=80,
            provider="cpu",
        )
        self._stream = self._spotter.create_stream()
        self._voiced = False
        self._tail_pending = False
        self._quiet_samples = 0
        self._need_quiet = int(0.28 * 16000)
        self._tail_samples = int(0.40 * 16000)
        self._speech_rms = 0.010
        self._quiet_rms = 0.006

    def reset(self) -> None:
        self._spotter.reset_stream(self._stream)
        self._voiced = False
        self._tail_pending = False
        self._quiet_samples = 0

    def _decode(self) -> bool:
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            keyword = (self._spotter.get_result(self._stream) or "").replace(" ", "")
            if keyword and (keyword == self.word or self.word in keyword):
                self.reset()
                return True
        return False

    def _feed(self, samples, sample_rate: int) -> bool:
        if samples is None or getattr(samples, "size", 0) == 0:
            return False
        self._stream.accept_waveform(sample_rate, samples)
        return self._decode()

    def accept_pcm(self, pcm: bytes, sample_rate: int = 16000) -> bool:
        if sample_rate != 16000:
            raise ValueError("wake-word audio must be 16 kHz")
        samples = pcm_s16le_to_float(pcm)
        samples, _raw_rms = boost_for_kws(samples)
        if self._feed(samples, sample_rate):
            return True
        rms = 0.0
        if getattr(samples, "size", 0):
            import numpy as np

            rms = float(np.sqrt(np.mean(np.square(samples))))
        if rms >= self._speech_rms:
            self._voiced = True
            self._quiet_samples = 0
            self._tail_pending = True
        elif self._voiced and rms < self._quiet_rms:
            self._quiet_samples += int(getattr(samples, "size", 0) or 0)
            if self._tail_pending and self._quiet_samples >= self._need_quiet:
                import numpy as np

                self._tail_pending = False
                tail = np.zeros(self._tail_samples, dtype=np.float32)
                if self._feed(tail, sample_rate):
                    return True
        return False


def describe_kws() -> dict[str, str | bool]:
    base = Path(model_dir())
    error = ""
    ready = (base / "encoder.onnx").is_file() and (base / "tokens.txt").is_file()
    try:
        _import_sherpa()
    except ImportError:
        ready = False
        error = "本机没有 sherpa-onnx。"
    if ready:
        try:
            verify_keyword_tokens(base / "tokens.txt")
        except RuntimeError as exc:
            ready = False
            error = str(exc)
    elif not error:
        error = f"找不到唤醒模型目录：{base}"
    return {
        "wake_word": wake_word(),
        "kws_model": str(base),
        "kws_ready": ready,
        "kws_error": error,
    }
