from __future__ import annotations

"""Doubao / Volcengine TTS.

V3 HTTP unidirectional (new console):
  https://openspeech.bytedance.com/api/v3/tts/unidirectional
  Docs: https://docs.volcengine.com/docs/6561/2532486

V1 HTTP (legacy AppID + Access Token):
  https://openspeech.bytedance.com/api/v1/tts
"""

import base64
import json
import uuid

import requests

from server import config


class DoubaoTTS:
    name = "doubao"

    def __init__(self):
        self.sample_rate = config.tts_sample_rate() or 16000

    def synthesize(self, text: str) -> bytes:
        return b"".join(self.synthesize_iter(text))

    def synthesize_iter(self, text: str):
        if not isinstance(text, str) or not text.strip():
            return
        cleaned = text.strip()
        api_key = config.doubao_api_key()
        appid = config.doubao_appid()
        token = config.doubao_access_token()
        if api_key or (appid and token):
            yielded = False
            try:
                for chunk in self._synth_v3_iter(cleaned):
                    yielded = True
                    yield chunk
                return
            except Exception as exc:
                if yielded:
                    print(f"⚠️ Doubao TTS v3 stopped after audio ({exc})", flush=True)
                    return
                print(f"⚠️ Doubao TTS v3 failed ({exc}); trying v1", flush=True)
        if appid and token:
            pcm = self._synth_v1(cleaned)
            if pcm:
                yield pcm
            return
        print("⚠️ Doubao TTS is not configured; set DOUBAO_TTS_API_KEY or APPID+ACCESS_TOKEN")

    def _headers_v3(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "X-Api-Resource-Id": config.doubao_resource_id(),
            "X-Api-Request-Id": str(uuid.uuid4()),
        }
        api_key = config.doubao_api_key()
        if api_key:
            headers["X-Api-Key"] = api_key
        else:
            headers["X-Api-App-Id"] = config.doubao_appid()
            headers["X-Api-Access-Key"] = config.doubao_access_token()
        return headers

    def _v3_payload(self, text: str) -> dict:
        voice = config.doubao_voice()
        if not voice:
            raise RuntimeError("DOUBAO_TTS_VOICE is empty; fill the speaker id from the console")
        return {
            "user": {"uid": "voice_hub"},
            "req_params": {
                "text": text,
                "speaker": voice,
                "audio_params": {
                    "format": "pcm",
                    "sample_rate": self.sample_rate,
                },
            },
        }

    def _synth_v3(self, text: str) -> bytes:
        return b"".join(self._synth_v3_iter(text))

    def _synth_v3_iter(self, text: str):
        url = config.doubao_v3_url()
        response = requests.post(
            url,
            headers=self._headers_v3(),
            json=self._v3_payload(text),
            timeout=30,
            stream=True,
        )
        try:
            if response.status_code != 200:
                raise RuntimeError(f"Doubao TTS HTTP {response.status_code}: {response.text[:200]}")
            got = False
            for raw in response.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if not line or line.startswith("event:"):
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                code = payload.get("code")
                if code not in (None, 0, 20000000):
                    raise RuntimeError(f"Doubao TTS error {code}: {payload.get('message')}")
                data = payload.get("data")
                if isinstance(data, str) and data:
                    try:
                        chunk = base64.b64decode(data)
                    except Exception:
                        continue
                    if chunk:
                        got = True
                        yield chunk
            if not got:
                raise RuntimeError("Doubao TTS v3 returned no audio")
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()

    def _synth_v1(self, text: str) -> bytes:
        token = config.doubao_access_token()
        voice = config.doubao_voice()
        if not voice:
            raise RuntimeError("DOUBAO_TTS_VOICE is empty; fill the speaker id from the console")
        reqid = str(uuid.uuid4())
        body = {
            "app": {
                "appid": config.doubao_appid(),
                "token": token,
                "cluster": config.doubao_cluster(),
            },
            "user": {"uid": "voice_hub"},
            "audio": {
                "voice_type": voice,
                "encoding": "pcm",
                "rate": self.sample_rate,
                "speed_ratio": 1.0,
                "volume_ratio": 1.0,
                "pitch_ratio": 1.0,
            },
            "request": {
                "reqid": reqid,
                "text": text,
                "text_type": "plain",
                "operation": "query",
            },
        }
        response = requests.post(
            config.doubao_v1_url(),
            headers={
                "Authorization": f"Bearer;{token}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=30,
        )
        payload = response.json()
        code = payload.get("code")
        if code not in (None, 0, 3000):
            raise RuntimeError(f"Doubao TTS v1 error {code}: {payload.get('message')}")
        data = payload.get("data")
        if not data:
            raise RuntimeError("Doubao TTS v1 returned no audio")
        return base64.b64decode(data)
