from __future__ import annotations

"""Cloud ASR placeholder. Set ASR_PROVIDER=online and ASR_ONLINE_URL later.

Keep the same transcribe(pcm, sample_rate) contract as LocalParaformerASR
so the rest of the server does not change when you swap engines.
"""

from server import config


class OnlineASR:
    name = "online"

    def transcribe(self, pcm: bytes, sample_rate: int = 16000) -> str:
        url = config.asr_online_url()
        if not url:
            raise RuntimeError(
                "ASR_PROVIDER=online but ASR_ONLINE_URL is empty. "
                "Use ASR_PROVIDER=sherpa_local until a cloud ASR is configured."
            )
        raise RuntimeError(
            f"Online ASR endpoint is reserved ({url}) but not wired yet. "
            "Keep ASR_PROVIDER=sherpa_local for now."
        )
