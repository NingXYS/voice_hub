import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server import config
from server.providers.asr_doubao import DoubaoASR, encode_full_client_request, encode_full_server_response, parse_response
from server.providers.asr_online import OnlineASR
from server.providers.llm_openai import OpenAICompatLLM
from server.providers.search_tavily import web_search
from server.providers.tts_doubao import DoubaoTTS


class ProviderContractTests(unittest.TestCase):
    def test_dialog_speaker_rejects_seed_tts_voice(self):
        with mock.patch("server.config.env", side_effect=lambda name, default="": {
            "DOUBAO_TTS_VOICE": "zh_female_vv_uranus_bigtts",
        }.get(name, default)):
            self.assertIn("jupiter", config.dialog_speaker())

    def test_dialog_reuses_doubao_tts_api_key(self):
        with mock.patch("server.config.env", side_effect=lambda name, default="": {
            "DIALOG_PROVIDER": "doubao",
            "DOUBAO_TTS_API_KEY": "ba0testkey",
        }.get(name, default)):
            self.assertTrue(config.dialog_configured())
            self.assertEqual(config.dialog_api_key(), "ba0testkey")

    def test_online_asr_refuses_until_url_configured(self):
        asr = OnlineASR()
        with mock.patch("server.config.asr_online_url", return_value=""):
            with self.assertRaises(RuntimeError):
                asr.transcribe(b"\x00\x00")

    def test_doubao_asr_without_key_is_speakable(self):
        asr = DoubaoASR()
        with mock.patch("server.config.doubao_asr_api_key", return_value=""), \
            mock.patch("server.config.doubao_appid", return_value=""), \
            mock.patch("server.config.doubao_access_token", return_value=""):
            with self.assertRaises(RuntimeError) as ctx:
                asr.transcribe(b"\x00\x00" * 160)
        self.assertIn("DOUBAO_ASR_API_KEY", str(ctx.exception))

    def test_sauc_response_parses_final_text(self):
        frame = encode_full_server_response(
            {"code": 0, "is_last_package": True, "result": {"text": "北京今天晴"}},
            last=True,
        )
        parsed = parse_response(frame)
        self.assertEqual(parsed["text"], "北京今天晴")
        self.assertTrue(parsed["is_last"])
        self.assertFalse(parsed["error"])
        header = encode_full_client_request({"user": {"uid": "t"}})
        self.assertGreater(len(header), 8)
        wrapped = encode_full_server_response(
            {"payload_msg": {"code": 0, "result": {"utterances": [{"text": "你好", "definite": True}]}}},
            last=False,
        )
        parsed_u = parse_response(wrapped)
        self.assertEqual(parsed_u["text"], "你好")
        self.assertFalse(parsed_u["is_last"])

    def test_doubao_asr_transcribe_async_reads_final_text(self):
        import asyncio

        class FakeWS:
            def __init__(self, replies):
                self.sent = []
                self._replies = list(replies)

            async def send(self, data):
                self.sent.append(data)

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._replies:
                    raise StopAsyncIteration
                return self._replies.pop(0)

            async def close(self):
                return None

        ack = encode_full_server_response({"code": 0}, last=False)
        frame = encode_full_server_response(
            {"code": 0, "is_last_package": True, "result": {"text": "你有几个关节"}},
            last=True,
        )
        fake = FakeWS([ack, frame])

        async def scenario():
            asr = DoubaoASR()
            with mock.patch("server.config.doubao_asr_api_key", return_value="key"), \
                mock.patch("server.config.doubao_asr_resource_id", return_value="volc.seedasr.sauc.duration"), \
                mock.patch("server.config.doubao_asr_url", return_value="wss://example.test/asr"), \
                mock.patch("server.providers.asr_doubao._connect", return_value=fake):
                return await asr.transcribe_async(b"\x00\x00" * 320)

        text = asyncio.run(scenario())
        self.assertEqual(text, "你有几个关节")
        self.assertGreaterEqual(len(fake.sent), 2)
        from server.providers.asr_doubao import encode_audio, FLAG_POS_SEQ
        import struct
        audio_frames = [item for item in fake.sent if len(item) >= 8 and (item[1] >> 4) == 0x02]
        self.assertTrue(audio_frames)
        seq = struct.unpack(">i", audio_frames[0][4:8])[0]
        self.assertEqual(seq, 2)

    def test_llm_without_key_returns_speakable_message(self):
        with mock.patch("server.config.llm_api_key", return_value=""):
            llm = OpenAICompatLLM(session=mock.Mock())
            text = llm.chat("你好")
        self.assertIn("百炼", text)
        llm.session.post.assert_not_called()

    def test_llm_without_model_name_asks_to_fill_it(self):
        with mock.patch("server.config.llm_api_key", return_value="sk-test"), \
            mock.patch("server.config.llm_model", return_value=""):
            llm = OpenAICompatLLM(session=mock.Mock())
            text = llm.chat("你好")
        self.assertIn("CHAT_LLM_MODEL", text)
        llm.session.post.assert_not_called()

    def test_llm_v4_base_url_does_not_insert_v1(self):
        llm = OpenAICompatLLM(session=mock.Mock())
        with mock.patch(
            "server.config.llm_base_url",
            return_value="https://open.bigmodel.cn/api/paas/v4",
        ):
            url = llm._url()
        self.assertEqual(url, "https://open.bigmodel.cn/api/paas/v4/chat/completions")
        self.assertNotIn("/v4/v1/", url)

    def test_dashscope_base_url_uses_v1_chat_completions(self):
        llm = OpenAICompatLLM(session=mock.Mock())
        with mock.patch(
            "server.config.llm_base_url",
            return_value="https://dashscope.aliyuncs.com/compatible-mode/v1",
        ):
            url = llm._url()
        self.assertEqual(
            url,
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        )

    def test_split_speakable_cuts_first_comma(self):
        from server.providers.llm_openai import split_speakable

        parts = split_speakable("左右两条手臂，各六个关节，一共十二个关节。")
        self.assertGreaterEqual(len(parts), 2)
        self.assertTrue(parts[0].endswith("，"))

    def test_llm_unauthorized_is_speakable(self):
        response = mock.Mock(status_code=401)
        response.iter_lines.return_value = []
        session = mock.Mock()
        session.post.return_value = response
        with mock.patch("server.config.llm_api_key", return_value="sk-bad"), \
            mock.patch("server.config.llm_model", return_value="qwen-flash"):
            llm = OpenAICompatLLM(session=session)
            text = llm.chat("你好")
        self.assertIn("密钥", text)
        self.assertIn("401", llm.last_error)

    def test_llm_disables_thinking(self):
        import json as json_mod

        response = mock.Mock(status_code=200)
        response.iter_lines.return_value = [
            "data: " + json_mod.dumps({"choices": [{"delta": {"content": "你好。"}}]}),
            "data: [DONE]",
        ]
        session = mock.Mock()
        session.post.return_value = response
        with mock.patch("server.config.llm_api_key", return_value="sk-test"), \
            mock.patch("server.config.llm_model", return_value="qwen-flash"):
            llm = OpenAICompatLLM(session=session)
            llm.chat("你好")
        body = session.post.call_args.kwargs["json"]
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertIs(body["enable_thinking"], False)
        self.assertEqual(body["extra_body"], {"enable_thinking": False})

    def test_search_without_key_is_speakable(self):
        with mock.patch("server.config.search_provider", return_value="metaso"), \
            mock.patch("server.config.search_api_key", return_value=""):
            self.assertIn("METASO_API_KEY", web_search("北京天气"))

    def test_metaso_formats_summary(self):
        response = mock.Mock()
        response.status_code = 200
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "summary": "北京今天晴。",
            "webpages": [{"title": "天气", "snippet": "气温适宜", "link": "https://example.com"}],
        }
        with mock.patch("server.config.search_provider", return_value="metaso"), \
            mock.patch("server.config.search_api_key", return_value="mk-test"), \
            mock.patch("server.providers.search_tavily.requests.post", return_value=response) as post:
            text = web_search("北京天气")
        self.assertIn("北京今天晴", text)
        self.assertIn("气温适宜", text)
        self.assertEqual(post.call_args.kwargs["json"]["q"], "北京天气")
        self.assertIs(post.call_args.kwargs["json"]["includeSummary"], True)
        self.assertEqual(post.call_args.kwargs["json"]["size"], 5)
        self.assertIn("Bearer mk-test", post.call_args.kwargs["headers"]["Authorization"])

    def test_metaso_bad_key_is_speakable(self):
        response = mock.Mock()
        response.status_code = 401
        response.json.return_value = {"errCode": 401, "errMsg": "认证失败"}
        with mock.patch("server.config.search_provider", return_value="metaso"), \
            mock.patch("server.config.search_api_key", return_value="mk-bad"), \
            mock.patch("server.providers.search_tavily.requests.post", return_value=response):
            text = web_search("北京天气")
        self.assertIn("密钥无效", text)

    def test_metaso_quota_exhausted_is_speakable(self):
        response = mock.Mock()
        response.status_code = 200
        response.json.return_value = {"errCode": 4002, "errMsg": "搜索次数已耗尽"}
        with mock.patch("server.config.search_provider", return_value="metaso"), \
            mock.patch("server.config.search_api_key", return_value="mk-test"), \
            mock.patch("server.providers.search_tavily.requests.post", return_value=response):
            text = web_search("北京天气")
        self.assertIn("次数已用完", text)

    def test_metaso_tls_error_is_speakable(self):
        import requests as req

        with mock.patch("server.config.search_provider", return_value="metaso"), \
            mock.patch("server.config.search_api_key", return_value="mk-test"), \
            mock.patch(
                "server.providers.search_tavily.requests.post",
                side_effect=req.exceptions.SSLError("UNEXPECTED_EOF"),
            ):
            text = web_search("北京天气")
        self.assertIn("连不上", text)

    def test_doubao_v3_yields_chunks(self):
        import base64 as b64mod
        import json as json_mod

        response = mock.Mock(status_code=200)
        response.iter_lines.return_value = [
            "data: " + json_mod.dumps({"code": 0, "data": b64mod.b64encode(b"AAA").decode()}),
            "data: " + json_mod.dumps({"code": 0, "data": b64mod.b64encode(b"BBB").decode()}),
        ]
        tts = DoubaoTTS()
        with mock.patch("server.config.doubao_api_key", return_value="key"), \
            mock.patch("server.config.doubao_voice", return_value="zh_female_demo"), \
            mock.patch("server.config.doubao_v3_url", return_value="https://example.test/v3"), \
            mock.patch("server.providers.tts_doubao.requests.post", return_value=response):
            chunks = list(tts.synthesize_iter("你好"))
        self.assertEqual(chunks, [b"AAA", b"BBB"])

    def test_doubao_v1_decodes_pcm(self):
        audio = b"PCMDATA"
        payload = {"code": 3000, "message": "ok", "data": __import__("base64").b64encode(audio).decode()}
        response = mock.Mock(status_code=200)
        response.json.return_value = payload
        response.iter_lines.return_value = []
        tts = DoubaoTTS()
        with mock.patch("server.config.doubao_api_key", return_value=""), \
            mock.patch("server.config.doubao_appid", return_value="app"), \
            mock.patch("server.config.doubao_access_token", return_value="token"), \
            mock.patch("server.config.doubao_voice", return_value="zh_female_demo"), \
            mock.patch("server.config.doubao_cluster", return_value="volcano_tts"), \
            mock.patch("server.config.doubao_v1_url", return_value="https://example.test/v1"), \
            mock.patch("server.providers.tts_doubao.requests.post", return_value=response) as post:
            pcm = tts.synthesize("你好")
        self.assertEqual(pcm, audio)
        headers = post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer;token")
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["audio"]["encoding"], "pcm")
        self.assertEqual(body["request"]["operation"], "query")

    def test_doubao_without_credentials_returns_empty(self):
        tts = DoubaoTTS()
        with mock.patch("server.config.doubao_api_key", return_value=""), \
            mock.patch("server.config.doubao_appid", return_value=""), \
            mock.patch("server.config.doubao_access_token", return_value=""):
            self.assertEqual(tts.synthesize("你好"), b"")
