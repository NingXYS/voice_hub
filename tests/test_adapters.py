import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.adapter_loader import load_adapter
from server.config import adapter_dir
from server.pipeline import Pipeline


class FakeLLM:
    configured = True

    def chat(self, user_text, on_sentence=None, extra_system="", enable_search=False, search_fn=None, history=None):
        if on_sentence:
            on_sentence("闲聊回复。")
        return "闲聊回复。"


class AdapterPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tianlian = load_adapter("tianlian_dual", adapter_dir())
        self.demo = load_adapter("demo_single", adapter_dir())
        self.pipeline = Pipeline(llm=FakeLLM(), search_fn=lambda q: "晴，气温适宜。")

    def test_knowledge_skips_llm(self):
        result = self.pipeline.handle_text("你有几个关节", self.tianlian)
        self.assertEqual(result.route, "knowledge")
        self.assertIn("关节", result.reply)
        self.assertEqual(result.commands, [])
        self.assertGreaterEqual(len(result.sentences), 2)
        self.assertTrue(result.sentences[0].endswith("，"))

    def test_intro_is_dual_arm_not_single_massage_bot(self):
        result = self.pipeline.handle_text("你是谁", self.tianlian)
        self.assertEqual(result.route, "knowledge")
        self.assertIn("双臂", result.reply)
        self.assertIn("按摩", result.reply)
        self.assertNotIn("只有一条手臂", result.reply)
        self.assertNotIn("单臂按摩", result.reply)

    def test_massage_faq_does_not_emit_commands(self):
        result = self.pipeline.handle_text("怎么开始按摩", self.tianlian)
        self.assertEqual(result.route, "knowledge")
        self.assertEqual(result.commands, [])
        self.assertIn("确认开始", result.reply)

    def test_massage_confirm_does_not_claim_dry_run_in_speech(self):
        result = self.pipeline.handle_text("确认开始", self.tianlian)
        self.assertEqual(result.route, "local_action")
        self.assertEqual(result.commands[0]["op"], "confirm_start")
        self.assertNotIn("演练", result.reply)
        self.assertIn("确认", result.reply)

    def test_massage_teach_emits_dry_run_command(self):
        result = self.pipeline.handle_text("开始记录点位", self.tianlian)
        self.assertEqual(result.route, "local_action")
        self.assertEqual(result.commands[0]["action"], "massage")
        self.assertEqual(result.commands[0]["op"], "start_teaching")
        self.assertIn("点位", result.reply)

    def test_massage_start_needs_part_and_force(self):
        missing = self.pipeline.handle_text("开始按摩", self.tianlian)
        self.assertEqual(missing.route, "clarify")
        self.assertEqual(missing.commands, [])
        started = self.pipeline.handle_text("开始按摩左肩，轻一点", self.tianlian)
        self.assertEqual(started.route, "local_action")
        self.assertEqual(started.commands[0]["op"], "start")
        self.assertEqual(started.commands[0]["parts"], ["left-shoulder"])
        self.assertEqual(started.commands[0]["force_n"], 30)
        self.assertIn("确认开始", started.reply)

    def test_massage_does_not_swallow_cartesian(self):
        result = self.pipeline.handle_text("左臂向上10厘米", self.tianlian)
        self.assertEqual(result.commands[0]["action"], "move_cartesian")

    def test_tianlian_local_action_emits_tianlian_json(self):
        result = self.pipeline.handle_text("左臂向上10厘米", self.tianlian)
        self.assertEqual(result.route, "local_action")
        self.assertEqual(result.commands[0]["action"], "move_cartesian")
        self.assertEqual(result.commands[0]["arm"], "left")
        self.assertEqual(result.commands[0]["z"], 0.1)
        self.assertEqual(result.reply, "好的，正在执行")

    def test_underspecified_control_clarifies(self):
        result = self.pipeline.handle_text("左臂向上", self.tianlian)
        self.assertEqual(result.route, "clarify")
        self.assertIn("请说明", result.reply)
        self.assertEqual(result.commands, [])

    def test_demo_arm_uses_different_json(self):
        result = self.pipeline.handle_text("向前50毫米", self.demo)
        self.assertEqual(result.route, "local_action")
        self.assertEqual(result.commands[0], {"cmd": "jog", "axis": "y", "mm": 50})

    def test_chat_does_not_emit_commands(self):
        result = self.pipeline.handle_text("你好呀", self.tianlian)
        self.assertEqual(result.route, "chat")
        self.assertEqual(result.commands, [])

    def test_weather_without_city_asks_location(self):
        result = self.pipeline.handle_text("今天天气怎么样", self.tianlian)
        self.assertEqual(result.route, "ask_city")
        self.assertIn("城市", result.reply)
        self.assertEqual(result.commands, [])
        self.assertTrue(result.takeover)

    def test_weather_speaks_filler_then_uses_search(self):
        seen: list[str] = []
        queries: list[str] = []

        class StreamLLM:
            configured = True

            def chat(self, *args, **kwargs):
                raise AssertionError("weather search must not call the chat LLM")

        pipe = Pipeline(llm=StreamLLM(), search_fn=lambda q: queries.append(q) or "长春多云，气温二十二度。")
        result = pipe.handle_text("长春天气怎么样", self.tianlian, on_sentence=seen.append)
        self.assertEqual(seen[0], "我查一下")
        self.assertTrue(any("多云" in item for item in seen))
        self.assertEqual(len(queries), 1)
        self.assertIn("长春天气怎么样", queries[0])
        self.assertRegex(queries[0], r"\d{4}年")
        self.assertEqual(result.route, "search")

    def test_midautumn_uses_local_calendar(self):
        from datetime import datetime

        from server.pipeline import holiday_countdown_reply, looks_like_search

        self.assertTrue(looks_like_search("还有多少天到中秋节"))
        reply = holiday_countdown_reply("几天到中秋节？", now=datetime(2026, 9, 4))
        self.assertIsNotNone(reply)
        self.assertIn("9月25日", reply)
        self.assertIn("21天", reply)
        self.assertIn("21天", holiday_countdown_reply("有多少天到中秋节放假呀", now=datetime(2026, 9, 4)))
        self.assertIn("就是中秋节", holiday_countdown_reply("中秋节是哪天", now=datetime(2026, 9, 25)))
        later = holiday_countdown_reply("几天到中秋节", now=datetime(2026, 9, 26))
        self.assertIn("9月15日", later)
        self.assertIsNone(holiday_countdown_reply("今天晚上吃什么"))
        self.assertIsNone(holiday_countdown_reply("中秋节"))
        self.assertIsNone(holiday_countdown_reply("中秋节放假安排"))
        self.assertIn("10月1日", holiday_countdown_reply("国庆节是几号", now=datetime(2026, 9, 4)))
        result = self.pipeline.handle_text("几天到中秋节？", self.tianlian)
        self.assertEqual(result.route, "calendar")
        self.assertIn("中秋", result.reply)
        self.assertTrue(result.takeover)
        self.assertEqual(self.pipeline.handle_text("今天晚上吃什么", self.tianlian).route, "chat")

    def test_search_stall_phrase_is_not_spoken(self):
        seen: list[str] = []

        class StreamLLM:
            configured = True

            def chat(self, *args, **kwargs):
                raise AssertionError("search must not call the chat LLM")

        pipe = Pipeline(llm=StreamLLM(), search_fn=lambda q: "资料：今天热点是本地新闻。")
        pipe.handle_text("今天有什么新闻", self.tianlian, on_sentence=seen.append)
        self.assertEqual(seen[0], "我查一下")
        self.assertNotIn("正在搜索…", seen)
        self.assertTrue(any("热点" in item or "新闻" in item for item in seen))

    def test_filler_is_ambient(self):
        result = self.pipeline.handle_text("嗯", self.tianlian)
        self.assertEqual(result.route, "ambient")

    def test_unconfigured_llm_does_not_block_on_dify(self):
        from unittest import mock

        class Down:
            configured = False

            def chat(self, *args, **kwargs):
                raise AssertionError("unconfigured LLM must not be called")

        with mock.patch("server.config.dialog_configured", return_value=False):
            pipe = Pipeline(llm=Down())
            result = pipe.handle_text("你好呀", self.tianlian)
            self.assertEqual(result.route, "chat")
            self.assertIn("CHAT_LLM_API_KEY", result.reply)
            self.assertEqual(result.commands, [])

    def test_chat_does_not_speak_bailian_key_when_dialog_on(self):
        from unittest import mock

        class KeyFail:
            configured = True

            def chat(self, *args, **kwargs):
                on_sentence = kwargs.get("on_sentence")
                msg = "聊天密钥无效。请到百炼控制台复制新的 CHAT_LLM_API_KEY（sk- 开头）。"
                if on_sentence:
                    on_sentence("聊天密钥无效。")
                    on_sentence("请到百炼控制台复制新的 CHAT_LLM_API_KEY（sk- 开头）。")
                return msg

        seen: list[str] = []
        with mock.patch("server.config.dialog_configured", return_value=True):
            pipe = Pipeline(llm=KeyFail())
            result = pipe.handle_text("今天晚上吃什么给我出个主意", self.tianlian, on_sentence=seen.append)
        blob = result.reply + "".join(seen)
        self.assertNotIn("CHAT_LLM_API_KEY", blob)
        self.assertNotIn("密钥无效", blob)
        self.assertNotIn("百炼", blob)
        self.assertIn("再说一次", result.reply)

    def test_unconfigured_llm_does_not_block_local_action(self):
        class Down:
            configured = False

            def chat(self, *args, **kwargs):
                raise AssertionError("local action must not call LLM")

        pipe = Pipeline(llm=Down())
        result = pipe.handle_text("右臂向前5厘米", self.tianlian)
        self.assertEqual(result.route, "local_action")
        self.assertEqual(result.commands[0]["arm"], "right")

    def test_changchun_is_remembered_for_tomorrow_weather(self):
        from server.memory import ConversationMemory

        queries: list[str] = []

        class Down:
            configured = False

            def chat(self, *args, **kwargs):
                raise AssertionError("remembered-city weather must not need Bailian")

        mem = ConversationMemory(robot_id="tianlian_dual")
        pipe = Pipeline(llm=Down(), search_fn=lambda q: queries.append(q) or "多云")
        first = pipe.handle_text("我在长春", self.tianlian, memory=mem)
        self.assertEqual(first.route, "memory")
        self.assertIn("长春", first.reply)
        self.assertEqual(mem.city, "长春")
        second = pipe.handle_text("你有几个关节", self.tianlian, memory=mem)
        self.assertEqual(second.route, "knowledge")
        third = pipe.handle_text("明天天气怎么样", self.tianlian, memory=mem)
        self.assertEqual(third.route, "search")
        self.assertEqual(len(queries), 1)
        self.assertIn("长春", queries[0])
        self.assertIn("明天", queries[0])
        self.assertIn("多云", third.reply)

    def test_spoken_search_reply_uses_snippet(self):
        from server.pipeline import spoken_search_reply

        text = spoken_search_reply("天气：长春今天多云，气温二十二度。\n其它：忽略", city="长春")
        self.assertIn("多云", text)
        self.assertIn("长春", text)
        self.assertIn("密钥", spoken_search_reply("秘塔搜索密钥无效或已过期。"))
        long_notes = (
            "长春天气预报：今天多云转晴，气温十八到二十六度。"
            "明天有小雨，气温十六到二十二度，出门带伞。\n"
            "其它：点击查看十五天趋势忽略这段。"
        )
        spoken = spoken_search_reply(long_notes, city="长春")
        self.assertIn("二十六", spoken)
        self.assertIn("小雨", spoken)
        self.assertNotIn("十五天", spoken)
        self.assertLessEqual(len(spoken), 240)
        ramble = spoken_search_reply(
            "2026年6月17日长春天气预报显示，当天为多云天气，最高气温29℃。"
            "此外，页面还提供了长春十大景点、美食、特产排行榜及城市文化信息。",
            city="长春",
        )
        self.assertIn("多云", ramble)
        self.assertNotIn("景点", ramble)
        self.assertNotIn("排行榜", ramble)
        almanac = spoken_search_reply(
            "长春，阳历2026年9月25日，丙午年阴历八月十五日【喜神】正南 【福神】东南 【财神】正南。",
            city="长春",
        )
        self.assertNotIn("喜神", almanac)
        self.assertNotIn("福神", almanac)
        self.assertIn("没有查到", almanac)
        horoscope = spoken_search_reply(
            "长春2026年5月5日天气预报显示，长春当日天气阴，气温15~27℃。"
            "当日为星期二，宜疗病、结婚、交易、入仓、求职，忌安葬、动土、针灸。",
            city="长春",
            query="2026年9月5日 明天长春的天气怎么样",
            now=__import__("datetime").datetime(2026, 9, 4),
        )
        self.assertNotIn("宜", horoscope)
        self.assertNotIn("忌", horoscope)
        self.assertNotIn("5月5日", horoscope)
        self.assertNotIn("结婚", horoscope)
        today_ok = spoken_search_reply(
            "长春明天多云，气温十八到二十六度。建议穿单层棉麻短套装。",
            city="长春",
            query="明天长春的天气怎么样",
            now=__import__("datetime").datetime(2026, 9, 4),
        )
        self.assertIn("多云", today_ok)
        self.assertNotIn("套装", today_ok)
        self.assertLessEqual(len(today_ok), 100)

    def test_working_is_not_a_city(self):
        from server.memory import extract_city

        self.assertEqual(extract_city("我在干活"), "")
        self.assertEqual(extract_city("我在长春"), "长春")


class MassageDryRunTests(unittest.TestCase):
    def test_execute_is_dry_run_by_default(self):
        from jetson_client.massage import execute_commands, teaching_capture

        os_env = __import__("os").environ
        os_env.pop("VOICE_ENABLE_MASSAGE", None)
        self.assertEqual(execute_commands([{"action": "massage", "op": "status"}]), "dry_run")
        captured = teaching_capture()
        self.assertTrue(captured.get("ok"))
        self.assertEqual(captured.get("status"), "dry_run")
        from jetson_client.massage import leftover_teaching_pids, reclaim_teaching

        self.assertIsInstance(leftover_teaching_pids(), list)
        self.assertIsInstance(reclaim_teaching(), list)


if __name__ == "__main__":
    unittest.main()
