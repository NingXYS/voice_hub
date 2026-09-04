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

    def test_midautumn_triggers_search_with_real_year(self):
        from datetime import datetime

        from server.pipeline import looks_like_search, search_query, today_label

        self.assertTrue(looks_like_search("还有多少天到中秋节"))
        year = str(datetime.now().year)
        self.assertIn(year, today_label())
        self.assertTrue(search_query("还有多少天到中秋节").startswith(year))

    def test_search_stall_phrase_is_not_spoken(self):
        seen: list[str] = []

        class StreamLLM:
            configured = True

            def chat(self, *args, **kwargs):
                raise AssertionError("search must not call the chat LLM")

        pipe = Pipeline(llm=StreamLLM(), search_fn=lambda q: "资料：中秋还没到。")
        pipe.handle_text("还有多少天到中秋节", self.tianlian, on_sentence=seen.append)
        self.assertEqual(seen[0], "我查一下")
        self.assertNotIn("正在搜索…", seen)
        self.assertTrue(any("中秋还没到" in item for item in seen))

    def test_filler_is_ambient(self):
        result = self.pipeline.handle_text("嗯", self.tianlian)
        self.assertEqual(result.route, "ambient")

    def test_unconfigured_llm_does_not_block_on_dify(self):
        class Down:
            configured = False

            def chat(self, *args, **kwargs):
                raise AssertionError("unconfigured LLM must not be called")

        pipe = Pipeline(llm=Down())
        result = pipe.handle_text("你好呀", self.tianlian)
        self.assertEqual(result.route, "chat")
        self.assertIn("CHAT_LLM_API_KEY", result.reply)
        self.assertEqual(result.commands, [])

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

    def test_working_is_not_a_city(self):
        from server.memory import extract_city

        self.assertEqual(extract_city("我在干活"), "")
        self.assertEqual(extract_city("我在长春"), "长春")


if __name__ == "__main__":
    unittest.main()
