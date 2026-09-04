import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.memory import ConversationMemory, extract_city, city_in_text
from server.pipeline import search_query
from datetime import datetime


class MemoryTests(unittest.TestCase):
    def test_extract_city_patterns(self):
        self.assertEqual(extract_city("我在长春"), "长春")
        self.assertEqual(extract_city("我住在北京"), "北京")
        self.assertEqual(extract_city("我是上海人"), "上海")
        self.assertEqual(extract_city("长春天气怎么样"), "长春")
        self.assertEqual(extract_city("我在干活"), "")
        self.assertEqual(city_in_text("明天去深圳出差"), "深圳")

    def test_persists_city_across_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            mem = ConversationMemory.load("tianlian_dual", folder)
            mem.observe_user("我在长春")
            mem.observe_assistant("好的，记住了。")
            self.assertEqual(mem.city, "长春")
            again = ConversationMemory.load("tianlian_dual", folder)
            self.assertEqual(again.city, "长春")
            self.assertEqual(again.turns[0]["content"], "我在长春")
            payload = json.loads((folder / "tianlian_dual.json").read_text(encoding="utf-8"))
            self.assertNotIn("sk-", json.dumps(payload))

    def test_spoken_search_reply_plain(self):
        from server.pipeline import spoken_search_reply

        self.assertEqual(spoken_search_reply(""), "网上没有查到有用的信息。")

    def test_search_query_adds_city_and_tomorrow(self):
        now = datetime(2026, 8, 31)
        query = search_query("明天天气怎么样", now=now, city="长春")
        self.assertIn("长春", query)
        self.assertIn("2026年9月1日", query)
        today = search_query("长春天气怎么样", now=now, city="长春")
        self.assertIn("2026年8月31日", today)
        self.assertNotIn("景点", today)

    def test_idempotent_observe(self):
        mem = ConversationMemory(robot_id="x")
        mem.observe_user("我在长春")
        mem.observe_user("我在长春")
        self.assertEqual(len(mem.turns), 1)


if __name__ == "__main__":
    unittest.main()
