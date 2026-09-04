import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path("/home/ning/model/lyz/src/tianlian_voice_bridge")))

from jetson_client.plugin import load_bridge


class PluginTests(unittest.TestCase):
    def test_loads_tianlian_and_demo(self):
        tianlian = load_bridge("tianlian_dual")
        demo = load_bridge("demo_single")
        good = [{
            "action": "move_cartesian",
            "arm": "right",
            "x": 0,
            "y": 0.05,
            "z": 0,
            "relative": True,
        }]
        self.assertEqual(tianlian.execute_commands(good), "dry_run")
        self.assertEqual(demo.execute_commands(good), "rejected")
        self.assertEqual(demo.execute_commands([{"cmd": "jog", "axis": "x", "mm": 10}]), "dry_run")
        self.assertEqual(tianlian.execute_commands([{"cmd": "jog", "axis": "x", "mm": 10}]), "rejected")


if __name__ == "__main__":
    unittest.main()
