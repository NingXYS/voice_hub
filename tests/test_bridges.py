import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path("/home/ning/model/lyz/src/tianlian_voice_bridge")))

from bridges.demo_single import validate_commands as demo_validate
from tianlian_voice_bridge import execute_commands, validate_commands


class BridgeIsolationTests(unittest.TestCase):
    def test_tianlian_accepts_own_json(self):
        commands = [{
            "action": "move_cartesian",
            "arm": "left",
            "x": 0,
            "y": 0,
            "z": 0.1,
            "relative": True,
        }]
        self.assertIsNotNone(validate_commands(commands))
        self.assertEqual(execute_commands(commands), "dry_run")

    def test_tianlian_rejects_demo_json(self):
        self.assertIsNone(validate_commands([{"cmd": "jog", "axis": "y", "mm": 50}]))
        self.assertEqual(execute_commands([{"cmd": "jog", "axis": "y", "mm": 50}]), "rejected")

    def test_demo_rejects_tianlian_json(self):
        tianlian = [{
            "action": "move_cartesian",
            "arm": "left",
            "x": 0,
            "y": 0,
            "z": 0.1,
            "relative": True,
        }]
        self.assertIsNone(demo_validate(tianlian))
        self.assertIsNotNone(demo_validate([{"cmd": "jog", "axis": "y", "mm": 50}]))

    def test_tianlian_rejects_extra_gripper_field(self):
        self.assertIsNone(
            validate_commands([{
                "action": "move_cartesian",
                "arm": "left",
                "x": 0,
                "y": 0,
                "z": 0.1,
                "relative": True,
                "gripper": "close",
            }])
        )

    def test_tianlian_rejects_missing_arm(self):
        self.assertIsNone(validate_commands([{"action": "home"}]))


if __name__ == "__main__":
    unittest.main()
