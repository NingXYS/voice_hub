import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path("/home/ning/model/lyz/src/tianlian_voice_bridge")))

from jetson_client.plugin import ensure_bridge_on_path, load_bridge


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

    def test_missing_module_is_speakable(self):
        with self.assertRaises(RuntimeError) as ctx:
            load_bridge("tianlian_dual", "no_such_voice_bridge_mod")
        self.assertIn("插件", str(ctx.exception))

    def test_ensure_bridge_on_path_uses_env(self):
        saved = os.environ.get("VOICE_HUB_BRIDGE_PATH")
        with tempfile.TemporaryDirectory() as tmp:
            pkg = Path(tmp) / "tianlian_voice_bridge"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("# probe\n", encoding="utf-8")
            os.environ["VOICE_HUB_BRIDGE_PATH"] = tmp
            resolved = str(Path(tmp).resolve())
            try:
                ensure_bridge_on_path()
                self.assertIn(resolved, sys.path)
            finally:
                if saved is None:
                    os.environ.pop("VOICE_HUB_BRIDGE_PATH", None)
                else:
                    os.environ["VOICE_HUB_BRIDGE_PATH"] = saved
                if resolved in sys.path:
                    sys.path.remove(resolved)


if __name__ == "__main__":
    unittest.main()
