from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BRIDGE = Path("/home/ning/model/lyz/src/tianlian_voice_bridge")
if BRIDGE.is_dir() and str(BRIDGE) not in sys.path:
    sys.path.insert(0, str(BRIDGE))

from device_ui.app import main

if __name__ == "__main__":
    main()
