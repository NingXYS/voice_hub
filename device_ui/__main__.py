from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jetson_client.plugin import ensure_bridge_on_path

ensure_bridge_on_path()

from device_ui.app import main

if __name__ == "__main__":
    main()
