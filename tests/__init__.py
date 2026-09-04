# test package
import os
import tempfile

os.environ.setdefault("VOICE_HUB_MEMORY_DIR", tempfile.mkdtemp(prefix="voice_hub_mem_"))
