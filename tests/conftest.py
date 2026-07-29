"""Put backend/ on the path so tests can import its modules directly."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))
