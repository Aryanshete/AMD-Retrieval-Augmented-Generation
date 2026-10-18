"""Make ``import sourcetrace`` work when running the tests straight from the repo."""

import sys
from pathlib import Path

APP = str(Path(__file__).resolve().parent.parent.parent / "app")
if APP not in sys.path:
    sys.path.insert(0, APP)
