# Repository layout, resolved from this file so the scripts do not depend on the cwd.
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
SCENE = ROOT / "scene"
RESULTS = ROOT / "results"
