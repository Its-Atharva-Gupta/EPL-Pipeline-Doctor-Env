import sys
from pathlib import Path

# Ensure the project root is always on sys.path so `import models` works
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
