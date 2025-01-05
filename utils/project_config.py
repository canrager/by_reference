import os
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
INTERIM_DIR = ROOT_DIR / "artifacts" / "interim"
PLOT_DIR = ROOT_DIR / "artifacts" / "plot"

# Make directories if they don't exist
dirs = [DATA_DIR, INTERIM_DIR, PLOT_DIR]
for d in dirs:
    if not os.path.exists(d):
        os.makedirs(d)