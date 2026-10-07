"""Repository paths shared by the analysis scripts (mirrors REPO_ROOT / directory in src/ocean/ocean_sim.jl)."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(os.environ.get("EXO_OCEAN_OUTPUT", REPO_ROOT / "output"))  # override with EXO_OCEAN_OUTPUT, as for the Julia runs
OCEAN_DIR = OUTPUT_DIR / "ocean"
FIG_DIR = REPO_ROOT / "figures"
