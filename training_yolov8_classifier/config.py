from pathlib import Path
import os

SHIPSNET_JSON = Path("PATH_TO/shipsnet.json")
AERIAL_ROOT = Path("PATH_TO/aerial_dataset")
BG_CHIPS_DIR = Path(AERIAL_ROOT) / "bg_chips"
PREPARED_DATASET = Path("data/chips_dataset")
RUNS_DIR = Path("runs/classification_logs")

CHIPS_PER_IMAGE = 30
VAL_RATIO = 0.15
TEST_RATIO = 0.15
