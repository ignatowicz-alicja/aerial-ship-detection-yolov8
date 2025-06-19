import datetime
from pathlib import Path
from config import SHIPSNET_JSON, AERIAL_ROOT, BG_CHIPS_DIR, PREPARED_DATASET, RUNS_DIR
from data_utils import load_shipsnet, generate_bg_chips, merge_and_split, _mkdir
from model_utils import train_model, evaluate

def main():
    shipsnet_extract = PREPARED_DATASET / "shipsnet_raw"
    chip_dataset = PREPARED_DATASET

    load_shipsnet(SHIPSNET_JSON, shipsnet_extract)
    generate_bg_chips(AERIAL_ROOT, BG_CHIPS_DIR)
    merge_and_split(chip_dataset, shipsnet_extract, BG_CHIPS_DIR)

    run_dir = RUNS_DIR / datetime.datetime.now().strftime("f1_%Y-%m-%d_%H-%M-%S")
    _mkdir(run_dir)

    model = train_model(chip_dataset, run_dir)
    evaluate(model, chip_dataset, run_dir)

if __name__ == "__main__":
    main()
