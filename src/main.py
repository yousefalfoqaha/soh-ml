import os
from pathlib import Path

import torch
from asammdf import MDF

MCUS_TRAIN = ["mcu1", "mcu2"]
MCUS_VALID = ["mcu3"]
MCUS_TEST = ["mcu4"]

train_filepaths = []

DATA_PATH = Path("../data")

for mcu in MCUS_TRAIN:
    mcu_path = Path.joinpath(DATA_PATH, mcu)
    for _, _, files in os.walk(mcu_path):
        if len(files) > 0:
            train_filepaths.append(files)
