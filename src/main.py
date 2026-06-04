import os
from pathlib import Path

import torch
from asammdf import MDF

from dataset import McusDataset

MCUS_TRAIN = ["mcu1", "mcu2"]
MCUS_VALID = ["mcu3"]
MCUS_TEST = ["mcu4"]

DATA_PATH = Path("../data")

samples = []

for mcu in MCUS_TRAIN:
    mcu_path = DATA_PATH / mcu

    for root, _, files in os.walk(mcu_path):
        print(root)
        # for file in files:
        #     sample_path = Path(root) / file
        #     sample_mdf = MDF(sample_path)
        #     df = sample_mdf.to_dataframe(channels=["U", "Temp[0]", "I"], raster="U")
