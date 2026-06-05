from pathlib import Path

import mf4_to_hdf
from dataset import McusDataset

MCUS_TRAIN = ["mcu1", "mcu2"]
MCUS_VALID = ["mcu3"]
MCUS_TEST = ["mcu4"]

DATA_PATH = Path("../data")
RASTER_FREQ = 0.1
TARGET_CHANNELS = ["U", "I", "Temp[1]", "Qneg"]


def main():
    mf4_to_hdf.convert(
        data_path=DATA_PATH,
        mcus=MCUS_TRAIN,
        raster=RASTER_FREQ,
        target_channels=TARGET_CHANNELS,
    )

    hdf_data_path = DATA_PATH.joinpath("hdf")
    dataset_train = McusDataset(
        mcus=MCUS_TRAIN, data_path=hdf_data_path, window_length=1000
    )

    for window, target in dataset_train:
        print("Window:", window, "Target", target)


if __name__ == "__main__":
    main()
