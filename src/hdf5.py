import os
from pathlib import Path

import h5py
from asammdf import MDF

TARGET_CHANNELS = ["U", "I", "Qneg", "Temp[1]"]
MCUS_TRAIN = ["mcu1", "mcu2"]
DATA_PATH = Path("../data")
RASTER_FREQ = 0.1


def inspect_hdf5_file(file_path: Path) -> None:
    """Recursively crawls and prints the internal structure of an HDF5 file."""
    print("=" * 60)
    print(f"Inspecting HDF5 File: {file_path}")
    print("=" * 60)

    def print_structure(name, obj):
        shift = "  " * name.count("/")

        if isinstance(obj, h5py.Group):
            print(f"{shift}📁 Group: {name}")
            if obj.attrs:
                print(f"{shift}   Attributes: {dict(obj.attrs)}")

        elif isinstance(obj, h5py.Dataset):
            print(f"{shift}   📄 Dataset: {obj.name.split('/')[-1]}")
            print(f"{shift}      Shape: {obj.shape} | Type: {obj.dtype}")

    with h5py.File(file_path, "r") as f:
        f.visititems(print_structure)
    print("=" * 60 + "\n")


for mcu in MCUS_TRAIN:
    mcu_path = DATA_PATH / mcu

    if not mcu_path.exists():
        print(f"Directory missing, skipping: {mcu_path}")
        continue

    for root, _, files in os.walk(mcu_path):
        print(f"\nScanning Directory: {root}")

        for file in files:
            if file.lower().endswith((".dat", ".mf4")):
                sample_path = Path(root) / file
                hdf5_path = sample_path.with_suffix(".hdf")

                print(f"Processing raw MDF: {sample_path.name}")

                mdf = MDF(name=sample_path, channels=TARGET_CHANNELS)

                if not hdf5_path.exists():
                    mdf.export(
                        fmt="hdf5",
                        filename=hdf5_path,
                        single_time_base=True,
                        raster=RASTER_FREQ,
                    )

                mdf.close()

                inspect_hdf5_file(hdf5_path)
