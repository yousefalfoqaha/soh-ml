from pathlib import Path
from typing import cast

import h5py
import numpy as np


class McuSample:
    def __init__(self, filepath: Path):
        self.filepath = filepath

        with h5py.File(filepath, "r") as f:
            group = f[filepath.name]
            assert isinstance(group, h5py.Group)
            signal = group["U"]
            assert isinstance(signal, h5py.Dataset)

            self.type = filepath.parts[4]
            self.n_samples = len(signal)

            soh_file = f.attrs.get("soh_file")
            self.soh = float(soh_file) if soh_file is not None else 1.0

    def __len__(self):
        return self.n_samples

    def load_window(self, start: int, end: int) -> np.ndarray:
        with h5py.File(self.filepath, "r") as f:
            group = f[self.filepath.name]
            assert isinstance(group, h5py.Group)

            voltage = cast(h5py.Dataset, group["U"])[start:end]
            current = cast(h5py.Dataset, group["I"])[start:end]
            temperature = cast(h5py.Dataset, group["Temp[1]"])[start:end]
            ambient_temperature = cast(h5py.Dataset, group["ClimaTemp"])[start:end]

            return np.stack([voltage, current, temperature, ambient_temperature])
