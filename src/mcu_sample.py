from pathlib import Path
from typing import cast

import h5py
import numpy as np


class McuSample:
    def __init__(self, filepath: Path):
        self.filepath = filepath

        with h5py.File(filepath, "r") as f:
            grp = f[filepath.name]
            assert isinstance(grp, h5py.Group)
            signal = grp["U"]
            assert isinstance(signal, h5py.Dataset)

            self.type = filepath.parts[4]
            self.n_samples = len(signal)

            soh_file = f.attrs.get("soh_file")
            self.soh = float(soh_file) if soh_file is not None else 1.0

    def __len__(self):
        return self.n_samples

    def load_window(self, start: int, end: int) -> np.ndarray:
        with h5py.File(self.filepath, "r") as f:
            grp = f[self.filepath.name]
            assert isinstance(grp, h5py.Group)

            u = cast(h5py.Dataset, grp["U"])[start:end]
            i = cast(h5py.Dataset, grp["I"])[start:end]
            t = cast(h5py.Dataset, grp["Temp[1]"])[start:end]
            ct = cast(h5py.Dataset, grp["ClimaTemp"])[start:end]

            return np.stack([u, i, t, ct])
