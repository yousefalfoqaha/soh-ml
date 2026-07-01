from pathlib import Path
from typing import cast

import h5py
import numpy as np


class McuSample:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._data: np.ndarray | None = None

        with h5py.File(filepath, "r") as f:
            group = f[filepath.name]
            assert isinstance(group, h5py.Group)
            signal = cast(h5py.Dataset, group["U"])

            self.type = filepath.parts[4]
            self.n_samples = len(signal)

            soh_file = f.attrs.get("soh_file")
            if soh_file is not None:
                self.soh = float(soh_file)

    def _load(self) -> np.ndarray:
        with h5py.File(self.filepath, "r") as f:
            group = f[self.filepath.name]
            assert isinstance(group, h5py.Group)
            return np.stack(
                [
                    cast(h5py.Dataset, group["U"])[:],
                    cast(h5py.Dataset, group["I"])[:],
                    cast(h5py.Dataset, group["Temp[1]"])[:],
                    cast(h5py.Dataset, group["ClimaTemp"])[:],
                    cast(h5py.Dataset, group["Q"])[:],
                ]
            ).T.astype(np.float32)

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self._data = self._load()
        return self._data

    def load_window(self, start: int, end: int) -> np.ndarray:
        return self.data[start:end, :]
