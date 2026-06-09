from pathlib import Path
from typing import cast

import h5py
import numpy as np


class McuSample:
    def __init__(self, filepath: Path, qnom: int):
        with h5py.File(filepath, "r") as f:
            self._group_path = self._find_channel_group(f)
            signal = f[f"{self._group_path}/U"]
            qneg = f[f"{self._group_path}/Qneg"]

            if not isinstance(signal, h5py.Dataset):
                raise ValueError("Expected U to be a Dataset")
            if not isinstance(qneg, h5py.Dataset):
                raise ValueError("Expected Qneg to be a Dataset")

            self.filepath = filepath
            self.n_samples = len(signal)
            self.soh = abs(float(np.min(qneg))) / qnom * 100

    def __len__(self):
        return self.n_samples

    @staticmethod
    def _find_channel_group(f: h5py.File) -> str:
        def visitor(path, obj):
            if isinstance(obj, h5py.Dataset):
                return path.rsplit("/", 1)[0]

        result = f.visititems(visitor)
        if result is None:
            raise KeyError("No datasets found in file")

        return result

    def load_window(self, start: int, end: int) -> np.ndarray:
        base_path = f"{self._group_path}/" if self._group_path else ""

        with h5py.File(self.filepath, "r") as f:
            u = cast(h5py.Dataset, f[f"{base_path}U"])[start:end]
            i = cast(h5py.Dataset, f[f"{base_path}I"])[start:end]
            t = cast(h5py.Dataset, f[f"{base_path}Temp[1]"])[start:end]
            q = cast(h5py.Dataset, f[f"{base_path}Qneg"])[start:end]
            qpos = cast(h5py.Dataset, f[f"{base_path}Qpos"])[start:end]
            ct = cast(h5py.Dataset, f[f"{base_path}ClimaTemp"])[start:end]

            return np.stack([u, i, t, q, qpos, ct])
