from pathlib import Path

import h5py
import numpy as np


class McuSample:
    def __init__(self, filepath: Path, qnom: int):
        with h5py.File(filepath, "r") as f:
            signal = f["U"]
            qneg = f["Qneg"]

            if not isinstance(signal, h5py.Dataset):
                raise ValueError("Expected U to be a Dataset")

            if not isinstance(qneg, h5py.Dataset):
                raise ValueError("Expected Qneg to be a Dataset")

            self.filepath = filepath
            self.n_samples = len(signal)
            self.soh = abs(float(np.min(qneg))) / qnom * 100

    def __len__(self):
        return self.n_samples

    def load_window(self, start: int, end: int) -> np.ndarray:
        with h5py.File(self.filepath, "r") as f:
            u = f["U"]
            i = f["I"]
            temp = f["Temp[1]"]

            if not isinstance(u, h5py.Dataset):
                raise ValueError("Expected U to be a Dataset")
            if not isinstance(i, h5py.Dataset):
                raise ValueError("Expected I to be a Dataset")
            if not isinstance(temp, h5py.Dataset):
                raise ValueError("Expected Temp[1] to be a Dataset")

            return np.stack([u[start:end], i[start:end], temp[start:end]])
