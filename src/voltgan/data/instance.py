from datetime import datetime
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from voltgan.config import MAX_SEQUENCE_LENGTH


class DischargeInstance:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self._data: np.ndarray | None = None

        with h5py.File(filepath, "r") as f:
            group = f[filepath.name]
            assert isinstance(group, h5py.Group)
            signal = cast(h5py.Dataset, group["U"])

            self.n_samples = len(signal)

            soh_file = f.attrs.get("curve_soh")
            ambient_temperature = f.attrs.get("ambient_temperature")
            datetime_str = f.attrs.get("datetime")
            protocol = f.attrs.get("protocol")
            phase = f.attrs.get("phase")
            assert isinstance(soh_file, (float, np.floating))
            assert isinstance(ambient_temperature, (float, np.floating))
            assert isinstance(datetime_str, str)
            assert isinstance(protocol, str)
            assert isinstance(phase, str)

            self.soh = float(soh_file)
            self.ambient_temperature = float(ambient_temperature)
            self.datetime = datetime.fromisoformat(datetime_str)
            self.protocol = protocol
            self.phase = phase

    def _load(self) -> np.ndarray:
        with h5py.File(self.filepath, "r") as f:
            group = f[self.filepath.name]
            assert isinstance(group, h5py.Group)

            data = np.stack(
                [
                    cast(h5py.Dataset, group["U"])[:],
                    cast(h5py.Dataset, group["I"])[:],
                    cast(h5py.Dataset, group["Temp[1]"])[:],
                ]
            ).T.astype(np.float32)

        return data[:MAX_SEQUENCE_LENGTH]

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self._data = self._load()

        return self._data
