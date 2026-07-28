from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class DischargeInstance:
    filepath: Path
    n_samples: int
    cell_id: str
    provider: str
    soh: float
    curve_soh: float
    ambient_temperature: float
    datetime: datetime
    protocol: str
    phase: str
    discharge_rate: float | None
    split: str | None
    dci: float

    _data_loader: Callable[[], np.ndarray] = field(repr=False)
    _data: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self._data = self._data_loader()
        return self._data

    @property
    def voltage(self) -> np.ndarray:
        return self.data[:, 0]

    @property
    def current(self) -> np.ndarray:
        return self.data[:, 1]

    @property
    def temperature(self) -> np.ndarray:
        return self.data[:, 2]

    @property
    def temp_center(self) -> int:
        return int(round(self.ambient_temperature / 5) * 5)
