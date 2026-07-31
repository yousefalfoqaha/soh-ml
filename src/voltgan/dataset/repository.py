from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from voltgan.config import (
    CURRENT_CHANNEL,
    HDF_DIR,
    TEMPERATURE_CHANNEL,
    VOLTAGE_CHANNEL,
)
from voltgan.dataset.instance import DischargeInstance
from voltgan.utils.discover import FileDiscoverer


class _HDFDataLoader:
    def __init__(self, filepath: Path, max_length: int | None = None):
        self.filepath = filepath
        self.max_length = max_length

    def __call__(self) -> np.ndarray:
        with h5py.File(self.filepath, "r") as f:
            group = cast(h5py.Group, f[self.filepath.name])
            data = np.stack(
                [
                    cast(h5py.Dataset, group[VOLTAGE_CHANNEL])[:],
                    cast(h5py.Dataset, group[CURRENT_CHANNEL])[:],
                    cast(h5py.Dataset, group[TEMPERATURE_CHANNEL])[:],
                ]
            ).T.astype(np.float32)

        if self.max_length is not None:
            return data[: self.max_length]
        return data


class InstanceRepository:
    def __init__(self, provider: str):
        self.provider = provider
        self._root = HDF_DIR / provider

    def _create_data_loader(
        self, filepath: Path, max_length: int | None
    ) -> Callable[[], np.ndarray]:
        return _HDFDataLoader(filepath, max_length)

    def load(
        self, cells: list[str], max_length: int | None = None
    ) -> list[DischargeInstance]:
        paths = FileDiscoverer.find(self._root, cells, (".hdf",))
        instances = []

        for p in paths:
            with h5py.File(p, "r") as f:
                attrs = f.attrs
                d_rate = attrs.get("discharge_rate")
                split = attrs.get("split")
                c_soh = attrs.get("curve_soh")

                inst = DischargeInstance(
                    filepath=p,
                    cell_id=str(attrs.get("cell_id", "")),
                    provider=str(attrs.get("provider", "")),
                    soh=float(attrs.get("soh", 0.0)),
                    curve_soh=float(c_soh) if c_soh is not None else 0.0,
                    ambient_temperature=float(attrs.get("ambient_temperature")),  # type: ignore
                    datetime=datetime.fromisoformat(str(attrs.get("datetime"))),
                    protocol=str(attrs.get("protocol")),
                    phase=str(attrs.get("phase")),
                    discharge_rate=float(d_rate)
                    if d_rate not in (None, "None")
                    else None,
                    split=str(split) if split not in (None, "None") else None,
                    dci=float(attrs.get("discharge_cycle_index", 0)),
                    _data_loader=self._create_data_loader(p, max_length),
                )
                instances.append(inst)

        return instances

    def exists(self, cell_id: str, filename: str) -> bool:
        return (self._root / cell_id / filename).exists()

    def save(
        self, cell_id: str, filename: str, data: dict[str, np.ndarray], metadata: dict
    ) -> None:
        filepath = self._root / cell_id / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, "w") as f:
            group = f.create_group(filename)

            for ch, arr in data.items():
                group.create_dataset(ch, data=arr)

            for key, val in metadata.items():
                f.attrs[key] = val if val is not None else "None"

    @staticmethod
    def update_metadata(filepath: Path, metadata: dict) -> None:
        with h5py.File(filepath, "a") as f:
            for key, val in metadata.items():
                f.attrs[key] = val if val is not None else "None"
