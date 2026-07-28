from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import cast

import h5py
import numpy as np

from voltgan.config import (
    CURRENT_CHANNEL,
    HDF_ROOT,
    MAX_SEQUENCE_LENGTH,
    TEMPERATURE_CHANNEL,
    VOLTAGE_CHANNEL,
)
from voltgan.dataset.instance import DischargeInstance
from voltgan.utils.discover import FileDiscoverer


class InstanceRepository:
    def __init__(self, provider: str):
        self.provider = provider
        self.root = HDF_ROOT / provider

    def _create_data_loader(self, filepath: Path) -> Callable[[], np.ndarray]:
        def loader() -> np.ndarray:
            with h5py.File(filepath, "r") as f:
                group = cast(h5py.Group, f[filepath.name])
                data = np.stack(
                    [
                        cast(h5py.Dataset, group[VOLTAGE_CHANNEL])[:],
                        cast(h5py.Dataset, group[CURRENT_CHANNEL])[:],
                        cast(h5py.Dataset, group[TEMPERATURE_CHANNEL])[:],
                    ]
                ).T.astype(np.float32)
            return data[:MAX_SEQUENCE_LENGTH]

        return loader

    def load(self, cells: list[str]) -> list[DischargeInstance]:
        paths = FileDiscoverer.find(self.root, cells, (".hdf",))
        instances = []

        for p in paths:
            with h5py.File(p, "r") as f:
                group = cast(h5py.Group, f[p.name])
                attrs = f.attrs

                d_rate = attrs.get("discharge_rate")
                split = attrs.get("split")
                c_soh = attrs.get("curve_soh")

                volt_ds = cast(h5py.Dataset, group[VOLTAGE_CHANNEL])

                inst = DischargeInstance(
                    filepath=p,
                    n_samples=len(volt_ds),
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
                    _data_loader=self._create_data_loader(p),
                )
                instances.append(inst)

        return instances

    def exists(self, cell_id: str, filename: str) -> bool:
        return (self.root / cell_id / filename).exists()

    def save(
        self, cell_id: str, filename: str, data: dict[str, np.ndarray], metadata: dict
    ) -> None:
        filepath = self.root / cell_id / filename
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
