from pathlib import Path

import h5py
import numpy as np

from voltgan.config import AMBIENT_TEMPERATURE_KEY
from voltgan.dataset.instance import DischargeInstance
from voltgan.utils.discover import FileDiscoverer


class InstanceRepository:
    def __init__(self, root: Path, extensions: tuple[str, ...] = (".hdf",)):
        self.root = root
        self.extensions = extensions

    def load(self, mcus: list[str]) -> list[DischargeInstance]:
        paths = FileDiscoverer.find(self.root, mcus, self.extensions)
        return [DischargeInstance(p) for p in paths]

    @staticmethod
    def save(filepath: Path, data: dict[str, np.ndarray], metadata: dict) -> None:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(filepath, "w") as f:
            group = f.create_group(filepath.name)

            for ch, arr in data.items():
                group.create_dataset(ch, data=arr)
                f.attrs[f"{ch}_mean"] = float(np.mean(arr))
                f.attrs[f"{ch}_m2"] = float(np.var(arr) * len(arr))
                f.attrs[f"{ch}_min"] = float(np.min(arr))
                f.attrs[f"{ch}_max"] = float(np.max(arr))

            for key, val in metadata.items():
                f.attrs[key] = val

            amb = metadata.get(AMBIENT_TEMPERATURE_KEY)
            if amb is not None:
                f.attrs[f"{AMBIENT_TEMPERATURE_KEY}_mean"] = float(amb)
                f.attrs[f"{AMBIENT_TEMPERATURE_KEY}_m2"] = 0.0
                f.attrs[f"{AMBIENT_TEMPERATURE_KEY}_min"] = float(amb)
                f.attrs[f"{AMBIENT_TEMPERATURE_KEY}_max"] = float(amb)

    @staticmethod
    def update_metadata(filepath: Path, metadata: dict) -> None:
        """Update attributes on an existing HDF5 file."""
        with h5py.File(filepath, "a") as f:
            for key, val in metadata.items():
                f.attrs[key] = val
