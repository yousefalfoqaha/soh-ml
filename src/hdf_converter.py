import os
from pathlib import Path

import h5py
import numpy as np
from asammdf import MDF


class HdfConverter:
    def __init__(
        self,
        data_path: Path,
        raster: float,
        channels: list[str],
    ):
        self.data_path = data_path
        self.raster = raster
        self.channels = channels

        self.mf4_root = self.data_path / "mf4"
        self.hdf_root = self.data_path / "hdf"

    def process_mcus(self, mcus: list[str]):
        for mcu in mcus:
            mcu_source_path = self.mf4_root / mcu

            if not mcu_source_path.exists():
                print(f"Source directory missing, skipping MCU: {mcu_source_path}")
                continue

            for root, _, files in os.walk(mcu_source_path):
                for file in files:
                    if file.lower().endswith((".dat", ".mf4")):
                        mf4_path = Path(root) / file
                        self._convert(mf4_path)

    def _convert(self, mf4_path: Path):
        relative_path = mf4_path.relative_to(self.mf4_root)
        hdf_path = (self.hdf_root / relative_path).with_suffix(".hdf")
        if hdf_path.exists():
            return

        hdf_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Converting {mf4_path.name}...")

        mdf = MDF(name=mf4_path, channels=self.channels)

        mdf.export(
            fmt="hdf5",
            filename=hdf_path.name,
            single_time_base=True,
            raster=self.raster,
            time_from_zero=True,
        )
        self._inject_metadata(hdf_path)
        os.rename(hdf_path.name, hdf_path)

        mdf.close()

    def _inject_metadata(self, hdf_path: Path):
        try:
            dataset_stage = hdf_path.parts[4].lower()
        except IndexError:
            dataset_stage = "unknown"

        with h5py.File(hdf_path.name, "a") as f:
            f.attrs["stage"] = dataset_stage

            group = f[hdf_path.name]
            assert isinstance(group, h5py.Group)
            signal = group.get("U")
            assert isinstance(signal, h5py.Dataset)

            total_rows = len(signal)
            f.attrs["total_rows"] = total_rows

            for channel in f.keys():
                if channel != "timestamps":
                    channel_obj = f[channel]
                    if isinstance(channel_obj, h5py.Dataset):
                        data = channel_obj[:]
                        f.attrs[f"{channel}_mean"] = float(np.mean(data))
                        f.attrs[f"{channel}_m2"] = float(np.var(data) * total_rows)
