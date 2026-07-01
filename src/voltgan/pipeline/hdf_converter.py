import os
from pathlib import Path

import h5py
from asammdf import MDF

from voltgan.pipeline.base import PipelineHandler, SampleContext


class HdfConvertHandler(PipelineHandler):
    def __init__(self, data_path: Path, raster: float, channels: list[str]):
        self.data_path = data_path
        self.raster = raster
        self.channels = channels
        self.mf4_root = data_path / "mf4"
        self.hdf_root = data_path / "hdf"

    @property
    def order(self) -> int:
        return 3

    def handle(self, context: SampleContext) -> SampleContext:
        instances = context.metadata.get("instances", [])
        if len(instances) == 0:
            return context

        mf4_path = context.source_path
        relative_path = mf4_path.relative_to(self.mf4_root)

        if len(instances) == 1:
            base_hdf_path = (self.hdf_root / relative_path).with_suffix(".hdf")
        else:
            base_hdf_path = (self.hdf_root / relative_path).with_suffix("")

        context.output_path = base_hdf_path

        target_files = []
        if len(instances) == 1:
            base_hdf_path.parent.mkdir(parents=True, exist_ok=True)
            target_files.append(base_hdf_path)
        else:
            base_hdf_path.mkdir(parents=True, exist_ok=True)
            for i in range(1, len(instances) + 1):
                target_files.append(base_hdf_path / f"{i}.hdf")

        if all(target.exists() for target in target_files):
            return context

        print(f"Converting to HDF...")
        mdf = MDF(name=mf4_path, channels=self.channels)
        try:
            for instance, target_file in zip(instances, target_files):
                if target_file.exists():
                    continue

                slice_mdf = mdf.cut(start=instance[0], stop=instance[1])
                slice_mdf.export(
                    fmt="hdf5",
                    filename=target_file.name,
                    single_time_base=True,
                    raster=self.raster,
                    time_from_zero=True,
                )

                os.rename(target_file.name, target_file)

                with h5py.File(target_file, "a") as f:
                    f.attrs["soh_file"] = instance[2]
        finally:
            mdf.close()

        return context
