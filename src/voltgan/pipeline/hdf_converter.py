import os
from pathlib import Path

import h5py

from voltgan.pipeline.base import PipelineHandler, SampleContext


class HdfConvertHandler(PipelineHandler):
    def __init__(self, data_path: Path, raster: float):
        self.data_path = data_path
        self.raster = raster
        self.mf4_root = data_path / "mf4"
        self.hdf_root = data_path / "hdf"

    @property
    def order(self) -> int:
        return 4

    def handle(self, context: SampleContext) -> SampleContext:
        instances = context.metadata.get("instances", [])
        mdf = context.mdf
        output_channels = context.metadata["output_channels"]

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

        for instance, target_file in zip(instances, target_files):
            if target_file.exists():
                continue

            start_t, end_t, soh, ambient_temperature = instance
            slice_mdf = mdf.cut(start=start_t, stop=end_t).filter(output_channels)

            try:
                slice_mdf.export(
                    fmt="hdf5",
                    filename=target_file.name,
                    single_time_base=True,
                    raster=self.raster,
                    time_from_zero=True,
                )
            finally:
                slice_mdf.close()

            os.rename(target_file.name, target_file)

            with h5py.File(target_file, "a") as f:
                f.attrs["soh_file"] = soh
                f.attrs["ambient_temperature"] = ambient_temperature

        return context
