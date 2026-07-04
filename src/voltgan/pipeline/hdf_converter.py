import os
from pathlib import Path

import h5py
import numpy as np

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

        df = mdf.to_dataframe(
            channels=output_channels,
            raster=None,
            time_from_zero=False,
        )

        for instance, target_file in zip(instances, target_files):
            if target_file.exists():
                continue

            start_t, end_t, soh, ambient_temperature = instance
            start_t = max(start_t, df.index[0])
            end_t = min(end_t, df.index[-1])
            instance_df = df.loc[start_t:end_t]

            original_index = instance_df.index.to_numpy() - start_t
            duration = original_index[-1]
            new_index = np.arange(0, duration + self.raster, self.raster)
            new_index = new_index[new_index <= duration]

            resampled = {
                channel: np.interp(
                    new_index, original_index, instance_df[channel].to_numpy()
                )
                for channel in instance_df.columns
            }

            with h5py.File(target_file, "w") as f:
                group = f.create_group(target_file.name)
                for channel, samples in resampled.items():
                    group.create_dataset(channel, data=samples)
                f.attrs["soh_file"] = soh
                f.attrs["ambient_temperature"] = ambient_temperature

        return context
