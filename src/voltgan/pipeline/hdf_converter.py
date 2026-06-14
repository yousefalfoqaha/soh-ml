import os
from pathlib import Path

import h5py
import numpy as np
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
        return 1

    def handle(self, ctx: SampleContext) -> SampleContext:
        mf4_path = ctx.source_path
        relative_path = mf4_path.relative_to(self.mf4_root)
        hdf_path = (self.hdf_root / relative_path).with_suffix(".hdf")

        if hdf_path.exists():
            try:
                dataset_stage = mf4_path.parts[4].lower()
            except IndexError:
                dataset_stage = "unknown"

            ctx.output_path = hdf_path
            ctx.stage = dataset_stage
            self._write_soh_metadata(hdf_path, ctx)
            return ctx

        hdf_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            dataset_stage = mf4_path.parts[4].lower()
        except IndexError:
            dataset_stage = "unknown"

        print(f"Converting [{dataset_stage}] {mf4_path.name}...")

        mdf = MDF(name=mf4_path, channels=self.channels)
        try:
            mdf.export(
                fmt="hdf5",
                filename=hdf_path.name,
                single_time_base=True,
                raster=self.raster,
                time_from_zero=True,
            )

            os.rename(hdf_path.name, hdf_path)
        finally:
            mdf.close()

        ctx.output_path = hdf_path
        ctx.stage = dataset_stage
        self._write_soh_metadata(hdf_path, ctx)
        print(f"Finished converting: {hdf_path.name}")
        return ctx

    def _write_soh_metadata(self, hdf_path: Path, ctx: SampleContext) -> None:
        if "soh_file" not in ctx.metadata:
            return

        soh_file = ctx.metadata.get("soh_file", 0.0)
        soh_method = ctx.metadata.get("soh_method", "")

        with h5py.File(hdf_path, "a") as f:
            if "soh_values" in f:
                del f["soh_values"]
            if "soh_timestamps" in f:
                del f["soh_timestamps"]
            f.attrs["soh_file"] = soh_file
            f.attrs["soh_method"] = soh_method

