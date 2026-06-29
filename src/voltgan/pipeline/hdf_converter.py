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
        return 2

    def handle(self, context: SampleContext) -> SampleContext:
        mf4_path = context.source_path
        relative_path = mf4_path.relative_to(self.mf4_root)
        hdf_path = (self.hdf_root / relative_path).with_suffix(".hdf")
        context.output_path = hdf_path

        if hdf_path.exists():
            return context

        hdf_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Converting to HDF...")

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

        self._write_soh_metadata(hdf_path, context)

        return context

    def _write_soh_metadata(
        self, hdf_path: Path, context: SampleContext
    ) -> SampleContext:
        if "soh_file" not in context.metadata:
            context.interrupted = "SoH not found in pipeline context"
            return context

        soh_file = context.metadata.get("soh_file", 0.0)
        with h5py.File(hdf_path, "a") as f:
            if "soh_values" in f:
                del f["soh_values"]
            if "soh_timestamps" in f:
                del f["soh_timestamps"]
            f.attrs["soh_file"] = soh_file

        return context
