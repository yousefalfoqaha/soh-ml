from pathlib import Path

import h5py
import numpy as np

from voltgan.pipeline.base import PipelineHandler, SampleContext


class StatsEnrichHandler(PipelineHandler):
    @property
    def order(self) -> int:
        return 6

    def handle(self, context: SampleContext) -> SampleContext:
        instances = context.metadata.get("instances")

        if not instances:
            return context

        target_files = context.metadata.get("target_files")
        if not target_files:
            context.interrupted = "no target files"
            return context

        for file_path in target_files:
            self._enrich_file(file_path)

        return context

    def _enrich_file(self, file_path: Path) -> None:
        with h5py.File(file_path, "a") as f:
            group = f[file_path.name]
            assert isinstance(group, h5py.Group)

            total_rows = None
            for channel in group.keys():
                dataset = group[channel]

                if not isinstance(dataset, h5py.Dataset):
                    continue

                total_rows = len(dataset)
                data = dataset[:]

                f.attrs[f"{channel}_mean"] = float(np.mean(data))
                f.attrs[f"{channel}_m2"] = float(np.var(data) * total_rows)
                f.attrs[f"{channel}_min"] = float(np.min(data))
                f.attrs[f"{channel}_max"] = float(np.max(data))

            f.attrs["total_rows"] = total_rows
            f.attrs["ambient_temperature_mean"] = f.attrs["ambient_temperature"]
            f.attrs["ambient_temperature_m2"] = 0.0
            f.attrs["ambient_temperature_min"] = f.attrs["ambient_temperature"]
            f.attrs["ambient_temperature_max"] = f.attrs["ambient_temperature"]
