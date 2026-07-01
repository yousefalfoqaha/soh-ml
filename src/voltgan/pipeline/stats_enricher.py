from pathlib import Path

import h5py
import numpy as np

from voltgan.pipeline.base import PipelineHandler, SampleContext


class StatsEnrichHandler(PipelineHandler):
    @property
    def order(self) -> int:
        return 4

    def handle(self, context: SampleContext) -> SampleContext:
        instances = context.metadata.get("instances")
        if not instances:
            return context

        output_path = context.output_path
        if output_path is None:
            context.interrupted = "no output path"
            return context

        if output_path.is_dir():
            target_files = list(output_path.glob("*.hdf"))
        else:
            target_files = [output_path]

        if not target_files:
            context.interrupted = f"no hdf files found at {output_path}"
            return context

        for file_path in target_files:
            self._enrich_file(file_path)

        print(f"Stats enriched.")

        return context

    def _enrich_file(self, file_path: Path) -> None:
        with h5py.File(file_path, "a") as f:
            group = f[file_path.name]
            assert isinstance(group, h5py.Group)

            total_rows = None
            for channel in group.keys():
                dataset = group[channel]
                assert isinstance(dataset, h5py.Dataset)
                if total_rows is None:
                    total_rows = len(dataset)
                data = dataset[:]

                f.attrs[f"{channel}_mean"] = float(np.mean(data))
                f.attrs[f"{channel}_m2"] = float(np.var(data) * total_rows)

            if total_rows is not None:
                f.attrs["total_rows"] = total_rows
