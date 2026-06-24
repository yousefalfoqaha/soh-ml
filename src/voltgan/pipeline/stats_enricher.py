import h5py
import numpy as np

from voltgan.pipeline.base import PipelineHandler, SampleContext


class StatsEnrichHandler(PipelineHandler):
    @property
    def order(self) -> int:
        return 3

    def handle(self, context: SampleContext) -> SampleContext:
        hdf_path = context.output_path
        if hdf_path is None:
            context.interrupted = "no output path"
            return context

        with h5py.File(hdf_path, "a") as f:
            f.attrs["stage"] = context.stage

            group = f[hdf_path.name]
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

            f.attrs["total_rows"] = total_rows

        print(f"  Stats enriched: {hdf_path.name}")
        return context
