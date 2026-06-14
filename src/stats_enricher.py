import h5py
import numpy as np

from pipeline import PipelineHandler, SampleContext


class StatsEnrichHandler(PipelineHandler):
    @property
    def order(self) -> int:
        return 2

    def handle(self, ctx: SampleContext) -> SampleContext:
        hdf_path = ctx.output_path
        if hdf_path is None:
            ctx.interrupted = "no output path"
            return ctx

        with h5py.File(hdf_path, "a") as f:
            f.attrs["stage"] = ctx.stage

            grp = f[hdf_path.name]
            assert isinstance(grp, h5py.Group)

            total_rows = None
            for channel in grp.keys():
                ds = grp[channel]
                assert isinstance(ds, h5py.Dataset)
                if total_rows is None:
                    total_rows = len(ds)
                data = ds[:]
                f.attrs[f"{channel}_mean"] = float(np.mean(data))
                f.attrs[f"{channel}_m2"] = float(np.var(data) * total_rows)

            f.attrs["total_rows"] = total_rows

        print(f"  Stats enriched: {hdf_path.name}")
        return ctx
