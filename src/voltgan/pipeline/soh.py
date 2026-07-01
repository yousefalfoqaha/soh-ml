import numpy as np
from asammdf import MDF

from voltgan.pipeline.base import PipelineHandler, SampleContext


class SohHandler(PipelineHandler):
    def __init__(self, nominal_capacity: float, raster: float):
        self.nominal_capacity = nominal_capacity
        self.raster = raster

    @property
    def order(self) -> int:
        return 2

    def handle(self, context: SampleContext) -> SampleContext:
        instances = context.metadata.get("instances")
        if not instances:
            return context

        mdf = MDF(name=context.source_path, channels=["I"])
        current_signal = mdf.get("I", raster=self.raster)
        current = current_signal.samples.astype(np.float32)
        timestamps = current_signal.timestamps.astype(np.float32)

        print(context.metadata["instances"])
        instances_with_soh = []
        for start_t, end_t in instances:
            mask = (timestamps >= start_t) & (timestamps <= end_t)
            if not np.any(mask):
                continue

            integrated = abs(float(np.trapezoid(current[mask], timestamps[mask])))
            soh = min(integrated / self.nominal_capacity, 1.0)

            print(f"Calculated SoH: {soh}")

            instances_with_soh.append((start_t, end_t, soh))

        context.metadata["instances"] = instances_with_soh

        return context
