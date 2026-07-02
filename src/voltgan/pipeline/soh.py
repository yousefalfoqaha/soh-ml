import numpy as np

from voltgan.pipeline.base import PipelineHandler, SampleContext


class SohHandler(PipelineHandler):
    def __init__(self, nominal_capacity: float):
        self.nominal_capacity = nominal_capacity

    @property
    def order(self) -> int:
        return 2

    def handle(self, context: SampleContext) -> SampleContext:
        instances = context.metadata.get("instances")
        if not instances:
            return context

        signal = context.mdf.get("I")
        samples = signal.samples.astype(np.float32)
        timestamps = signal.timestamps.astype(np.float32)

        print(context.metadata["instances"])
        instances_with_soh = []
        for start_t, end_t in instances:
            mask = (timestamps >= start_t) & (timestamps <= end_t)

            if not np.any(mask):
                continue

            integrated = abs(float(np.trapezoid(samples[mask], timestamps[mask])))
            soh = min(integrated / self.nominal_capacity, 1.0)

            print(f"Calculated SoH: {soh}")

            instances_with_soh.append((start_t, end_t, soh))

        context.metadata["instances"] = instances_with_soh

        return context
