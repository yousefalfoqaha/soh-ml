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

        max_soh = 0.0
        valid_instances = []

        for start_t, end_t in instances:
            mask = (timestamps >= start_t) & (timestamps <= end_t)

            if not np.any(mask):
                continue

            integrated = abs(float(np.trapezoid(samples[mask], timestamps[mask])))
            soh = min(integrated / self.nominal_capacity, 1.0)

            if soh > max_soh:
                max_soh = soh

            valid_instances.append((start_t, end_t))

        context.metadata["instances"] = [
            (start_t, end_t, max_soh) for start_t, end_t in valid_instances
        ]

        return context
