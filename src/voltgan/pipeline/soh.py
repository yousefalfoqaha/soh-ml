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

        mdf = context.mdf

        current_signal = mdf.get("I")
        current_samples = current_signal.samples.astype(np.float32)
        current_timestamps = current_signal.timestamps.astype(np.float32)

        result_instances = []
        for start_t, end_t in instances:
            cur_mask = (current_timestamps >= start_t) & (
                current_timestamps <= end_t
            )
            if not np.any(cur_mask):
                continue

            masked_current = current_samples[cur_mask]
            masked_timestamps = current_timestamps[cur_mask]

            integrated = abs(
                float(np.trapezoid(masked_current, masked_timestamps))
            )
            soh = min(integrated / self.nominal_capacity, 1.0)

            neg_current = masked_current[masked_current < 0]
            if neg_current.size:
                mean_neg_current = float(np.mean(np.abs(neg_current)))
            else:
                mean_neg_current = 0.0

            result_instances.append((start_t, end_t, soh, mean_neg_current))

        context.metadata["instances"] = result_instances
        return context