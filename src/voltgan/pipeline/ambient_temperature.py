import numpy as np

from voltgan.pipeline.base import PipelineHandler, SampleContext


class AmbientTemperatureHandler(PipelineHandler):
    @property
    def order(self) -> int:
        return 3

    def handle(self, context: SampleContext) -> SampleContext:
        instances = context.metadata.get("instances", [])
        if not instances:
            return context

        channel = context.metadata["ambient_temperature_channel"]
        signal = context.mdf.get(channel)
        samples = signal.samples.astype(np.float32)
        timestamps = signal.timestamps.astype(np.float32)

        instances_with_temperature = []
        for start_t, end_t, soh in instances:
            mask = (timestamps >= start_t) & (timestamps <= end_t)
            masked = samples[mask]

            if masked.size:
                value = float(masked[masked.size // 2])
            else:
                value = float("nan")

            print(f"Ambient temp.: {value}")

            instances_with_temperature.append((start_t, end_t, soh, value))

        context.metadata["instances"] = instances_with_temperature

        return context
