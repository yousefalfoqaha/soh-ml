from asammdf import MDF

from voltgan.pipeline.base import PipelineHandler, SampleContext

_INTERNAL_TIME_CHANNELS = [
    "sgl_charge_time_start",
    "sgl_charge_time_end",
    "sgl_discharge_time_start",
    "sgl_discharge_time_end",
    "sgl_pulse",
]


class ChannelValidationHandler(PipelineHandler):
    def __init__(self, required_channels: list[str], ambient_temperature_channel: str):
        self.required_channels = required_channels
        self.ambient_temperature_channel = ambient_temperature_channel

    @property
    def order(self) -> int:
        return 0

    def handle(self, context: SampleContext) -> SampleContext:
        mf4_path = context.source_path

        context.metadata["output_channels"] = self.required_channels
        context.metadata["ambient_temperature_channel"] = (
            self.ambient_temperature_channel
        )

        superset = list(
            dict.fromkeys(
                [
                    *self.required_channels,
                    self.ambient_temperature_channel,
                    *_INTERNAL_TIME_CHANNELS,
                ]
            )
        )
        mdf = MDF(name=mf4_path, channels=superset)
        context.mdf = mdf

        context.metadata["time_channels"] = {
            ch for ch in _INTERNAL_TIME_CHANNELS if ch in mdf.channels_db
        }

        mandatory = [*self.required_channels, self.ambient_temperature_channel]
        missing = [ch for ch in mandatory if ch not in mdf.channels_db]

        if missing:
            context.interrupted = (
                f"Missing required channels in {mf4_path.name}: {missing}"
            )

        return context
