from asammdf import MDF

from voltgan.pipeline.base import PipelineHandler, SampleContext


class ChannelValidationHandler(PipelineHandler):
    def __init__(self, required_channels: list[str]):
        self.required_channels = required_channels

    @property
    def order(self) -> int:
        return 0

    def handle(self, context: SampleContext) -> SampleContext:
        mf4_path = context.source_path
        mdf = MDF(name=mf4_path, channels=self.required_channels)
        try:
            missing = [ch for ch in self.required_channels if ch not in mdf.channels_db]
        finally:
            mdf.close()

        context.interrupted = f"Missing required channels in {mf4_path.name}: {missing}"

        return context
