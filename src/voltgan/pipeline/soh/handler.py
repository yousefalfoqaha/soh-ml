from asammdf import MDF

from voltgan.pipeline.base import PipelineHandler, SampleContext
from voltgan.pipeline.soh.discharge_time import DischargeTimeMergeStrategy
from voltgan.pipeline.soh.pulse_integration import PulseIntegrationStrategy
from voltgan.pipeline.soh.sohc import SOHCStrategy
from voltgan.pipeline.soh.voltage_threshold import VoltageThresholdStrategy

_STRATEGIES = [
    DischargeTimeMergeStrategy(),
    SOHCStrategy(),
    PulseIntegrationStrategy(),
    VoltageThresholdStrategy(),
]


class Mf4SohHandler(PipelineHandler):
    def __init__(self, qnom: float = 18000.0, raster: float = 0.1):
        self.qnom = qnom
        self.raster = raster
        self.strategies = _STRATEGIES

    @property
    def order(self) -> int:
        return 2

    def handle(self, context: SampleContext) -> SampleContext:
        mf4_path = context.source_path
        try:
            mdf = MDF(name=mf4_path)
        except Exception:
            context.interrupted = f"cannot open mf4: {mf4_path}"
            return context

        try:
            for strategy in self.strategies:
                if strategy.can_handle(mdf):
                    result = strategy.calculate(mdf, self.qnom, self.raster)
                    break
            else:
                context.interrupted = "No SoH strategy suitable"
                return context
        finally:
            mdf.close()

        context.metadata["soh_file"] = result.soh_file
        context.metadata["soh_method"] = result.method

        print(
            f"  SoH [{result.method}]: soh_file={result.soh_file:.3f}"
            + f" in {mf4_path.name}"
        )

        return context
