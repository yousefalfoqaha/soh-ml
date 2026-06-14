from asammdf import MDF

from voltgan.pipeline.base import PipelineHandler, SampleContext
from voltgan.pipeline.soh.base import SohResult
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
        return 0

    def handle(self, ctx: SampleContext) -> SampleContext:
        mf4_path = ctx.source_path
        try:
            mdf = MDF(name=mf4_path)
        except Exception:
            ctx.interrupted = f"cannot open mf4: {mf4_path}"
            return ctx

        try:
            for strategy in self.strategies:
                if strategy.can_handle(mdf):
                    result = strategy.calculate(mdf, self.qnom, self.raster)
                    break
            else:
                result = SohResult(soh_file=0.0, method="no_strategy")
        finally:
            mdf.close()

        ctx.metadata["soh_file"] = result.soh_file
        ctx.metadata["soh_method"] = result.method

        print(
            f"  SoH [{result.method}]: soh_file={result.soh_file:.3f}"
            + f" in {mf4_path.name}"
        )

        return ctx
