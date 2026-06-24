from asammdf import MDF

from voltgan.pipeline.base import PipelineHandler, SampleContext
from voltgan.pipeline.soh_strategies import (
    DischargeTimeMergeStrategy,
    PulseIntegrationStrategy,
    SOHCStrategy,
    VoltageThresholdStrategy,
)

_STRATEGIES = [
    DischargeTimeMergeStrategy(),
    SOHCStrategy(),
    PulseIntegrationStrategy(),
    VoltageThresholdStrategy(),
]


class SohHandler(PipelineHandler):
    def __init__(self, nominal_charge: float = 18000.0, raster: float = 0.1):
        self.nominal_charge = nominal_charge
        self.raster = raster
        self.strategies = _STRATEGIES

    @property
    def order(self) -> int:
        return 1

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
                    context = strategy.calculate(
                        mdf, self.nominal_charge, self.raster, context
                    )

                    if context.interrupted:
                        return context

                    print(
                        f"  SoH [{strategy.__class__.__name__}]: soh_file={context.metadata['soh_file']:.3f}"
                        + f" in {mf4_path.name}"
                    )

            else:
                context.interrupted = "No SoH strategy was picked"
                return context
        finally:
            mdf.close()
