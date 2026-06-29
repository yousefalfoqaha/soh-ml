from asammdf import MDF

from voltgan.pipeline.base import PipelineHandler, SampleContext
from voltgan.pipeline.soh_strategies import (
    DischargeTimeStrategy,
    PulseTestStrategy,
    SOHCStrategy,
    VoltageThresholdStrategy,
)

_STRATEGIES = [
    DischargeTimeStrategy(),
    SOHCStrategy(),
    PulseTestStrategy(),
    VoltageThresholdStrategy(),
]


class SohHandler(PipelineHandler):
    def __init__(self, nominal_capacity: float = 18000.0, raster: float = 0.1):
        self.nominal_capacity = nominal_capacity
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
                        mdf, self.nominal_capacity, self.raster, context
                    )

                    if context.interrupted:
                        return context

                    print(
                        f"  SoH [{strategy.__class__.__name__}]: soh_file={context.metadata['soh_file']:.3f}"
                    )
                    break
            else:
                context.interrupted = f"No SoH strategy was picked {mf4_path.name}"
                return context
        finally:
            mdf.close()

        return context
