from asammdf import MDF

from voltgan.pipeline.base import SampleContext
from voltgan.pipeline.soh_strategies.base import SohStrategy
from voltgan.pipeline.soh_strategies.utils import _safe_get_channel


class SOHCStrategy(SohStrategy):
    def can_handle(self, mdf: MDF) -> bool:
        return (
            "sgl_SOHC" in mdf.channels_db
            and "sgl_discharge_time_start" not in mdf.channels_db
            and "sgl_pulse" not in mdf.channels_db
        )

    def calculate(
        self, mdf: MDF, nominal_capacity: float, raster: float, context: SampleContext
    ) -> SampleContext:
        sohc = _safe_get_channel(mdf, "sgl_SOHC")
        if sohc is None or len(sohc) == 0:
            context.interrupted = "SoHC empty"
            return context

        soh_file = min(float(sohc[-1]) / 100.0, 1.0)
        context.metadata["soh_file"] = soh_file

        return context
