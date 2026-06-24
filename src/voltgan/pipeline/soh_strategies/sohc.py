from asammdf import MDF

from voltgan.pipeline.soh_strategies.base import SohResult, SohStrategy
from voltgan.pipeline.soh_strategies.utils import _safe_get_channel


class SOHCStrategy(SohStrategy):
    def can_handle(self, mdf: MDF) -> bool:
        return (
            "sgl_SOHC" in mdf.channels_db
            and "sgl_discharge_time_start" not in mdf.channels_db
            and "sgl_pulse" not in mdf.channels_db
        )

    def calculate(self, mdf: MDF, nominal_charge: float, raster: float) -> SohResult:
        sohc = _safe_get_channel(mdf, "sgl_SOHC")
        if sohc is None or len(sohc) == 0:
            return SohResult(soh_file=0.0, method="sohc_empty")

        soh_file = min(float(sohc[-1]) / 100.0, 1.0)

        return SohResult(
            soh_file=soh_file,
            method="sohc",
        )
