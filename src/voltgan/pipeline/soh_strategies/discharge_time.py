import numpy as np
from asammdf import MDF

from voltgan.pipeline.base import SampleContext
from voltgan.pipeline.soh_strategies.base import SohStrategy
from voltgan.pipeline.soh_strategies.utils import (
    _merge_intervals,
    _rasterized_current,
    _safe_get_channel,
)


class DischargeTimeStrategy(SohStrategy):
    MERGE_GAP_SECONDS = 1000.0

    def can_handle(self, mdf: MDF) -> bool:
        channels = mdf.channels_db
        return (
            "sgl_discharge_time_start" in channels
            and "sgl_discharge_time_end" in channels
        )

    def calculate(
        self, mdf: MDF, nominal_capacity: float, raster: float, context: SampleContext
    ) -> SampleContext:
        start = _safe_get_channel(mdf, "sgl_discharge_time_start")
        end = _safe_get_channel(mdf, "sgl_discharge_time_end")

        if start is None or end is None or len(start) == 0:
            context.interrupted = "Discharge time merge empty"
            return context

        result = _rasterized_current(mdf, raster)
        if result is None:
            context.interrupted = "Discharge time merge no current"
            return context

        current, timestamps = result
        merged = _merge_intervals(start, end, self.MERGE_GAP_SECONDS)

        soh_values = []
        for start_t, end_t in merged:
            mask = (timestamps >= start_t) & (timestamps <= end_t)
            if not np.any(mask):
                continue
            integrated = abs(float(np.trapezoid(current[mask], timestamps[mask])))
            soh = min(integrated / nominal_capacity, 1.0)
            if soh >= 0.05:
                soh_values.append(soh)

        if soh_values == []:
            context.interrupted = "No SoH values found"
            return context

        context.metadata["soh_file"] = max(soh_values)

        return context
