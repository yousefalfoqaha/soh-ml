import numpy as np
from asammdf import MDF

from voltgan.pipeline.soh.base import SohResult, SohStrategy
from voltgan.pipeline.soh.utils import (
    _merge_intervals,
    _rasterized_current,
    _safe_get_channel,
)


class DischargeTimeMergeStrategy(SohStrategy):
    MERGE_GAP_SECONDS = 1000.0

    def can_handle(self, mdf: MDF) -> bool:
        channels = mdf.channels_db
        return (
            "sgl_discharge_time_start" in channels
            and "sgl_discharge_time_end" in channels
        )

    def calculate(self, mdf: MDF, qnom: float, raster: float) -> SohResult:
        t_start = _safe_get_channel(mdf, "sgl_discharge_time_start")
        t_end = _safe_get_channel(mdf, "sgl_discharge_time_end")

        if t_start is None or t_end is None or len(t_start) == 0:
            return SohResult(soh_file=0.0, method="discharge_time_merge_empty")

        result = _rasterized_current(mdf, raster)
        if result is None:
            return SohResult(soh_file=0.0, method="discharge_time_merge_no_current")
        current, timestamps = result

        merged = _merge_intervals(t_start, t_end, self.MERGE_GAP_SECONDS)

        soh_values = []
        for start_t, end_t in merged:
            mask = (timestamps >= start_t) & (timestamps <= end_t)
            if not np.any(mask):
                continue
            integrated = abs(float(np.trapezoid(current[mask], timestamps[mask])))
            soh = min(integrated / qnom, 1.0)
            if soh >= 0.05:
                soh_values.append(soh)

        soh_file = max(soh_values) if soh_values else 0.0

        return SohResult(
            soh_file=soh_file,
            method="discharge_time_merge",
        )