import numpy as np
from asammdf import MDF

from voltgan.pipeline.soh.base import SohResult, SohStrategy
from voltgan.pipeline.soh.utils import _rasterized_current


class PulseIntegrationStrategy(SohStrategy):
    def can_handle(self, mdf: MDF) -> bool:
        return (
            "sgl_pulse" in mdf.channels_db
            and "sgl_discharge_time_start" not in mdf.channels_db
        )

    def calculate(self, mdf: MDF, nominal_charge: float, raster: float) -> SohResult:
        try:
            pulse_signal = mdf.get("sgl_pulse")
        except Exception:
            return SohResult(soh_file=0.0, method="pulse_integration_no_pulse")

        result = _rasterized_current(mdf, raster)
        if result is None:
            return SohResult(soh_file=0.0, method="pulse_integration_no_current")
        current, timestamps = result

        pulse_timestamps = pulse_signal.timestamps
        pulse_samples = pulse_signal.samples

        start_indices = pulse_timestamps[pulse_samples == 1.0]
        end_indices = pulse_timestamps[pulse_samples == 2.0]

        n_pairs = min(len(start_indices), len(end_indices))
        if n_pairs == 0:
            return SohResult(soh_file=0.0, method="pulse_integration_no_pairs")

        total_discharge = 0.0
        for j in range(n_pairs):
            mask = (timestamps >= start_indices[j]) & (timestamps <= end_indices[j])
            if not np.any(mask):
                continue
            total_discharge += abs(float(np.trapezoid(current[mask], timestamps[mask])))

        soh_file = min(total_discharge / nominal_charge, 1.0)

        return SohResult(
            soh_file=soh_file,
            method="pulse_integration",
        )

