import numpy as np
from asammdf import MDF

from voltgan.pipeline.soh.base import SohResult, SohStrategy
from voltgan.pipeline.soh.utils import (
    _contiguous_regions,
    _rasterized_current,
    _rasterized_voltage,
)


class VoltageThresholdStrategy(SohStrategy):
    V_HIGH_THRESHOLD = 4.1
    V_LOW_THRESHOLD = 2.7
    MIN_CYCLE_VOLTAGE_DELTA = 1.0
    MIN_CYCLE_DURATION_SECONDS = 900.0
    MIN_SOH = 0.05

    def can_handle(self, mdf: MDF) -> bool:
        return True

    def calculate(self, mdf: MDF, qnom: float, raster: float) -> SohResult:
        result_v = _rasterized_voltage(mdf, raster)
        result_i = _rasterized_current(mdf, raster)

        if result_v is None or result_i is None:
            return SohResult(soh_file=0.0, method="voltage_threshold_no_data")

        voltage, _ = result_v
        current, timestamps = result_i

        pairs = self._detect_discharge_cycles(voltage)

        soh_values = []

        for peak_idx, trough_idx in pairs:
            duration = timestamps[trough_idx] - timestamps[peak_idx]
            if duration < self.MIN_CYCLE_DURATION_SECONDS:
                continue

            segment_current = current[peak_idx : trough_idx + 1]
            segment_time = timestamps[peak_idx : trough_idx + 1]
            integrated_charge = abs(float(np.trapezoid(segment_current, segment_time)))
            soh = integrated_charge / qnom
            if soh < self.MIN_SOH:
                continue
            soh_values.append(soh)

        soh_file = max(soh_values) if soh_values else 0.0

        return SohResult(
            soh_file=soh_file,
            method="voltage_threshold",
        )

    def _detect_discharge_cycles(self, voltage: np.ndarray) -> list[tuple[int, int]]:
        high_regions = _contiguous_regions(voltage >= self.V_HIGH_THRESHOLD)
        low_regions = _contiguous_regions(voltage <= self.V_LOW_THRESHOLD)

        maxima: list[tuple[int, float]] = []
        for start, end in high_regions:
            idx = start + int(np.argmax(voltage[start:end]))
            maxima.append((idx, float(voltage[idx])))

        minima: list[tuple[int, float]] = []
        for start, end in low_regions:
            idx = start + int(np.argmin(voltage[start:end]))
            minima.append((idx, float(voltage[idx])))

        events = sorted(maxima + minima, key=lambda e: e[0])

        pairs: list[tuple[int, int]] = []
        current_peak_idx = None
        current_peak_val = -np.inf

        for idx, val in events:
            if val >= self.V_HIGH_THRESHOLD:
                if val > current_peak_val:
                    current_peak_idx = idx
                    current_peak_val = val
            else:
                if current_peak_idx is not None:
                    delta = current_peak_val - val
                    if delta >= self.MIN_CYCLE_VOLTAGE_DELTA:
                        pairs.append((current_peak_idx, idx))
                        current_peak_idx = None
                        current_peak_val = -np.inf

        return pairs