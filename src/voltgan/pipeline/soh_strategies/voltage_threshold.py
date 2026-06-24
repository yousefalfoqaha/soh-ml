import numpy as np
from asammdf import MDF

from voltgan.pipeline.base import SampleContext
from voltgan.pipeline.soh_strategies.base import SohStrategy
from voltgan.pipeline.soh_strategies.utils import (
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

    def calculate(
        self, mdf: MDF, nominal_charge: float, raster: float, context: SampleContext
    ) -> SampleContext:
        voltage_signal = _rasterized_voltage(mdf, raster)
        current_signal = _rasterized_current(mdf, raster)

        if voltage_signal is None or current_signal is None:
            context.interrupted = "No SoH is suitable"
            return context

        voltage, _ = voltage_signal
        current, timestamps = current_signal

        discharge_start_end_pairs = self._detect_discharge_cycles(voltage)

        soh_values = []

        for peak_idx, trough_idx in discharge_start_end_pairs:
            duration = timestamps[trough_idx] - timestamps[peak_idx]
            if duration < self.MIN_CYCLE_DURATION_SECONDS:
                continue

            segment_current = current[peak_idx : trough_idx + 1]
            segment_time = timestamps[peak_idx : trough_idx + 1]
            integrated_charge = abs(float(np.trapezoid(segment_current, segment_time)))
            soh = integrated_charge / nominal_charge
            if soh < self.MIN_SOH:
                continue
            soh_values.append(soh)

        if soh_values == []:
            context.interrupted = "No SoH is suitable"
            return context

        context.metadata["soh_file"] = max(soh_values)

        return context

    def _detect_discharge_cycles(self, voltage: np.ndarray) -> list[tuple[int, int]]:
        high_regions = _contiguous_regions(voltage >= self.V_HIGH_THRESHOLD)
        low_regions = _contiguous_regions(voltage <= self.V_LOW_THRESHOLD)

        maxima: list[tuple[int, float]] = []
        for start, end in high_regions:
            i = start + int(np.argmax(voltage[start:end]))
            maxima.append((i, float(voltage[i])))

        minima: list[tuple[int, float]] = []
        for start, end in low_regions:
            i = start + int(np.argmin(voltage[start:end]))
            minima.append((i, float(voltage[i])))

        events = sorted(maxima + minima, key=lambda e: e[0])

        pairs: list[tuple[int, int]] = []
        current_peak_i = None
        current_peak_value = -np.inf

        for i, val in events:
            if val >= self.V_HIGH_THRESHOLD:
                if val > current_peak_value:
                    current_peak_i = i
                    current_peak_value = val
            else:
                if current_peak_i is not None:
                    delta = current_peak_value - val
                    if delta >= self.MIN_CYCLE_VOLTAGE_DELTA:
                        pairs.append((current_peak_i, i))
                        current_peak_i = None
                        current_peak_value = -np.inf

        return pairs
