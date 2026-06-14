from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from asammdf import MDF

from pipeline import PipelineHandler, SampleContext


@dataclass
class SohResult:
    soh_file: float
    method: str = ""


class SohStrategy(ABC):
    @abstractmethod
    def can_handle(self, mdf: MDF) -> bool: ...

    @abstractmethod
    def calculate(self, mdf: MDF, qnom: float, raster: float) -> SohResult: ...


def _safe_get_channel(mdf: MDF, name: str) -> np.ndarray | None:
    channels = mdf.channels_db
    if name not in channels:
        return None
    try:
        sig = mdf.get(name)
        return sig.samples
    except Exception:
        return None


def _rasterized_current(
    mdf: MDF, raster: float
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        sig = mdf.get("I", raster=raster)
        return sig.samples.astype(np.float64), sig.timestamps.astype(np.float64)
    except Exception:
        return None


def _rasterized_voltage(
    mdf: MDF, raster: float
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        sig = mdf.get("U", raster=raster)
        return sig.samples.astype(np.float64), sig.timestamps.astype(np.float64)
    except Exception:
        return None


def _merge_intervals(
    starts: np.ndarray, ends: np.ndarray, gap: float
) -> list[tuple[float, float]]:
    intervals = sorted(zip(starts, ends), key=lambda x: x[0])
    merged: list[tuple[float, float]] = []
    for s, e in intervals:
        if merged and s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((float(s), float(e)))
    return merged


class DischargeTimeMergeStrategy(SohStrategy):
    MERGE_GAP_SECONDS = 3000.0

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


class SOHCStrategy(SohStrategy):
    def can_handle(self, mdf: MDF) -> bool:
        return (
            "sgl_SOHC" in mdf.channels_db
            and "sgl_discharge_time_start" not in mdf.channels_db
            and "sgl_pulse" not in mdf.channels_db
        )

    def calculate(self, mdf: MDF, qnom: float, raster: float) -> SohResult:
        sohc = _safe_get_channel(mdf, "sgl_SOHC")
        if sohc is None or len(sohc) == 0:
            return SohResult(soh_file=0.0, method="sohc_empty")

        soh_file = min(float(sohc[-1]) / 100.0, 1.0)

        return SohResult(
            soh_file=soh_file,
            method="sohc",
        )


class PulseIntegrationStrategy(SohStrategy):
    def can_handle(self, mdf: MDF) -> bool:
        return (
            "sgl_pulse" in mdf.channels_db
            and "sgl_discharge_time_start" not in mdf.channels_db
        )

    def calculate(self, mdf: MDF, qnom: float, raster: float) -> SohResult:
        try:
            sig_pulse = mdf.get("sgl_pulse")
        except Exception:
            return SohResult(soh_file=0.0, method="pulse_integration_no_pulse")

        result = _rasterized_current(mdf, raster)
        if result is None:
            return SohResult(soh_file=0.0, method="pulse_integration_no_current")
        current, timestamps = result

        pulse_ts = sig_pulse.timestamps
        pulse_vals = sig_pulse.samples

        start_indices = pulse_ts[pulse_vals == 1.0]
        end_indices = pulse_ts[pulse_vals == 2.0]

        n_pairs = min(len(start_indices), len(end_indices))
        if n_pairs == 0:
            return SohResult(soh_file=0.0, method="pulse_integration_no_pairs")

        total_discharge = 0.0
        for j in range(n_pairs):
            mask = (timestamps >= start_indices[j]) & (timestamps <= end_indices[j])
            if not np.any(mask):
                continue
            total_discharge += abs(float(np.trapezoid(current[mask], timestamps[mask])))

        soh_file = min(total_discharge / qnom, 1.0)

        return SohResult(
            soh_file=soh_file,
            method="pulse_integration",
        )


def _contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    regions = []
    start = None
    for i in range(len(mask)):
        if mask[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                regions.append((start, i))
                start = None
    if start is not None:
        regions.append((start, len(mask)))
    return regions


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


_STRATEGIES: list[SohStrategy] = [
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
