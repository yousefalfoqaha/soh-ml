import numpy as np
from asammdf import MDF

from voltgan.pipeline.base import PipelineHandler, SampleContext


class ExtractDischargePeriodsHandler(PipelineHandler):
    def __init__(self, channels):
        self.required_channels = channels

    @property
    def order(self) -> int:
        return 1

    def handle(self, context: SampleContext) -> SampleContext:
        mdf = MDF(name=context.source_path)

        try:
            discharge_start = mdf.get("sgl_discharge_time_start").samples.astype(
                np.float32
            )
            discharge_end = mdf.get("sgl_discharge_time_end").samples.astype(np.float32)
            discharge_exists = True
        except Exception:
            discharge_start = np.array([], dtype=np.float32)
            discharge_end = np.array([], dtype=np.float32)
            discharge_exists = False

        try:
            charge_start = mdf.get("sgl_charge_time_start").samples.astype(np.float32)
            has_charge_channel = True
        except Exception:
            charge_start = np.array([], dtype=np.float32)
            has_charge_channel = False

        try:
            pulse_signal = mdf.get("sgl_pulse")
            has_pulse_channel = True
        except Exception:
            pulse_signal = None
            has_pulse_channel = False

        has_signal = discharge_exists or has_pulse_channel

        if not has_charge_channel and not has_signal:
            print("no charge channel, and no signals")
            context.metadata["instances"] = [(-np.inf, np.inf)]

            return context

        if has_charge_channel and not has_signal:
            context.metadata["instances"] = []

            return context

        if not has_charge_channel or np.all(charge_start == charge_start.mean()):
            print("no charge channel or 1 charge")
            windows = [(-np.inf, np.inf)]
        else:
            print("multiple charges")
            windows = list(zip(charge_start[:-1], charge_start[1:]))

        instances = self._extract_instances(
            windows, discharge_start, discharge_end, pulse_signal, has_pulse_channel
        )

        context.metadata["instances"] = instances

        return context

    def _extract_instances(
        self, windows, discharge_start, discharge_end, pulse_signal, has_pulse_channel
    ):
        instances = []
        for window_start, window_end in windows:
            discharge_pair = self._discharge_pair_in_window(
                window_start, window_end, discharge_start, discharge_end
            )
            if discharge_pair is not None:
                instances.append(discharge_pair)
                continue

            if has_pulse_channel:
                pulse_pair = self._pulse_pair_in_window(
                    window_start, window_end, pulse_signal
                )
                if pulse_pair is not None:
                    instances.append(pulse_pair)

        return instances

    def _discharge_pair_in_window(
        self, window_start, window_end, discharge_start, discharge_end
    ):
        mask = (discharge_start >= window_start) & (discharge_start < window_end)
        if not np.any(mask):
            return None

        return (discharge_start[mask].min(), discharge_end[mask].max())

    def _pulse_pair_in_window(self, window_start, window_end, pulse_signal):
        pulse_timestamps = pulse_signal.timestamps
        mask = (pulse_timestamps >= window_start) & (pulse_timestamps < window_end)
        if not np.any(mask):
            return None

        return (pulse_timestamps[mask].min(), pulse_timestamps[mask].max())
