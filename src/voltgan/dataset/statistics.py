import json
from pathlib import Path

import numpy as np

from voltgan.config import (
    CURRENT_CHANNEL,
    SOH_KEY,
    TEMP_DELTA_KEY,
    TEMPERATURE_CHANNEL,
    VOLTAGE_CHANNEL,
)
from voltgan.dataset.instance import DischargeInstance
from voltgan.utils.box_table import print_box_table


class StatisticsCalculator:
    def __init__(self, save_path: Path | None = None):
        self.save_path = save_path
        self.stats = {}

    def compute(
        self, instances: list[DischargeInstance]
    ) -> dict[str, dict[str, float]]:

        sums = {
            ch: 0.0
            for ch in [
                VOLTAGE_CHANNEL,
                CURRENT_CHANNEL,
                TEMPERATURE_CHANNEL,
                TEMP_DELTA_KEY,
            ]
        }
        sq_sums = {
            ch: 0.0
            for ch in [
                VOLTAGE_CHANNEL,
                CURRENT_CHANNEL,
                TEMPERATURE_CHANNEL,
                TEMP_DELTA_KEY,
            ]
        }

        total_rows = 0
        soh_values: list[float] = []

        for inst in instances:
            n_rows = inst.n_samples
            total_rows += n_rows
            soh_values.append(inst.curve_soh)

            v = inst.voltage
            i = inst.current
            t = inst.temperature
            t_delta = t - inst.ambient_temperature

            for key, arr in zip(
                [VOLTAGE_CHANNEL, CURRENT_CHANNEL, TEMPERATURE_CHANNEL, TEMP_DELTA_KEY],
                [v, i, t, t_delta],
            ):
                sums[key] += float(np.sum(arr))
                sq_sums[key] += float(np.sum(arr**2))

        if total_rows == 0:
            raise ValueError("No data points found.")

        result: dict[str, dict[str, float]] = {}

        for key in sums.keys():
            mean = sums[key] / total_rows
            variance = (sq_sums[key] / total_rows) - (mean**2)

            result[key] = {
                "mean": float(mean),
                "standard_deviation": float(np.sqrt(max(variance, 1e-8))),
            }

        if not soh_values:
            raise ValueError("No curve_soh attributes found across discovered files.")

        soh_array = np.asarray(soh_values, dtype=np.float64)
        result[SOH_KEY] = {
            "mean": float(soh_array.mean()),
            "standard_deviation": float(np.sqrt(max(soh_array.var(), 1e-8))),
        }

        self._print_stats(result, total_rows)
        self.stats = result
        return result

    def save(self) -> None:
        if self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.save_path, "w") as f:
                json.dump(self.stats, f, indent=2)

    @staticmethod
    def _print_stats(stats: dict[str, dict[str, float]], total_rows: int):
        print(f"\nTotal time steps: {total_rows:,}")
        if not stats:
            return

        headers = ["Channel", "Mean", "Standard Deviation"]
        rows = [
            [channel, f"{s['mean']:.4f}", f"{s['standard_deviation']:.4f}"]
            for channel, s in stats.items()
        ]
        print_box_table(
            headers,
            rows,
            alignments=["left", "right", "right"],
            min_widths=[12, 12, 20],
        )
        print()
