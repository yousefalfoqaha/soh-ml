import json
from pathlib import Path

import numpy as np

from voltgan.config import (
    AMBIENT_TEMPERATURE_KEY,
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

    def calculate_mean_std(
        self, instances: list[DischargeInstance]
    ) -> dict[str, dict[str, float]]:
        """Calculates traditional Mean and Standard Deviation (for paper/analysis)."""
        stat_keys = [
            VOLTAGE_CHANNEL,
            CURRENT_CHANNEL,
            TEMPERATURE_CHANNEL,
            TEMP_DELTA_KEY,
            AMBIENT_TEMPERATURE_KEY,
        ]

        sums = {ch: 0.0 for ch in stat_keys}
        sq_sums = {ch: 0.0 for ch in stat_keys}

        total_rows = 0
        soh_values: list[float] = []

        for inst in instances:
            n_rows = len(inst)
            if n_rows == 0:
                continue

            total_rows += n_rows
            soh_values.append(inst.curve_soh)

            v = inst.voltage
            i = inst.current
            t = inst.temperature
            t_delta = t - inst.ambient_temperature
            amb = inst.ambient_temperature

            for key, val in zip(stat_keys, [v, i, t, t_delta, amb]):
                if key == AMBIENT_TEMPERATURE_KEY:
                    sums[key] += float(val * n_rows)
                    sq_sums[key] += float((val**2) * n_rows)
                else:
                    sums[key] += float(np.sum(val))
                    sq_sums[key] += float(np.sum(val**2))

        if total_rows == 0:
            raise ValueError("No data points found.")

        result: dict[str, dict[str, float]] = {}

        for key in sums.keys():
            mean = sums[key] / total_rows
            variance = (sq_sums[key] / total_rows) - (mean**2)
            std = float(np.sqrt(max(variance, 1e-8)))
            result[key] = {"mean": float(mean), "standard_deviation": std}

        soh_array = np.asarray(soh_values, dtype=np.float64)
        result[SOH_KEY] = {
            "mean": float(soh_array.mean()),
            "standard_deviation": float(np.sqrt(max(soh_array.var(), 1e-8))),
        }

        self._print_stats(
            "Mean & Std (Traditional)", result, total_rows, ["Mean", "Std Dev"]
        )
        return result

    def compute(
        self, instances: list[DischargeInstance]
    ) -> dict[str, dict[str, float]]:
        """Calculates Robust Min-Max bounds (1st and 99th Percentiles) for neural network scaling."""
        stat_keys = [
            VOLTAGE_CHANNEL,
            CURRENT_CHANNEL,
            TEMPERATURE_CHANNEL,
            TEMP_DELTA_KEY,
            AMBIENT_TEMPERATURE_KEY,
            SOH_KEY,
        ]

        data_buffers = {key: [] for key in stat_keys}
        total_rows = 0

        for inst in instances:
            n_rows = len(inst)
            if n_rows == 0:
                continue

            total_rows += n_rows

            data_buffers[VOLTAGE_CHANNEL].append(inst.voltage)
            data_buffers[CURRENT_CHANNEL].append(inst.current)
            data_buffers[TEMPERATURE_CHANNEL].append(inst.temperature)
            data_buffers[TEMP_DELTA_KEY].append(
                inst.temperature - inst.ambient_temperature
            )
            data_buffers[AMBIENT_TEMPERATURE_KEY].append(
                np.full(n_rows, inst.ambient_temperature)
            )
            data_buffers[SOH_KEY].append(np.full(n_rows, inst.curve_soh))

        if total_rows == 0:
            raise ValueError("No data points found.")

        result: dict[str, dict[str, float]] = {}

        for key in stat_keys:
            concat_data = np.concatenate(data_buffers[key])

            p01 = float(np.percentile(concat_data, 1))
            p99 = float(np.percentile(concat_data, 99))

            result[key] = {
                "p01": p01,
                "p99": p99,
            }

        self._print_stats(
            "Robust Min-Max Percentiles", result, total_rows, ["1st %ile", "99th %ile"]
        )
        self.stats = result
        return result

    def save(self) -> None:
        if self.save_path is not None:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.save_path, "w") as f:
                json.dump(self.stats, f, indent=2)

    @staticmethod
    def _print_stats(
        title: str,
        stats: dict[str, dict[str, float]],
        total_rows: int,
        col_names: list[str],
    ):
        print(f"\n{title} | Total time steps: {total_rows:,}")
        if not stats:
            return

        headers = ["Channel", col_names[0], col_names[1]]

        k1 = "mean" if "mean" in next(iter(stats.values())) else "p01"
        k2 = (
            "standard_deviation"
            if "standard_deviation" in next(iter(stats.values()))
            else "p99"
        )

        rows = [
            [channel, f"{s[k1]:.4f}", f"{s[k2]:.4f}"] for channel, s in stats.items()
        ]
        print_box_table(
            headers,
            rows,
            alignments=["left", "right", "right"],
            min_widths=[12, 12, 20],
        )
        print()
