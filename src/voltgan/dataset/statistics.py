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

    def compute(
        self, instances: list[DischargeInstance]
    ) -> dict[str, dict[str, float]]:
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
            n_rows = inst.n_samples
            if n_rows == 0:
                continue

            total_rows += n_rows
            soh_values.append(inst.curve_soh)

            v = inst.voltage
            i = inst.current
            t = inst.temperature
            t_delta = t - inst.ambient_temperature
            amb = inst.ambient_temperature

            for key, val in zip(
                stat_keys,
                [v, i, t, t_delta, amb],
            ):
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

            if np.isnan(mean) or np.isinf(mean) or np.isnan(std) or np.isinf(std):
                raise ValueError(
                    f"Invalid or NaN/Inf statistics detected for channel '{key}': mean={mean}, std={std}"
                )

            if key == VOLTAGE_CHANNEL and (mean < 1.0 or mean > 5.0):
                raise ValueError(f"Unrealistic voltage mean detected: {mean:.4f}")
            if key == TEMPERATURE_CHANNEL and (mean < -50.0 or mean > 150.0):
                raise ValueError(f"Unrealistic temperature mean detected: {mean:.4f}")
            if key == AMBIENT_TEMPERATURE_KEY and (mean < -50.0 or mean > 100.0):
                raise ValueError(
                    f"Unrealistic ambient temperature mean detected: {mean:.4f}"
                )

            result[key] = {
                "mean": float(mean),
                "standard_deviation": std,
            }

        if not soh_values:
            raise ValueError("No curve_soh attributes found across discovered files.")

        soh_array = np.asarray(soh_values, dtype=np.float64)
        soh_mean = float(soh_array.mean())
        soh_std = float(np.sqrt(max(soh_array.var(), 1e-8)))

        if (
            np.isnan(soh_mean)
            or np.isinf(soh_mean)
            or soh_mean <= 0.0
            or soh_mean > 1.0
        ):
            raise ValueError(
                f"Unrealistic or invalid SOH mean detected: {soh_mean:.4f}"
            )

        result[SOH_KEY] = {
            "mean": soh_mean,
            "standard_deviation": soh_std,
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
