import json
from pathlib import Path

import h5py
import numpy as np

from voltgan.config import (
    AMBIENT_TEMPERATURE_KEY,
    SOH_KEY,
    TEMP_DELTA_KEY,
    TEMPERATURE_CHANNEL,
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
        channel_stats: dict[str, dict[str, float]] = {}
        total_rows = 0
        soh_values: list[float] = []

        for instance in instances:
            with h5py.File(instance.filepath, "r") as f:
                n_rows = int(f.attrs.get("total_rows", 0))
                total_rows += n_rows

                group = f[instance.filepath.name]
                keys_to_process = [
                    k
                    for k in group.keys()
                    if isinstance(group[k], h5py.Dataset) and not k.startswith("soh_")
                ]
                keys_to_process.append(AMBIENT_TEMPERATURE_KEY)

                for key in keys_to_process:
                    mean = float(f.attrs.get(f"{key}_mean", 0.0))
                    m2 = float(f.attrs.get(f"{key}_m2", 0.0))
                    if key not in channel_stats:
                        channel_stats[key] = {"sum": 0.0, "m2_sum": 0.0, "sum_sq": 0.0}
                    channel_stats[key]["sum"] += mean * n_rows
                    channel_stats[key]["m2_sum"] += m2
                    channel_stats[key]["sum_sq"] += mean * mean * n_rows

                soh_file = f.attrs.get("curve_soh")
                if soh_file is not None:
                    soh_values.append(float(soh_file))

                if TEMP_DELTA_KEY not in channel_stats:
                    channel_stats[TEMP_DELTA_KEY] = {
                        "sum": 0.0,
                        "m2_sum": 0.0,
                        "sum_sq": 0.0,
                    }
                tr_file_mean = float(f.attrs[f"{TEMPERATURE_CHANNEL}_mean"]) - float(
                    f.attrs[AMBIENT_TEMPERATURE_KEY]
                )
                channel_stats[TEMP_DELTA_KEY]["sum"] += tr_file_mean * n_rows
                channel_stats[TEMP_DELTA_KEY]["sum_sq"] += (
                    tr_file_mean * tr_file_mean * n_rows
                )
                channel_stats[TEMP_DELTA_KEY]["m2_sum"] += float(
                    f.attrs.get(f"{TEMPERATURE_CHANNEL}_m2", 0.0)
                )

        if total_rows == 0:
            raise ValueError("No data points found.")

        result: dict[str, dict[str, float]] = {}
        for channel, stats in channel_stats.items():
            mean = stats["sum"] / total_rows
            variance = (
                stats["m2_sum"] + stats["sum_sq"] - total_rows * mean * mean
            ) / total_rows
            result[channel] = {
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
