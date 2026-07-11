import json
from pathlib import Path

import h5py
import numpy as np

from voltgan.data.instance import DischargeInstance
from voltgan.utils.box_table import print_box_table


class Standardizer:
    def __init__(self, stats_path: Path):
        self.stats_path = stats_path
        self.stats = {}

    def compute(self, instances: list[DischargeInstance]) -> dict[str, dict[str, float]]:
        channel_stats: dict[str, dict[str, float]] = {}
        total_rows = 0

        soh_values: list[float] = []
        temp_rises: list[float] = []

        for instance in instances:
            with h5py.File(instance.filepath, "r") as f:
                n_rows = int(f.attrs.get("total_rows", 0))
                total_rows += n_rows

                group = f[instance.filepath.name]
                assert isinstance(group, h5py.Group)

                keys_to_process = []
                for key in group.keys():
                    if key.startswith("soh_"):
                        continue
                    if isinstance(group[key], h5py.Dataset):
                        keys_to_process.append(key)
                keys_to_process.append("ambient_temperature")

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

                temp_min = float(f.attrs.get("Temp[1]_min", 0.0))
                temp_max = float(f.attrs.get("Temp[1]_max", 0.0))
                temp_rises.append(temp_max - temp_min)

        if total_rows == 0:
            raise ValueError("No data points found.")

        result: dict[str, dict[str, float]] = {}
        for channel, stats in channel_stats.items():
            mean = stats["sum"] / total_rows
            variance = (
                stats["m2_sum"] + stats["sum_sq"] - total_rows * mean * mean
            ) / total_rows
            standard_deviation = float(np.sqrt(max(variance, 1e-8)))
            result[channel] = {
                "mean": float(mean),
                "standard_deviation": standard_deviation,
            }

        if not soh_values:
            raise ValueError("No curve_soh attributes found across discovered files.")

        soh_array = np.asarray(soh_values, dtype=np.float64)
        soh_mean = float(soh_array.mean())
        soh_std = float(np.sqrt(max(soh_array.var(), 1e-8)))
        result["soh"] = {
            "mean": soh_mean,
            "standard_deviation": soh_std,
        }

        temp_rise_array = np.asarray(temp_rises, dtype=np.float64)
        result["temp_delta"] = {
            "mean": float(temp_rise_array.mean()),
            "standard_deviation": float(np.sqrt(max(temp_rise_array.var(), 1e-8))),
        }

        self._print_stats(result, total_rows)
        self.stats = result
        return result

    def save(self) -> None:
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)

    @staticmethod
    def _print_stats(stats: dict[str, dict[str, float]], total_rows: int):
        print(f"\nTotal time steps: {total_rows:,}")
        if not stats:
            print("No stats available to display.")
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