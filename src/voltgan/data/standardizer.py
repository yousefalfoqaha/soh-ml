import json
from pathlib import Path

import h5py
import numpy as np

from voltgan.pipeline.base import discover


class Standardizer:
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.stats = {}

    def compute(self, mcus: list[str]) -> dict[str, dict[str, float]]:
        channel_stats: dict[str, dict[str, float]] = {}
        total_rows = 0

        data_path = self.data_path / "hdf"
        for hdf_path in discover(data_path, mcus, (".hdf",)):
            with h5py.File(hdf_path, "r") as f:
                n_rows = int(f.attrs.get("total_rows", 0))

                total_rows += n_rows

                group = f[hdf_path.name]
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

        self._print_stats(result, total_rows)
        self.stats = result

        return result

    def save(self) -> None:
        stats_path = self.data_path / "stats.json"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)

    @staticmethod
    def _print_stats(stats: dict[str, dict[str, float]], total_rows: int):
        print(f"\nTotal time steps: {total_rows:,}")

        if not stats:
            print("No stats available to display.")
            return

        col1_w = max(len(channel) for channel in stats.keys())
        col1_w = max(col1_w, 12)
        col2_w = 12
        col3_w = 20

        top_border = f"┌{'─' * (col1_w + 2)}┬{'─' * (col2_w + 2)}┬{'─' * (col3_w + 2)}┐"
        middle_border = (
            f"├{'─' * (col1_w + 2)}┼{'─' * (col2_w + 2)}┼{'─' * (col3_w + 2)}┤"
        )
        bottom_border = (
            f"└{'─' * (col1_w + 2)}┴{'─' * (col2_w + 2)}┴{'─' * (col3_w + 2)}┘"
        )

        print(top_border)
        print(
            f"│ {'Channel':<{col1_w}} │ {'Mean':>{col2_w}} │ {'Standard Deviation':>{col3_w}} │"
        )
        print(middle_border)

        for channel, s in stats.items():
            mean_val = s.get("mean", 0.0)
            std_val = s.get("standard_deviation", 0.0)
            print(
                f"│ {channel:<{col1_w}} │ {mean_val:>{col2_w}.4f} │ {std_val:>{col3_w}.4f} │"
            )

        print(bottom_border)
        print()
