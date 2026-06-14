from pathlib import Path

import h5py
import numpy as np

from pipeline import discover


class Standardizer:
    def __init__(self, data_path: Path):
        self.data_path = data_path

    def compute(self, mcus: list[str]) -> dict[str, dict[str, float]]:
        print("\nCalculating global standardization stats...")

        channel_stats: dict[str, dict[str, float]] = {}
        total_rows = 0

        for hdf_path in discover(self.data_path, mcus, (".hdf",)):
            with h5py.File(hdf_path, "r") as f:
                n = int(f.attrs.get("total_rows", 0))
                if n == 0:
                    continue
                total_rows += n

                grp = f[hdf_path.name]
                assert isinstance(grp, h5py.Group)

                for key in grp.keys():
                    if key.startswith("soh_"):
                        continue

                    ds = grp[key]
                    if not isinstance(ds, h5py.Dataset):
                        continue

                    mean = float(f.attrs.get(f"{key}_mean", 0.0))
                    m2 = float(f.attrs.get(f"{key}_m2", 0.0))

                    if key not in channel_stats:
                        channel_stats[key] = {"sum": 0.0, "m2_sum": 0.0}

                    channel_stats[key]["sum"] += mean * n
                    channel_stats[key]["m2_sum"] += m2

        if total_rows == 0:
            raise ValueError("No data points found.")

        result: dict[str, dict[str, float]] = {}
        for channel, stats in channel_stats.items():
            mean = stats["sum"] / total_rows
            var = stats["m2_sum"] / total_rows
            std = float(np.sqrt(max(var, 1e-8)))
            result[channel] = {"mean": float(mean), "std": std}

        self._print_stats(result, total_rows)
        return result

    @staticmethod
    def _print_stats(stats: dict[str, dict[str, float]], total_rows: int):
        print(f"  Total time steps: {total_rows:,}")
        for channel, s in stats.items():
            print(f"  {channel:12s} -> Mean: {s['mean']:10.4f} | Std: {s['std']:10.4f}")

