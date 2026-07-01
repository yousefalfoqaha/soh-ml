from pathlib import Path

import numpy as np
import torch

from voltgan.data.mcu_sample import McuSample
from voltgan.pipeline.base import discover


class McusDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mcus: list[str],
        data_path: Path,
        window_length: int,
        stride: int,
        stats: dict,
    ):
        self.window_length = window_length

        self._means = np.array(
            [
                stats["U"]["mean"],
                stats["I"]["mean"],
                stats["Temp[1]"]["mean"],
                stats["ClimaTemp"]["mean"],
                stats["Q"]["mean"],
            ],
            dtype=np.float32,
        )
        self._standard_deviations = np.array(
            [
                stats["U"]["standard_deviation"],
                stats["I"]["standard_deviation"],
                stats["Temp[1]"]["standard_deviation"],
                stats["ClimaTemp"]["standard_deviation"],
                stats["Q"]["standard_deviation"],
            ],
            dtype=np.float32,
        )

        self.samples: list[McuSample] = []
        self.window_map: list[tuple[int, int]] = []

        for hdf_path in discover(data_path, mcus, (".hdf",)):
            sample = McuSample(filepath=hdf_path)
            sample_i = len(self.samples)
            self.samples.append(sample)

            n_windows = (sample.n_samples - window_length) // stride + 1
            for i in range(n_windows):
                self.window_map.append((sample_i, i * stride))

        print(f"Mapped {len(self.window_map):,} windows from {len(mcus)} MCU(s).")

    def __len__(self):
        return len(self.window_map)

    def __getitem__(self, i):
        sample_i, start_i = self.window_map[i]
        sample = self.samples[sample_i]
        end_i = start_i + self.window_length

        # (window_length, 4)
        raw = sample.load_window(start_i, end_i)
        window = (raw - self._means) / self._standard_deviations

        # (window_length,)
        voltage = torch.from_numpy(window[:, 0]).float()
        temperature = torch.from_numpy(window[:, 2]).float()
        current = torch.from_numpy(window[:, 1]).float()
        ambient_temperature = torch.from_numpy(window[:, 3]).float()
        charge = torch.from_numpy(window[:, 4]).float()

        # (window_length, 3)
        X = torch.stack([current, ambient_temperature, charge], dim=1)

        # (1,)
        conditions = torch.tensor([sample.soh], dtype=torch.float32)

        # (window_length, 2)
        y = torch.stack([voltage, temperature], dim=1)

        return conditions, y
