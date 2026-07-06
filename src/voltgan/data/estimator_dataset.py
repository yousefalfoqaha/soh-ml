from pathlib import Path

import numpy as np
import torch

from voltgan.config import WINDOW_SIZE
from voltgan.data.instance import DischargeInstance
from voltgan.pipeline.base import discover


class EstimatorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mcus: list[str],
        data_path: Path,
        stats: dict,
    ):
        self._means = np.array(
            [
                stats["U"]["mean"],
                stats["I"]["mean"],
                stats["Temp[1]"]["mean"],
                stats["ambient_temperature"]["mean"],
                stats["soh"]["mean"],
            ],
            dtype=np.float32,
        )
        self._standard_deviations = np.array(
            [
                stats["U"]["standard_deviation"],
                stats["I"]["standard_deviation"],
                stats["Temp[1]"]["standard_deviation"],
                stats["ambient_temperature"]["standard_deviation"],
                stats["soh"]["standard_deviation"],
            ],
            dtype=np.float32,
        )

        self.windows = []
        for hdf_path in discover(data_path, mcus, (".hdf",)):
            sample = DischargeInstance(filepath=hdf_path)
            for start in range(0, len(sample.data) - WINDOW_SIZE + 1, WINDOW_SIZE):
                self.windows.append((sample, start, start + WINDOW_SIZE))
        print(f"Loaded {len(self.windows)} windows from {len(mcus)} MCU(s).")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        instance, start, end = self.windows[i]
        data = instance.data[start:end]

        raw_voltage = data[:, 0]
        raw_current = data[:, 1]
        raw_temperature = data[:, 2]

        voltage = (raw_voltage - self._means[0]) / self._standard_deviations[0]
        current = (raw_current - self._means[1]) / self._standard_deviations[1]
        temperature = (raw_temperature - self._means[2]) / self._standard_deviations[2]

        voltage = torch.from_numpy(voltage).float()
        current = torch.from_numpy(current).float()
        temperature = torch.from_numpy(temperature).float()

        # (instance_length, 3)
        X = torch.stack([voltage, current, temperature], dim=1)

        soh_standardized = (instance.soh - self._means[4]) / self._standard_deviations[
            4
        ]

        # (1,)
        y = torch.tensor([soh_standardized], dtype=torch.float32)

        return X, y
