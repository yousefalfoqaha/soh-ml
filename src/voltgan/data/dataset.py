from pathlib import Path

import numpy as np
import torch

from voltgan.config import WINDOW_SIZE
from voltgan.data.instance import DischargeInstance
from voltgan.pipeline.base import discover


class DischargeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mcus: list[str],
        data_path: Path,
        stats: dict,
        windows: bool = False,
    ):
        self._means = np.array(
            [
                stats["U"]["mean"],
                stats["I"]["mean"],
                stats["Temp[1]"]["mean"],
                stats["ambient_temperature"]["mean"],
            ],
            dtype=np.float32,
        )

        self._stds = np.array(
            [
                stats["U"]["standard_deviation"],
                stats["I"]["standard_deviation"],
                stats["Temp[1]"]["standard_deviation"],
                stats["ambient_temperature"]["standard_deviation"],
            ],
            dtype=np.float32,
        )

        self.instances = []

        for hdf_path in discover(data_path, mcus, (".hdf",)):
            instance = DischargeInstance(filepath=hdf_path)
            data_len = len(instance.data)

            if not windows:
                self.instances.append((instance, 0, data_len))
                continue

            for start in range(0, data_len - WINDOW_SIZE + 1):
                self.instances.append((instance, start, start + WINDOW_SIZE))

        print(f"Loaded {len(self.instances)} samples from {len(mcus)} MCU(s).")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        instance, start, end = self.instances[i]

        data = instance.data[start:end]

        raw_voltage = data[:, 0]
        raw_current = data[:, 1]
        raw_temperature = data[:, 2]

        voltage = (raw_voltage - self._means[0]) / self._stds[0]
        current = (raw_current - self._means[1]) / self._stds[1]
        temperature = (raw_temperature - self._means[2]) / self._stds[2]

        ambient_temperature = (
            instance.ambient_temperature - self._means[3]
        ) / self._stds[3]

        X = torch.from_numpy(current).float().unsqueeze(1)

        y = torch.stack(
            [
                torch.from_numpy(voltage).float(),
                torch.from_numpy(temperature).float(),
            ],
            dim=1,
        )

        conditions = torch.tensor(
            [instance.soh, ambient_temperature],
            dtype=torch.float32,
        )

        return X, conditions, y
