from pathlib import Path

import numpy as np
import torch

from voltgan.data.instance import DischargeInstance
from voltgan.pipeline.base import discover


class DischargeDataset(torch.utils.data.Dataset):
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

        self.instances: list[DischargeInstance] = []
        for hdf_path in discover(data_path, mcus, (".hdf",)):
            sample = DischargeInstance(filepath=hdf_path)
            self.instances.append(sample)
        print(f"Loaded {len(self.instances)} instances from {len(mcus)} MCU(s).")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        instance = self.instances[i]

        raw_voltage = instance.data[:, 0]
        raw_current = instance.data[:, 1]
        raw_temperature = instance.data[:, 2]

        voltage = (raw_voltage - self._means[0]) / self._standard_deviations[0]
        current = (raw_current - self._means[1]) / self._standard_deviations[1]
        temperature = (raw_temperature - self._means[2]) / self._standard_deviations[2]

        ambient_temperature = (
            instance.ambient_temperature - self._means[3]
        ) / self._standard_deviations[3]

        soh = (instance.soh - self._means[4]) / self._standard_deviations[4]

        voltage = torch.from_numpy(voltage).float()
        current = torch.from_numpy(current).float()
        temperature = torch.from_numpy(temperature).float()

        # (instance_length, 1)
        X = torch.stack([current], dim=1)

        # (2,)
        conditions = torch.tensor([soh, ambient_temperature], dtype=torch.float32)

        # (instance_length, 2)
        y = torch.stack([voltage, temperature], dim=1)

        return X, conditions, y
