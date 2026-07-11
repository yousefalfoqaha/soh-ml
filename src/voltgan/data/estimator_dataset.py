import numpy as np
import torch

from voltgan.config import WINDOW_SIZE
from voltgan.data.instance import DischargeInstance


class EstimatorDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        instances: list[DischargeInstance],
        stats: dict,
    ):
        self._means = np.array(
            [
                stats["U"]["mean"],
                stats["I"]["mean"],
                stats["ambient_temperature"]["mean"],
                stats["soh"]["mean"],
            ],
            dtype=np.float32,
        )
        self._standard_deviations = np.array(
            [
                stats["U"]["standard_deviation"],
                stats["I"]["standard_deviation"],
                stats["ambient_temperature"]["standard_deviation"],
                stats["soh"]["standard_deviation"],
            ],
            dtype=np.float32,
        )

        self._temp_delta_mean = stats["temp_delta"]["mean"]
        self._temp_delta_std = stats["temp_delta"]["standard_deviation"]

        self.windows = []
        for instance in instances:
            for start in range(0, len(instance.data) - WINDOW_SIZE + 1, WINDOW_SIZE):
                self.windows.append((instance, start, start + WINDOW_SIZE))
        print(f"Loaded {len(self.windows)} windows from {len(instances)} instances.")

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

        thermal_rise = raw_temperature - instance.ambient_temperature
        temperature = (thermal_rise - self._temp_delta_mean) / self._temp_delta_std

        voltage = torch.from_numpy(voltage).float()
        current = torch.from_numpy(current).float()
        temperature = torch.from_numpy(temperature).float()

        ambient_temperature = (
            instance.ambient_temperature - self._means[2]
        ) / self._standard_deviations[2]

        # (instance_length, 3)
        X = torch.stack([voltage, current, temperature], dim=1)

        # (1,)
        conditions = torch.tensor([ambient_temperature], dtype=torch.float32)

        soh_standardized = (instance.soh - self._means[3]) / self._standard_deviations[
            3
        ]

        # (1,)
        y = torch.tensor([soh_standardized], dtype=torch.float32)

        return X, conditions, y