import numpy as np
import torch

from voltgan.data.instance import DischargeInstance


class DischargeDataset(torch.utils.data.Dataset):
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

        self.instances = instances
        print(f"Loaded {len(self.instances)} instances.")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        instance = self.instances[i]

        raw_voltage = instance.data[:, 0]
        raw_current = instance.data[:, 1]
        raw_temperature = instance.data[:, 2]

        voltage = (raw_voltage - self._means[0]) / self._standard_deviations[0]
        current = (raw_current - self._means[1]) / self._standard_deviations[1]

        thermal_rise = raw_temperature - instance.ambient_temperature
        temperature = (thermal_rise - self._temp_delta_mean) / self._temp_delta_std

        ambient_temperature = (
            instance.ambient_temperature - self._means[2]
        ) / self._standard_deviations[2]

        soh = (instance.soh - self._means[3]) / self._standard_deviations[3]

        voltage = torch.from_numpy(voltage).float()
        current = torch.from_numpy(current).float()
        temperature = torch.from_numpy(temperature).float()

        # (instance_length, 1)
        X = torch.stack([current], dim=1)

        # (2,)
        conditions = torch.tensor([soh, ambient_temperature], dtype=torch.float32)

        # (instance_length, 1)
        y = torch.stack([voltage], dim=1)

        return X, conditions, y
