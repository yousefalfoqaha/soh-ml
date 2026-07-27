import numpy as np
import torch

from voltgan.config import (
    AMBIENT_TEMPERATURE_KEY,
    CURRENT_CHANNEL,
    SOH_KEY,
    TEMP_DELTA_KEY,
    VOLTAGE_CHANNEL,
    WINDOW_SIZE,
)
from voltgan.dataset.instance import DischargeInstance


class EstimatorDataset(torch.utils.data.Dataset):
    def __init__(self, instances: list[DischargeInstance], stats: dict):
        self._means = np.array(
            [
                stats[VOLTAGE_CHANNEL]["mean"],
                stats[CURRENT_CHANNEL]["mean"],
                stats[AMBIENT_TEMPERATURE_KEY]["mean"],
                stats[SOH_KEY]["mean"],
            ],
            dtype=np.float32,
        )

        self._standard_deviations = np.array(
            [
                stats[VOLTAGE_CHANNEL]["standard_deviation"],
                stats[CURRENT_CHANNEL]["standard_deviation"],
                stats[AMBIENT_TEMPERATURE_KEY]["standard_deviation"],
                stats[SOH_KEY]["standard_deviation"],
            ],
            dtype=np.float32,
        )

        self._temp_delta_mean = stats[TEMP_DELTA_KEY]["mean"]
        self._temp_delta_std = stats[TEMP_DELTA_KEY]["standard_deviation"]

        self.windows = []
        stride = 500
        for instance in instances:
            for start in range(0, len(instance.data) - WINDOW_SIZE + 1, stride):
                self.windows.append((instance, start, start + WINDOW_SIZE))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        instance, start, end = self.windows[i]
        data = instance.data[start:end]

        voltage = (data[:, 0] - self._means[0]) / self._standard_deviations[0]
        current = (data[:, 1] - self._means[1]) / self._standard_deviations[1]

        thermal_rise = data[:, 2] - instance.ambient_temperature
        temperature = (thermal_rise - self._temp_delta_mean) / self._temp_delta_std

        X = torch.stack(
            [
                torch.from_numpy(voltage).float(),
                torch.from_numpy(current).float(),
                torch.from_numpy(temperature).float(),
            ],
            dim=1,
        )

        amb_temp_std = (
            instance.ambient_temperature - self._means[2]
        ) / self._standard_deviations[2]
        conditions = torch.tensor([amb_temp_std], dtype=torch.float32)

        soh_std = (instance.soh - self._means[3]) / self._standard_deviations[3]
        y = torch.tensor([soh_std], dtype=torch.float32)

        return X, conditions, y

    def materialize(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str]]:
        """Flatten every window into two big tensors plus mapping lists.

        Returns ``(X, conditions, window_to_inst, window_to_proto)`` where
        ``window_to_inst`` uses ``id(instance)`` and ``window_to_proto`` is the
        protocol string of each window's source instance.
        """
        X_list: list[torch.Tensor] = []
        cond_list: list[torch.Tensor] = []
        window_to_inst: list[int] = []
        window_to_proto: list[str] = []
        for i, (inst, _, _) in enumerate(self.windows):
            X_i, cond_i, _ = self[i]
            X_list.append(X_i.unsqueeze(0))
            cond_list.append(cond_i.unsqueeze(0))
            window_to_inst.append(id(inst))
            window_to_proto.append(inst.protocol)
        return (
            torch.cat(X_list, dim=0),
            torch.cat(cond_list, dim=0),
            window_to_inst,
            window_to_proto,
        )
