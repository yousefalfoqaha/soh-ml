from __future__ import annotations

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
    """Lazy-loading dataset for battery discharge windows utilizing Robust Min-Max Scaling."""

    def __init__(self, instances: list[DischargeInstance], stats: dict):
        self.v_p01, self.v_p99 = (
            stats[VOLTAGE_CHANNEL]["p01"],
            stats[VOLTAGE_CHANNEL]["p99"],
        )
        self.i_p01, self.i_p99 = (
            stats[CURRENT_CHANNEL]["p01"],
            stats[CURRENT_CHANNEL]["p99"],
        )
        self.td_p01, self.td_p99 = (
            stats[TEMP_DELTA_KEY]["p01"],
            stats[TEMP_DELTA_KEY]["p99"],
        )
        self.amb_p01, self.amb_p99 = (
            stats[AMBIENT_TEMPERATURE_KEY]["p01"],
            stats[AMBIENT_TEMPERATURE_KEY]["p99"],
        )
        self.soh_p01, self.soh_p99 = stats[SOH_KEY]["p01"], stats[SOH_KEY]["p99"]

        self.windows = []
        stride = 500
        for instance in instances:
            for start in range(0, len(instance.data) - WINDOW_SIZE + 1, stride):
                self.windows.append((instance, start, start + WINDOW_SIZE))

        print(f"Loaded {len(self.windows)} windows from {len(instances)} instances.")

    @staticmethod
    def _robust_scale(
        x: np.ndarray | float, p01: float, p99: float
    ) -> np.ndarray | float:
        """Clips data to the 1st/99th percentiles and scales it to strictly [-1, 1]."""
        denominator = p99 - p01
        if denominator == 0:
            denominator = 1e-8

        clipped_x = np.clip(x, p01, p99)
        return 2.0 * (clipped_x - p01) / denominator - 1.0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        instance, start, end = self.windows[i]
        data = instance.data[start:end]

        v_scaled = self._robust_scale(data[:, 0], self.v_p01, self.v_p99)
        i_scaled = self._robust_scale(data[:, 1], self.i_p01, self.i_p99)

        thermal_rise = data[:, 2] - instance.ambient_temperature
        t_scaled = self._robust_scale(thermal_rise, self.td_p01, self.td_p99)

        X = torch.stack(
            [
                torch.from_numpy(v_scaled).float(),
                torch.from_numpy(i_scaled).float(),
                torch.from_numpy(t_scaled).float(),
            ],
            dim=1,
        )

        amb_scaled = self._robust_scale(
            instance.ambient_temperature, self.amb_p01, self.amb_p99
        )
        conditions = torch.tensor([amb_scaled], dtype=torch.float32)

        soh_scaled = self._robust_scale(instance.curve_soh, self.soh_p01, self.soh_p99)
        y = torch.tensor([soh_scaled], dtype=torch.float32)

        return X, conditions, y
