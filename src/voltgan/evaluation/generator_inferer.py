from __future__ import annotations

import h5py
import numpy as np
import torch
import torch.nn.functional as F

from voltgan.config import (
    AMBIENT_TEMPERATURE_KEY,
    CONV_HIDDEN_LAYERS,
    CURRENT_CHANNEL,
    NOISE_DIM,
    SOH_KEY,
    TEMP_DELTA_KEY,
    TEMPERATURE_CHANNEL,
    VOLTAGE_CHANNEL,
)
from voltgan.models import GeneratorClient


class GeneratorInferer:
    """Runs single-instance generator inference given a loaded HDF file.

    Standardizes inputs, forwards through `GeneratorClient`, destandardizes the
    (voltage, thermal_delta) outputs. Pure computation — no plotting (the CLI
    orchestrator owns matplotlib).
    """

    def __init__(self, client: GeneratorClient, stats: dict):
        self.client = client
        self.stats = stats

    def predict(
        self,
        filepath,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Read `filepath`, run inference, return tuples for plotting.

        Returns `(voltage_true, voltage_pred, temperature_true, temperature_pred,
                  soh_condition, ambient_condition)`."""
        with h5py.File(filepath, "r") as f:
            group = f[filepath.name]
            assert isinstance(group, h5py.Group)

            def _load(ch: str) -> np.ndarray:
                dataset = group[ch]
                assert isinstance(dataset, h5py.Dataset)
                return dataset[:]

            current = _load(CURRENT_CHANNEL)
            voltage = _load(VOLTAGE_CHANNEL)
            temperature = _load(TEMPERATURE_CHANNEL)

            soh_attr = float(f.attrs.get("curve_soh", 1.0))
            ambient_attr = float(f.attrs.get(AMBIENT_TEMPERATURE_KEY, 25.0))

        current_std = self._standardize(current, self.stats[CURRENT_CHANNEL])
        voltage_std = self._standardize(voltage, self.stats[VOLTAGE_CHANNEL])
        thermal_rise = temperature - ambient_attr
        temp_delta_std = self._standardize(thermal_rise, self.stats[TEMP_DELTA_KEY])

        voltage_true = self._destandardize(voltage_std, self.stats[VOLTAGE_CHANNEL])
        temperature_true = (
            self._destandardize(temp_delta_std, self.stats[TEMP_DELTA_KEY])
            + ambient_attr
        )

        voltage_pred_std, temp_delta_pred_std = self._forward(current_std, soh_attr, ambient_attr)
        voltage_pred = self._destandardize(voltage_pred_std, self.stats[VOLTAGE_CHANNEL])
        temperature_pred = (
            self._destandardize(temp_delta_pred_std, self.stats[TEMP_DELTA_KEY]) + ambient_attr
        )
        return voltage_true, voltage_pred, temperature_true, temperature_pred, soh_attr, ambient_attr

    def predict_with_overrides(
        self,
        filepath,
        *,
        soh_override: float | None = None,
        ambient_override: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Same as `predict`, but allow overriding the conditioning values."""
        with h5py.File(filepath, "r") as f:
            group = f[filepath.name]
            assert isinstance(group, h5py.Group)

            def _load(ch: str) -> np.ndarray:
                dataset = group[ch]
                assert isinstance(dataset, h5py.Dataset)
                return dataset[:]

            current = _load(CURRENT_CHANNEL)
            voltage = _load(VOLTAGE_CHANNEL)
            temperature = _load(TEMPERATURE_CHANNEL)

            soh_attr = float(f.attrs.get("curve_soh", 1.0))
            ambient_attr = float(f.attrs.get(AMBIENT_TEMPERATURE_KEY, 25.0))

        soh_cond = soh_override if soh_override is not None else soh_attr
        ambient_cond = ambient_override if ambient_override is not None else ambient_attr

        current_std = self._standardize(current, self.stats[CURRENT_CHANNEL])
        voltage_std = self._standardize(voltage, self.stats[VOLTAGE_CHANNEL])
        thermal_rise = temperature - ambient_attr
        temp_delta_std = self._standardize(thermal_rise, self.stats[TEMP_DELTA_KEY])

        voltage_true = self._destandardize(voltage_std, self.stats[VOLTAGE_CHANNEL])
        temperature_true = (
            self._destandardize(temp_delta_std, self.stats[TEMP_DELTA_KEY])
            + ambient_attr
        )

        voltage_pred_std, temp_delta_pred_std = self._forward(current_std, soh_cond, ambient_cond)
        voltage_pred = self._destandardize(voltage_pred_std, self.stats[VOLTAGE_CHANNEL])
        temperature_pred = (
            self._destandardize(temp_delta_pred_std, self.stats[TEMP_DELTA_KEY]) + ambient_cond
        )
        return voltage_true, voltage_pred, temperature_true, temperature_pred, soh_cond, ambient_cond

    @torch.no_grad()
    def _forward(
        self, current_std: np.ndarray, soh_cond: float, ambient_cond: float
    ) -> tuple[np.ndarray, np.ndarray]:
        orig_length = current_std.shape[0]

        X = (
            torch.tensor(current_std, dtype=torch.float32)
            .unsqueeze(-1)
            .unsqueeze(0)
            .to(self.client.device)
        )

        amb_std = (ambient_cond - self.stats[AMBIENT_TEMPERATURE_KEY]["mean"]) / self.stats[
            AMBIENT_TEMPERATURE_KEY
        ]["standard_deviation"]
        soh_std = (soh_cond - self.stats[SOH_KEY]["mean"]) / self.stats[SOH_KEY][
            "standard_deviation"
        ]
        conditions = torch.tensor([[soh_std, amb_std]], dtype=torch.float32).to(
            self.client.device
        )

        downsample_factor = 5**CONV_HIDDEN_LAYERS
        remainder = X.size(1) % downsample_factor
        if remainder != 0:
            pad_len = downsample_factor - remainder
            X = F.pad(X, (0, 0, 0, pad_len), value=0.0)

        noise = torch.rand(1, NOISE_DIM, device=self.client.device)
        y_hat = self.client(X, conditions, noise)
        pred = y_hat.squeeze(0).cpu().numpy()[:orig_length]
        return pred[:, 0], pred[:, 1]

    @staticmethod
    def _standardize(arr: np.ndarray, s: dict) -> np.ndarray:
        return (arr - s["mean"]) / s["standard_deviation"]

    @staticmethod
    def _destandardize(arr: np.ndarray, s: dict) -> np.ndarray:
        return arr * s["standard_deviation"] + s["mean"]