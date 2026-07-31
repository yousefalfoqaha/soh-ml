from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from voltgan.config import BATCH_SIZE, SOH_KEY
from voltgan.dataset import EstimatorDataset
from voltgan.evaluation.metrics import PredictionResult
from voltgan.models import SohEstimatorClient


class InferenceEngine:
    def __init__(
        self,
        client: SohEstimatorClient,
        dataset: EstimatorDataset,
        stats: dict,
        batch_size: int = BATCH_SIZE,
    ):
        self.client = client
        self.dataset = dataset
        self.batch_size = batch_size

        # 1. Update to use the Robust Min-Max percentiles instead of mean/std
        self.soh_p01 = stats[SOH_KEY]["p01"]
        self.soh_p99 = stats[SOH_KEY]["p99"]

    @staticmethod
    def aggregate_per_instance(
        preds: np.ndarray, window_to_inst: list[int]
    ) -> dict[int, float]:
        """Average per-window predictions down to per-instance means."""
        bucket: dict[int, list[float]] = defaultdict(list)
        for i, inst_id in enumerate(window_to_inst):
            bucket[inst_id].append(float(preds[i]))
        return {inst_id: float(np.mean(v)) for inst_id, v in bucket.items()}

    def _unscale_predictions(self, scaled_preds: np.ndarray) -> np.ndarray:
        """Applies the inverse of the Robust Min-Max scaling to return physical SoH."""
        denom = self.soh_p99 - self.soh_p01
        if denom == 0:
            denom = 1e-8

        # Inverse of: 2.0 * (x - p01) / denom - 1.0
        return ((scaled_preds + 1.0) / 2.0) * denom + self.soh_p01

    def run_batch_predictions(
        self, X: torch.Tensor, conditions: torch.Tensor
    ) -> np.ndarray:
        """Batched forward on raw (scaled) tensors; returns unscaled physical preds."""
        parts: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                bX = X[i : i + self.batch_size].to(self.client.device)
                bC = conditions[i : i + self.batch_size].to(self.client.device)
                pred = self.client(bX, bC).squeeze(-1)
                parts.append(pred.detach().cpu().numpy())

        raw_scaled = np.concatenate(parts)

        # 2. Return the unscaled values
        return self._unscale_predictions(raw_scaled)

    def run_predictions(self) -> list[PredictionResult]:
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        all_preds: list[np.ndarray] = []
        with torch.no_grad():
            for batch_X, batch_cond, _ in loader:
                pred = self.client(
                    batch_X.to(self.client.device),
                    batch_cond.to(self.client.device),
                ).squeeze(-1)
                all_preds.append(pred.cpu().numpy())

        raw_scaled_preds = np.concatenate(all_preds)

        # 3. Unscale the predictions before aggregating
        unscaled_preds = self._unscale_predictions(raw_scaled_preds)

        window_to_inst = [id(inst) for inst, _, _ in self.dataset.windows]
        inst_preds = self.aggregate_per_instance(unscaled_preds, window_to_inst)

        seen: set[int] = set()
        results: list[PredictionResult] = []
        for inst, _, _ in self.dataset.windows:
            inst_id = id(inst)
            if inst_id in seen:
                continue
            seen.add(inst_id)
            preds = inst_preds.get(inst_id)
            if preds is not None:
                results.append(PredictionResult(inst, preds))
        return results
