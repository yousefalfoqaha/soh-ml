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
        self.soh_mean = stats[SOH_KEY]["mean"]
        self.soh_std = stats[SOH_KEY]["standard_deviation"]

    @staticmethod
    def aggregate_per_instance(
        preds: np.ndarray, window_to_inst: list[int]
    ) -> dict[int, float]:
        """Average per-window predictions down to per-instance means."""
        bucket: dict[int, list[float]] = defaultdict(list)
        for i, inst_id in enumerate(window_to_inst):
            bucket[inst_id].append(float(preds[i]))
        return {inst_id: float(np.mean(v)) for inst_id, v in bucket.items()}

    def run_batch_predictions(
        self, X: torch.Tensor, conditions: torch.Tensor
    ) -> np.ndarray:
        """Batched forward on raw (standardized) tensors; returns destandardized preds."""
        parts: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                bX = X[i : i + self.batch_size].to(self.client.device)
                bC = conditions[i : i + self.batch_size].to(self.client.device)
                pred = self.client(bX, bC).squeeze(-1)
                parts.append(pred.detach().cpu().numpy())
        raw = np.concatenate(parts)
        return raw * self.soh_std + self.soh_mean

    def run_predictions(self) -> list[PredictionResult]:
        loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        all_preds: list[np.ndarray] = []
        with torch.no_grad():
            for batch_X, batch_cond, _ in loader:
                pred = self.client(
                    batch_X.to(self.client.device),
                    batch_cond.to(self.client.device),
                ).squeeze(-1)
                all_preds.append(pred.cpu().numpy())

        raw_preds = np.concatenate(all_preds)
        destd_preds = raw_preds * self.soh_std + self.soh_mean

        window_to_inst = [id(inst) for inst, _, _ in self.dataset.windows]
        inst_preds = self.aggregate_per_instance(destd_preds, window_to_inst)

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
