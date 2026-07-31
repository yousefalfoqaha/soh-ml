from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from voltgan.dataset import EstimatorDataset
from voltgan.evaluation import InferenceEngine
from voltgan.evaluation.metrics import MetricsAggregator, MetricSet, PredictionResult
from voltgan.utils.latex import TableRow


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    tensor_type: str  # "X" or "cond"
    channels: tuple[int, ...]


@dataclass
class PfiResult:
    feature_name: str
    protocol: str
    deltas: list[float]

    @property
    def mean_delta(self) -> float:
        return float(np.mean(self.deltas)) if self.deltas else 0.0


@dataclass
class PfiReport:
    """Rich domain object representing the results of a PFI evaluation."""

    features: list[FeatureSpec]
    protocols: list[str]
    baseline_metrics: dict[str, MetricSet]
    results: dict[str, dict[str, PfiResult]]

    @property
    def active_protocols(self) -> list[str]:
        """Returns protocols that actually had windows evaluated."""
        return [
            p
            for p in self.protocols
            if p in self.baseline_metrics and self.baseline_metrics[p].cycles > 0
        ]

    @property
    def ranked_features(self) -> list[FeatureSpec]:
        """Ranks features by their mean delta impact across all active protocols."""
        active = self.active_protocols
        if not active:
            return self.features

        def _mean_impact(f: FeatureSpec) -> float:
            impacts = [self.results[f.name][p].mean_delta for p in active]
            return float(np.mean(impacts))

        return sorted(self.features, key=_mean_impact, reverse=True)

    def to_latex_delta_rows(self) -> list[TableRow]:
        """Generates the transposed LaTeX TableRow objects (Rows=Protocols, Cols=Features)."""
        active_protos = self.active_protocols
        ranked = self.ranked_features
        rows = []

        for p in active_protos:
            cells = [p]

            best_feat = None
            best_mean = -float("inf")
            for f in ranked:
                res = self.results[f.name].get(p)
                if res and res.deltas and res.mean_delta > best_mean:
                    best_mean = res.mean_delta
                    best_feat = f.name

            for f in ranked:
                res = self.results[f.name].get(p)
                if not res or not res.deltas:
                    cells.append("--")
                else:
                    m = res.mean_delta
                    s = f"+{m:.4f}" if m >= 0 else f"{m:.4f}"
                    if f.name == best_feat and m > 0:
                        s = rf"\textbf{{{s}}}"
                    cells.append(s)

            rows.append(TableRow(cells=cells))

        return rows


class PermutationImportanceEvaluator:
    def __init__(
        self, engine: InferenceEngine, dataset: EstimatorDataset, repeats: int = 5
    ):
        self.engine = engine
        self.dataset = dataset
        self.repeats = repeats

    @staticmethod
    def _materialize_dataset(
        dataset: EstimatorDataset,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str]]:
        """Extracts bulk tensors from the dataset for rapid permutation."""
        X_list, cond_list, window_to_inst, window_to_proto = [], [], [], []
        for i, (inst, _, _) in enumerate(dataset.windows):
            X_i, cond_i, _ = dataset[i]
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

    def run(self, features: list[FeatureSpec], protocols: list[str]) -> PfiReport:
        X, conditions, window_to_inst, window_to_proto = self._materialize_dataset(
            self.dataset
        )
        inst_by_id = {id(inst): inst for inst, _, _ in self.dataset.windows}

        base_preds = self.engine.run_batch_predictions(X, conditions)
        base_inst_pred = self.engine.aggregate_per_instance(base_preds, window_to_inst)

        protocol_windows = defaultdict(list)
        # New: Track micro-strata indices within each protocol
        protocol_strata_windows = defaultdict(lambda: defaultdict(list))

        for w_idx, proto in enumerate(window_to_proto):
            protocol_windows[proto].append(w_idx)

            # Build the strict stratum key based on protocol, temp, and discharge rate
            inst = inst_by_id[window_to_inst[w_idx]]
            stratum = f"{proto}_{inst.temp_center}_{inst.discharge_rate}"
            protocol_strata_windows[proto][stratum].append(w_idx)

        protocol_instance_ids = {
            p: {window_to_inst[i] for i in idxs} for p, idxs in protocol_windows.items()
        }

        baseline = {}
        for p in protocols:
            ids = protocol_instance_ids.get(p, set())
            results = [
                PredictionResult(inst_by_id[iid], base_inst_pred[iid])
                for iid in ids
                if iid in base_inst_pred
            ]
            baseline[p] = MetricsAggregator.compute(p, results)

        # 3. Permutation
        pfi_results: dict[str, dict[str, PfiResult]] = {f.name: {} for f in features}

        for spec in features:
            for p in protocols:
                p_windows = protocol_windows.get(p, [])
                if not p_windows or baseline[p].cycles == 0:
                    pfi_results[spec.name][p] = PfiResult(spec.name, p, [])
                    continue

                deltas = []
                for _ in range(self.repeats):
                    X_perm, cond_perm = X.clone(), conditions.clone()

                    # Permute strictly within each micro-stratum to preserve physical validity
                    for stratum, bucket_idx in protocol_strata_windows[p].items():
                        # If a stratum only has 1 window, it cannot be permuted with itself
                        if len(bucket_idx) < 2:
                            continue

                        local = torch.randperm(len(bucket_idx))

                        if spec.tensor_type == "X":
                            for ch in spec.channels:
                                X_perm[bucket_idx, :, ch] = X_perm[bucket_idx, :, ch][
                                    local
                                ]
                        else:
                            cond_perm[bucket_idx, 0] = cond_perm[bucket_idx, 0][local]

                    perm_preds = self.engine.run_batch_predictions(X_perm, cond_perm)
                    perm_inst_pred = self.engine.aggregate_per_instance(
                        perm_preds, window_to_inst
                    )
                    perm_results = [
                        PredictionResult(inst_by_id[iid], perm_inst_pred[iid])
                        for iid in protocol_instance_ids[p]
                        if iid in perm_inst_pred
                    ]

                    perm_metric = MetricsAggregator.compute(p, perm_results)
                    if perm_metric.cycles > 0:
                        deltas.append(perm_metric.rmse - baseline[p].rmse)

                pfi_results[spec.name][p] = PfiResult(spec.name, p, deltas)

        return PfiReport(
            features=features,
            protocols=protocols,
            baseline_metrics=baseline,
            results=pfi_results,
        )
