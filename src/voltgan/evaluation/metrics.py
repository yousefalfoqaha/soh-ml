from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from voltgan.dataset.instance import DischargeInstance


@dataclass(frozen=True)
class PredictionResult:
    instance: DischargeInstance
    predicted_soh: float

    @property
    def abs_pct_error(self) -> float:
        return abs(self.predicted_soh - self.instance.soh) / self.instance.soh * 100


@dataclass(frozen=True)
class MetricSet:
    label: str
    cycles: int
    soh_min: float
    soh_max: float
    rmse: float
    mae: float
    r2: float | None
    pct_err: float

    def cells(self, *, bold: bool = False) -> list[str]:
        """Return 7 LaTeX-formatted cells: [label, SoH range, RMSE, MAE, R², %Err, Cycles]."""
        soh_inner = f"{self.soh_min * 100:.1f}$--${self.soh_max * 100:.1f}"
        soh_body = f"${soh_inner}$"
        r2_str = f"{self.r2:.2f}" if self.r2 is not None else "--"
        pct_str = f"{self.pct_err:.1f}" + r"\%"

        if bold:
            label_str = rf"\textbf{{{self.label}}}"
            soh_str = rf"\textbf{{{soh_body}}}"
            rmse_str = rf"$\mathbf{{{self.rmse:.3f}}}$"
            mae_str = rf"$\mathbf{{{self.mae:.3f}}}$"
            r2_full = rf"$\mathbf{{{r2_str}}}$" if r2_str != "--" else "--"
            pct_full = rf"\textbf{{{pct_str}}}"
            cyc_str = rf"\textbf{{{self.cycles}}}"
        else:
            label_str = self.label
            soh_str = soh_body
            rmse_str = f"{self.rmse:.3f}"
            mae_str = f"{self.mae:.3f}"
            r2_full = r2_str if r2_str == "--" else f"${r2_str}$"
            pct_full = pct_str
            cyc_str = str(self.cycles)

        return [label_str, soh_str, rmse_str, mae_str, r2_full, pct_full, cyc_str]


class MetricsAggregator:
    """Aggregates PredictionResults into statistical MetricSets."""

    @staticmethod
    def compute(label: str, results: list[PredictionResult]) -> MetricSet:
        if not results:
            return MetricSet(
                label=label,
                cycles=0,
                soh_min=float("nan"),
                soh_max=float("nan"),
                rmse=float("nan"),
                mae=float("nan"),
                r2=None,
                pct_err=float("nan"),
            )

        actuals = np.array([r.instance.soh for r in results])
        preds = np.array([r.predicted_soh for r in results])

        cycles = len(results)
        r2 = float(r2_score(actuals, preds)) if cycles >= 2 else None

        return MetricSet(
            label=label,
            cycles=cycles,
            soh_min=float(actuals.min()),
            soh_max=float(actuals.max()),
            rmse=float(np.sqrt(mean_squared_error(actuals, preds))),
            mae=float(mean_absolute_error(actuals, preds)),
            r2=r2,
            pct_err=float(np.mean([r.abs_pct_error for r in results])),
        )

    @classmethod
    def group_and_compute(
        cls,
        results: list[PredictionResult],
        group_fn: Callable[[PredictionResult], Any],
    ) -> list[MetricSet]:
        """Groups results dynamically and computes a MetricSet for each group."""
        groups = defaultdict(list)
        for r in results:
            groups[group_fn(r)].append(r)

        metrics = []
        for key in sorted(groups.keys()):
            metrics.append(cls.compute(label=str(key), results=groups[key]))

        return metrics
