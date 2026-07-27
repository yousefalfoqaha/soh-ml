from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class CurveFitResult:
    model: Callable[[float], float]
    ref_points: list[tuple[float, float]]
    rmse: float
    start_dci: float
    end_dci: float
    start_soh: float
    end_soh: float


class SohCurveFitter:
    def __init__(
        self,
        ref_temp_range: tuple[float, float],
        ref_current_range: tuple[float, float],
    ):
        self.ref_temp_range = ref_temp_range
        self.ref_current_range = ref_current_range

    @staticmethod
    def _poly4(x, a, b, c, d, e):
        return a + b * x + c * x**2 + d * x**3 + e * x**4

    def filter_reference(
        self, records: list[tuple[float, float, float, float]]
    ) -> list[tuple[float, float]]:
        """Return (dci, soh) pairs passing the temperature/current reference filter."""
        t_lo, t_hi = self.ref_temp_range
        c_lo, c_hi = self.ref_current_range
        return [
            (r[0], r[1])
            for r in records
            if not np.isnan(r[1]) and (t_lo <= r[2] <= t_hi) and (c_lo <= r[3] <= c_hi)
        ]

    def fit(
        self, records: list[tuple[float, float, float, float]]
    ) -> CurveFitResult | None:
        """
        Expects records as (dci, soh, ambient_temperature, mean_neg_current).
        Returns a CurveFitResult if enough reference points are found.
        """
        ref_points = self.filter_reference(records)

        if len(ref_points) < 5:
            return None

        ref_points.sort(key=lambda p: p[0])
        dci = np.array([p[0] for p in ref_points], dtype=float)
        soh = np.array([p[1] for p in ref_points], dtype=float)

        popt, _ = curve_fit(
            self._poly4, dci, soh, p0=[soh[0], -0.001, -1e-5, 1e-7, 1e-9]
        )
        model = lambda x: float(self._poly4(x, *popt))

        rmse = float(np.sqrt(np.mean((np.array([model(x) for x in dci]) - soh) ** 2)))

        return CurveFitResult(
            model=model,
            ref_points=ref_points,
            rmse=rmse,
            start_dci=float(dci[0]),
            end_dci=float(dci[-1]),
            start_soh=float(model(float(dci[0]))),
            end_soh=float(model(float(dci[-1]))),
        )
