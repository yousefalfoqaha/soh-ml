from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

from voltgan.dataset.instance import DischargeInstance
from voltgan.dataset.repository import InstanceRepository


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
        reference_temperature: float,
        reference_discharge_rate: float,
    ):
        self.reference_temperature = reference_temperature
        self.reference_discharge_rate = reference_discharge_rate

        self._reference_temperature_range = (
            reference_temperature - 2.0,
            reference_temperature + 2.0,
        )

    @staticmethod
    def _poly4(x, a, b, c, d, e):
        return a + b * x + c * x**2 + d * x**3 + e * x**4

    def filter_reference(
        self, instances: list[DischargeInstance]
    ) -> list[DischargeInstance]:
        """Return instances passing temperature, rate, and protocol filters."""
        t_lo, t_hi = self._reference_temperature_range
        return [
            inst
            for inst in instances
            if not np.isnan(inst.soh)
            and (t_lo <= inst.ambient_temperature <= t_hi)
            and (inst.discharge_rate is not None)
            and abs(inst.discharge_rate - self.reference_discharge_rate) < 0.15
            and inst.protocol == "Constant"
        ]

    def fit(self, instances: list[DischargeInstance]) -> CurveFitResult | None:
        """
        Fits a degradation curve using valid reference points from the instances.
        Returns a CurveFitResult if enough reference points are found.
        """
        ref_instances = self.filter_reference(instances)

        if len(ref_instances) < 5:
            return None

        ref_instances.sort(key=lambda inst: inst.dci)
        dci = np.array([inst.dci for inst in ref_instances], dtype=float)
        soh = np.array([inst.soh for inst in ref_instances], dtype=float)

        popt, _ = curve_fit(
            self._poly4, dci, soh, p0=[soh[0], -0.001, -1e-5, 1e-7, 1e-9]
        )
        model = lambda x: float(self._poly4(x, *popt))

        rmse = float(np.sqrt(np.mean((np.array([model(x) for x in dci]) - soh) ** 2)))

        return CurveFitResult(
            model=model,
            ref_points=[(float(d), float(s)) for d, s in zip(dci, soh)],
            rmse=rmse,
            start_dci=float(dci[0]),
            end_dci=float(dci[-1]),
            start_soh=float(model(float(dci[0]))),
            end_soh=float(model(float(dci[-1]))),
        )

    def apply(self, repo: InstanceRepository, mcus: list[str]) -> None:
        """Loads instances from the repo, fits the curves, and updates the HDF metadata."""
        for mcu in mcus:
            instances = repo.load([mcu])
            if not instances:
                continue

            fit_result = self.fit(instances)

            if not fit_result:
                print(f"[{mcu}] Skipped curve fitting - insufficient reference points.")
                continue

            print(
                f"[{mcu}] Fitted deg4 curve | {len(fit_result.ref_points)} pts | RMSE: {fit_result.rmse:.5f}"
            )

            for inst in instances:
                fitted_soh = min(max(float(fit_result.model(inst.dci)), 0.0), 1.0)
                repo.update_metadata(inst.filepath, {"curve_soh": fitted_soh})
