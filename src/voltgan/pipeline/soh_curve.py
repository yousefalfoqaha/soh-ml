from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import curve_fit

from voltgan.utils.reference import select_reference_points

_NAN = float("nan")


def _poly4(x, a, b, c, d, e):
    return a + b * x + c * x**2 + d * x**3 + e * x**4


def fit_soh_curve(
    ref_points: list[tuple[float, float]],
) -> tuple[Callable[[float], float], float, float, float, float] | None:
    """Fit a degree-4 SoH degradation curve from reference points.

    Uses Levenberg-Marquardt (``scipy.optimize.curve_fit``).

    Returns ``(model, first_dci, last_dci, first_soh, last_soh)`` where
    ``model`` is a callable ``model(x) -> float`` for any cycle index
    (including extrapolation beyond the last reference point), or ``None``.
    """
    if len(ref_points) < 5:
        return None

    ref_points = sorted(ref_points, key=lambda p: p[0])
    dci = np.array([p[0] for p in ref_points], dtype=float)
    soh = np.array([p[1] for p in ref_points], dtype=float)

    popt, _ = curve_fit(_poly4, dci, soh, p0=[soh[0], -0.001, -1e-5, 1e-7, 1e-9])

    model = lambda x: float(_poly4(x, *popt))

    first_dci = float(dci[0])
    last_dci = float(dci[-1])
    first_soh = float(model(first_dci))
    last_soh = float(model(last_dci))

    return model, first_dci, last_dci, first_soh, last_soh


def fit_soh_curves(
    hdf_root: Path,
    mcus: list[str],
    ref_temp_range: tuple[float, float],
    ref_current_range: tuple[float, float],
) -> None:
    ref_lo, ref_hi = ref_temp_range
    cur_lo, cur_hi = ref_current_range

    for mcu in mcus:
        mcu_path = hdf_root / mcu
        if not mcu_path.exists():
            continue

        hdf_paths = sorted(mcu_path.rglob("*.hdf"))
        if not hdf_paths:
            continue

        records: list[tuple[float, float, float, float, Path]] = []
        for hdf_path in hdf_paths:
            with h5py.File(hdf_path, "r") as f:
                dci = float(f.attrs["discharge_cycle_index"])
                soh = float(f.attrs.get("soh", _NAN))
                amb = float(f.attrs["ambient_temperature"])
                mni = float(f.attrs["mean_neg_current"])

            records.append((dci, soh, amb, mni, hdf_path))

        if not records:
            continue

        ref_points = select_reference_points(
            [r[:4] for r in records], ref_temp_range, ref_current_range
        )

        fit = fit_soh_curve(ref_points)
        if fit is None:
            print(
                f"[soh_curve] {mcu}: only {len(ref_points)} reference point(s), "
                f"skipping curve fit."
            )
            continue

        model, first_dci, last_dci, first_soh, last_soh = fit

        rmse = float(
            np.sqrt(
                np.mean(
                    (
                        np.array([model(p[0]) for p in ref_points])
                        - np.array([p[1] for p in ref_points])
                    )
                    ** 2
                )
            )
        )

        print(
            f"[soh_curve] {mcu}: fitted deg4 with {len(ref_points)} reference "
            f"points, cycle {first_dci:.0f} -> {last_dci:.0f}, "
            f"SoH {first_soh:.4f} -> {last_soh:.4f}, RMSE={rmse:.5f}"
        )

        for dci, raw_soh, amb, mni, hdf_path in records:
            fitted_soh = float(model(dci))
            fitted_soh = min(max(fitted_soh, 0.0), 1.0)

            with h5py.File(hdf_path, "a") as f:
                f.attrs["curve_soh"] = fitted_soh
