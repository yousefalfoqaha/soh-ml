from pathlib import Path

import h5py
import numpy as np
from scipy.interpolate import PchipInterpolator

_NAN = float("nan")


def select_reference_points(
    records: list[tuple[float, float, float, float]],
    ref_temp_range: tuple[float, float],
    ref_current_range: tuple[float, float],
) -> list[tuple[float, float]]:
    """Return every qualifying (discharge_cycle_index, soh) measurement."""
    ref_lo, ref_hi = ref_temp_range
    cur_lo, cur_hi = ref_current_range

    ref_points: list[tuple[float, float]] = []
    for dci, soh, amb, mni in records:
        if ref_lo <= amb <= ref_hi and cur_lo <= mni <= cur_hi:
            ref_points.append((dci, soh))
    return ref_points


def fit_soh_curve(
    ref_points: list[tuple[float, float]],
) -> tuple[PchipInterpolator, float, float, float, float] | None:
    """Fit a PCHIP curve from reference points.

    Points sharing the same discharge cycle index get tiny increments
    so each measurement is its own data point (no averaging).

    Returns (curve, first_dci, last_dci, first_soh, last_soh) or None.
    """
    if len(ref_points) < 2:
        return None

    ref_points = sorted(ref_points, key=lambda p: p[0])

    dci_seen: dict[float, int] = {}
    ref_dci: list[float] = []
    ref_soh: list[float] = []
    for dci, soh in ref_points:
        offset = dci_seen.get(dci, 0)
        dci_seen[dci] = offset + 1
        ref_dci.append(dci + offset * 1e-3)
        ref_soh.append(soh)

    ref_dci_arr = np.array(ref_dci)
    ref_soh_arr = np.array(ref_soh)

    curve = PchipInterpolator(ref_dci_arr, ref_soh_arr)

    first_dci = float(ref_dci_arr[0])
    last_dci = float(ref_dci_arr[-1])
    first_soh = float(ref_soh_arr[0])
    last_soh = float(ref_soh_arr[-1])

    return curve, first_dci, last_dci, first_soh, last_soh


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
                soh = float(f.attrs["soh"])
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

        curve, first_dci, last_dci, first_soh, last_soh = fit

        print(
            f"[soh_curve] {mcu}: fitted curve with {len(ref_points)} reference "
            f"points, cycle {first_dci:.0f} → {last_dci:.0f}, "
            f"SoH {first_soh:.4f} → {last_soh:.4f}"
        )

        for dci, raw_soh, amb, mni, hdf_path in records:
            is_ref = ref_lo <= amb <= ref_hi and cur_lo <= mni <= cur_hi

            if dci <= first_dci:
                fitted_soh = first_soh
            elif dci >= last_dci:
                fitted_soh = last_soh
            else:
                fitted_soh = float(curve(dci))
            fitted_soh = min(max(fitted_soh, 0.0), 1.0)

            with h5py.File(hdf_path, "a") as f:
                f.attrs["curve_soh"] = fitted_soh
                f.attrs["soh"] = raw_soh if is_ref else _NAN

