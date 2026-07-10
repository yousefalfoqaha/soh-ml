from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from voltgan.config import (
    HDF_ROOT,
    PLOTS_PATH,
    REFERENCE_CURRENT_RANGE,
    REFERENCE_TEMPERATURE_RANGE,
)
from voltgan.pipeline.base import discover
from voltgan.pipeline.soh_curve import fit_soh_curve, select_reference_points


def _load_records(mcus: list[str]) -> dict[str, list[tuple[float, float, float, float, float]]]:
    per_mcu: dict[str, list[tuple[float, float, float, float, float]]] = {}
    for hdf_path in discover(HDF_ROOT, mcus, (".hdf",)):
        mcu = hdf_path.relative_to(HDF_ROOT).parts[0]
        with h5py.File(hdf_path, "r") as f:
            dci_attr = f.attrs.get("discharge_cycle_index")
            if dci_attr is None:
                continue
            dci = float(dci_attr)
            soh_attr = f.attrs.get("soh")
            soh = float(soh_attr) if soh_attr is not None else float("nan")
            curve_soh = float(f.attrs.get("curve_soh", float("nan")))
            amb = float(f.attrs["ambient_temperature"])
            mni = float(f.attrs.get("mean_neg_current", 0.0))
        per_mcu.setdefault(mcu, []).append((dci, soh, curve_soh, amb, mni))
    return per_mcu


def _plot_mcu(
    mcu: str,
    records: list[tuple[float, float, float, float, float]],
) -> Path:
    ref_lo, ref_hi = REFERENCE_TEMPERATURE_RANGE
    cur_lo, cur_hi = REFERENCE_CURRENT_RANGE

    all_dci = np.array([r[0] for r in records], dtype=float)
    all_soh = np.array([r[1] for r in records], dtype=float)
    all_curve = np.array([r[2] for r in records], dtype=float)
    all_amb = np.array([r[3] for r in records], dtype=float)

    ref_mask = np.array(
        [ref_lo <= amb <= ref_hi and cur_lo <= mni <= cur_hi
         for _, _, _, amb, mni in records]
    )
    ref_dci = all_dci[ref_mask]
    ref_soh = all_soh[ref_mask]

    order = np.argsort(ref_dci)
    ref_dci = ref_dci[order]
    ref_soh = ref_soh[order]

    non_ref_mask = ~ref_mask & ~np.isnan(all_curve)
    non_ref_dci = all_dci[non_ref_mask]
    non_ref_curve = all_curve[non_ref_mask]
    non_ref_amb = all_amb[non_ref_mask]

    fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")

    ax.scatter(non_ref_dci, non_ref_curve, c=non_ref_amb, cmap="coolwarm",
               s=15, alpha=0.5, label="Non-reference files")
    ax.plot(ref_dci, ref_soh, "ro", markersize=4, label="Reference points")

    fit = fit_soh_curve(list(zip(ref_dci, ref_soh)))
    if fit is not None:
        curve = fit[0]
        dense_dci = np.linspace(float(ref_dci.min()), float(ref_dci.max()), 500)
        dense_soh = curve(dense_dci)
        ax.plot(dense_dci, dense_soh, "k-", alpha=0.7, lw=1.2, label="PCHIP curve")

    ax.legend(fontsize=9)
    ax.set_xlabel("Discharge Cycles")
    ax.set_ylabel("SoH")
    ax.set_title(
        f"{mcu}  |  {len(ref_dci)} reference  |  {len(non_ref_dci)} non-reference"
    )

    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    out = PLOTS_PATH / f"soh_curve_{mcu}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the SoH degradation curve per MCU."
    )
    parser.add_argument(
        "--mcus",
        nargs="*",
        default=None,
        help="Optional subset of MCUs to plot. Defaults to all.",
    )
    args = parser.parse_args()

    mcus = args.mcus if args.mcus else [p.name for p in HDF_ROOT.iterdir() if p.is_dir()]

    per_mcu = _load_records(mcus)
    if not per_mcu:
        print("Nothing to plot: no HDF files with discharge_cycle_index found.")
        return

    for mcu in sorted(per_mcu):
        out = _plot_mcu(mcu, per_mcu[mcu])
        print(f"Plot saved -> {out}")


if __name__ == "__main__":
    main()