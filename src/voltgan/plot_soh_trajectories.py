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


def _load_records(
    mcus: list[str],
) -> dict[str, list[tuple[float, float, float, float]]]:
    """Return per-mcu list of (discharge_cycle_index, soh, ambient_temp, mean_neg_current)."""
    per_mcu: dict[str, list[tuple[float, float, float, float]]] = {}
    for hdf_path in discover(HDF_ROOT, mcus, (".hdf",)):
        mcu = hdf_path.relative_to(HDF_ROOT).parts[0]
        with h5py.File(hdf_path, "r") as f:
            dci_attr = f.attrs.get("discharge_cycle_index")
            if dci_attr is None:
                continue
            dci = float(dci_attr)
            soh_attr = f.attrs.get("soh")
            soh = float(soh_attr) if soh_attr is not None else float("nan")
            amb = float(f.attrs["ambient_temperature"])
            mni = float(f.attrs["mean_neg_current"])
        per_mcu.setdefault(mcu, []).append((dci, soh, amb, mni))
    return per_mcu


def _plot_trajectories(
    per_mcu: dict[str, list[tuple[float, float, float, float]]],
) -> Path:
    mcus = sorted(per_mcu)
    colors = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")

    for i, mcu in enumerate(mcus):
        records = per_mcu[mcu]
        if not records:
            continue

        ref_points = select_reference_points(
            records,
            REFERENCE_TEMPERATURE_RANGE,
            REFERENCE_CURRENT_RANGE,
        )
        fit = fit_soh_curve(ref_points)

        if fit is None:
            print(f"  {mcu}: only {len(ref_points)} ref point(s), skipping curve")
            continue

        curve, first_dci, last_dci, first_soh, last_soh = fit

        all_dci = np.array([r[0] for r in records], dtype=float)
        min_dci = float(all_dci.min())
        max_dci = float(all_dci.max())

        dense_dci = np.linspace(min_dci, max_dci, 500)
        dense_soh = np.full_like(dense_dci, np.nan)

        for j, t in enumerate(dense_dci):
            if t <= first_dci:
                dense_soh[j] = first_soh
            elif t >= last_dci:
                dense_soh[j] = last_soh
            else:
                dense_soh[j] = float(curve(float(t)))

        dense_soh = np.clip(dense_soh, 0.0, 1.0)

        ax.plot(
            dense_dci,
            dense_soh,
            color=colors(i % 10),
            lw=1.5,
            alpha=0.85,
            label=f"{mcu} ({len(records)} files)",
        )

    ax.set_xlabel("Discharge Cycles")
    ax.set_ylabel("SoH (curve-fitted)")
    ax.set_title("SoH Trajectories per MCU")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    out = PLOTS_PATH / "soh_trajectories.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot fitted SoH trajectories for all MCUs on a single chart."
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
    if not any(per_mcu.values()):
        print("Nothing to plot: no HDF files found.")
        return

    out = _plot_trajectories(per_mcu)
    print(f"Plot saved -> {out}")


if __name__ == "__main__":
    main()