from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from voltgan.config import (
    ALL_MCUS,
    CONFERENCE_PATH,
    CURRENT_CHANNEL,
    FEATURE_DISPLAY_NAMES,
    HDF_ROOT,
    PHASE_ORDER,
    REFERENCE_CURRENT_RANGE,
    REFERENCE_TEMPERATURE_RANGE,
)
from voltgan.dataset import (
    DatasetAnalyzer,
    InstanceRepository,
    SohCurveFitter,
    StatisticsCalculator,
)
from voltgan.utils import HLine, LatexTable, TableRow

_DISCHARGE_PROTOCOL_PANELS = [
    (
        "Constant",
        "ChDch2 2024-10-22_05.31.47 Pulse_Test_SamsungINR2170050E_Cell 5 Zelltester_1.hdf",
    ),
    (
        "Pulse",
        "2024-10-24_15.20.25 Pulse_Test_SamsungINR2170050E_Cell 5 Zelltester_1.hdf",
    ),
    (
        "HPPC",
        "HPPC 2024-09-15_16.21.33 Pulse_Test_SamsungINR2170050E_Cell 5 Zelltester_1_3.hdf",
    ),
    (
        "WLTC",
        "WLTC 2024-09-07 21.45.06 Pulse_Test_SamsungINR2170050E_Cell 5 Zelltester_1.hdf",
    ),
]
_MCU_DIR = Path("/mnt/ssd/datasets/wuppertal/hdf/mcu5")
_MAX_PULSE_STEPS = 1000


def main() -> None:
    repo = InstanceRepository(root=HDF_ROOT)
    instances = repo.load(ALL_MCUS)
    print(f"Loaded {len(instances)} instances")

    # feature statistics table
    stats = StatisticsCalculator().compute(instances)
    feature_rows = []

    for key, display_name in FEATURE_DISPLAY_NAMES.items():
        s = stats.get(key)
        if s is not None:
            feature_rows.append(
                TableRow(
                    cells=[
                        display_name,
                        f"{s['mean']:.4f}",
                        f"{s['standard_deviation']:.4f}",
                    ]
                )
            )

    LatexTable(
        out_path=CONFERENCE_PATH / "feature_stats.tex",
        caption="FEATURE STATISTICS",
        label="tab:feature_stats",
        align="lcc",
        headers=["Feature", r"Mean ($\mu$)", r"Std. Dev. ($\sigma$)"],
        items=feature_rows,
    ).write()

    # temperature distribution table
    dist = DatasetAnalyzer.compute_temp_distribution(instances, PHASE_ORDER)
    n = len(dist.temp_bands)

    LatexTable(
        out_path=CONFERENCE_PATH / "temp_distribution.tex",
        caption="TEMPERATURE BAND DISTRIBUTION",
        label="tab:temp_phase_matrix",
        align="l" + "c" * n,
        headers=["Phase", *[f"${tc}$" for tc in dist.temp_bands]],
        items=[
            *[
                TableRow(cells=[label, *[str(c) for c in row_counts]])
                for label, row_counts in zip(dist.phase_order, dist.matrix)
            ],
            HLine(),
            TableRow(cells=["Total", *[str(t) for t in dist.col_totals]], bold=True),
        ],
    ).write()

    # mcu soh summary table
    fitter = SohCurveFitter(
        ref_temp_range=REFERENCE_TEMPERATURE_RANGE,
        ref_current_range=REFERENCE_CURRENT_RANGE,
    )

    mcu_summaries = DatasetAnalyzer.compute_mcu_summaries(instances, fitter)

    LatexTable(
        out_path=CONFERENCE_PATH / "mcu_soh_summary.tex",
        caption="MCU SOH RANGE AND CYCLE COUNT",
        label="tab:mcu_soh_summary",
        align="lcc",
        headers=["MCU", r"SoH Range (\%)", "Cycles"],
        items=[TableRow(cells=rec.to_latex_cells()) for rec in mcu_summaries],
    ).write()

    # soh trajectories plot
    by_mcu = defaultdict(list)
    for inst in instances:
        mcu = inst.filepath.relative_to(HDF_ROOT).parts[0]
        by_mcu[mcu].append(
            (inst.dci, inst.soh, inst.ambient_temperature, inst.mean_neg_current)
        )

    mcus = sorted(by_mcu)
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
    traj_mins: list[float] = []

    for i, mcu in enumerate(mcus):
        records = by_mcu[mcu]
        if not records:
            continue

        fit = fitter.fit(records)
        if fit is None:
            print(f"  {mcu}: not enough reference points, skipping curve")
            continue

        all_dci = np.array([r[0] for r in records], dtype=float)
        min_dci = float(all_dci.min())
        max_dci = float(all_dci.max())
        dense_dci = np.linspace(min_dci, max_dci, 500)
        dense_soh = np.clip([fit.model(float(t)) for t in dense_dci], 0.0, 1.0)

        ax.plot(
            dense_dci,
            dense_soh,
            color=colors(i % 10),
            lw=1.5,
            alpha=0.85,
            label=f"{mcu} ({len(records)} files)",
        )
        traj_mins.append(float(np.min(dense_soh)))

    if traj_mins:
        y_min = min(traj_mins)
        padding = (1.05 - y_min) * 0.05
        ax.set_ylim(y_min - padding, 1.05)

    ax.set_xlabel("Discharge Cycles")
    ax.set_ylabel("SoH")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)

    CONFERENCE_PATH.mkdir(parents=True, exist_ok=True)
    out_traj = CONFERENCE_PATH / "soh_trajectories.pdf"
    fig.savefig(out_traj, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {out_traj}")

    # discharge protocols plot
    fig2, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
    positions = [
        ("Constant", axes[0, 0]),
        ("Pulse", axes[0, 1]),
        ("HPPC", axes[1, 0]),
        ("WLTC", axes[1, 1]),
    ]
    panels = dict(_DISCHARGE_PROTOCOL_PANELS)

    for name, ax2 in positions:
        filename = panels[name]
        path = _MCU_DIR / filename
        with h5py.File(path, "r") as f:
            group = f[filename]
            assert isinstance(group, h5py.Group)
            ds = group[CURRENT_CHANNEL]
            assert isinstance(ds, h5py.Dataset)
            current = ds[:]

        if name == "Pulse":
            current = current[:_MAX_PULSE_STEPS]

        time = np.arange(len(current))
        ax2.plot(time, current, linewidth=0.5, color="black")
        ax2.set_title(name, fontsize=10, fontweight="bold")
        ax2.set_xlabel("Time (s)", fontsize=8)
        ax2.set_ylabel("Current (A)", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.grid(True, alpha=0.3)

    out_proto = CONFERENCE_PATH / "discharge_protocols.pdf"
    fig2.savefig(out_proto, bbox_inches="tight")
    plt.close(fig2)
    print(f"Plot saved -> {out_proto}")


if __name__ == "__main__":
    main()
