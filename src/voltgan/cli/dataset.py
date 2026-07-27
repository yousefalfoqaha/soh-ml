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
    DischargeInstance,
    InstanceRepository,
    SohCurveFitter,
    StatisticsCalculator,
)
from voltgan.utils import LatexTableWriter

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

    _write_feature_stats(instances)
    _write_temp_distribution(instances)
    _write_mcu_summary(instances)
    _plot_soh_trajectories(instances)
    _plot_discharge_protocols()


def _write_feature_stats(instances: list[DischargeInstance]) -> None:
    stats = StatisticsCalculator().compute(instances)

    rows: list[list[str]] = []
    for key, display_name in FEATURE_DISPLAY_NAMES.items():
        s = stats.get(key)
        if s is None:
            continue
        rows.append(
            [display_name, f"{s['mean']:.4f}", f"{s['standard_deviation']:.4f}"]
        )

    (
        LatexTableWriter(CONFERENCE_PATH / "feature_stats.tex")
        .caption("FEATURE STATISTICS")
        .label("tab:feature_stats")
        .align("lcc")
        .hline()
        .header(["Feature", "Mean ($\\mu$)", "Std. Dev. ($\\sigma$)"])
        .rows(rows)
        .write()
    )


def _write_temp_distribution(instances: list[DischargeInstance]) -> None:
    counts: dict[tuple[str, int], int] = defaultdict(int)
    for inst in instances:
        counts[(inst.phase, inst.temp_center)] += 1

    temp_bands = sorted({tc for (_, tc) in counts})
    n = len(temp_bands)
    matrix = [
        [counts.get((phase, tc), 0) for tc in temp_bands] for phase in PHASE_ORDER
    ]
    col_totals = [sum(matrix[i][j] for i in range(len(PHASE_ORDER))) for j in range(n)]

    align = "l" + "c" * n
    w = (
        LatexTableWriter(CONFERENCE_PATH / "temp_distribution.tex")
        .caption("TEMPERATURE BAND DISTRIBUTION")
        .label("tab:temp_phase_matrix")
        .align(align)
        .hline()
    )
    w.row(
        [rf"\multicolumn{{{n}}}{{c}}{{\textbf{{Temperature Bands ($^{{\circ}}$C)}}}}"]
    )
    w.row([r"\textbf{Phase}", *[f"${tc}$" for tc in temp_bands]])
    for label, row_counts in zip(PHASE_ORDER, matrix):
        w.row([label, *[str(c) for c in row_counts]])
    w.bold_row([r"\textbf{Total}", *[str(t) for t in col_totals]])
    w.write()


def _write_mcu_summary(instances: list[DischargeInstance]) -> None:
    by_mcu: dict[str, list[DischargeInstance]] = defaultdict(list)
    for inst in instances:
        mcu = inst.filepath.relative_to(HDF_ROOT).parts[0]
        by_mcu[mcu].append(inst)

    fitter = SohCurveFitter(
        ref_temp_range=REFERENCE_TEMPERATURE_RANGE,
        ref_current_range=REFERENCE_CURRENT_RANGE,
    )

    rows: list[list[str]] = []
    for mcu_name, insts in by_mcu.items():
        if not insts:
            continue
        records = [
            (i.dci, i.soh, i.ambient_temperature, i.mean_neg_current) for i in insts
        ]
        ref_points = fitter.filter_reference(records)
        if not ref_points:
            print(f"[mcu-summary] {mcu_name}: no reference points, skipping.")
            continue
        soh_values = [p[1] for p in ref_points]
        soh_range = f"${max(soh_values) * 100:.1f}$--${min(soh_values) * 100:.1f}$"
        rows.append([mcu_name.replace("mcu", ""), soh_range, str(len(insts))])

    (
        LatexTableWriter(CONFERENCE_PATH / "mcu_soh_summary.tex")
        .caption("MCU SOH RANGE AND CYCLE COUNT")
        .label("tab:mcu_soh_summary")
        .align("lcc")
        .hline()
        .header(["MCU", "SoH Range (\\%)", "Cycles"])
        .rows(rows)
        .write()
    )


def _plot_soh_trajectories(instances: list[DischargeInstance]) -> None:
    fitter = SohCurveFitter(
        ref_temp_range=REFERENCE_TEMPERATURE_RANGE,
        ref_current_range=REFERENCE_CURRENT_RANGE,
    )

    by_mcu: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
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
    out = CONFERENCE_PATH / "soh_trajectories.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {out}")


def _plot_discharge_protocols() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
    positions = [
        ("Constant", axes[0, 0]),
        ("Pulse", axes[0, 1]),
        ("HPPC", axes[1, 0]),
        ("WLTC", axes[1, 1]),
    ]
    panels = dict(_DISCHARGE_PROTOCOL_PANELS)

    for name, ax in positions:
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
        ax.plot(time, current, linewidth=0.5, color="black")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Current (A)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    CONFERENCE_PATH.mkdir(parents=True, exist_ok=True)
    out = CONFERENCE_PATH / "discharge_protocols.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {out}")
