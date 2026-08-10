from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from voltgan.config import (
    CONFERENCE_DIR,
    EVALUATION_PROVIDER,
    FEATURE_DISPLAY_NAMES,
    OXFORD_TESTING_MCUS,
    OXFORD_TRAINING_MCUS,
    OXFORD_VALIDATION_MCUS,
    PHASE_ORDER,
    PROTOCOL_ORDER,
    REFERENCE_DISCHARGE_RATE,
    REFERENCE_TEMPERATURE,
    TESTING_MCUS,
    TRAINING_MCUS,
    TRAINING_PROVIDER,
    VALIDATION_MCUS,
)
from voltgan.dataset import (
    InstanceRepository,
    SohCurveFitter,
    StatisticsCalculator,
)
from voltgan.utils import HLine, LatexTable, RowItem, TableRow

_DISCHARGE_PROTOCOL_PANELS = [
    (
        "Constant",
        "Cyc026_Initial_Constant_1.0C_Temp25_20241022.hdf",
    ),
    (
        "Pulse",
        "Cyc009_Initial_Pulse_1.0C_Temp25_20240910.hdf",
    ),
    (
        "HPPC",
        "Cyc020_Initial_HPPC_Temp25_20240915.hdf",
    ),
    (
        "WLTC",
        "Cyc029_Initial_WLTC_Temp25_20241024.hdf",
    ),
]
_MAX_PULSE_STEPS = 1000


def main() -> None:
    repo = InstanceRepository(provider=TRAINING_PROVIDER)
    instances = repo.load(TRAINING_MCUS + VALIDATION_MCUS + TESTING_MCUS)
    print(f"Loaded {len(instances)} instances")

    # feature statistics table
    stats = StatisticsCalculator().calculate_mean_std(instances)
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
        out_path=CONFERENCE_DIR / "feature_stats.tex",
        caption="FEATURE STATISTICS",
        label="tab:feature_stats",
        align="lcc",
        headers=["Feature", r"Mean ($\mu$)", r"Std. Dev. ($\sigma$)"],
        items=feature_rows,
    ).write()

    # temperature x protocol distribution table
    tp_counts: defaultdict[tuple[int, str], int] = defaultdict(int)
    for inst in instances:
        tp_counts[(inst.temp_center, inst.protocol)] += 1

    temp_bands = sorted({tc for (tc, _) in tp_counts})
    protocols = [
        p for p in PROTOCOL_ORDER if any((tc, p) in tp_counts for tc in temp_bands)
    ]
    num_protos = len(protocols)

    tp_rows: list[TableRow | HLine] = []
    col_totals = [0] * num_protos
    for tc in temp_bands:
        row_cells = []
        for j, proto in enumerate(protocols):
            c = tp_counts.get((tc, proto), 0)
            row_cells.append(str(c))
            col_totals[j] += c
        row_total = sum(int(c) for c in row_cells)
        tp_rows.append(
            TableRow(
                cells=[
                    rf"${tc}^{{\circ}}\text{{C}}$",
                    *row_cells,
                    str(row_total),
                ]
            )
        )

    LatexTable(
        out_path=CONFERENCE_DIR / "temp_distribution.tex",
        caption="TEMPERATURE AND PROTOCOL DISTRIBUTION",
        label="tab:temp_protocol_matrix",
        align="l" + "c" * (num_protos + 1),
        headers=["Temp Band", *protocols, "Total"],
        items=[
            *tp_rows,
            HLine(),
            TableRow(
                cells=["Total", *[str(t) for t in col_totals], str(sum(col_totals))],
                bold=True,
            ),
        ],
    ).write()

    # mcu soh summary table (per-phase reference-condition SoH ranges)
    fitter = SohCurveFitter(
        reference_temperature=REFERENCE_TEMPERATURE,
        reference_discharge_rate=REFERENCE_DISCHARGE_RATE,
    )

    by_mcu: defaultdict[str, list] = defaultdict(list)
    for inst in instances:
        by_mcu[inst.cell_id].append(inst)

    def _mcu_num(mcu_str: str) -> int:
        match = re.search(r"\d+", mcu_str)
        return int(match.group()) if match else 0

    mcu_rows: list[RowItem] = []
    for mcu_name, insts in sorted(by_mcu.items(), key=lambda x: _mcu_num(x[0])):
        if not insts:
            continue

        label = mcu_name.replace("mcu", "").replace("cell", "")
        ref_insts = fitter.filter_reference(insts)

        phase_cells: list[str] = []
        for phase in PHASE_ORDER:
            phase_ref = [i for i in ref_insts if i.phase == phase]
            if not phase_ref:
                phase_cells.append("--")
            else:
                soh_vals = [i.soh for i in phase_ref]
                phase_cells.append(
                    f"${max(soh_vals) * 100:.1f}$--${min(soh_vals) * 100:.1f}$"
                )

        mcu_rows.append(TableRow(cells=[label, *phase_cells, str(len(insts))]))

    LatexTable(
        out_path=CONFERENCE_DIR / "mcu_soh_summary.tex",
        caption="MCU SOH RANGE BY PHASE",
        label="tab:mcu_soh_summary",
        align="lcccc",
        headers=["MCU", "Initial", "Aging", "Post-Aging", "Cycles"],
        items=mcu_rows,
    ).write()

    # oxford soh summary table (per-cell SoH range and cycle count)
    oxford_repo = InstanceRepository(provider=EVALUATION_PROVIDER)
    oxford_cells = (
        OXFORD_TRAINING_MCUS + OXFORD_VALIDATION_MCUS + OXFORD_TESTING_MCUS
    )
    oxford_instances = oxford_repo.load(oxford_cells)
    print(f"Loaded {len(oxford_instances)} Oxford instances")

    oxford_by_cell: defaultdict[str, list] = defaultdict(list)
    for inst in oxford_instances:
        oxford_by_cell[inst.cell_id].append(inst)

    def _cell_num(cell_str: str) -> int:
        match = re.search(r"\d+", cell_str)
        return int(match.group()) if match else 0

    oxford_rows: list[RowItem] = []
    for cell_name, cell_insts in sorted(
        oxford_by_cell.items(), key=lambda x: _cell_num(x[0])
    ):
        if not cell_insts:
            continue

        label = cell_name.replace("cell", "")
        soh_vals = [i.soh for i in cell_insts if not np.isnan(i.soh)]
        if soh_vals:
            soh_range = f"${max(soh_vals) * 100:.1f}$--${min(soh_vals) * 100:.1f}$"
        else:
            soh_range = "--"
        oxford_rows.append(
            TableRow(cells=[label, soh_range, str(len(cell_insts))])
        )

    LatexTable(
        out_path=CONFERENCE_DIR / "oxford_soh_summary.tex",
        caption="OXFORD CELL SOH RANGE AND CYCLE COUNT",
        label="tab:oxford_soh_summary",
        align="lcc",
        headers=["Cell", "SoH Range (\\%)", "Cycles"],
        items=oxford_rows,
    ).write()

    # soh trajectories plot
    by_mcu = defaultdict(list)
    for instance in instances:
        by_mcu[instance.cell_id].append(instance)

    mcus = sorted(by_mcu)
    colors = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(14, 7), layout="constrained")
    traj_mins: list[float] = []

    for i, mcu in enumerate(mcus):
        mcu_instances = by_mcu[mcu]
        if not mcu_instances:
            continue

        fit = fitter.fit(mcu_instances)
        if fit is None:
            print(f"  {mcu}: not enough reference points, skipping curve")
            continue

        all_cycle_index = np.array([inst.dci for inst in mcu_instances], dtype=float)
        min_cycle_index = float(all_cycle_index.min())
        max_cycle_index = float(all_cycle_index.max())
        dense_cycle_index = np.linspace(min_cycle_index, max_cycle_index, 500)
        dense_soh = np.clip([fit.model(float(t)) for t in dense_cycle_index], 0.0, 1.0)

        ax.plot(
            dense_cycle_index,
            dense_soh,
            color=colors(i % 10),
            lw=1.5,
            alpha=0.85,
            label=f"{mcu} ({len(mcu_instances)} files)",
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

    CONFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    out_traj = CONFERENCE_DIR / "soh_trajectories.pdf"
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

        instance = next((i for i in instances if i.filepath.name == filename), None)

        if not instance:
            print(f"Warning: Panel file {filename} not found in loaded instances.")
            continue

        current = instance.current

        if name == "Pulse":
            current = current[:_MAX_PULSE_STEPS]

        time = np.arange(len(current))
        ax2.plot(time, current, linewidth=0.5, color="black")
        ax2.set_title(name, fontsize=10, fontweight="bold")
        ax2.set_xlabel("Time (s)", fontsize=8)
        ax2.set_ylabel("Current (A)", fontsize=8)
        ax2.tick_params(labelsize=7)
        ax2.grid(True, alpha=0.3)

    out_proto = CONFERENCE_DIR / "discharge_protocols.pdf"
    fig2.savefig(out_proto, bbox_inches="tight")
    plt.close(fig2)
    print(f"Plot saved -> {out_proto}")


if __name__ == "__main__":
    main()
