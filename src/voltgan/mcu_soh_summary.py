from __future__ import annotations

import numpy as np

from voltgan.config import (
    CONFERENCE_PATH,
    HDF_ROOT,
    TESTING_MCUS,
    TRAINING_MCUS,
    VALIDATION_MCUS,
)
from voltgan.utils.reference import count_instances, load_reference_points

_ALL_MCUS = TRAINING_MCUS + VALIDATION_MCUS + TESTING_MCUS


def main() -> None:
    rows: list[str] = []

    for mcu in _ALL_MCUS:
        ref_points = load_reference_points([mcu], HDF_ROOT)
        if not ref_points:
            print(f"[mcu-summary] {mcu}: no reference points, skipping.")
            continue

        soh_values = [p[1] for p in ref_points]
        soh_min = min(soh_values)
        soh_max = max(soh_values)
        total = count_instances([mcu], HDF_ROOT)

        mcu_num = mcu.replace("mcu", "")
        soh_range = f"${soh_max * 100:.1f}$--${soh_min * 100:.1f}$"
        rows.append(f"{mcu_num} & {soh_range} & {total}" + r" \\")

    lines = [
        r"\begin{table}[H]",
        r"    \caption{MCU SOH RANGE AND CYCLE COUNT}",
        r"    \label{tab:mcu_soh_summary}",
        r"    \begin{center}",
        r"        \footnotesize",
        r"        \begin{tabular}{lcc}",
        r"            \hline",
        r"            \textbf{MCU} & \textbf{SoH Range (\%)} & \textbf{Cycles} \\",
        r"            \hline",
        *rows,
        r"            \hline",
        r"        \end{tabular}",
        r"    \end{center}",
        r"\end{table}",
    ]

    tex_path = CONFERENCE_PATH / "mcu_soh_summary.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"LaTeX table saved -> {tex_path}")


if __name__ == "__main__":
    main()