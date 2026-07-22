from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime

from voltgan.config import HDF_ROOT, PROJECT_ROOT
from voltgan.utils.discover import load_instances

_AGING_START = datetime(2025, 2, 12)
_AGING_END = datetime(2025, 3, 8)
_PHASE_ORDER = ["Initial", "Aging", "Post-Aging"]
_ALL_MCUS = [f"mcu{i}" for i in range(1, 9)]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX temperature band distribution table."
    )
    parser.add_argument(
        "--mcus",
        nargs="+",
        default=_ALL_MCUS,
        help="MCUs to include (default: all mcu1-mcu8).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    instances = load_instances(HDF_ROOT, args.mcus)
    print(f"Loaded {len(instances)} instances from {args.mcus}")

    counts: dict[tuple[str, int], int] = defaultdict(int)
    for inst in instances:
        dt = inst.datetime
        phase = (
            "Initial"
            if dt < _AGING_START
            else "Aging"
            if dt <= _AGING_END
            else "Post-Aging"
        )
        temp_center = int(round(inst.ambient_temperature / 5) * 5)
        counts[(phase, temp_center)] += 1

    temp_bands = sorted({tc for (_, tc) in counts})
    n_bands = len(temp_bands)

    col_spec = "l" + "c" * n_bands
    multicolumn = rf"               & \multicolumn{{{n_bands}}}{{c}}{{\textbf{{Temperature Bands ($^{{\circ}}$C)}}}} \\"
    band_header = (
        r"            \textbf{Phase}"
        + "".join(f" & ${tc}$" for tc in temp_bands)
        + r" \\"
    )

    lines = [
        r"\begin{table}[htbp]",
        r"    \caption{TEMPERATURE BAND DISTRIBUTION}",
        r"    \label{tab:temp_phase_matrix}",
        r"    \begin{center}",
        rf"        \begin{{tabular}}{{{col_spec}}}",
        r"            \hline",
        multicolumn,
        band_header,
        r"            \hline",
    ]

    col_totals = [0] * n_bands

    for phase in _PHASE_ORDER:
        row_counts = [counts.get((phase, tc), 0) for tc in temp_bands]
        for i, c in enumerate(row_counts):
            col_totals[i] += c
        cells = "".join(f" & {c}" for c in row_counts)
        lines.append(f"            {phase:<14}{cells} \\\\")

    lines.append(r"            \hline")

    total_cells = "".join(f" & {t}" for t in col_totals)
    lines.append(rf"            \textbf{{Total}}{total_cells} \\")
    lines.append(r"            \hline")
    lines.append(r"        \end{tabular}")
    lines.append(r"    \end{center}")
    lines.append(r"\end{table}")

    tex_path = PROJECT_ROOT / "conference" / "temp_distribution.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"LaTeX table saved -> {tex_path}")


if __name__ == "__main__":
    main()