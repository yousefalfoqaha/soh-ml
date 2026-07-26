from __future__ import annotations

import argparse

from voltgan.config import (
    AMBIENT_TEMPERATURE_KEY,
    CURRENT_CHANNEL,
    HDF_ROOT,
    PROJECT_ROOT,
    SOH_KEY,
    TEMP_DELTA_KEY,
    TEMPERATURE_CHANNEL,
    VOLTAGE_CHANNEL,
)
from voltgan.data import StatisticsCalculator
from voltgan.utils.discover import load_instances

_ALL_MCUS = [f"mcu{i}" for i in range(1, 9)]

_FEATURE_ORDER = [
    (VOLTAGE_CHANNEL, r"Voltage ($V$)"),
    (CURRENT_CHANNEL, r"Current ($I$)"),
    (TEMPERATURE_CHANNEL, r"Cell Temperature ($T$)"),
    (AMBIENT_TEMPERATURE_KEY, r"Ambient Temperature ($T_{\text{amb}}$)"),
    (TEMP_DELTA_KEY, r"Thermal Delta ($\Delta T_{\text{cell}}$)"),
    (SOH_KEY, r"State of Health ($\text{SoH}$)"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX feature statistics table."
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

    statistics = StatisticsCalculator(save_path=None)
    stats = statistics.compute(instances)

    lines = [
        r"\begin{table}[htbp]",
        r"    \caption{FEATURE STATISTICS}",
        r"    \label{tab:feature_stats}",
        r"    \begin{center}",
        r"        \begin{tabular}{lcc}",
        r"            \hline",
        r"            \textbf{Feature}                         & \textbf{Mean ($\mu$)} & \textbf{Std. Dev. ($\sigma$)} \\",
        r"            \hline",
    ]

    for key, display_name in _FEATURE_ORDER:
        s = stats.get(key)
        if s is None:
            continue
        lines.append(
            f"            {display_name:<40} & "
            f"{s['mean']:.4f}                & "
            f"{s['standard_deviation']:.4f}                        \\\\"
        )

    lines.append(r"            \hline")
    lines.append(r"        \end{tabular}")
    lines.append(r"    \end{center}")
    lines.append(r"\end{table}")

    tex_path = PROJECT_ROOT / "conference" / "feature_stats.tex"
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"LaTeX table saved -> {tex_path}")


if __name__ == "__main__":
    main()
