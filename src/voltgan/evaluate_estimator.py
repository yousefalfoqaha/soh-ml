from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from voltgan.config import (
    BATCH_SIZE,
    ESTIMATOR_BASE_CHANNELS,
    ESTIMATOR_CHECKPOINT_PATH,
    ESTIMATOR_GRU_HIDDEN_SIZE,
    ESTIMATOR_GRU_N_LAYERS,
    ESTIMATOR_INPUT_FEATURES,
    ESTIMATOR_KERNEL_SIZE,
    ESTIMATOR_N_CONDITIONS,
    ESTIMATOR_STRIDE,
    HDF_ROOT,
    PHASE_ORDER,
    PROJECT_ROOT,
    STATS_PATH,
)
from voltgan.data import EstimatorDataset
from voltgan.models import SohEstimator
from voltgan.utils.discover import load_instances

_PROTOCOL_ORDER = ["Constant", "HPPC", "Pulse", "WLTC"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the SoH estimator and generate LaTeX results tables."
    )
    parser.add_argument(
        "--mcus",
        nargs="+",
        required=True,
        help="MCUs to evaluate, e.g. --mcus mcu3 mcu8",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda' or 'cpu'. Auto-detected if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open(STATS_PATH) as f:
        stats = json.load(f)

    model = SohEstimator(
        input_features=ESTIMATOR_INPUT_FEATURES,
        n_conditions=ESTIMATOR_N_CONDITIONS,
        base_channels=ESTIMATOR_BASE_CHANNELS,
        stride=ESTIMATOR_STRIDE,
        kernel_size=ESTIMATOR_KERNEL_SIZE,
        gru_hidden_size=ESTIMATOR_GRU_HIDDEN_SIZE,
        gru_n_layers=ESTIMATOR_GRU_N_LAYERS,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(torch.load(ESTIMATOR_CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"Loaded weights from {ESTIMATOR_CHECKPOINT_PATH}")

    instances = load_instances(HDF_ROOT, args.mcus)
    print(f"Loaded {len(instances)} instances from {args.mcus}")

    dataset = EstimatorDataset(instances, stats)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    all_preds = []
    with torch.no_grad():
        for batch_X, batch_cond, _ in loader:
            preds = model(batch_X.to(device), batch_cond.to(device)).squeeze(-1)
            all_preds.append(preds.cpu().numpy())
    all_preds = np.concatenate(all_preds)

    soh_mean = stats["soh"]["mean"]
    soh_std = stats["soh"]["standard_deviation"]
    preds_destd = all_preds * soh_std + soh_mean

    instance_preds: dict[int, list[float]] = defaultdict(list)
    for i, (inst, _, _) in enumerate(dataset.windows):
        instance_preds[id(inst)].append(float(preds_destd[i]))

    results = []
    for inst in instances:
        preds = instance_preds.get(id(inst))
        if not preds:
            continue
        mean_pred = float(np.mean(preds))
        actual = inst.soh
        phase = inst.phase
        protocol = inst.protocol
        temp_center = int(round(inst.ambient_temperature / 5) * 5)
        results.append(
            {
                "phase": phase,
                "protocol": protocol,
                "temp_center": temp_center,
                "actual_soh": actual,
                "predicted_soh": mean_pred,
                "abs_pct_error": abs(mean_pred - actual) / actual * 100,
            }
        )

    phase_groups: dict[str, list[dict]] = defaultdict(list)
    temp_groups: dict[int, list[dict]] = defaultdict(list)
    protocol_groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        phase_groups[r["phase"]].append(r)
        temp_groups[r["temp_center"]].append(r)
        protocol_groups[r["protocol"]].append(r)

    def _metrics(group: list[dict]) -> dict:
        actual = np.array([r["actual_soh"] for r in group])
        pred = np.array([r["predicted_soh"] for r in group])
        r2 = float(r2_score(actual, pred)) if len(group) >= 2 else None
        temps = [r["temp_center"] for r in group]
        return {
            "cycles": len(group),
            "soh_range": f"${actual.min() * 100:.1f}$--${actual.max() * 100:.1f}$",
            "temp_range": f"${min(temps)}$--${max(temps)}$",
            "rmse": float(np.sqrt(mean_squared_error(actual, pred))),
            "mae": float(mean_absolute_error(actual, pred)),
            "r2": r2,
            "pct_err": float(np.mean([r["abs_pct_error"] for r in group])),
        }

    def _row(label: str, m: dict, bold: bool = False) -> str:
        r2_str = f"{m['r2']:.2f}" if m["r2"] is not None else "--"
        pct_str = f"{m['pct_err']:.1f}" + r"\%"
        soh_str = rf"\textbf{{{m['soh_range']}}}" if bold else m["soh_range"]
        if bold:
            label_str = rf"\textbf{{{label}}}"
            rmse_str = rf"$\mathbf{{{m['rmse']:.3f}}}$"
            mae_str = rf"$\mathbf{{{m['mae']:.3f}}}$"
            r2_full = rf"$\mathbf{{{r2_str}}}$" if r2_str != "--" else "--"
            pct_full = rf"$\mathbf{{{pct_str}}}$"
            cyc_str = rf"\textbf{{{m['cycles']}}}"
        else:
            label_str = label
            rmse_str = f"{m['rmse']:.3f}"
            mae_str = f"{m['mae']:.3f}"
            r2_full = r2_str if r2_str == "--" else f"${r2_str}$"
            pct_full = f"{pct_str}"
            cyc_str = str(m["cycles"])
        return (
            f"{label_str} & {soh_str} & {rmse_str} & {mae_str} & "
            f"{r2_full} & {pct_full} & {cyc_str}" + r" \\"
        )

    def _write_table(
        file_name: str,
        caption: str,
        label: str,
        first_header: str,
        rows: list[str],
        overall_m: dict,
    ) -> None:
        lines = [
            r"\begin{table}[H]",
            rf"    \caption{{{caption}}}",
            rf"    \label{{{label}}}",
            r"    \begin{center}",
            r"        \footnotesize",
            r"        \begin{tabular}{lcccccc}",
            r"            \hline",
            rf"            \textbf{{{first_header}}} & \textbf{{SoH}} & \textbf{{RMSE}} & \textbf{{MAE}} & \textbf{{R\textsuperscript{{2}}}} & \textbf{{\%Err}} & \textbf{{Cycles}} \\",
            r"            \hline",
            *rows,
            r"            \hline",
            _row("Overall", overall_m, bold=True),
            r"            \hline",
            r"        \end{tabular}",
            r"    \end{center}",
            r"\end{table}",
        ]
        tex_path = PROJECT_ROOT / "conference" / f"{file_name}.tex"
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text("\n".join(lines) + "\n")
        print(f"LaTeX table saved -> {tex_path}")

    overall_m = _metrics(results)

    phase_rows = [
        _row(phase, _metrics(phase_groups[phase]))
        for phase in PHASE_ORDER
        if phase_groups.get(phase)
    ]
    _write_table(
        "baseline_results",
        "BASELINE ESTIMATOR PERFORMANCE",
        "tab:baseline_results",
        "Phase",
        phase_rows,
        overall_m,
    )

    def _subheader(text: str) -> str:
        return rf"\multicolumn{{7}}{{c}}{{\textbf{{{text}}}}}" + r" \\"

    temp_rows = [
        _row(rf"${tc}^{{\circ}}\text{{C}}$", _metrics(temp_groups[tc]))
        for tc in sorted(temp_groups.keys())
    ]
    protocol_rows = [
        _row(proto, _metrics(protocol_groups[proto]))
        for proto in _PROTOCOL_ORDER
        if protocol_groups.get(proto)
    ]
    combined_lines = [
        r"\begin{table}[H]",
        r"    \caption{ESTIMATOR PERFORMANCE BY TEMPERATURE AND PROTOCOL}",
        r"    \label{tab:temp_protocol_results}",
        r"    \begin{center}",
        r"        \footnotesize",
        r"        \begin{tabular}{lcccccc}",
        r"            \hline",
        (
            r"            \textbf{Slice} & \textbf{SoH (\%)} & "
            r"\textbf{RMSE} & \textbf{MAE} & \textbf{R\textsuperscript{2}} & "
            r"\textbf{\%Err} & \textbf{Cycles} \\"
        ),
        r"            \hline",
        _subheader("Temperature Bands"),
        *temp_rows,
        r"            \hline",
        _subheader("Discharge Protocols"),
        *protocol_rows,
        r"            \hline",
        _row("Overall", overall_m, bold=True),
        r"            \hline",
        r"        \end{tabular}",
        r"    \end{center}",
        r"\end{table}",
    ]
    combined_tex_path = PROJECT_ROOT / "conference" / "temp_protocol_results.tex"
    combined_tex_path.write_text("\n".join(combined_lines) + "\n")
    print(f"LaTeX table saved -> {combined_tex_path}")


if __name__ == "__main__":
    main()
