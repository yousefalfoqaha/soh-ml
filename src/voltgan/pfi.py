from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from voltgan.config import (
    BATCH_SIZE,
    CONFERENCE_PATH,
    ESTIMATOR_BASE_CHANNELS,
    ESTIMATOR_CHECKPOINT_PATH,
    ESTIMATOR_GRU_HIDDEN_SIZE,
    ESTIMATOR_GRU_N_LAYERS,
    ESTIMATOR_INPUT_FEATURES,
    ESTIMATOR_KERNEL_SIZE,
    ESTIMATOR_N_CONDITIONS,
    ESTIMATOR_STRIDE,
    SOH_KEY,
    STATS_PATH,
    HDF_ROOT,
    RANDOM_SEED,
)
from voltgan.data import EstimatorDataset
from voltgan.models import SohEstimator
from voltgan.utils.discover import load_instances

_PROTOCOL_ORDER = ["Constant", "HPPC", "Pulse", "WLTC"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stratified Permutation Feature Importance for the SoH estimator."
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
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of random shuffles per feature per protocol (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def _materialize_dataset(
    dataset: EstimatorDataset,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[str]]:
    X_list, cond_list, y_list = [], [], []
    window_to_inst: list[int] = []
    window_to_proto: list[str] = []
    for idx, (inst, _, _) in enumerate(dataset.windows):
        X_i, cond_i, y_i = dataset[idx]
        X_list.append(X_i.unsqueeze(0))
        cond_list.append(cond_i.unsqueeze(0))
        y_list.append(y_i.unsqueeze(0))
        window_to_inst.append(id(inst))
        window_to_proto.append(inst.protocol)
    return (
        torch.cat(X_list, dim=0),
        torch.cat(cond_list, dim=0),
        torch.cat(y_list, dim=0),
        window_to_inst,
        window_to_proto,
    )


def _run_inference(
    X: torch.Tensor,
    conditions: torch.Tensor,
    model: torch.nn.Module,
    stats: dict,
    device: str,
) -> np.ndarray:
    soh_mean = stats[SOH_KEY]["mean"]
    soh_std = stats[SOH_KEY]["standard_deviation"]
    all_preds = []
    for i in range(0, len(X), BATCH_SIZE):
        batch_X = X[i : i + BATCH_SIZE].to(device)
        batch_cond = conditions[i : i + BATCH_SIZE].to(device)
        preds = model(batch_X, batch_cond).squeeze(-1)
        all_preds.append(preds.detach().cpu())
    all_preds = torch.cat(all_preds).numpy()
    return all_preds * soh_std + soh_mean


def _aggregate_per_instance(
    preds_destd: np.ndarray, window_to_inst: list[int]
) -> dict[int, float]:
    bucket: dict[int, list[float]] = defaultdict(list)
    for i, inst_id in enumerate(window_to_inst):
        bucket[inst_id].append(float(preds_destd[i]))
    return {inst_id: float(np.mean(vals)) for inst_id, vals in bucket.items()}


def _bucket_metrics(
    inst_pred: dict[int, float], inst_actual: dict[int, float], indices: set[int]
) -> dict:
    actuals, preds = [], []
    for inst_id in indices:
        if inst_id in inst_pred and inst_id in inst_actual:
            actuals.append(inst_actual[inst_id])
            preds.append(inst_pred[inst_id])
    if not actuals:
        return {"cycles": 0, "rmse": None, "mae": None, "pct_err": None}
    actuals_arr = np.array(actuals)
    preds_arr = np.array(preds)
    rmse = float(np.sqrt(mean_squared_error(actuals_arr, preds_arr)))
    mae = float(mean_absolute_error(actuals_arr, preds_arr))
    pct_err = float(np.mean(np.abs(preds_arr - actuals_arr) / actuals_arr * 100))
    return {"cycles": len(actuals), "rmse": rmse, "mae": mae, "pct_err": pct_err}


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
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

    instances = load_instances(HDF_ROOT, args.mcus)
    print(f"Loaded {len(instances)} instances from {args.mcus}")

    dataset = EstimatorDataset(instances, stats)
    X, conditions, y, window_to_inst, window_to_proto = _materialize_dataset(dataset)
    N = len(X)
    print(f"Materialized {N} windows")

    inst_actual = {id(inst): inst.soh for inst in instances}
    protocol_windows: dict[str, list[int]] = defaultdict(list)
    for w_idx, proto in enumerate(window_to_proto):
        protocol_windows[proto].append(w_idx)
    protocol_instance_ids: dict[str, set[int]] = {
        p: {window_to_inst[i] for i in idxs} for p, idxs in protocol_windows.items()
    }

    base_preds = _run_inference(X, conditions, model, stats, device)
    base_inst_pred = _aggregate_per_instance(base_preds, window_to_inst)

    baseline: dict[str, dict] = {}
    for p in _PROTOCOL_ORDER:
        baseline[p] = _bucket_metrics(
            base_inst_pred, inst_actual, protocol_instance_ids.get(p, set())
        )

    all_indices = set(window_to_inst)
    overall_base = _bucket_metrics(base_inst_pred, inst_actual, all_indices)
    print(f"Ran baseline inference ({overall_base['cycles']} instances)")

    feature_specs = [
        ("$V$", "X", (0,)),
        ("$I$", "X", (1,)),
        ("$T$", "X", (2,)),
        ("$T_{amb}$", "cond", ()),
    ]

    pfi_grid: dict[str, dict[str, dict]] = {
        f: {p: {"rmse": [], "mae": []} for p in _PROTOCOL_ORDER}
        for f, _, _ in feature_specs
    }

    for fname, tensor_type, channels in feature_specs:
        for p in _PROTOCOL_ORDER:
            bucket_idx = protocol_windows.get(p, [])
            if not bucket_idx or baseline[p]["rmse"] is None:
                continue
            for _ in range(args.repeats):
                X_perm = X.clone()
                cond_perm = conditions.clone()
                local_perm = torch.randperm(len(bucket_idx))
                if tensor_type == "X":
                    for ch in channels:
                        src = X_perm[bucket_idx, :, ch]
                        X_perm[bucket_idx, :, ch] = src[local_perm]
                else:
                    src = cond_perm[bucket_idx, 0]
                    cond_perm[bucket_idx, 0] = src[local_perm]

                perm_preds = _run_inference(X_perm, cond_perm, model, stats, device)
                perm_inst_pred = _aggregate_per_instance(perm_preds, window_to_inst)
                m = _bucket_metrics(
                    perm_inst_pred, inst_actual, protocol_instance_ids[p]
                )
                pfi_grid[fname][p]["rmse"].append(m["rmse"])
                pfi_grid[fname][p]["mae"].append(m["mae"])

    pfi_summary: list[dict] = []
    for fname, _, _ in feature_specs:
        row = {"feature": fname, "protocols": {}}
        for p in _PROTOCOL_ORDER:
            entry = pfi_grid[fname][p]
            base = baseline[p]
            if not entry["rmse"] or base["rmse"] is None:
                row["protocols"][p] = None
                continue
            rmse_arr = np.array(entry["rmse"])
            mae_arr = np.array(entry["mae"])
            row["protocols"][p] = {
                "perm_rmse": float(rmse_arr.mean()),
                "perm_rmse_std": float(rmse_arr.std(ddof=1)),
                "delta_rmse": float(rmse_arr.mean() - base["rmse"]),
                "delta_rmse_std": float(rmse_arr.std(ddof=1)),
                "perm_mae": float(mae_arr.mean()),
                "delta_mae": float(mae_arr.mean() - base["mae"]),
            }
        valid_means = [
            v["delta_rmse"] for v in row["protocols"].values() if v is not None
        ]
        row["mean_delta_rmse"] = float(np.mean(valid_means)) if valid_means else 0.0
        pfi_summary.append(row)

    pfi_summary.sort(key=lambda r: r["mean_delta_rmse"], reverse=True)

    active_protocols = [p for p in _PROTOCOL_ORDER if protocol_windows.get(p)]
    tex_rows = []
    for row in pfi_summary:
        cells = [row["feature"]]
        best_proto = max(
            (
                (p, row["protocols"][p]["delta_rmse"])
                for p in active_protocols
                if row["protocols"][p] is not None
            ),
            key=lambda kv: kv[1],
            default=(None, None),
        )
        for p in active_protocols:
            v = row["protocols"][p]
            if v is None:
                cells.append("--")
            else:
                s = (
                    f"+{v['delta_rmse']:.4f}"
                    if v["delta_rmse"] >= 0
                    else f"{v['delta_rmse']:.4f}"
                )
                if p == best_proto[0] and v["delta_rmse"] > 0:
                    s = rf"\textbf{{{s}}}"
                cells.append(s)
        tex_rows.append(" & ".join(cells) + r" \\")

    header_cols = " & ".join(
        [r"\textbf{Feature}"] + [rf"\textbf{{{p}}}" for p in active_protocols]
    )
    align_cols = "l" + "r" * len(active_protocols)

    grid_lines = [
        r"\begin{table}[H]",
        r"    \caption{Stratified PFI Results}",
        r"    \label{tab:pfi_results}",
        r"    \begin{center}",
        r"        \footnotesize",
        r"        \begin{tabular}{align_cols}",
        r"            \hline",
        header_cols + r" \\",
        r"            \hline",
        *tex_rows,
        r"            \hline",
        r"        \end{tabular}",
        r"    \end{center}",
        r"\end{table}",
    ]
    grid_path = CONFERENCE_PATH / "pfi_results.tex"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.write_text("\n".join(grid_lines) + "\n")
    print(f"Table saved -> {grid_path}")

    def _fmt(v: float | None, fmt: str = ".4f") -> str:
        return f"{v:{fmt}}" if v is not None else "--"

    def _pct(v: float | None) -> str:
        return f"{v:.1f}\\%" if v is not None else "--"

    base_rows = []
    for p in _PROTOCOL_ORDER:
        if p not in baseline or baseline[p]["cycles"] == 0:
            continue
        m = baseline[p]
        base_rows.append(
            f"{p} & {m['cycles']} & {_fmt(m['rmse'])} & "
            f"{_fmt(m['mae'])} & {_pct(m['pct_err'])} \\\\"
        )
    base_rows.append(
        rf"\textbf{{Overall}} & \textbf{{{overall_base['cycles']}}} & "
        rf"$\mathbf{{{overall_base['rmse']:.4f}}}$ & "
        rf"$\mathbf{{{overall_base['mae']:.4f}}}$ & "
        rf"$\mathbf{{{overall_base['pct_err']:.1f}\%}}$ \\\\"
    )

    base_lines = [
        r"\begin{table}[H]",
        r"    \caption{Per-Protocol Baseline Estimator Performance}",
        r"    \label{tab:pfi_baseline}",
        r"    \begin{center}",
        r"        \footnotesize",
        r"        \begin{tabular}{@{}lcccc@{}}",
        r"            \hline",
        (
            r"            \textbf{Protocol} & \textbf{Cycles} & "
            r"\textbf{RMSE} & \textbf{MAE} & \textbf{\%Err} \\"
        ),
        r"            \hline",
        *base_rows,
        r"            \hline",
        r"        \end{tabular}",
        r"    \end{center}",
        r"\end{table}",
    ]
    base_path = CONFERENCE_PATH / "pfi_baseline.tex"
    base_path.write_text("\n".join(base_lines) + "\n")
    print(f"Table saved -> {base_path}")

    feature_labels = [r["feature"] for r in pfi_summary]
    n_features = len(feature_labels)
    n_protocols = len(active_protocols)
    group_centers = np.arange(n_features)
    bar_width = 0.8 / max(n_protocols, 1)
    cmap = plt.get_cmap("tab10")
    proto_colors = {p: cmap(i) for i, p in enumerate(active_protocols)}

    fig, ax = plt.subplots(figsize=(8.5, 4.5), layout="constrained")
    for j, p in enumerate(active_protocols):
        deltas = []
        errs = []
        for row in pfi_summary:
            v = row["protocols"].get(p)
            if v is None:
                deltas.append(0.0)
                errs.append(0.0)
            else:
                deltas.append(v["delta_rmse"])
                errs.append(v["delta_rmse_std"])
        offsets = (j - (n_protocols - 1) / 2) * bar_width
        ax.barh(
            group_centers + offsets,
            deltas,
            height=bar_width,
            xerr=errs,
            color=proto_colors[p],
            edgecolor="black",
            capsize=2,
            label=p,
        )

    ax.set_yticks(group_centers)
    ax.set_yticklabels(feature_labels)
    ax.set_xlabel(r"$\Delta$RMSE (Permuted within Protocol $-$ Baseline)")
    ax.set_title("Stratified PFI")
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(fontsize=9, loc="best", title="Protocol")

    pdf_path = CONFERENCE_PATH / "pfi_importance.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved -> {pdf_path}")


if __name__ == "__main__":
    main()
