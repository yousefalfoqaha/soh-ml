from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from voltgan.config import (
    CONV_CHANNELS,
    CONV_KERNEL_SIZES,
    CONV_STRIDES,
    ESTIMATOR_CHECKPOINT_PATH,
    ESTIMATOR_N_CONDITIONS,
    GRU_HIDDEN_SIZE,
    GRU_N_LAYERS,
    HDF_ROOT,
    INPUT_FEATURES,
    PLOTS_PATH,
    STATS_PATH,
    WINDOW_SIZE,
)
from voltgan.data import DischargeInstance
from voltgan.models import SohEstimator
from voltgan.pipeline.base import discover
from voltgan.utils.box_table import print_box_table

_REF_TEMP = (23.0, 27.0)
_MODERATE_TEMP = (-10.0, 10.0)


def _load_model(device: str) -> torch.nn.Module:
    if not ESTIMATOR_CHECKPOINT_PATH.exists():
        raise ValueError("No estimator checkpoint found.")

    model = SohEstimator(
        input_features=INPUT_FEATURES,
        n_conditions=ESTIMATOR_N_CONDITIONS,
        conv_channels=CONV_CHANNELS,
        conv_kernel_sizes=CONV_KERNEL_SIZES,
        conv_strides=CONV_STRIDES,
        gru_hidden_size=GRU_HIDDEN_SIZE,
        gru_n_layers=GRU_N_LAYERS,
        dropout=0.0,
    ).to(device)

    state_dict = torch.load(ESTIMATOR_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded weights from {ESTIMATOR_CHECKPOINT_PATH}")

    model.eval()
    return model


def _standardize(item, s: dict) -> np.ndarray:
    return (item - s["mean"]) / s["standard_deviation"]


def _destandardize(arr: np.ndarray, s: dict) -> np.ndarray:
    return arr * s["standard_deviation"] + s["mean"]


def _make_windows(x_std: np.ndarray, window_size: int) -> np.ndarray:
    n_windows = x_std.shape[0] // window_size
    trimmed = x_std[: n_windows * window_size]
    return trimmed.reshape(n_windows, window_size, x_std.shape[1])


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    windows: np.ndarray,
    conditions: np.ndarray,
    device: str,
) -> np.ndarray:
    x_tensor = torch.tensor(windows, dtype=torch.float32).to(device)
    conditions_tensor = torch.tensor(conditions, dtype=torch.float32).to(device)
    prediction_tensor = model(x_tensor, conditions_tensor)
    return prediction_tensor.squeeze(-1).cpu().numpy()


def _temp_label(ambient: float) -> str:
    if _REF_TEMP[0] <= ambient <= _REF_TEMP[1]:
        return "reference (23-27C)"
    if _MODERATE_TEMP[0] <= ambient <= _MODERATE_TEMP[1]:
        return "moderate (-10 to 10C)"
    return "extreme (<-10 or >10C)"


def _soh_band_label(soh: float) -> str:
    if soh >= 0.85:
        return "SoH > 85%"
    if soh >= 0.75:
        return "SoH 75-85%"
    return "SoH < 75%"


def _evaluate_mcu(
    mcu: str,
    model: torch.nn.Module,
    stats: dict,
    device: str,
) -> list[dict]:
    results: list[dict] = []
    for hdf_path in discover(HDF_ROOT, [mcu], (".hdf",)):
        instance = DischargeInstance(hdf_path)
        raw = instance.data

        voltage_std = _standardize(raw[:, 0], stats["U"])
        current_std = _standardize(raw[:, 1], stats["I"])
        temperature_std = _standardize(raw[:, 2], stats["Temp[1]"])
        ambient_std = _standardize(
            instance.ambient_temperature, stats["ambient_temperature"]
        )

        x_std = np.stack([voltage_std, current_std, temperature_std], axis=1).astype(
            np.float32
        )
        conditions_std = np.full((1, 1), ambient_std, dtype=np.float32)

        windows = _make_windows(x_std, WINDOW_SIZE)
        n_windows = windows.shape[0]
        if n_windows == 0:
            continue

        conditions_tiled = np.tile(conditions_std, (n_windows, 1))

        preds_std = run_inference(model, windows, conditions_tiled, device)
        preds = _destandardize(preds_std, stats["soh"])

        actual_soh = float(instance.soh)
        mean_pred = float(np.mean(preds))
        std_pred = float(np.std(preds))
        mean_error = mean_pred - actual_soh
        pct_error = abs(mean_error) / actual_soh * 100

        results.append(
            {
                "mcu": mcu,
                "filename": str(hdf_path.relative_to(HDF_ROOT)),
                "n_samples": instance.n_samples,
                "n_windows": n_windows,
                "ambient_temperature": float(instance.ambient_temperature),
                "actual_soh": actual_soh,
                "predicted_soh": mean_pred,
                "std_soh": std_pred,
                "error": mean_error,
                "abs_pct_error": pct_error,
                "temp_label": _temp_label(float(instance.ambient_temperature)),
                "soh_band": _soh_band_label(actual_soh),
            }
        )

    return results


def _group_metrics(group: list[dict]) -> dict:
    actual = np.array([r["actual_soh"] for r in group])
    predicted = np.array([r["predicted_soh"] for r in group])
    errors = predicted - actual

    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    mae = float(mean_absolute_error(actual, predicted))
    max_abs_error = float(np.max(np.abs(errors)))
    r2 = float(r2_score(actual, predicted)) if len(group) >= 2 else None
    mean_error = float(np.mean(errors))
    mean_abs_pct_error = float(np.mean([r["abs_pct_error"] for r in group]))
    pred_std = float(np.mean([r["std_soh"] for r in group]))

    return {
        "files": len(group),
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs_error,
        "r2": r2,
        "mean_error": mean_error,
        "mean_abs_pct_error": mean_abs_pct_error,
        "pred_std": pred_std,
    }


_EVAL_HEADERS = ["Group", "Files", "RMSE", "MAE", "MaxAE", "R2", "MeanErr", "%Err", "PredStd"]
_EVAL_ALIGNMENTS = ["left"] + ["right"] * 8


def _group_row(label: str, m: dict) -> list[str]:
    r2_str = f"{m['r2']:.4f}" if m["r2"] is not None else "--"
    return [
        label,
        str(m["files"]),
        f"{m['rmse']:.5f}",
        f"{m['mae']:.5f}",
        f"{m['max_abs_error']:.5f}",
        r2_str,
        f"{m['mean_error']:+.4f}",
        f"{m['mean_abs_pct_error']:.2f}%",
        f"{m['pred_std']:.4f}",
    ]


def _print_group_table(title: str, order: list[str], groups: dict[str, list[dict]]) -> dict[str, dict]:
    print(f"\n== {title} ==")

    rows: list[list[str]] = []
    out: dict[str, dict] = {}
    for label in order:
        group = groups.get(label, [])
        if not group:
            continue
        m = _group_metrics(group)
        out[label] = m
        rows.append(_group_row(label, m))

    if rows:
        print_box_table(_EVAL_HEADERS, rows, alignments=_EVAL_ALIGNMENTS)

    return out


def _print_summary(results: list[dict], mcu_tag: str) -> dict:
    if not results:
        print("No results to summarize.")
        return {}

    temp_order = [
        "reference (23-27C)",
        "moderate (-10 to 10C)",
        "extreme (<-10 or >10C)",
    ]
    soh_order = ["SoH > 85%", "SoH 75-85%", "SoH < 75%"]

    temp_groups: dict[str, list[dict]] = {}
    soh_groups: dict[str, list[dict]] = {}
    for r in results:
        temp_groups.setdefault(r["temp_label"], []).append(r)
        soh_groups.setdefault(r["soh_band"], []).append(r)

    by_temp = _print_group_table("By temperature", temp_order, temp_groups)
    by_soh = _print_group_table("By SoH band", soh_order, soh_groups)

    overall = _group_metrics(results)
    print("\n== Overall ==")
    overall_headers = _EVAL_HEADERS[1:]
    overall_alignments = _EVAL_ALIGNMENTS[1:]
    print_box_table(
        overall_headers,
        [],
        alignments=overall_alignments,
        footer_row=_group_row("ALL", overall)[1:],
    )

    return {
        "mcus": mcu_tag.split("_"),
        "n_files": len(results),
        "overall": overall,
        "by_temperature": by_temp,
        "by_soh_band": by_soh,
    }


def _write_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved -> {path}")


def _write_metrics_json(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics JSON saved -> {path}")


def _plot_scatter(results: list[dict], path: Path) -> None:
    if not results:
        return

    actual = np.array([r["actual_soh"] for r in results])
    predicted = np.array([r["predicted_soh"] for r in results])
    colors = np.array([r["ambient_temperature"] for r in results])

    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")

    lo = min(actual.min(), predicted.min()) - 0.02
    hi = max(actual.max(), predicted.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, label="y = x")

    sc = ax.scatter(actual, predicted, c=colors, cmap="coolwarm", s=40, alpha=0.8)
    plt.colorbar(sc, ax=ax, label="Ambient temperature (C)")

    for r in results:
        ax.errorbar(
            r["actual_soh"],
            r["predicted_soh"],
            yerr=r["std_soh"],
            fmt="none",
            ecolor="gray",
            alpha=0.3,
            capsize=2,
        )

    ax.set_xlabel("Actual SoH")
    ax.set_ylabel("Predicted SoH")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend()

    mcus = sorted(set(r["mcu"] for r in results))
    ax.set_title(f"Predicted vs Actual SoH ({', '.join(mcus)})")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot saved -> {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the SoH estimator across MCU files."
    )
    parser.add_argument(
        "--mcus",
        nargs="+",
        required=True,
        help="MCUs to evaluate, e.g. --mcus mcu6 mcu7",
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

    if not STATS_PATH.exists():
        raise FileNotFoundError(
            f"Stats file not found at {STATS_PATH}. Run training first."
        )
    with open(STATS_PATH) as f:
        stats = json.load(f)

    model = _load_model(device)

    all_results: list[dict] = []
    for mcu in args.mcus:
        print(f"\nEvaluating {mcu}...")
        results = _evaluate_mcu(mcu, model, stats, device)
        print(f"  {len(results)} files evaluated")
        all_results.extend(results)

    mcu_tag = "_".join(args.mcus)
    metrics = _print_summary(all_results, mcu_tag)
    _write_csv(all_results, PLOTS_PATH / f"evaluation_{mcu_tag}.csv")
    _plot_scatter(all_results, PLOTS_PATH / f"eval_scatter_{mcu_tag}.png")
    _write_metrics_json(metrics, PLOTS_PATH / f"metrics_{mcu_tag}.json")


if __name__ == "__main__":
    main()