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
    CONV_BASE_CHANNELS,
    CONV_CHANNEL_MULTS,
    CONV_KERNEL_SIZE,
    GENERATOR_CHECKPOINT_PATH,
    GENERATOR_STATS_PATH,
    HDF_ROOT,
    LATENT_DIM,
    LATENT_LENGTH,
    PADDED_LENGTH,
    PLOTS_PATH,
)
from voltgan.data.instance import DischargeInstance
from voltgan.models import BatterySequenceGenerator
from voltgan.utils.box_table import print_box_table
from voltgan.utils.discover import discover

_REF_TEMP = (23.0, 27.0)
_MODERATE_TEMP = (-10.0, 10.0)


def _load_model(device: str) -> torch.nn.Module:
    if not GENERATOR_CHECKPOINT_PATH.exists():
        raise ValueError("No generator checkpoint found.")

    model = BatterySequenceGenerator(
        padded_length=PADDED_LENGTH,
        latent_length=LATENT_LENGTH,
        latent_dim=LATENT_DIM,
        conv_base_channels=CONV_BASE_CHANNELS,
        conv_channel_mults=CONV_CHANNEL_MULTS,
        conv_kernel_size=CONV_KERNEL_SIZE,
    ).to(device)

    state_dict = torch.load(GENERATOR_CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded weights from {GENERATOR_CHECKPOINT_PATH}")

    model.eval()
    return model


def _standardize(item, s: dict) -> np.ndarray:
    return (item - s["mean"]) / s["standard_deviation"]


def _destandardize(arr: np.ndarray, s: dict) -> np.ndarray:
    return arr * s["standard_deviation"] + s["mean"]


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


@torch.no_grad()
def _run_inference(
    model: torch.nn.Module,
    voltage_std: np.ndarray,
    temp_delta_std: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    y_input = np.stack([voltage_std, temp_delta_std], axis=1).astype(np.float32)
    y_tensor = torch.tensor(y_input, dtype=torch.float32).unsqueeze(0).to(device)

    pred_tensor = model(y_tensor)
    n = y_input.shape[0]
    pred = pred_tensor.squeeze(0).cpu().numpy()[:n]

    voltage_pred_std = pred[:, 0]
    temp_pred_std = pred[:, 1]
    return voltage_pred_std, temp_pred_std


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

        current_std = _standardize(raw[:, 1], stats["I"])
        voltage_std = _standardize(raw[:, 0], stats["U"])
        thermal_rise = raw[:, 2] - instance.ambient_temperature
        temp_delta_std = _standardize(thermal_rise, stats["temp_delta"])
        ambient_std = float(
            _standardize(instance.ambient_temperature, stats["ambient_temperature"])
        )
        soh_std = float(_standardize(instance.soh, stats["soh"]))

        voltage_pred_std, temp_delta_pred_std = _run_inference(
            model, voltage_std, temp_delta_std, device
        )

        voltage_true = _destandardize(voltage_std, stats["U"])
        voltage_pred = _destandardize(voltage_pred_std, stats["U"])
        temp_true = raw[:, 2]
        temp_delta_pred = _destandardize(temp_delta_pred_std, stats["temp_delta"])
        temp_pred = temp_delta_pred + instance.ambient_temperature

        u_rmse = float(np.sqrt(mean_squared_error(voltage_true, voltage_pred)))
        u_mae = float(mean_absolute_error(voltage_true, voltage_pred))
        u_max_ae = float(np.max(np.abs(voltage_true - voltage_pred)))
        u_r2 = float(r2_score(voltage_true, voltage_pred))

        t_rmse = float(np.sqrt(mean_squared_error(temp_true, temp_pred)))
        t_mae = float(mean_absolute_error(temp_true, temp_pred))
        t_max_ae = float(np.max(np.abs(temp_true - temp_pred)))
        t_r2 = float(r2_score(temp_true, temp_pred))

        results.append(
            {
                "mcu": mcu,
                "filename": str(hdf_path.relative_to(HDF_ROOT)),
                "n_samples": instance.n_samples,
                "ambient_temperature": float(instance.ambient_temperature),
                "actual_soh": float(instance.soh),
                "U_rmse": u_rmse,
                "U_mae": u_mae,
                "U_max_ae": u_max_ae,
                "U_r2": u_r2,
                "T_rmse": t_rmse,
                "T_mae": t_mae,
                "T_max_ae": t_max_ae,
                "T_r2": t_r2,
                "temp_label": _temp_label(float(instance.ambient_temperature)),
                "soh_band": _soh_band_label(float(instance.soh)),
            }
        )

    return results


def _group_metrics(group: list[dict], prefix: str) -> dict:
    rmse = float(np.mean([r[f"{prefix}_rmse"] for r in group]))
    mae = float(np.mean([r[f"{prefix}_mae"] for r in group]))
    max_ae = float(np.mean([r[f"{prefix}_max_ae"] for r in group]))
    r2 = float(np.mean([r[f"{prefix}_r2"] for r in group]))

    return {
        "files": len(group),
        "rmse": rmse,
        "mae": mae,
        "max_ae": max_ae,
        "r2": r2,
    }


_VOLTAGE_HEADERS = ["Group", "Files", "RMSE", "MAE", "MaxAE", "R2"]
_TEMP_HEADERS = ["Group", "Files", "RMSE", "MAE", "MaxAE", "R2"]
_TABLE_ALIGNMENTS = ["left", "right", "right", "right", "right", "right"]


def _group_row(label: str, m: dict) -> list[str]:
    return [
        label,
        str(m["files"]),
        f"{m['rmse']:.5f}",
        f"{m['mae']:.5f}",
        f"{m['max_ae']:.5f}",
        f"{m['r2']:.4f}",
    ]


def _print_group_table(
    title: str,
    prefix: str,
    order: list[str],
    groups: dict[str, list[dict]],
) -> dict[str, dict]:
    print(f"\n== {title} ==")

    rows: list[list[str]] = []
    out: dict[str, dict] = {}
    for label in order:
        group = groups.get(label, [])
        if not group:
            continue
        m = _group_metrics(group, prefix)
        out[label] = m
        rows.append(_group_row(label, m))

    if rows:
        headers = _VOLTAGE_HEADERS
        print_box_table(headers, rows, alignments=_TABLE_ALIGNMENTS)

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

    print("\n" + "=" * 60)
    print("VOLTAGE PREDICTION METRICS")
    print("=" * 60)
    v_by_temp = _print_group_table(
        "Voltage by temperature", "U", temp_order, temp_groups
    )
    v_by_soh = _print_group_table("Voltage by SoH band", "U", soh_order, soh_groups)
    v_overall = _group_metrics(results, "U")
    print("\n== Voltage overall ==")
    print_box_table(
        _VOLTAGE_HEADERS[1:],
        [],
        alignments=_TABLE_ALIGNMENTS[1:],
        footer_row=_group_row("ALL", v_overall)[1:],
    )

    print("\n" + "=" * 60)
    print("TEMPERATURE PREDICTION METRICS")
    print("=" * 60)
    t_by_temp = _print_group_table(
        "Temperature by temperature", "T", temp_order, temp_groups
    )
    t_by_soh = _print_group_table("Temperature by SoH band", "T", soh_order, soh_groups)
    t_overall = _group_metrics(results, "T")
    print("\n== Temperature overall ==")
    print_box_table(
        _TEMP_HEADERS[1:],
        [],
        alignments=_TABLE_ALIGNMENTS[1:],
        footer_row=_group_row("ALL", t_overall)[1:],
    )

    return {
        "mcus": mcu_tag.split("_"),
        "n_files": len(results),
        "voltage": {
            "overall": v_overall,
            "by_temperature": v_by_temp,
            "by_soh_band": v_by_soh,
        },
        "temperature": {
            "overall": t_overall,
            "by_temperature": t_by_temp,
            "by_soh_band": t_by_soh,
        },
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

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), layout="constrained")

    for ax, signal, label in [
        (axes[0], "U", "Voltage (V)"),
        (axes[1], "T", "Temperature (C)"),
    ]:
        rmse_vals = [r[f"{signal}_rmse"] for r in results]
        colors = [r["ambient_temperature"] for r in results]

        sc = ax.scatter(
            range(len(rmse_vals)),
            rmse_vals,
            c=colors,
            cmap="coolwarm",
            s=40,
            alpha=0.8,
        )
        ax.set_xlabel("File index")
        ax.set_ylabel(f"{signal} RMSE")
        ax.set_title(f"{label} RMSE per file")
        ax.grid(True, alpha=0.3)

    plt.colorbar(sc, ax=axes, label="Ambient temperature (C)", shrink=0.8)

    mcus = sorted(set(r["mcu"] for r in results))
    fig.suptitle(f"Generator prediction error ({', '.join(mcus)})", fontsize=12)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Scatter plot saved -> {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the battery sequence generator across MCU files."
    )
    parser.add_argument(
        "--mcus",
        nargs="+",
        required=True,
        help="MCUs to evaluate, e.g. --mcus mcu7",
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

    if not GENERATOR_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Generator stats file not found at {GENERATOR_STATS_PATH}. "
            "Run train-generator first."
        )
    with open(GENERATOR_STATS_PATH) as f:
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
    _write_csv(all_results, PLOTS_PATH / f"evaluation_generator_{mcu_tag}.csv")
    _plot_scatter(all_results, PLOTS_PATH / f"eval_generator_scatter_{mcu_tag}.png")
    _write_metrics_json(metrics, PLOTS_PATH / f"metrics_generator_{mcu_tag}.json")


if __name__ == "__main__":
    main()

