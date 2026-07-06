from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

from voltgan.config import (
    GENERATOR_CHECKPOINT_PATH,
    GENERATOR_N_CONDITIONS,
    HDF_ROOT,
    HIDDEN_SIZE,
    INPUT_FEATURES,
    MAX_SEQUENCE_LENGTH,
    N_LAYERS,
    PLOTS_PATH,
    STATS_PATH,
)
from voltgan.models import BatterySequenceGenerator

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _read_hdf(
    hdf_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    with h5py.File(hdf_path, "r") as f:
        group = f[hdf_path.name]
        assert isinstance(group, h5py.Group)

        def _load(ch: str) -> np.ndarray:
            dataset = group[ch]
            assert isinstance(dataset, h5py.Dataset)

            return dataset[:MAX_SEQUENCE_LENGTH]

        current = _load("I")
        voltage = _load("U")
        temperature = _load("Temp[1]")

        soh = float(f.attrs.get("soh_file", 1.0))
        ambient_temperature = float(f.attrs.get("ambient_temperature", 25.0))

    return (
        current,
        voltage,
        temperature,
        soh,
        ambient_temperature,
    )


def _standardize(arr: np.ndarray, s: dict) -> np.ndarray:
    return (arr - s["mean"]) / s["standard_deviation"]


def _destandardize(arr: np.ndarray, s: dict) -> np.ndarray:
    return arr * s["standard_deviation"] + s["mean"]


def _load_model(device: str) -> torch.nn.Module:
    model = BatterySequenceGenerator(
        input_features=INPUT_FEATURES,
        n_conditions=GENERATOR_N_CONDITIONS,
        hidden_size=HIDDEN_SIZE,
        n_layers=N_LAYERS,
        dropout=0.0,
    ).to(device)

    if GENERATOR_CHECKPOINT_PATH.exists():
        model.load_state_dict(
            torch.load(GENERATOR_CHECKPOINT_PATH, map_location=device)
        )
        print(f"Loaded weights from {GENERATOR_CHECKPOINT_PATH}")
    else:
        raise ValueError("No model.pt found.")

    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    current: np.ndarray,
    conditions: np.ndarray,
    device: str,
) -> np.ndarray:

    # (sequence_length, 1) -> (1, sequence_length, 1)
    x_input = np.stack([current], axis=1).astype(np.float32)
    x_tensor = torch.tensor(x_input, dtype=torch.float32).unsqueeze(0).to(device)

    # (1, conditions_size)
    conditions_tensor = (
        torch.tensor(conditions, dtype=torch.float32).view(1, -1).to(device)
    )

    prediction_tensor, _ = model(x_tensor, conditions_tensor, None)

    # (sequence_length, output_features)
    return prediction_tensor.squeeze(0).cpu().numpy()


def _plot(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
    stats: dict,
    stem: str,
) -> None:
    voltage_true = _destandardize(y_true[:, 0], stats["U"])
    temperature_true = _destandardize(y_true[:, 1], stats["Temp[1]"])

    voltage_prediction = _destandardize(y_prediction[:, 0], stats["U"])
    temperature_prediction = _destandardize(y_prediction[:, 1], stats["Temp[1]"])

    timesteps = np.arange(len(y_true))

    fig, (ax_1, ax_2) = plt.subplots(2, 1, figsize=(16, 8), layout="constrained")

    ax_1.plot(timesteps, voltage_true, color="black", lw=0.8, label="True U")
    ax_1.plot(
        timesteps,
        voltage_prediction,
        color="red",
        lw=0.8,
        # linestyle="--",
        alpha=0.8,
        label="Pred U",
    )
    ax_1.set_title("Voltage")
    ax_1.set_ylabel("Voltage (V)")
    ax_1.set_xlabel("Time steps (0.1 s)")
    ax_1.legend(fontsize=9)
    ax_1.grid(True, alpha=0.3)

    ax_2.plot(timesteps, temperature_true, color="darkred", lw=0.8, label="True Temp")
    ax_2.plot(
        timesteps,
        temperature_prediction,
        color="blue",
        lw=0.8,
        # linestyle="--",
        alpha=0.8,
        label="Pred Temp",
    )
    ax_2.set_title("Temperature")
    ax_2.set_ylabel("Temperature (°C)")
    ax_2.set_xlabel("Time steps (0.1 s)")
    ax_2.legend(fontsize=9)
    ax_2.grid(True, alpha=0.3)

    fig.suptitle(
        f"{stem}  |  Total steps = {len(y_true):,}",
        fontsize=11,
    )

    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    out = PLOTS_PATH / "infer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BatterySequenceGenerator inference on a single HDF file."
    )
    parser.add_argument(
        "--hdf",
        type=Path,
        required=True,
        help='Path relative to dataset/hdf/, e.g. "mcu1/aging/sample01/2025-02-12_13.11.28 Aging_….hdf"',
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

    hdf_path = HDF_ROOT / args.hdf
    if not hdf_path.exists():
        raise FileNotFoundError(f"HDF file not found: {hdf_path}")

    print(f"Reading {hdf_path} …")
    (
        current,
        voltage,
        temperature,
        soh,
        ambient_temperature,
    ) = _read_hdf(hdf_path)
    print(
        f"  {len(current):,} samples  |  SoH={soh:.4f}  |  "
        f"U ∈ [{voltage.min():.2f}, {voltage.max():.2f}] V  |  "
        f"I ∈ [{current.min():.2f}, {current.max():.2f}] A"
    )

    if not STATS_PATH.exists():
        raise FileNotFoundError(
            f"Stats file not found at {STATS_PATH}. Run training first to generate it."
        )
    with open(STATS_PATH) as f:
        stats = json.load(f)
    print(f"Stats loaded from {STATS_PATH}")

    current_std = _standardize(current, stats["I"])
    voltage_std = _standardize(voltage, stats["U"])
    temperature_std = _standardize(temperature, stats["Temp[1]"])

    y_true_std = np.stack([voltage_std, temperature_std], axis=1).astype(np.float32)

    model = _load_model(device)

    amb_std = (ambient_temperature - stats["ambient_temperature"]["mean"]) / stats[
        "ambient_temperature"
    ]["standard_deviation"]

    soh_std = (soh - stats["soh"]["mean"]) / stats["soh"]["standard_deviation"]

    conditions = np.array([soh_std, amb_std])
    y_prediction_std = run_inference(model, current_std, conditions, device)

    stem = hdf_path.stem[:40]
    _plot(y_true_std, y_prediction_std, stats, stem)


if __name__ == "__main__":
    main()
