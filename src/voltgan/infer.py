from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

from voltgan.models.generator_gru import GeneratorGru

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_PATH = _PROJECT_ROOT / "dataset"
_HDF_ROOT = _DATA_PATH / "hdf"
_STATS_PATH = _DATA_PATH / "stats.json"
_CHECKPOINT = _PROJECT_ROOT / "model.pt"
_PLOTS_PATH = _PROJECT_ROOT / "plots"

WINDOW_LENGTH = 500
STRIDE = 500

# gru
INPUT_FEATURES = 3
N_CONDITIONS = 1
HIDDEN_SIZE = 128
OUTPUT_FEATURES = 2
N_LAYERS = 2

NOISE_DIM = 32
CONDITION_DIM = 8


def _read_hdf(
    hdf_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    with h5py.File(hdf_path, "r") as f:
        group = f[hdf_path.name]

        def _load(ch: str) -> np.ndarray:
            return group[ch][:]

        current = _load("I")
        climate_temperature = _load("ClimaTemp")
        charge = _load("Q")
        voltage = _load("U")
        temperature = _load("Temp[1]")
        soh = float(f.attrs.get("soh_file", 1.0))

    return current, climate_temperature, charge, voltage, temperature, soh


def _standardize(arr: np.ndarray, s: dict) -> np.ndarray:
    return (arr - s["mean"]) / s["standard_deviation"]


def _destandardize(arr: np.ndarray, s: dict) -> np.ndarray:
    return arr * s["standard_deviation"] + s["mean"]


def _make_windows(
    current: np.ndarray,
    climate_temperature: np.ndarray,
    charge: np.ndarray,
    voltage: np.ndarray,
    temperature: np.ndarray,
    stats: dict,
    window_length: int,
    stride: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    current = _standardize(current, stats["I"])
    climate_temperature = _standardize(climate_temperature, stats["ClimaTemp"])
    voltage = _standardize(voltage, stats["U"])
    battery_temperature = _standardize(temperature, stats["Temp[1]"])

    X_windows, y_windows = [], []

    start = 0
    while start + window_length <= len(current):
        end = start + window_length
        X_windows.append(
            np.stack(
                [current[start:end], climate_temperature[start:end], charge[start:end]],
                axis=1,
            ).astype(np.float32)
        )
        y_windows.append(
            np.stack(
                [voltage[start:end], battery_temperature[start:end]], axis=1
            ).astype(np.float32)
        )
        start += stride

    return X_windows, y_windows


def _load_model(device: str) -> torch.nn.Module:
    model = GeneratorGru(
        n_conditions=N_CONDITIONS,
        hidden_size=HIDDEN_SIZE,
        output_features=OUTPUT_FEATURES,
        n_layers=N_LAYERS,
        dropout=0.0,
        noise_dim=NOISE_DIM,
        condition_dim=CONDITION_DIM,
    ).to(device)

    if _CHECKPOINT.exists():
        model.load_state_dict(torch.load(_CHECKPOINT, map_location=device))
        print(f"Loaded weights from {_CHECKPOINT}")
    else:
        raise ValueError("No model.pt found.")

    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    y_windows: list[np.ndarray],
    conditions: np.ndarray,
    device: str,
) -> list[np.ndarray]:
    predictions = []
    hidden_state = None

    # (1, conditions_size)
    conditions_tensor = (
        torch.tensor(conditions, dtype=torch.float32).view(1, 1).to(device)
    )

    for i in range(len(y_windows)):
        # (window_length, input_size)
        y = y_windows[i]

        # (1, window_length, input_size)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)

        # 1: (1, window_length, output_size)
        # 2: (n_layers, 1, hidden_size)
        batch_size = y_tensor.size(0)
        sequence_length = y_tensor.size(1)
        noise = torch.randn([batch_size, sequence_length, NOISE_DIM], device=device)
        prediction_tensor, hidden_state = model(conditions_tensor, noise, hidden_state)

        prediction = prediction_tensor.squeeze(0).cpu().numpy()
        predictions.append(prediction)

    return predictions


def _plot(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
    stats: dict,
    stem: str,
    n_windows: int,
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
        linestyle="--",
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
        linestyle="--",
        alpha=0.8,
        label="Pred Temp",
    )
    ax_2.set_title("Temperature")
    ax_2.set_ylabel("Temperature (°C)")
    ax_2.set_xlabel("Time steps (0.1 s)")
    ax_2.legend(fontsize=9)
    ax_2.grid(True, alpha=0.3)

    fig.suptitle(
        f"{stem}  |  {n_windows} windows × {WINDOW_LENGTH} steps = {len(y_true):,} total",
        fontsize=11,
    )

    _PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    out = _PLOTS_PATH / "infer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BatteryEncoderTransformer inference on a single HDF file."
    )
    parser.add_argument(
        "--hdf",
        type=Path,
        required=True,
        help='Path relative to dataset/hdf/, e.g. "mcu1/aging/sample01/2025-02-12_13.11.28 Aging_….hdf"',
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=STRIDE,
        help=f"Stride between windows (default {STRIDE}).",
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

    hdf_path = _HDF_ROOT / args.hdf
    if not hdf_path.exists():
        raise FileNotFoundError(f"HDF file not found: {hdf_path}")

    print(f"Reading {hdf_path} …")
    current, climate_temperature, charge, voltage, temperature, soh = _read_hdf(
        hdf_path
    )
    print(
        f"  {len(current):,} samples  |  SoH={soh:.4f}  |  "
        f"U ∈ [{voltage.min():.2f}, {voltage.max():.2f}] V  |  "
        f"I ∈ [{current.min():.2f}, {current.max():.2f}] A"
    )

    if not _STATS_PATH.exists():
        raise FileNotFoundError(
            f"Stats file not found at {_STATS_PATH}. Run training first to generate it."
        )
    with open(_STATS_PATH) as f:
        stats = json.load(f)
    print(f"Stats loaded from {_STATS_PATH}")

    X_windows, y_windows = _make_windows(
        current,
        climate_temperature,
        charge,
        voltage,
        temperature,
        stats,
        WINDOW_LENGTH,
        args.stride,
    )
    n_windows = len(X_windows)
    print(f"Windows: {n_windows}  (length={WINDOW_LENGTH}, stride={args.stride})")
    if n_windows == 0:
        raise ValueError(
            f"Signal too short ({len(current)} samples) for window_length={WINDOW_LENGTH}."
        )

    model = _load_model(device)
    conditions = np.array(soh)
    pred_windows = run_inference(model, y_windows, conditions, device)

    y_true = np.concatenate(y_windows, axis=0)
    y_prediction = np.concatenate(pred_windows, axis=0)

    stem = hdf_path.stem[:40]
    _plot(y_true, y_prediction, stats, stem, n_windows)


if __name__ == "__main__":
    main()
