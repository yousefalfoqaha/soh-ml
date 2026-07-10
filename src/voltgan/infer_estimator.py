from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch

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
    STATS_PATH,
    WINDOW_SIZE,
)
from voltgan.data import DischargeInstance
from voltgan.models import SohEstimator


def _standardize(item, s: dict) -> np.ndarray:
    return (item - s["mean"]) / s["standard_deviation"]


def _destandardize(arr: np.ndarray, s: dict) -> np.ndarray:
    return arr * s["standard_deviation"] + s["mean"]


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


def _make_windows(x_std: np.ndarray, window_size: int) -> np.ndarray:
    """Slice (seq_len, features) into non-overlapping (n_windows, window_size,
    features) chunks. Any trailing samples that don't fill a full window are
    dropped."""
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
    # (n_windows, window_size, input_features)
    x_tensor = torch.tensor(windows, dtype=torch.float32).to(device)
    conditions_tensor = torch.tensor(conditions, dtype=torch.float32).to(device)

    prediction_tensor = model(x_tensor, conditions_tensor)

    # (n_windows,)
    return prediction_tensor.squeeze(-1).cpu().numpy()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SohEstimator inference on a single HDF file."
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
    instance = DischargeInstance(hdf_path)
    print(
        f"  {instance.n_samples:,} samples  |  Actual SoH={instance.soh:.4f}  |  "
        f"Ambient T={instance.ambient_temperature:.2f} °C"
    )

    if not STATS_PATH.exists():
        raise FileNotFoundError(
            f"Stats file not found at {STATS_PATH}. Run training first to generate it."
        )
    with open(STATS_PATH) as f:
        stats = json.load(f)
    print(f"Stats loaded from {STATS_PATH}")

    # instance.data is (seq_len, 3) as [U, I, Temp[1]]
    raw = instance.data
    voltage_std = _standardize(raw[:, 0], stats["U"])
    current_std = _standardize(raw[:, 1], stats["I"])
    temperature_std = _standardize(raw[:, 2], stats["Temp[1]"])
    ambient_temperature = _standardize(
        instance.ambient_temperature, stats["ambient_temperature"]
    )

    x_std = np.stack([voltage_std, current_std, temperature_std], axis=1).astype(
        np.float32
    )

    conditions_std = np.stack([[ambient_temperature]]).astype(np.float32)

    model = _load_model(device)

    windows = _make_windows(x_std, WINDOW_SIZE)
    n_windows = windows.shape[0]
    n_dropped = x_std.shape[0] - n_windows * WINDOW_SIZE
    print(
        f"Sliced into {n_windows} window(s) of {WINDOW_SIZE:,} samples "
        f"({n_dropped:,} trailing samples dropped)."
    )
    if n_windows == 0:
        raise ValueError(
            f"Sequence length ({x_std.shape[0]:,}) is shorter than WINDOW_SIZE "
            f"({WINDOW_SIZE:,}); no full windows to run inference on."
        )

    predicted_soh_per_window_std = run_inference(model, windows, conditions_std, device)
    predicted_soh_per_window = _destandardize(
        predicted_soh_per_window_std, stats["soh"]
    )

    actual_soh = float(instance.soh)

    print()
    for i, predicted_soh in enumerate(predicted_soh_per_window):
        error = predicted_soh - actual_soh
        print(
            f"Window {i:03d} | Predicted SoH: {predicted_soh:.4f} | "
            f"Actual SoH: {actual_soh:.4f} | "
            f"Error: {error:+.4f} ({abs(error) / actual_soh * 100:.2f}%)"
        )

    mean_predicted_soh = float(np.mean(predicted_soh_per_window))
    std_predicted_soh = float(np.std(predicted_soh_per_window))
    mean_error = mean_predicted_soh - actual_soh

    print()
    print(f"Mean Predicted SoH: {mean_predicted_soh:.4f} (± {std_predicted_soh:.4f})")
    print(f"Actual SoH:         {actual_soh:.4f}")
    print(
        f"Mean Error:         {mean_error:+.4f}  "
        f"({abs(mean_error) / actual_soh * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()
