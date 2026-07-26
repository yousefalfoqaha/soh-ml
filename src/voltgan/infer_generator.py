from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

from voltgan.config import (
    AMBIENT_TEMPERATURE_KEY,
    CONV_BASE_CHANNELS,
    CONV_HIDDEN_LAYERS,
    CONV_KERNEL_SIZE,
    CURRENT_CHANNEL,
    GENERATOR_CHECKPOINT_PATH,
    GENERATOR_STATS_PATH,
    HDF_ROOT,
    LATENT_SIZE,
    N_CONDITIONS_GAN,
    NOISE_DIM,
    PLOTS_PATH,
    SOH_KEY,
    TEMP_DELTA_KEY,
    TEMPERATURE_CHANNEL,
    VOLTAGE_CHANNEL,
)
from voltgan.models import Generator

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _read_hdf(
    hdf_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    with h5py.File(hdf_path, "r") as f:
        group = f[hdf_path.name]
        assert isinstance(group, h5py.Group)

        def _load(ch: str) -> np.ndarray:
            dataset = group[ch]
            assert isinstance(dataset, h5py.Dataset)
            return dataset[:]

        current = _load(CURRENT_CHANNEL)
        voltage = _load(VOLTAGE_CHANNEL)
        temperature = _load(TEMPERATURE_CHANNEL)

        soh = float(f.attrs.get("curve_soh", 1.0))
        ambient_temperature = float(f.attrs.get(AMBIENT_TEMPERATURE_KEY, 25.0))

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
    model = Generator(
        input_features=1,
        n_conditions=N_CONDITIONS_GAN,
        base_channels=CONV_BASE_CHANNELS,
        noise_dim=NOISE_DIM,
        kernel_size=CONV_KERNEL_SIZE,
        latent_size=LATENT_SIZE,
        dropout=0.0,
    ).to(device)

    if GENERATOR_CHECKPOINT_PATH.exists():
        model.load_state_dict(
            torch.load(GENERATOR_CHECKPOINT_PATH, map_location=device)
        )
        print(f"Loaded weights from {GENERATOR_CHECKPOINT_PATH}")
    else:
        raise ValueError(f"No checkpoint found at {GENERATOR_CHECKPOINT_PATH}")

    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    current_std: np.ndarray,
    conditions_std: np.ndarray,
    device: str,
) -> np.ndarray:
    orig_length = current_std.shape[0]

    X_tensor = (
        torch.tensor(current_std, dtype=torch.float32)
        .unsqueeze(-1)
        .unsqueeze(0)
        .to(device)
    )
    conditions_tensor = torch.tensor(conditions_std, dtype=torch.float32).to(device)

    downsample_factor = 5**CONV_HIDDEN_LAYERS
    remainder = X_tensor.size(1) % downsample_factor
    if remainder != 0:
        pad_len = downsample_factor - remainder
        X_tensor = F.pad(X_tensor, (0, 0, 0, pad_len), value=0.0)

    noise = torch.rand(1, NOISE_DIM, device=device)
    y_hat = model(X_tensor, conditions_tensor, noise)

    return y_hat.squeeze(0).cpu().numpy()[:orig_length]


def _plot(
    y_true: np.ndarray,
    y_prediction: np.ndarray,
    stats: dict,
    ambient_temperature: float,
    stem: str,
    soh_condition: float,
    ambient_condition: float,
) -> None:
    voltage_true = _destandardize(y_true[:, 0], stats[VOLTAGE_CHANNEL])
    temperature_true = (
        _destandardize(y_true[:, 1], stats[TEMP_DELTA_KEY]) + ambient_temperature
    )

    voltage_prediction = _destandardize(y_prediction[:, 0], stats[VOLTAGE_CHANNEL])
    temperature_prediction = (
        _destandardize(y_prediction[:, 1], stats[TEMP_DELTA_KEY]) + ambient_condition
    )

    timesteps = np.arange(len(y_true))

    fig, (ax_1, ax_2) = plt.subplots(2, 1, figsize=(16, 8), layout="constrained")

    ax_1.plot(timesteps, voltage_true, color="black", lw=0.8, label="True U")
    ax_1.plot(
        timesteps,
        voltage_prediction,
        color="red",
        lw=0.8,
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
        alpha=0.8,
        label="Pred Temp",
    )
    ax_2.set_title("Temperature")
    ax_2.set_ylabel("Temperature (°C)")
    ax_2.set_xlabel("Time steps (0.1 s)")
    ax_2.legend(fontsize=9)
    ax_2.grid(True, alpha=0.3)

    fig.suptitle(
        f"{stem}  |  steps={len(y_true):,}  |  "
        f"conditions: SoH={soh_condition:.4f}, amb={ambient_condition:.2f}°C",
        fontsize=11,
    )

    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    out = PLOTS_PATH / "infer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Generator inference on a single HDF file."
    )
    parser.add_argument(
        "--hdf",
        type=Path,
        required=True,
        help='Path relative to dataset/hdf/, e.g. "mcu1/aging/sample01/...hdf"',
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda' or 'cpu'. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--ambient-temperature",
        type=float,
        default=None,
        help="Override the ambient temperature (°C) used as the conditioning value.",
    )
    parser.add_argument(
        "--soh",
        type=float,
        default=None,
        help="Override the SoH used as the conditioning value.",
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

    if not GENERATOR_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Stats file not found at {GENERATOR_STATS_PATH}. Run training first."
        )
    with open(GENERATOR_STATS_PATH) as f:
        stats = json.load(f)
    print(f"Stats loaded from {GENERATOR_STATS_PATH}")

    current_std = _standardize(current, stats[CURRENT_CHANNEL])
    voltage_std = _standardize(voltage, stats[VOLTAGE_CHANNEL])
    thermal_rise = temperature - ambient_temperature
    temp_delta_std = _standardize(thermal_rise, stats[TEMP_DELTA_KEY])

    y_true_std = np.stack([voltage_std, temp_delta_std], axis=1).astype(np.float32)

    model = _load_model(device)

    soh_condition = args.soh if args.soh is not None else soh
    ambient_condition = (
        args.ambient_temperature
        if args.ambient_temperature is not None
        else ambient_temperature
    )

    amb_std = (ambient_condition - stats[AMBIENT_TEMPERATURE_KEY]["mean"]) / stats[
        AMBIENT_TEMPERATURE_KEY
    ]["standard_deviation"]
    soh_std = (soh_condition - stats[SOH_KEY]["mean"]) / stats[SOH_KEY][
        "standard_deviation"
    ]
    conditions_std = np.array([[soh_std, amb_std]], dtype=np.float32)

    notes = []
    if args.soh is not None:
        notes.append(f"SoH {soh:.4f} -> {args.soh:.4f}")
    if args.ambient_temperature is not None:
        notes.append(
            f"amb {ambient_temperature:.2f}°C -> {args.ambient_temperature:.2f}°C"
        )
    if notes:
        print(f"Condition overrides: {' | '.join(notes)}")
    print(
        f"Effective inference conditions: SoH={soh_condition:.4f}, amb={ambient_condition:.2f}°C"
    )

    y_prediction_std = run_inference(model, current_std, conditions_std, device)

    stem = hdf_path.stem[:40]
    _plot(
        y_true_std,
        y_prediction_std,
        stats,
        ambient_temperature,
        stem,
        soh_condition=soh_condition,
        ambient_condition=ambient_condition,
    )


if __name__ == "__main__":
    main()
