from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from voltgan.models import BatteryEncoderTransformer

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DATA_PATH = _PROJECT_ROOT / "dataset"
_HDF_ROOT = _DATA_PATH / "hdf"
_STATS_PATH = _DATA_PATH / "stats.json"
_CHECKPOINT = _PROJECT_ROOT / "model.pt"
_PLOTS_PATH = _PROJECT_ROOT / "plots"

WINDOW_LENGTH = 8000
STRIDE = 4000

EMBEDDING_DIM = 128
FEEDFORWARD_DIM = 512
N_HEADS = 8
N_BLOCKS = 2


def _read_hdf(
    hdf_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    with h5py.File(hdf_path, "r") as f:
        group = f[hdf_path.name]

        def _load(ch: str) -> np.ndarray:
            return group[ch][:]

        current = _load("I")
        clima_temp = _load("ClimaTemp")
        voltage = _load("U")
        temperature = _load("Temp[1]")
        soh = float(f.attrs.get("soh_file", 1.0))

    return current, clima_temp, voltage, temperature, soh


def _std(arr: np.ndarray, s: dict) -> np.ndarray:
    return (arr - s["mean"]) / s["standard_deviation"]


def _destd(arr: np.ndarray, s: dict) -> np.ndarray:
    return arr * s["standard_deviation"] + s["mean"]


def _make_windows(
    current: np.ndarray,
    clima_temp: np.ndarray,
    voltage: np.ndarray,
    temperature: np.ndarray,
    soh: float,
    stats: dict,
    window_length: int,
    stride: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    i_std = _std(current, stats["I"])
    c_std = _std(clima_temp, stats["ClimaTemp"])
    u_std = _std(voltage, stats["U"])
    t_std = _std(temperature, stats["Temp[1]"])

    X_windows, ic_windows, y_raw_windows = [], [], []

    start = 0
    while start + window_length <= len(current):
        end = start + window_length
        X_windows.append(
            np.stack([i_std[start:end], c_std[start:end]], axis=1).astype(np.float32)
        )
        ic_windows.append(np.array([u_std[start], t_std[start], soh], dtype=np.float32))
        y_raw_windows.append(
            np.stack([voltage[start:end], temperature[start:end]], axis=1)
        )
        start += stride

    return X_windows, ic_windows, y_raw_windows


def _load_model(device: str) -> BatteryEncoderTransformer:
    model = BatteryEncoderTransformer(
        embedding_dim=EMBEDDING_DIM,
        n_heads=N_HEADS,
        n_blocks=N_BLOCKS,
        window_length=WINDOW_LENGTH,
        feedforward_dim=FEEDFORWARD_DIM,
        dropout=0.0,
    ).to(device)

    if _CHECKPOINT.exists():
        model.load_state_dict(torch.load(_CHECKPOINT, map_location=device))
        print(f"Loaded weights from {_CHECKPOINT}")
    else:
        print(f"No checkpoint found at {_CHECKPOINT} – running with random weights.")

    model.eval()
    return model


@torch.no_grad()
def run_inference(
    model: BatteryEncoderTransformer,
    X_windows: list[np.ndarray],
    ic_windows: list[np.ndarray],
    device: str,
) -> list[np.ndarray]:
    predictions = []
    ic = ic_windows[0]
    soh = ic[2]

    for X in X_windows:
        X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
        ic_t = torch.tensor(ic, dtype=torch.float32).unsqueeze(0).to(device)

        # (window_length, 2)
        pred = model(X_t, ic_t).squeeze(0).cpu().numpy()
        predictions.append(pred)

        ic = np.array([pred[-1, 0], pred[-1, 1], soh], dtype=np.float32)

    return predictions


def _plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stats: dict,
    stem: str,
    n_windows: int,
) -> None:
    u_true = y_true[:, 0]
    t_true = y_true[:, 1]
    u_pred = _destd(y_pred[:, 0], stats["U"])
    t_pred = _destd(y_pred[:, 1], stats["Temp[1]"])

    ts = np.arange(len(y_true))

    fig, (ax_u, ax_t) = plt.subplots(2, 1, figsize=(16, 8), layout="constrained")

    ax_u.plot(ts, u_true, color="black", lw=0.8, label="True U")
    ax_u.plot(
        ts, u_pred, color="red", lw=0.8, linestyle="--", alpha=0.8, label="Pred U"
    )
    ax_u.set_title("Voltage")
    ax_u.set_ylabel("Voltage (V)")
    ax_u.set_xlabel("Time steps (0.1 s)")
    ax_u.legend(fontsize=9)
    ax_u.grid(True, alpha=0.3)

    ax_t.plot(ts, t_true, color="darkred", lw=0.8, label="True Temp")
    ax_t.plot(
        ts, t_pred, color="blue", lw=0.8, linestyle="--", alpha=0.8, label="Pred Temp"
    )
    ax_t.set_title("Temperature")
    ax_t.set_ylabel("Temperature (°C)")
    ax_t.set_xlabel("Time steps (0.1 s)")
    ax_t.legend(fontsize=9)
    ax_t.grid(True, alpha=0.3)

    fig.suptitle(
        f"{stem}  |  {n_windows} windows × {WINDOW_LENGTH} steps = {len(y_true):,} total",
        fontsize=11,
    )

    _PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    out = _PLOTS_PATH / f"infer_{stem}.png"
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
    current, clima_temp, voltage, temperature, soh = _read_hdf(hdf_path)
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

    X_windows, ic_windows, y_raw_windows = _make_windows(
        current,
        clima_temp,
        voltage,
        temperature,
        soh,
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
    pred_windows = run_inference(model, X_windows, ic_windows, device)

    # (total_steps, 2)
    y_true = np.concatenate(y_raw_windows, axis=0)
    y_pred = np.concatenate(pred_windows, axis=0)

    stem = hdf_path.stem[:40]
    _plot(y_true, y_pred, stats, stem, n_windows)


if __name__ == "__main__":
    main()
