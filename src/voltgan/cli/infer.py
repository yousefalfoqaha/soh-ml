from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from voltgan.config import (
    GENERATOR_CHECKPOINT_PATH,
    GENERATOR_STATS_PATH,
    HDF_ROOT,
    PLOTS_PATH,
)
from voltgan.evaluation import GeneratorInferer
from voltgan.models import GeneratorClient


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: voltgan infer generator <hdf-rel-path> [sqlite...]")

    hdf_rel = Path(sys.argv[2])
    soh_override: float | None = None
    ambient_override: float | None = None

    rest = sys.argv[3:]
    for tok in rest:
        if tok.startswith("--soh="):
            soh_override = float(tok.split("=", 1)[1])
        elif tok.startswith("--ambient="):
            ambient_override = float(tok.split("=", 1)[1])

    hdf_path = HDF_ROOT / hdf_rel
    if not hdf_path.exists():
        raise FileNotFoundError(f"HDF file not found: {hdf_path}")
    print(f"Reading {hdf_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if not GENERATOR_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Stats file not found at {GENERATOR_STATS_PATH}. Run training first."
        )
    with open(GENERATOR_STATS_PATH) as f:
        stats = json.load(f)

    client = GeneratorClient(device=device, checkpoint_path=GENERATOR_CHECKPOINT_PATH)
    inferer = GeneratorInferer(client=client, stats=stats)

    if soh_override is not None:
        print(f"SoH override: {soh_override:.4f}")
    if ambient_override is not None:
        print(f"Ambient override: {ambient_override:.2f}°C")

    (
        voltage_true,
        voltage_pred,
        temperature_true,
        temperature_pred,
        soh_cond,
        ambient_cond,
    ) = inferer.predict_with_overrides(
        hdf_path,
        soh_override=soh_override,
        ambient_override=ambient_override,
    )
    print(
        f"Effective conditions: SoH={soh_cond:.4f}, amb={ambient_cond:.2f}°C  |  "
        f"{len(voltage_true):,} samples  |  "
        f"U_pred ∈ [{voltage_pred.min():.2f}, {voltage_pred.max():.2f}] V"
    )

    # -- Plot --
    timesteps = np.arange(len(voltage_true))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), layout="constrained")

    ax1.plot(timesteps, voltage_true, color="black", lw=0.8, label="True U")
    ax1.plot(timesteps, voltage_pred, color="red", lw=0.8, alpha=0.8, label="Pred U")
    ax1.set_title("Voltage")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_xlabel("Time steps (0.1 s)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(timesteps, temperature_true, color="darkred", lw=0.8, label="True Temp")
    ax2.plot(
        timesteps, temperature_pred, color="blue", lw=0.8, alpha=0.8, label="Pred Temp"
    )
    ax2.set_title("Temperature")
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_xlabel("Time steps (0.1 s)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"{hdf_path.stem[:40]}  |  steps={len(voltage_true):,}  |  "
        f"conditions: SoH={soh_cond:.4f}, amb={ambient_cond:.2f}°C",
        fontsize=11,
    )

    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    out = PLOTS_PATH / "infer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out}")