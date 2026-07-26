from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from voltgan.config import CONFERENCE_PATH, CURRENT_CHANNEL

_PROTOCOLS = [
    ("Constant", "ChDch2 2024-10-22_05.31.47 Pulse_Test_SamsungINR2170050E_Cell 5 Zelltester_1.hdf"),
    ("Pulse", "2024-10-24_15.20.25 Pulse_Test_SamsungINR2170050E_Cell 5 Zelltester_1.hdf"),
    ("HPPC", "HPPC 2024-09-15_16.21.33 Pulse_Test_SamsungINR2170050E_Cell 5 Zelltester_1_3.hdf"),
    ("WLTC", "WLTC 2024-09-07 21.45.06 Pulse_Test_SamsungINR2170050E_Cell 5 Zelltester_1.hdf"),
]
_MCU_DIR = Path("/mnt/ssd/datasets/wuppertal/hdf/mcu5")
_MAX_PULSE_STEPS = 1000


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")

    positions = [
        ("Constant", axes[0, 0]),
        ("Pulse", axes[0, 1]),
        ("HPPC", axes[1, 0]),
        ("WLTC", axes[1, 1]),
    ]

    for name, ax in positions:
        filename = dict(_PROTOCOLS)[name]
        path = _MCU_DIR / filename

        with h5py.File(path, "r") as f:
            group = f[filename]
            current = group[CURRENT_CHANNEL][:]

        if name == "Pulse":
            current = current[:_MAX_PULSE_STEPS]

        time = np.arange(len(current))

        ax.plot(time, current, linewidth=0.5, color="black")
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Current (A)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    CONFERENCE_PATH.mkdir(parents=True, exist_ok=True)
    out = CONFERENCE_PATH / "discharge_protocols.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved -> {out}")


if __name__ == "__main__":
    main()